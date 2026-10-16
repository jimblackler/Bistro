#!/usr/bin/env python3
"""Builds the Android-ready Bistro assets from the original ORCA download.

Pipeline:
  1. (optional) Download + extract Bistro_v5_2.zip from NVIDIA ORCA into source_assets/.
  2. Analyse base-colour alpha channels so the converter can classify materials (opaque/mask/blend).
  3. Build and run the C++ scene converter (FBX -> clustered binary scene.bin).
  4. Convert every referenced DDS texture (BC1/BC3/BC5, unsupported on most mobile GPUs) into ASTC
     with a full, properly filtered mip chain (sRGB-correct, coverage-preserving for alpha-tested
     foliage, renormalised for normal maps).
  5. Pre-process the HDR environment map: extract the sun, build a sky texture, a GGX pre-filtered
     specular environment and SH9 diffuse irradiance coefficients.

Usage:  .venv/bin/python tools/build_assets.py [--download] [--max-size 1024] [--jobs N]
Requires: numpy, pillow, assimp (brew), clang++, astcenc (tools/bin or PATH).
"""

import argparse
import math
import multiprocessing
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdr_io import read_hdr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(ROOT, 'source_assets', 'Bistro_v5_2')
WORK_DIR = os.path.join(ROOT, 'build_assets')
OUT_DIR = os.path.join(ROOT, 'app', 'src', 'main', 'assets', 'bistro')
TEX_OUT_DIR = os.path.join(OUT_DIR, 'textures')
DOWNLOAD_URL = 'https://developer.nvidia.com/bistro'

KIND_BASECOLOR, KIND_NORMAL, KIND_ORM, KIND_EMISSIVE = 0, 1, 2, 3
FORMAT_ASTC, FORMAT_RGBA16F = 0, 1

# Per-kind settings: (max dimension scale relative to --max-size, ASTC block, sRGB)
KIND_SETTINGS = {
    KIND_BASECOLOR: (1.0, (6, 6), True),
    KIND_NORMAL: (1.0, (6, 6), False),
    KIND_ORM: (0.5, (8, 8), False),
    KIND_EMISSIVE: (0.5, (6, 6), True),
}


def log(msg):
    print(msg, flush=True)


def find_astcenc():
    local = os.path.join(ROOT, 'tools', 'bin', 'astcenc')
    if os.path.exists(local):
        return local
    for name in ('astcenc', 'astcenc-neon', 'astcenc-avx2', 'astcenc-sse4.1'):
        path = shutil.which(name)
        if path:
            return path
    sys.exit('astcenc not found. Download it from https://github.com/ARM-software/astc-encoder/releases '
             'and place it in tools/bin/.')


# --------------------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------------------

def download_source():
    zip_path = os.path.join(ROOT, 'source_assets', 'Bistro_v5_2.zip')
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    if not os.path.exists(zip_path):
        log('Downloading %s ...' % DOWNLOAD_URL)
        urllib.request.urlretrieve(DOWNLOAD_URL, zip_path)
    if not os.path.isdir(SOURCE_DIR):
        log('Extracting %s ...' % zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(os.path.dirname(SOURCE_DIR))


# --------------------------------------------------------------------------------------------------
# Colour helpers
# --------------------------------------------------------------------------------------------------

def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def downsample(img):
    """2x2 box filter (handles odd sizes by edge replication)."""
    h, w = img.shape[:2]
    if h % 2:
        img = np.concatenate([img, img[-1:]], axis=0)
    if w % 2:
        img = np.concatenate([img, img[:, -1:]], axis=1)
    h2, w2 = img.shape[0] // 2, img.shape[1] // 2
    out = img.reshape(h2, 2, w2, 2, *img.shape[2:]).mean(axis=(1, 3))
    return out


def resize_to(img, max_dim):
    """Repeated box-downsampling until the image fits within max_dim."""
    while max(img.shape[0], img.shape[1]) > max_dim:
        img = downsample(img)
    return img


def load_dds(path):
    im = Image.open(path)
    im.load()
    return np.asarray(im.convert('RGBA'), dtype=np.float32) / 255.0


# --------------------------------------------------------------------------------------------------
# Alpha analysis
# --------------------------------------------------------------------------------------------------

def texture_stats_worker(name):
    path = os.path.join(SOURCE_DIR, 'Textures', name + '.dds')
    im = Image.open(path)
    im.load()
    px = np.asarray(im.convert('RGBA'))
    a = px[:, :, 3]
    cut = float((a < 128).mean())
    translucent = float(((a > 13) & (a < 242)).mean())
    rgb = px[:, :, :3].reshape(-1, 3).astype(np.float32) / 255.0
    mean = rgb.mean(axis=0)
    std = float(rgb.std(axis=0).max())
    return name, cut, translucent, mean, std


def compute_texture_stats(jobs):
    """Per-texture statistics used by the converter: alpha coverage (material classification) and
    mean/stddev (constant ORM textures are folded into material factors)."""
    out_path = os.path.join(WORK_DIR, 'texture_stats.txt')
    if os.path.exists(out_path):
        return out_path
    names = sorted(f[:-4] for f in os.listdir(os.path.join(SOURCE_DIR, 'Textures'))
                   if f.endswith('_BaseColor.dds') or f.endswith('_Specular.dds'))
    log('Analysing %d base colour / specular textures...' % len(names))
    with multiprocessing.Pool(jobs) as pool:
        results = pool.map(texture_stats_worker, names)
    with open(out_path, 'w') as f:
        for name, cut, translucent, mean, std in results:
            f.write('%s %.6f %.6f %.5f %.5f %.5f %.5f\n' % (name, cut, translucent, mean[0], mean[1], mean[2], std))
    return out_path


# --------------------------------------------------------------------------------------------------
# Scene conversion
# --------------------------------------------------------------------------------------------------

def build_converter():
    src = os.path.join(ROOT, 'tools', 'scene_converter', 'convert.cpp')
    header = os.path.join(ROOT, 'app', 'src', 'main', 'cpp', 'scene_format.h')
    exe = os.path.join(WORK_DIR, 'bistro_convert')
    if os.path.exists(exe) and os.path.getmtime(exe) > max(os.path.getmtime(src), os.path.getmtime(header)):
        return exe
    prefix = subprocess.run(['brew', '--prefix'], capture_output=True, text=True).stdout.strip() or '/usr/local'
    cmd = ['clang++', '-std=c++17', '-O2', '-Wall', '-I' + prefix + '/include', src, '-o', exe,
           '-L' + prefix + '/lib', '-lassimp']
    log('Building converter: ' + ' '.join(cmd))
    subprocess.check_call(cmd)
    return exe


def run_converter(alpha_stats):
    exe = build_converter()
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = [exe, OUT_DIR, WORK_DIR, alpha_stats,
           os.path.join(SOURCE_DIR, 'BistroExterior.fbx'),
           os.path.join(SOURCE_DIR, 'BistroInterior.fbx')]
    log('Running converter...')
    subprocess.check_call(cmd)
    textures = []
    with open(os.path.join(WORK_DIR, 'texture_list.txt')) as f:
        for line in f:
            name, kind = line.split()
            textures.append((name, int(kind)))
    return textures


# --------------------------------------------------------------------------------------------------
# Texture conversion
# --------------------------------------------------------------------------------------------------

def coverage(alpha, cutoff=0.5):
    return float((alpha >= cutoff).mean())


def preserve_coverage(alpha, target, cutoff=0.5):
    """Scales alpha so that the fraction of texels passing the alpha test matches 'target'."""
    if target <= 0.0 or target >= 1.0:
        return alpha
    lo, hi = 0.0, 8.0
    for _ in range(16):
        mid = (lo + hi) / 2
        if coverage(np.clip(alpha * mid, 0, 1), cutoff) < target:
            lo = mid
        else:
            hi = mid
    return np.clip(alpha * hi, 0, 1)


def build_mips(img, kind, alpha_tested):
    """Returns a list of uint8 RGBA mip levels."""
    levels = []
    if kind in (KIND_BASECOLOR, KIND_EMISSIVE):
        lin = img.copy()
        lin[:, :, :3] = srgb_to_linear(img[:, :, :3])
        target = coverage(img[:, :, 3]) if alpha_tested else None
        cur = lin
        while True:
            out = cur.copy()
            out[:, :, :3] = linear_to_srgb(cur[:, :, :3])
            if alpha_tested and len(levels) > 0:
                out[:, :, 3] = preserve_coverage(cur[:, :, 3], target)
            levels.append(out)
            if cur.shape[0] == 1 and cur.shape[1] == 1:
                break
            cur = downsample(cur)
    elif kind == KIND_NORMAL:
        # BC5 stores X,Y; reconstruct Z, filter as vectors and renormalise.
        x = img[:, :, 0] * 2 - 1
        y = img[:, :, 1] * 2 - 1
        z = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
        cur = np.stack([x, y, z], axis=-1)
        while True:
            n = cur / np.maximum(np.linalg.norm(cur, axis=-1, keepdims=True), 1e-6)
            out = np.ones(cur.shape[:2] + (4,), dtype=np.float32)
            out[:, :, :3] = n * 0.5 + 0.5
            levels.append(out)
            if cur.shape[0] == 1 and cur.shape[1] == 1:
                break
            cur = downsample(cur)
    else:
        cur = img
        while True:
            levels.append(cur)
            if cur.shape[0] == 1 and cur.shape[1] == 1:
                break
            cur = downsample(cur)
    return [np.clip(np.rint(l * 255), 0, 255).astype(np.uint8) for l in levels]


def write_tex(path, fmt, width, height, block, srgb, payloads):
    header_size = 32 + 8 * len(payloads)
    offset = (header_size + 15) & ~15
    level_table = []
    for p in payloads:
        level_table.append((offset, len(p)))
        offset = (offset + len(p) + 15) & ~15
    with open(path + '.tmp', 'wb') as f:
        f.write(b'BTEX')
        f.write(struct.pack('<7I', fmt, width, height, len(payloads), block[0], block[1], 1 if srgb else 0))
        for off, size in level_table:
            f.write(struct.pack('<2I', off, size))
        for (off, size), p in zip(level_table, payloads):
            f.write(b'\0' * (off - f.tell()))
            f.write(p)
    os.replace(path + '.tmp', path)


def texture_worker(args):
    name, kind, max_size, astcenc, mask_names = args
    src = os.path.join(SOURCE_DIR, 'Textures', name + '.dds')
    dst = os.path.join(TEX_OUT_DIR, name + '.tex')
    if os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(src):
        return name, os.path.getsize(dst), True
    scale, block, srgb = KIND_SETTINGS[kind]
    img = load_dds(src)
    if kind == KIND_NORMAL and img.shape[0] <= 4:
        pass
    img = resize_to(img, max(4, int(max_size * scale)))
    if max(img.shape[:2]) <= 64:
        block = (4, 4)
    levels = build_mips(img, kind, name in mask_names)
    payloads = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, lvl in enumerate(levels):
            png = os.path.join(tmp, '%d.png' % i)
            out = os.path.join(tmp, '%d.astc' % i)
            Image.fromarray(lvl, 'RGBA').save(png, compress_level=1)
            cmd = [astcenc, '-cs' if srgb else '-cl', png, out, '%dx%d' % block, '-medium', '-silent']
            if kind == KIND_NORMAL:
                cmd.append('-normal')
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            with open(out, 'rb') as f:
                data = f.read()
            payloads.append(data[16:])  # strip the 16-byte .astc header
    write_tex(dst, FORMAT_ASTC, img.shape[1], img.shape[0], block, srgb, payloads)
    return name, os.path.getsize(dst), False


def convert_textures(textures, max_size, jobs, alpha_stats_path):
    os.makedirs(TEX_OUT_DIR, exist_ok=True)
    astcenc = find_astcenc()
    mask_names = set()
    with open(alpha_stats_path) as f:
        for line in f:
            fields = line.split()
            name, cut = fields[0], fields[1]
            if name.endswith('_BaseColor') and float(cut) > 0.002:
                mask_names.add(name)
    wanted = {name + '.tex' for name, _ in textures}
    for f in os.listdir(TEX_OUT_DIR):
        if f not in wanted:
            os.remove(os.path.join(TEX_OUT_DIR, f))
    log('Converting %d textures to ASTC with %d jobs...' % (len(textures), jobs))
    start = time.time()
    total = 0
    work = [(name, kind, max_size, astcenc, mask_names) for name, kind in textures]
    with multiprocessing.Pool(jobs) as pool:
        for i, (name, size, cached) in enumerate(pool.imap_unordered(texture_worker, work)):
            total += size
            if not cached:
                log('  [%d/%d] %s (%d KB)' % (i + 1, len(work), name, size // 1024))
    log('Textures done in %.1fs, total %.1f MB' % (time.time() - start, total / 1e6))


# --------------------------------------------------------------------------------------------------
# Environment map
# --------------------------------------------------------------------------------------------------
# Equirectangular convention (must match shaders/common.glsl):
#   u = atan2(d.x, -d.z) / (2 pi) + 0.5,  v = acos(d.y) / pi

def equirect_dirs(w, h):
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    phi = (u - 0.5) * 2 * np.pi
    theta = v * np.pi
    st = np.sin(theta)[:, None]
    d = np.stack([st * np.sin(phi)[None, :], np.repeat(np.cos(theta)[:, None], w, axis=1),
                  -st * np.cos(phi)[None, :]], axis=-1)
    return d.astype(np.float32)


def solid_angles(w, h):
    v = (np.arange(h) + 0.5) / h
    return (np.sin(v * np.pi) * (2 * np.pi / w) * (np.pi / h)).astype(np.float32)[:, None].repeat(w, axis=1)


def dir_to_uv(d):
    u = np.arctan2(d[..., 0], -d[..., 2]) / (2 * np.pi) + 0.5
    v = np.arccos(np.clip(d[..., 1], -1, 1)) / np.pi
    return u, v


def sample_bilinear(img, u, v):
    h, w = img.shape[:2]
    x = u * w - 0.5
    y = np.clip(v * h - 0.5, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x0 %= w
    x1 = (x0 + 1) % w
    y1 = np.minimum(y0 + 1, h - 1)
    return ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) +
            (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)


def hammersley(n):
    i = np.arange(n, dtype=np.uint32)
    b = i.copy()
    b = ((b << 16) | (b >> 16)) & 0xFFFFFFFF
    b = ((b & 0x55555555) << 1) | ((b & 0xAAAAAAAA) >> 1)
    b = ((b & 0x33333333) << 2) | ((b & 0xCCCCCCCC) >> 2)
    b = ((b & 0x0F0F0F0F) << 4) | ((b & 0xF0F0F0F0) >> 4)
    b = ((b & 0x00FF00FF) << 8) | ((b & 0xFF00FF00) >> 8)
    return (i + 0.5) / n, b.astype(np.float64) / 2 ** 32


def prefilter_level(pyramid, w, h, roughness, samples=96):
    n = equirect_dirs(w, h).reshape(-1, 3).astype(np.float64)
    if roughness <= 0.0:
        u, v = dir_to_uv(n)
        return sample_bilinear(pyramid[0], u, v).reshape(h, w, 3)
    a = roughness * roughness
    xi1, xi2 = hammersley(samples)
    phi = 2 * np.pi * xi1
    cos_t = np.sqrt((1 - xi2) / (1 + (a * a - 1) * xi2))
    sin_t = np.sqrt(1 - cos_t * cos_t)
    hl = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=-1)  # tangent space
    up = np.where(np.abs(n[:, 2:3]) < 0.999, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    tx = np.cross(up, n)
    tx /= np.linalg.norm(tx, axis=1, keepdims=True)
    ty = np.cross(n, tx)
    base_w = pyramid[0].shape[1]
    base_h = pyramid[0].shape[0]
    texel_sa = 4 * np.pi / (base_w * base_h)
    acc = np.zeros((n.shape[0], 3))
    wsum = np.zeros((n.shape[0], 1))
    for s in range(samples):
        hv = tx * hl[s, 0] + ty * hl[s, 1] + n * hl[s, 2]
        ndh = hl[s, 2]
        l = 2 * ndh * hv - n
        ndl = np.sum(n * l, axis=1, keepdims=True)
        # pdf of the reflected direction (V = N): D * NdH / (4 VdH) = D / 4
        d = a * a / (np.pi * ((ndh * ndh) * (a * a - 1) + 1) ** 2)
        pdf = d / 4
        sa_sample = 1.0 / (samples * pdf + 1e-6)
        lod = max(0.0, 0.5 * math.log2(sa_sample / texel_sa) + 1.0)
        lod = min(lod, len(pyramid) - 1)
        lo = int(math.floor(lod))
        hi = min(lo + 1, len(pyramid) - 1)
        f = lod - lo
        u, v = dir_to_uv(l)
        c = sample_bilinear(pyramid[lo], u, v) * (1 - f) + sample_bilinear(pyramid[hi], u, v) * f
        wgt = np.clip(ndl, 0, None)
        acc += c * wgt
        wsum += wgt
    return (acc / np.maximum(wsum, 1e-6)).reshape(h, w, 3)


def sh9_irradiance(img):
    """Projects radiance onto SH9 and convolves with the clamped cosine lobe (Ramamoorthi & Hanrahan)."""
    h, w = img.shape[:2]
    d = equirect_dirs(w, h).reshape(-1, 3).astype(np.float64)
    sa = solid_angles(w, h).reshape(-1, 1).astype(np.float64)
    c = img.reshape(-1, 3).astype(np.float64)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    basis = np.stack([
        0.282095 * np.ones_like(x),
        0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1),
        1.092548 * x * z, 0.546274 * (x * x - y * y)], axis=1)
    coeffs = basis.T @ (c * sa)  # 9 x 3
    band = np.array([np.pi, 2 * np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3,
                     np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4])
    return coeffs * band[:, None]


def to_half_rgba(img):
    out = np.ones(img.shape[:2] + (4,), dtype=np.float16)
    out[:, :, :3] = np.clip(img, 0, 60000).astype(np.float16)
    return out.tobytes()


def process_environment():
    sky_path = os.path.join(OUT_DIR, 'sky.tex')
    spec_path = os.path.join(OUT_DIR, 'env_specular.tex')
    env_bin = os.path.join(OUT_DIR, 'environment.bin')
    src = os.path.join(SOURCE_DIR, 'san_giuseppe_bridge_4k.hdr')
    if all(os.path.exists(p) and os.path.getmtime(p) > os.path.getmtime(__file__) for p in
           (sky_path, spec_path, env_bin)):
        return
    log('Processing environment map...')
    env = read_hdr(src)
    h, w = env.shape[:2]
    lum = env @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    # Sun extraction: very bright texels form the sun disc. Integrate their irradiance and direction,
    # then replace them with the surrounding sky radiance.
    sun_mask = lum > 200.0
    sa = solid_angles(w, h)
    dirs = equirect_dirs(w, h)
    sun_irr = (env[sun_mask] * sa[sun_mask][:, None]).sum(axis=0)
    sun_dir = (dirs[sun_mask] * (lum[sun_mask] * sa[sun_mask])[:, None]).sum(axis=0)
    sun_dir /= np.linalg.norm(sun_dir)
    ys, xs = np.nonzero(sun_mask)
    y0, y1 = max(0, ys.min() - 24), min(h, ys.max() + 25)
    x0, x1 = max(0, xs.min() - 24), min(w, xs.max() + 25)
    ring = env[y0:y1, x0:x1][~sun_mask[y0:y1, x0:x1]]
    fill = np.median(ring, axis=0)
    env[sun_mask] = fill
    # Soft clamp any remaining extreme highlights to keep the IBL noise-free.
    lum2 = env @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    scale = np.where(lum2 > 64.0, 64.0 / np.maximum(lum2, 1e-6), 1.0)
    env *= scale[:, :, None]
    log('  sun dir %s, irradiance %s, sun texels %d' % (sun_dir, sun_irr, sun_mask.sum()))

    # Sky texture: 2048x1024 with full mip chain (half float).
    sky = resize_to(env, 2048)
    levels = [sky]
    while levels[-1].shape[1] > 1:
        levels.append(downsample(levels[-1]))
    write_tex(sky_path, FORMAT_RGBA16F, sky.shape[1], sky.shape[0], (1, 1), False,
              [to_half_rgba(l) for l in levels])

    # Specular pre-filtered environment: 256x128 base, roughness = mip / (count - 1).
    pyramid = [resize_to(env, 512)]
    while pyramid[-1].shape[1] > 4:
        pyramid.append(downsample(pyramid[-1]))
    spec_levels = []
    count = 6
    for i in range(count):
        lw, lh = 256 >> i, 128 >> i
        r = i / (count - 1)
        spec_levels.append(prefilter_level(pyramid, lw, lh, r).astype(np.float32))
        log('  prefiltered level %d (%dx%d, roughness %.2f)' % (i, lw, lh, r))
    write_tex(spec_path, FORMAT_RGBA16F, 256, 128, (1, 1), False, [to_half_rgba(l) for l in spec_levels])

    sh = sh9_irradiance(resize_to(env, 256))
    with open(env_bin, 'wb') as f:
        f.write(b'BENV')
        f.write(struct.pack('<3f', *sun_dir))
        f.write(struct.pack('<3f', *sun_irr))
        f.write(struct.pack('<27f', *sh.astype(np.float32).reshape(-1)))
    log('  SH9 L0 = %s' % sh[0])


# --------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--download', action='store_true', help='download and extract the Bistro archive')
    parser.add_argument('--max-size', type=int, default=1024, help='maximum texture dimension')
    parser.add_argument('--jobs', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--skip-textures', action='store_true')
    args = parser.parse_args()

    if args.download or not os.path.isdir(SOURCE_DIR):
        download_source()
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = compute_texture_stats(args.jobs)
    textures = run_converter(stats)
    if not args.skip_textures:
        convert_textures(textures, args.max_size, args.jobs, stats)
    process_environment()
    for f in ('LICENSE.txt', 'README.txt'):
        shutil.copy(os.path.join(SOURCE_DIR, f), os.path.join(OUT_DIR, 'BISTRO_' + f))
    log('All assets written to %s' % OUT_DIR)


if __name__ == '__main__':
    main()
