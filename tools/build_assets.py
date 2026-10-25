#!/usr/bin/env python3
"""Builds the Android-ready runtime assets for all environments under environments/*/.

Pipeline (per environment defined by environments/<id>/environment.json):
  1. (optional) Download + extract source archive into source_assets/.
  2. Analyse base-colour, specular, and emissive DDS textures (alpha coverage & constant folding).
  3. Build and invoke the C++ scene converter (FBX -> clustered binary scene.bin).
  4. Convert referenced DDS textures (BC1/BC3/BC5) into ASTC (.tex) with filtered mip chains.
  5. Pre-process the HDR or procedural studio environment map into sky.tex, env_specular.tex,
     and environment.bin (sun + SH9 diffuse irradiance), and write manifest.json.

Usage:  .venv/bin/python tools/build_assets.py [--download] [--env ID] [--force] [--max-size 1024] [--jobs N]
Requires: numpy, pillow, assimp (brew), clang++, astcenc (tools/bin or PATH).
"""

import argparse
import ctypes
import json
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
ENVIRONMENTS_DIR = os.path.join(ROOT, 'environments')
WORK_ROOT = os.path.join(ROOT, 'build_assets')
ASSETS_ENV_ROOT = os.path.join(ROOT, 'app', 'src', 'main', 'assets', 'environments')

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


def load_environments(only_env=None):
    envs = []
    for entry in sorted(os.listdir(ENVIRONMENTS_DIR)):
        cfg_path = os.path.join(ENVIRONMENTS_DIR, entry, 'environment.json')
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            if only_env and cfg['id'] != only_env:
                continue
            envs.append(cfg)
    envs.sort(key=lambda c: c.get('order', 0))
    return envs


# --------------------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------------------

def download_source(cfg):
    source_dir = os.path.join(ROOT, cfg['sourceDir'])
    if 'sourceArchiveZip' in cfg and 'sourceArchiveUrl' in cfg:
        zip_path = os.path.join(ROOT, cfg['sourceArchiveZip'])
        url = cfg['sourceArchiveUrl']
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        if not os.path.exists(zip_path):
            log('Downloading %s -> %s ...' % (url, zip_path))
            urllib.request.urlretrieve(url, zip_path)
        if not os.path.isdir(source_dir):
            log('Extracting %s ...' % zip_path)
            with zipfile.ZipFile(zip_path) as z:
                names = z.namelist()
                top_dirs = {n.split('/')[0] for n in names if '/' in n}
                if os.path.basename(source_dir) in top_dirs:
                    z.extractall(os.path.dirname(source_dir))
                else:
                    os.makedirs(source_dir, exist_ok=True)
                    z.extractall(source_dir)


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


def load_image(path):
    im = Image.open(path)
    im.load()
    return np.asarray(im.convert('RGBA'), dtype=np.float32) / 255.0


TEX_EXTENSIONS = ('.dds', '.png', '.jpg', '.jpeg', '.tga', '.tif')


def discover_textures(tex_dir):
    """Recursively maps texture stem -> full path under tex_dir."""
    tex_map = {}
    for root, _, files in os.walk(tex_dir):
        for f in sorted(files):
            low = f.lower()
            if low.endswith(TEX_EXTENSIONS):
                stem = os.path.splitext(f)[0]
                if stem not in tex_map:
                    tex_map[stem] = os.path.join(root, f)
    return tex_map


# --------------------------------------------------------------------------------------------------
# Alpha analysis
# --------------------------------------------------------------------------------------------------

def texture_stats_worker(args):
    name, path = args
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


def compute_texture_stats(tex_dir, work_dir, jobs):
    """Per-texture statistics used by the converter: alpha coverage (material classification) and
    mean/stddev (constant ORM/BaseColor/Emissive textures are folded into material factors)."""
    out_path = os.path.join(work_dir, 'texture_stats.txt')
    if os.path.exists(out_path):
        return out_path
    tex_map = discover_textures(tex_dir)
    work = [
        (name, path) for name, path in sorted(tex_map.items())
        if not name.lower().endswith('_normal') and not name.lower().endswith('_normal_processed')
    ]
    log('Analysing %d textures in %s...' % (len(work), tex_dir))
    with multiprocessing.Pool(jobs) as pool:
        results = pool.map(texture_stats_worker, work)
    with open(out_path, 'w') as f:
        for name, cut, translucent, mean, std in results:
            f.write('%s %.6f %.6f %.5f %.5f %.5f %.5f\n' % (name, cut, translucent, mean[0], mean[1], mean[2], std))
    return out_path


# --------------------------------------------------------------------------------------------------
# Scene conversion
# --------------------------------------------------------------------------------------------------

def build_converter():
    os.makedirs(WORK_ROOT, exist_ok=True)
    src = os.path.join(ROOT, 'tools', 'scene_converter', 'convert.cpp')
    sky_hdr = os.path.join(ROOT, 'tools', 'scene_converter', 'sky_visibility.h')
    header = os.path.join(ROOT, 'app', 'src', 'main', 'cpp', 'scene_format.h')
    dylib = os.path.join(WORK_ROOT, 'libscene_convert.dylib')
    newest_src = max(os.path.getmtime(src), os.path.getmtime(sky_hdr), os.path.getmtime(header))
    if os.path.exists(dylib) and os.path.getmtime(dylib) > newest_src:
        return dylib
    prefix = subprocess.run(['brew', '--prefix'], capture_output=True, text=True).stdout.strip() or '/usr/local'
    cmd = ['clang++', '-std=c++17', '-O2', '-Wall', '-dynamiclib', '-I' + prefix + '/include',
           src, '-o', dylib, '-L' + prefix + '/lib', '-lassimp']
    log('Building converter library: ' + ' '.join(cmd))
    subprocess.check_call(cmd)
    return dylib


def write_convert_config(cfg, work_dir):
    cfg_file = os.path.join(work_dir, 'convert_config.txt')
    with open(cfg_file, 'w') as f:
        f.write('ao_rays %d\n' % cfg.get('aoRays', 48))
        f.write('ao_max_dist %.4f\n' % cfg.get('aoMaxDistance', 80.0))
        f.write('ao_min_vis %.4f\n' % cfg.get('aoMinVisibility', 0.0))
        f.write('ao_min_area %.6f\n' % cfg.get('aoMinOccluderArea', 0.0))
        f.write('weld_grid %.6f\n' % cfg.get('weldGrid', 0.0))
        f.write('foliage_weld_grid %.6f\n' % cfg.get('foliageWeldGrid', 0.0))
        f.write('fold_constant_basecolor %d\n' % (1 if cfg.get('foldConstantBaseColor', False) else 0))
        f.write('add_curtain_glass %d\n' % (1 if cfg.get('addCurtainGlass', False) else 0))
        f.write('deduplicate_within_scene %d\n' % (1 if cfg.get('deduplicateWithinScene', False) else 0))
        for sub in cfg.get('skipNodeSubstrings', []):
            f.write('skip_node %s\n' % sub)
        sun_ov = cfg.get('sunOverride')
        if sun_ov:
            d = sun_ov['direction']
            c = sun_ov['color']
            f.write('sun_override %.5f %.5f %.5f %.5f %.5f %.5f\n' % (d[0], d[1], d[2], c[0], c[1], c[2]))
        for pl in cfg.get('pointLights', []):
            p = pl['position']
            r = pl['range']
            c = pl['color']
            f.write('point_light %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n' % (p[0], p[1], p[2], r, c[0], c[1], c[2]))


def run_converter(cfg, source_dir, out_dir, work_dir, alpha_stats, force=False):
    scene_bin = os.path.join(out_dir, 'scene.bin')
    tex_list_path = os.path.join(work_dir, 'texture_list.txt')
    fbx_paths = [os.path.join(source_dir, f) for f in cfg['fbxFiles']]
    if not force and os.path.exists(scene_bin) and os.path.exists(tex_list_path):
        # Reuse existing scene.bin + texture_list.txt
        log('Using cached %s' % scene_bin)
    elif not force and os.path.exists(scene_bin) and not os.path.exists(tex_list_path):
        # Extract texture list directly from existing scene.bin header + TextureRecord table!
        with open(scene_bin, 'rb') as f:
            hdr = f.read(84)
            magic, ver, vc, ic, cc, mc, tc, ckc = struct.unpack('<4s7I', hdr[:32])
            rec_data = f.read(tc * 100)
        textures = []
        with open(tex_list_path, 'w') as out_f:
            for i in range(tc):
                raw_name, kind = struct.unpack_from('<96sI', rec_data, i * 100)
                name = raw_name.split(b'\0', 1)[0].decode('utf-8')
                textures.append((name, kind))
                out_f.write('%s %d\n' % (name, kind))
        log('Using existing %s (%d textures)' % (scene_bin, len(textures)))
        return textures
    else:
        dylib = build_converter()
        os.makedirs(out_dir, exist_ok=True)
        write_convert_config(cfg, work_dir)
        argv_strs = ['bistro_convert', out_dir, work_dir, alpha_stats] + fbx_paths
        log('Running converter for %s...' % cfg['id'])
        lib = ctypes.CDLL(dylib)
        lib.bistro_convert_main.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        lib.bistro_convert_main.restype = ctypes.c_int
        c_argv = (ctypes.c_char_p * len(argv_strs))(*[s.encode('utf-8') for s in argv_strs])
        rc = lib.bistro_convert_main(len(argv_strs), c_argv)
        if rc != 0:
            sys.exit('Scene converter failed with code %d' % rc)

    textures = []
    with open(tex_list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, kind = line.rsplit(None, 1)
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


def build_mips(img, kind, alpha_tested, name=''):
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
        # BC5 / tangent normal map stores X,Y; reconstruct Z, filter as vectors and renormalise.
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
        low = name.lower()
        # Single-channel roughness textures (not packed ORM/MetallicRoughness) have R=G=B=roughness;
        # force B (metalness) = 0.0 so dielectric surfaces don't become metallic in scene.frag.
        if kind == KIND_ORM and 'rough' in low and 'metal' not in low and 'orm' not in low:
            cur = img.copy()
            cur[:, :, 1] = img[:, :, 0]
            cur[:, :, 2] = 0.0
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


_ASTC_LIB = None


def build_astcenc_bundle(astcenc_path, dylib_path):
    """Extracts the arm64 slice of astcenc as an MH_BUNDLE so worker processes can call it via dlopen."""
    bundle_path = os.path.join(WORK_ROOT, 'astcenc.bundle')
    with open(astcenc_path, 'rb') as f:
        fat = f.read()
    magic, nfat = struct.unpack('>II', fat[:8])
    arm = None
    if magic == 0xCAFEBABE:
        for i in range(nfat):
            cputype, cpusubtype, offset, size, align = struct.unpack('>IIIII', fat[8 + i * 20:8 + (i + 1) * 20])
            if cputype == 0x0100000C:  # CPU_TYPE_ARM64
                arm = bytearray(fat[offset:offset + size])
                break
    elif struct.unpack('<I', fat[:4])[0] == 0xFEEDFACF:
        arm = bytearray(fat)
    if arm is None:
        return None, 0, dylib_path
    mh_magic, cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags, reserved = struct.unpack('<8I', arm[:32])
    pos = 32
    entryoff = 0
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack('<II', arm[pos:pos + 8])
        if cmd == 0x19:  # LC_SEGMENT_64
            segname = arm[pos + 8:pos + 24].rstrip(b'\0')
            if segname == b'__PAGEZERO':
                struct.pack_into('<QQ', arm, pos + 24, 0, 0)
        elif cmd == 0x80000028:  # LC_MAIN
            entryoff, _ = struct.unpack('<QQ', arm[pos + 8:pos + 24])
        pos += cmdsize
    if not os.path.exists(bundle_path) or os.path.getmtime(bundle_path) < os.path.getmtime(astcenc_path):
        struct.pack_into('<II', arm, 12, 8, ncmds)  # MH_BUNDLE = 8
        struct.pack_into('<I', arm, 24, flags & ~0x200000)
        with open(bundle_path, 'wb') as f:
            f.write(arm)
        subprocess.run(['codesign', '-f', '-s', '-', bundle_path], check=True, stderr=subprocess.DEVNULL)
    return bundle_path, entryoff, dylib_path


def texture_worker(args):
    global _ASTC_LIB
    name, src, kind, max_size, astc_info, mask_names, tex_out_dir = args
    astcenc, bundle_path, entryoff, dylib_path = astc_info
    dst = os.path.join(tex_out_dir, name + '.tex')
    if os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(src):
        return name, os.path.getsize(dst), True
    scale, block, srgb = KIND_SETTINGS[kind]
    img = load_image(src)
    img = resize_to(img, max(4, int(max_size * scale)))
    if max(img.shape[:2]) <= 64:
        block = (4, 4)
    levels = build_mips(img, kind, name in mask_names, name=name)
    payloads = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, lvl in enumerate(levels):
            png = os.path.join(tmp, '%d.png' % i)
            out = os.path.join(tmp, '%d.astc' % i)
            Image.fromarray(lvl, 'RGBA').save(png, compress_level=1)
            cmd = [astcenc, '-cs' if srgb else '-cl', png, out, '%dx%d' % block, '-medium', '-silent']
            if kind == KIND_NORMAL:
                cmd.append('-normal')
            if bundle_path and entryoff:
                if _ASTC_LIB is None:
                    _ASTC_LIB = ctypes.CDLL(dylib_path)
                    _ASTC_LIB.run_astc.argtypes = [
                        ctypes.c_char_p, ctypes.c_ulong, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
                    ]
                    _ASTC_LIB.run_astc.restype = ctypes.c_int
                c_argv = (ctypes.c_char_p * len(cmd))(*[s.encode('utf-8') for s in cmd])
                rc = _ASTC_LIB.run_astc(bundle_path.encode('utf-8'), entryoff, len(cmd), c_argv)
                if rc != 0:
                    raise RuntimeError('astcenc failed with rc=%d on %s' % (rc, name))
            else:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            with open(out, 'rb') as f:
                data = f.read()
            payloads.append(data[16:])  # strip the 16-byte .astc header
    write_tex(dst, FORMAT_ASTC, img.shape[1], img.shape[0], block, srgb, payloads)
    return name, os.path.getsize(dst), False


def convert_textures(textures, tex_src_dir, tex_out_dir, max_size, jobs, alpha_stats_path):
    os.makedirs(tex_out_dir, exist_ok=True)
    astcenc = find_astcenc()
    dylib_path = build_converter()
    bundle_path, entryoff, dylib_path = build_astcenc_bundle(astcenc, dylib_path)
    astc_info = (astcenc, bundle_path, entryoff, dylib_path)
    tex_map = discover_textures(tex_src_dir)
    tex_map_lower = {k.lower(): v for k, v in tex_map.items()}
    mask_names = set()
    with open(alpha_stats_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(None, 6)
            if len(parts) == 7:
                name, cut = parts[0], parts[1]
                if float(cut) > 0.002:
                    mask_names.add(name)
    wanted = {name + '.tex' for name, _ in textures}
    for f in os.listdir(tex_out_dir):
        if f not in wanted:
            os.remove(os.path.join(tex_out_dir, f))
    log('Converting %d textures to ASTC with %d jobs...' % (len(textures), jobs))
    start = time.time()
    total = 0
    work = []
    for name, kind in textures:
        src = tex_map.get(name) or tex_map_lower.get(name.lower())
        if not src:
            raise FileNotFoundError('Missing source texture %r in %s' % (name, tex_src_dir))
        work.append((name, src, kind, max_size, astc_info, mask_names, tex_out_dir))
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


def build_procedural_hdr(spec, w=2048, h=1024):
    dirs = equirect_dirs(w, h)
    y = dirs[..., 1:2]
    zenith = np.array(spec['zenithColor'], dtype=np.float32)
    horizon = np.array(spec['horizonColor'], dtype=np.float32)
    nadir = np.array(spec['nadirColor'], dtype=np.float32)
    upper = horizon * (1.0 - np.clip(y, 0.0, 1.0)) + zenith * np.clip(y, 0.0, 1.0)
    lower = horizon * (1.0 - np.clip(-y, 0.0, 1.0)) + nadir * np.clip(-y, 0.0, 1.0)
    env = np.where(y >= 0.0, upper, lower)
    for lobe in spec.get('lobes', []):
        ld = np.array(lobe['direction'], dtype=np.float32)
        ld /= np.linalg.norm(ld)
        lc = np.array(lobe['color'], dtype=np.float32)
        exp = float(lobe.get('exponent', 16.0))
        dp = np.clip(np.sum(dirs * ld, axis=-1, keepdims=True), 0.0, 1.0)
        env += lc * (dp ** exp)
    return env.astype(np.float32)


def process_environment(cfg, source_dir, out_dir, force=False):
    sky_path = os.path.join(out_dir, 'sky.tex')
    spec_path = os.path.join(out_dir, 'env_specular.tex')
    env_bin = os.path.join(out_dir, 'environment.bin')
    if not force and all(os.path.exists(p) for p in (sky_path, spec_path, env_bin)):
        log('Using existing environment maps in %s' % out_dir)
        return
    log('Processing environment map for %s...' % cfg['id'])
    if cfg.get('hdrFile'):
        src = os.path.join(source_dir, cfg['hdrFile'])
        env = read_hdr(src)
        h, w = env.shape[:2]
        lum = env @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

        # Sun extraction: very bright texels form the sun disc. Integrate their irradiance and direction,
        # then replace them with the surrounding sky radiance.
        sun_mask = lum > 200.0
        sa = solid_angles(w, h)
        dirs = equirect_dirs(w, h)
        if np.any(sun_mask):
            sun_irr = (env[sun_mask] * sa[sun_mask][:, None]).sum(axis=0)
            sun_dir = (dirs[sun_mask] * (lum[sun_mask] * sa[sun_mask])[:, None]).sum(axis=0)
            sun_dir /= np.linalg.norm(sun_dir)
            ys, xs = np.nonzero(sun_mask)
            y0, y1 = max(0, ys.min() - 24), min(h, ys.max() + 25)
            x0, x1 = max(0, xs.min() - 24), min(w, xs.max() + 25)
            ring = env[y0:y1, x0:x1][~sun_mask[y0:y1, x0:x1]]
            fill = np.median(ring, axis=0)
            env[sun_mask] = fill
        else:
            top_mask = lum >= np.percentile(lum, 99.95)
            sun_dir = (dirs[top_mask] * (lum[top_mask] * sa[top_mask])[:, None]).sum(axis=0)
            sun_dir /= max(np.linalg.norm(sun_dir), 1e-6)
            if sun_dir[1] < 0.2:
                sun_dir = np.array([0.45, 0.78, 0.42], dtype=np.float32)
                sun_dir /= np.linalg.norm(sun_dir)
            sun_irr = np.array([6.5, 6.0, 5.2], dtype=np.float32)
        if cfg.get('sunOverride'):
            sun_ov = cfg['sunOverride']
            sun_dir = np.array(sun_ov['direction'], dtype=np.float32)
            sun_dir /= np.linalg.norm(sun_dir)
            sun_irr = np.array(sun_ov['color'], dtype=np.float32)
        # Soft clamp any remaining extreme highlights to keep the IBL noise-free.
        lum2 = env @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        scale = np.where(lum2 > 64.0, 64.0 / np.maximum(lum2, 1e-6), 1.0)
        env *= scale[:, :, None]
        log('  sun dir %s, irradiance %s, sun texels %d' % (sun_dir, sun_irr, sun_mask.sum()))
    else:
        env = build_procedural_hdr(cfg['proceduralHdr'], 2048, 1024)
        sun_ov = cfg.get('sunOverride', {'direction': [0.2, 0.9, 0.3], 'color': [1.0, 1.1, 1.3]})
        sun_dir = np.array(sun_ov['direction'], dtype=np.float32)
        sun_dir /= np.linalg.norm(sun_dir)
        sun_irr = np.array(sun_ov['color'], dtype=np.float32)
        log('  procedural studio HDR, sun dir %s, irradiance %s' % (sun_dir, sun_irr))

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


def build_environment(cfg, args):
    env_id = cfg['id']
    source_dir = os.path.join(ROOT, cfg['sourceDir'])
    tex_src_dir = os.path.join(source_dir, cfg['textureDir'])
    work_dir = os.path.join(WORK_ROOT, env_id)
    out_dir = os.path.join(ASSETS_ENV_ROOT, env_id)
    tex_out_dir = os.path.join(out_dir, 'textures')

    if args.download or not os.path.isdir(source_dir):
        download_source(cfg)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if cfg.get('usdBuildScript'):
        first_scene = os.path.join(source_dir, cfg['fbxFiles'][0])
        if args.force or not os.path.exists(first_scene):
            script_path = os.path.join(ROOT, cfg['usdBuildScript'])
            log('Running USD scene builder %s ...' % script_path)
            subprocess.check_call([sys.executable, script_path])

    # Write runtime manifest.json for Android discovery and camera/render configuration.
    manifest_path = os.path.join(out_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    preview_src = os.path.join(ENVIRONMENTS_DIR, env_id, 'preview.png')
    preview_dst = os.path.join(out_dir, 'preview.png')
    if os.path.exists(preview_src):
        shutil.copyfile(preview_src, preview_dst)

    stats = compute_texture_stats(tex_src_dir, work_dir, args.jobs)
    textures = run_converter(cfg, source_dir, out_dir, work_dir, stats, force=args.force)
    if not args.skip_textures:
        convert_textures(textures, tex_src_dir, tex_out_dir, args.max_size, args.jobs, stats)
    process_environment(cfg, source_dir, out_dir, force=args.force)
    for key, dst_name in (('licenseFile', 'LICENSE.txt'), ('readmeFile', 'README.txt')):
        rel = cfg.get(key, dst_name)
        src_doc = os.path.join(source_dir, rel)
        dst_doc = os.path.join(out_dir, dst_name)
        if os.path.exists(src_doc):
            if os.path.exists(dst_doc):
                os.chmod(dst_doc, 0o644)
            shutil.copyfile(src_doc, dst_doc)
    log('Environment "%s" ready in %s' % (env_id, out_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--download', action='store_true', help='download and extract missing source archives')
    parser.add_argument('--env', type=str, default=None, help='only build a specific environment ID')
    parser.add_argument('--force', action='store_true', help='force rebuild of scene.bin and environment.bin')
    parser.add_argument('--max-size', type=int, default=1024, help='maximum texture dimension')
    parser.add_argument('--jobs', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--skip-textures', action='store_true')
    args = parser.parse_args()

    envs = load_environments(args.env)
    if not envs:
        sys.exit('No environments found matching %r' % args.env)
    for cfg in envs:
        build_environment(cfg, args)


if __name__ == '__main__':
    main()
