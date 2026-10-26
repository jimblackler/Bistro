#!/usr/bin/env python3
"""Converts the Intel Jungle Ruins USD scene into glTF 2.0 (JungleRuins_Scene.gltf + .bin).

Uses /usr/bin/usdcat (system USD tool) to stream-parse the binary .usd and .usda layers:
  - Pyramid.usd (all 19 GeomSubset materials)
  - River.usd (reflective river surface)
  - Terrain_Cinematic.usd (16 textured terrain tiles, adaptive decimation)
  - Banyan.usd (hero Banyan tree with bark, vines, and two-sided leaves)
  - PointInstancer layers (QueenForest, RiverForest, RiverSapling, Shrub, Pyramid_Grass_B,
    Pyramid_Moss, Anthurium, ShrubSorrel) around the Pyramid & River clearing.

Also merges separate *_albedo + *_opacity JPEG pairs in textures/Trees/ into RGBA PNGs so
the asset pipeline preserves alpha cutouts on tree leaves.
"""

import json
import os
import struct
import subprocess
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUNGLE_DIR = os.path.join(ROOT, 'source_assets', 'JungleRuins')
USD_DIR = os.path.join(JUNGLE_DIR, 'USD')
TEX_DIR = os.path.join(JUNGLE_DIR, 'textures')

# Center the scene around (550, 780, 0) in Blender Z-up meters:
#   Pyramid is at (543.7, 809.8, 9.5) -> Y-up (-6.3, 9.5, -29.8)
#   defaultCam is at (572.5, 766.5, 1.4) -> Y-up (+22.5, 1.4, +13.5)
ORIGIN_X = 550.0
ORIGIN_Y = 780.0
ORIGIN_Z = 0.0


def log(msg):
    print(msg, flush=True)


def prepare_tree_leaf_textures():
    """Combines separate RGB albedo + grayscale opacity JPEGs into RGBA PNGs."""
    trees_dir = os.path.join(TEX_DIR, 'Trees')
    pairs = [
        (
            'PX_Chamaedorea_elegans_Leaf_01_BaseColor.png',
            'PX_Chamaedorea_elegans_Leaf_01_albedo_E.jpg',
            'PX_Chamaedorea_elegans_Leaf_01_opacity.jpg',
        ),
        (
            'PX_Imperata_cylindrica_Leaf_01_BaseColor.png',
            'PX_Imperata_Cylindrica_Leaf_01_Albedo_Processed.png',
            'PX_Imperata_cylindrica_Leaf_01_opacity.jpg',
        ),
        (
            'PX_Prunus_x_yedoensis_Leaf_02_BaseColor.png',
            'PX_Prunus_x_yedoensis_Leaf_02_albedo.jpg',
            'PX_Prunus_x_yedoensis_Leaf_02_opacity.jpg',
        ),
        (
            'PX_Prunus_x_yedoensis_Leaf_03_BaseColor.png',
            'PX_Prunus_x_yedoensis_Leaf_03_albedo.jpg',
            'PX_Prunus_x_yedoensis_Leaf_03_opacity.jpg',
        ),
    ]
    created_any = False
    for out_name, rgb_name, alpha_name in pairs:
        out_path = os.path.join(trees_dir, out_name)
        if os.path.exists(out_path):
            continue
        rgb_path = os.path.join(trees_dir, rgb_name)
        alpha_path = os.path.join(trees_dir, alpha_name)
        if not os.path.exists(rgb_path) or not os.path.exists(alpha_path):
            continue
        rgb = Image.open(rgb_path).convert('RGB')
        a = Image.open(alpha_path).convert('L')
        if a.size != rgb.size:
            a = a.resize(rgb.size, Image.Resampling.BILINEAR)
        rgba = Image.merge('RGBA', (*rgb.split(), a))
        rgba.save(out_path)
        log(f'  Created RGBA leaf texture {out_name}')
        created_any = True
    if created_any:
        stats_file = os.path.join(ROOT, 'build_assets', 'jungle', 'texture_stats.txt')
        if os.path.exists(stats_file):
            os.remove(stats_file)


# Material definitions: mat_name -> (baseColor_relpath, normal_relpath, roughness_relpath, double_sided, alpha_mode, base_factor, rough_factor)
MATERIALS = {
    # River
    'River': (
        'textures/River/River_BaseColor.png',
        None,
        None,
        False,
        'OPAQUE',
        [0.14, 0.22, 0.18, 1.0],
        0.06,
    ),
    # Banyan
    'MI_Banyan_Bark': (
        'textures/Trees/PX_Banyan_Pinus_Rigida_Bark_02_Albedo_Processed.png',
        'textures/Trees/PX_Banyan_Pinus_Rigida_Bark_02_Normal_Processed.png',
        'textures/Trees/PX_Banyan_Pinus_Rigida_Bark_02_Roughness_Processed.png',
        False,
        'OPAQUE',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    'MI_Banyan_Vines': (
        'textures/Trees/PX_Banyan_Pinus_Rigida_Vines_02_Albedo_Processed.png',
        'textures/Trees/PX_Banyan_Pinus_Rigida_Vines_02_Normal_Processed.png',
        'textures/Trees/PX_Banyan_Pinus_Rigida_Vines_02_Roughness_Processed.png',
        False,
        'OPAQUE',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    'MI_Banyan_Leaves_TwoSided': (
        'textures/Trees/PX_Prunus_x_yedoensis_Leaf_02_BaseColor.png',
        'textures/Trees/PX_Prunus_x_yedoensis_Leaf_02_normal.jpg',
        'textures/Trees/PX_Prunus_x_yedoensis_Leaf_02_roughness.jpg',
        True,
        'MASK',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    # RiverFamily Trees (RiverForest, RiverSapling, RiverSeedling)
    'MI_RiverFamily_Trunk': (
        'textures/Trees/PX_Chamaecyparis_Pisifera_Bark_01_Albedo_Processed.png',
        'textures/Trees/PX_Chamaecyparis_Pisifera_Bark_01_Normal_Processed.png',
        'textures/Trees/PX_Chamaecyparis_Pisifera_Bark_01_Roughness_Processed.png',
        False,
        'OPAQUE',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    'MI_RiverFamily_Branches': (
        'textures/Trees/PX_Pinus_Rigida_Bark_02_Albedo_Processed.png',
        'textures/Trees/PX_Pinus_Rigida_Bark_02_Normal_Processed.png',
        'textures/Trees/PX_Pinus_Rigida_Bark_02_Roughness_Processed.png',
        False,
        'OPAQUE',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    'MI_RiverFamily_Leaves_TwoSided': (
        'textures/Trees/PX_Prunus_x_yedoensis_Leaf_03_BaseColor.png',
        'textures/Trees/PX_Prunus_x_yedoensis_Leaf_03_normal.jpg',
        'textures/Trees/PX_Prunus_x_yedoensis_Leaf_03_roughness.jpg',
        True,
        'MASK',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    # QueenPalm (QueenForest)
    'MI_QueenPalm_Trunk': (
        'textures/Trees/palm_bark_diff_4k.jpg',
        'textures/Trees/PX_Chamaecyparis_Pisifera_Bark_01_Normal_Processed.png',
        'textures/Trees/PX_Chamaecyparis_Pisifera_Bark_01_Roughness_Processed.png',
        False,
        'OPAQUE',
        [1.0, 1.0, 1.0, 1.0],
        0.85,
    ),
    'MI_QueenPalm_OldBranches': (
        'textures/Trees/PX_Imperata_cylindrica_Leaf_01_BaseColor.png',
        'textures/Trees/PX_Imperata_cylindrica_Leaf_01_normal.jpg',
        'textures/Trees/PX_Imperata_cylindrica_Leaf_01_roughness.jpg',
        True,
        'MASK',
        [1.0, 1.0, 1.0, 1.0],
        0.85,
    ),
    'MI_QueenPalm_DeadLeaves_TwoSided': (
        'textures/Trees/PX_Imperata_cylindrica_Leaf_01_BaseColor.png',
        'textures/Trees/PX_Imperata_cylindrica_Leaf_01_normal.jpg',
        'textures/Trees/PX_Imperata_cylindrica_Leaf_01_roughness.jpg',
        True,
        'MASK',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    'MI_QueenPalm_Berries': (
        'textures/Trees/MI_QueenForest_Berries_Albedo_Processed.png',
        'textures/Trees/MI_QueenForest_Berries_Normal_Processed.png',
        'textures/Trees/MI_QueenForest_Berries_Roughness_Processed.png',
        False,
        'OPAQUE',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    'MI_QueenPalm_Leaves_TwoSided': (
        'textures/Trees/PX_Chamaedorea_elegans_Leaf_01_BaseColor.png',
        'textures/Trees/PX_Chamaedorea_elegans_Leaf_01_normal.jpg',
        'textures/Trees/PX_Chamaedorea_elegans_Leaf_01_roughness.jpg',
        True,
        'MASK',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    # Plants
    'MI_Anthurium_01_TwoSided': (
        'textures/Plants/Anthurium_Botany_01_BaseColor.tif',
        'textures/Plants/Anthurium_Botany_01_Normal.png',
        'textures/Plants/Anthurium_Botany_01_Roughness.png',
        True,
        'MASK',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    'MI_Grass_01_TwoSided': (
        'textures/Plants/Grass_Medium_01_BaseColor.tif',
        'textures/Plants/Grass_Medium_01_Normal.png',
        'textures/Plants/Grass_Medium_01_Roughness.png',
        True,
        'MASK',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    'MI_Grass_02_TwoSided': (
        'textures/Plants/Grass_Medium_02_BaseColor.tif',
        'textures/Plants/Grass_Medium_02_Normal.png',
        'textures/Plants/Grass_Medium_02_Roughness.png',
        True,
        'MASK',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    'MI_Moss_01_TwoSided': (
        'textures/Plants/Moss_01_BaseColor.tif',
        'textures/Plants/Moss_01_Normal.png',
        'textures/Plants/Moss_01_Roughness.png',
        True,
        'MASK',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    'MI_Nettle_01_TwoSided': (
        'textures/Plants/Nettle_BaseColor.tif',
        'textures/Plants/Nettle_Normal.png',
        'textures/Plants/Nettle_Roughness.png',
        True,
        'MASK',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    'MI_Shrub_04_TwoSided': (
        'textures/Plants/Shrub_04_BaseColor.tif',
        'textures/Plants/Shrub_04_Normal.png',
        'textures/Plants/Shrub_04_Roughness.png',
        True,
        'MASK',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    'MI_Shrub_Sorrel_01_TwoSided': (
        'textures/Plants/Shrub_Sorrel_01_BaseColor.tif',
        'textures/Plants/Shrub_Sorrel_01_Normal.png',
        'textures/Plants/Shrub_Sorrel_01_Roughness.png',
        True,
        'MASK',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
}

# Add all 19 Pyramid materials
for sub in (
    'BottomFlat', 'Ground', 'TableStand', 'TableTop', 'Ceiling',
    'StairsStraight', 'StairsCorner', 'WallTop_L', 'WallTop_R', 'Corner',
    'Column', 'StairRail', 'StairsBottom', 'StairsTop', 'RailCorner',
    'RailStraight', 'BottomWallCorner', 'BottomWallSide', 'Topper',
):
    tex_sub = sub.replace('WallTop_L', 'WallTopL').replace('WallTop_R', 'WallTopR')
    MATERIALS[f'MI_Pyramid_{sub}'] = (
        f'textures/Pyramid/Pyramid_{tex_sub}_BaseColor.png',
        f'textures/Pyramid/Pyramid_{tex_sub}_Normal.png',
        f'textures/Pyramid/Pyramid_{tex_sub}_Roughness.png',
        False,
        'OPAQUE',
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    )

# Add all 4 Terrain_Cinematic tile materials
for gx, gy in (('3', '3'), ('3', '4'), ('4', '3'), ('4', '4')):
    MATERIALS[f'MI_Terrain_X{gx}_Y{gy}'] = (
        f'textures/Terrain/Terrain_Cinematic_BaseColor_{gx}_{gy}.jpg',
        f'textures/Terrain/Terrain_Cinematic_Normal_{gx}_{gy}.jpg',
        'textures/Terrain/Metallic-Roughness.png',
        False,
        'OPAQUE',
        [1.0, 1.0, 1.0, 1.0],
        0.9,
    )


def parse_float_array(line, cols):
    eq = line.find('=')
    start = line.find('[', eq if eq != -1 else 0)
    end = line.rfind(']')
    if start == -1 or end == -1 or end <= start + 1:
        return np.zeros((0, cols), dtype=np.float32)
    raw = line[start + 1:end].replace('(', '').replace(')', '')
    arr = np.fromstring(raw, sep=',', dtype=np.float32)
    return arr.reshape(-1, cols)


def parse_int_array(line):
    eq = line.find('=')
    start = line.find('[', eq if eq != -1 else 0)
    end = line.rfind(']')
    if start == -1 or end == -1 or end <= start + 1:
        return np.zeros(0, dtype=np.int32)
    return np.fromstring(line[start + 1:end], sep=',', dtype=np.int32)


def open_usd_lines(path):
    if path.endswith('.usda'):
        return open(path, 'r', encoding='utf-8'), None
    proc = subprocess.Popen(['/usr/bin/usdcat', path], stdout=subprocess.PIPE, text=True)
    return proc.stdout, proc


def parse_usd_meshes(path):
    """Stream-parses a USD/USDA file and returns a list of mesh dicts."""
    stream, proc = open_usd_lines(path)
    meshes = []
    cur_xform_tr = np.zeros(3, dtype=np.float64)
    cur_xform_sc = np.ones(3, dtype=np.float64)
    cur_mesh = None
    cur_subset_name = None
    cur_subset_indices = None

    for line in stream:
        s = line.strip()
        if s.startswith('def Xform "'):
            if cur_mesh is not None:
                meshes.append(cur_mesh)
                cur_mesh = None
            name = s.split('"')[1]
            if name != 'root':
                cur_xform_tr = np.zeros(3, dtype=np.float64)
                cur_xform_sc = np.ones(3, dtype=np.float64)
        elif cur_mesh is None and (s.startswith('double3 xformOp:translate =') or s.startswith('float3 xformOp:translate =')):
            vals = s.split('=')[1].strip().strip('()').split(',')
            cur_xform_tr = np.array([float(v) for v in vals], dtype=np.float64)
        elif cur_mesh is None and s.startswith('float3 xformOp:scale ='):
            vals = s.split('=')[1].strip().strip('()').split(',')
            sc = np.array([float(v) for v in vals], dtype=np.float64)
            if not np.allclose(sc, 100.0):
                cur_xform_sc = sc
        elif s.startswith('def Mesh "'):
            if cur_mesh is not None:
                meshes.append(cur_mesh)
            mname = s.split('"')[1]
            cur_mesh = {
                'name': mname,
                'translate': cur_xform_tr.copy(),
                'scale': cur_xform_sc.copy(),
                'counts': None,
                'indices': None,
                'points': None,
                'normals': None,
                'uvs': None,
                'default_mat': None,
                'subsets': [],
            }
            cur_subset_name = None
        elif cur_mesh is not None:
            if s.startswith('int[] faceVertexCounts ='):
                cur_mesh['counts'] = parse_int_array(s)
            elif s.startswith('int[] faceVertexIndices ='):
                cur_mesh['indices'] = parse_int_array(s)
            elif s.startswith('normal3f[] normals ='):
                cur_mesh['normals'] = parse_float_array(s, 3)
            elif s.startswith('point3f[] points ='):
                cur_mesh['points'] = parse_float_array(s, 3)
            elif s.startswith('texCoord2f[] primvars:st ='):
                cur_mesh['uvs'] = parse_float_array(s, 2)
            elif s.startswith('def GeomSubset "'):
                cur_subset_name = s.split('"')[1]
                cur_subset_indices = None
            elif s.startswith('int[] indices =') and cur_subset_name is not None:
                cur_subset_indices = parse_int_array(s)
            elif s.startswith('rel material:binding ='):
                mat_path = s.split('<')[1].split('>')[0]
                mat_name = mat_path.split('/')[-1]
                if cur_subset_name is not None and cur_subset_indices is not None:
                    cur_mesh['subsets'].append((mat_name, cur_subset_indices))
                    cur_subset_name = None
                    cur_subset_indices = None
                elif cur_mesh['default_mat'] is None:
                    cur_mesh['default_mat'] = mat_name

    if cur_mesh is not None:
        meshes.append(cur_mesh)
    stream.close()
    if proc is not None:
        proc.wait()
    return meshes


def build_triangles_from_mesh(mesh, face_subset=None):
    """Extracts (pos, norm, uv) arrays of shape (T*3, 3/3/2) in local Z-up space for a mesh or subset."""
    counts = mesh['counts']
    indices = mesh['indices']
    points = mesh['points']
    normals = mesh['normals']
    uvs = mesh['uvs']
    if counts is None or indices is None or points is None:
        return None

    starts = np.zeros(len(counts), dtype=np.int32)
    if len(counts) > 1:
        np.cumsum(counts[:-1], out=starts[1:])

    if face_subset is not None:
        f_idx = face_subset
    else:
        f_idx = np.arange(len(counts), dtype=np.int32)

    f_counts = counts[f_idx]
    f_starts = starts[f_idx]

    # Triangles (c == 3)
    m3 = f_counts == 3
    s3 = f_starts[m3]
    c3 = np.stack([s3, s3 + 1, s3 + 2], axis=1).reshape(-1)

    # Quads (c == 4) -> (0, 1, 2) and (0, 2, 3)
    m4 = f_counts == 4
    s4 = f_starts[m4]
    c4 = np.stack([s4, s4 + 1, s4 + 2, s4, s4 + 2, s4 + 3], axis=1).reshape(-1)

    if len(c3) > 0 and len(c4) > 0:
        corners = np.concatenate([c3, c4])
    elif len(c3) > 0:
        corners = c3
    else:
        corners = c4

    if len(corners) == 0:
        return None

    v_idx = indices[corners]
    pos = points[v_idx]
    if normals is not None and len(normals) == len(indices):
        nrm = normals[corners]
    elif normals is not None and len(normals) == len(points):
        nrm = normals[v_idx]
    else:
        nrm = np.zeros_like(pos)
        nrm[:, 2] = 1.0

    if uvs is not None and len(uvs) == len(indices):
        st = uvs[corners]
    elif uvs is not None and len(uvs) == len(points):
        st = uvs[v_idx]
    else:
        st = np.zeros((len(corners), 2), dtype=np.float32)

    return pos, nrm, st


def subsample_triangles(pos, nrm, st, stride):
    """Subsamples whole triangles/quads (every 2 triangles = 1 quad leaf) without welding across leaves."""
    if stride <= 1 or len(pos) < 6:
        return pos, nrm, st
    # Group in pairs of triangles (quads) so quad leaf cards stay intact
    n_tris = len(pos) // 3
    if n_tris % 2 == 0:
        q_pos = pos.reshape(-1, 6, 3)[::stride].reshape(-1, 3)
        q_nrm = nrm.reshape(-1, 6, 3)[::stride].reshape(-1, 3)
        q_st = st.reshape(-1, 6, 2)[::stride].reshape(-1, 2)
        if len(q_pos) > 0:
            return q_pos, q_nrm, q_st
    t_pos = pos.reshape(-1, 3, 3)[::stride].reshape(-1, 3)
    t_nrm = nrm.reshape(-1, 3, 3)[::stride].reshape(-1, 3)
    t_st = st.reshape(-1, 3, 2)[::stride].reshape(-1, 2)
    return t_pos, t_nrm, t_st


def decimate_triangles(pos, nrm, st, grid_size):
    """Fast collision-free spatial vertex-clustering decimation in NumPy."""
    if grid_size <= 0.0 or len(pos) == 0:
        return pos, nrm, st
    q = np.floor(pos / grid_size).astype(np.int64)
    ox = (q[:, 0] + 1048576) & 0x1FFFFF
    oy = (q[:, 1] + 1048576) & 0x1FFFFF
    oz = (q[:, 2] + 1048576) & 0x1FFFFF
    h = (ox << 42) | (oy << 21) | oz
    h_tri = h.reshape(-1, 3)
    keep = (h_tri[:, 0] != h_tri[:, 1]) & (h_tri[:, 1] != h_tri[:, 2]) & (h_tri[:, 2] != h_tri[:, 0])
    if not np.any(keep):
        return pos[:3], nrm[:3], st[:3]
    _, first_idx, inv_idx = np.unique(h, return_index=True, return_inverse=True)
    snapped_pos = pos[first_idx][inv_idx].reshape(-1, 3, 3)[keep]
    snapped_nrm = nrm[first_idx][inv_idx].reshape(-1, 3, 3)[keep]
    snapped_st = st.reshape(-1, 3, 2)[keep]
    return snapped_pos.reshape(-1, 3), snapped_nrm.reshape(-1, 3), snapped_st.reshape(-1, 2)


def dedupe_triangles(pos, nrm, st):
    """Removes coincident duplicate triangles and coplanar folded triangles sharing an edge.

    The Jungle Ruins USD foliage contains many overlapping triangles (and vertex clustering creates
    coplanar diagonal overlaps on flat leaf quads). Coincident or coplanar overlapping triangles z-fight
    across parallel TBDR tile processors, which shows up as shimmering leaves and bark.
    """
    n_tri = len(pos) // 3
    if n_tri < 2:
        return pos, nrm, st
    q = np.round(pos * 1000.0).astype(np.int64).reshape(n_tri, 3, 3)
    order = np.lexsort((q[:, :, 2], q[:, :, 1], q[:, :, 0]), axis=1)
    key = np.take_along_axis(q, order[:, :, None], axis=1).reshape(n_tri, 9)
    key_view = np.ascontiguousarray(key).view(np.dtype((np.void, key.dtype.itemsize * 9))).ravel()
    _, keep = np.unique(key_view, return_index=True)
    keep.sort()
    tris = pos.reshape(n_tri, 3, 3)[keep]
    nrms = nrm.reshape(n_tri, 3, 3)[keep]
    sts = st.reshape(n_tri, 3, 2)[keep]
    n_tri = len(tris)
    if n_tri < 2:
        return tris.reshape(-1, 3), nrms.reshape(-1, 3), sts.reshape(-1, 2)

    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    fn = np.cross(v1 - v0, v2 - v0)
    area2 = np.linalg.norm(fn, axis=1)
    alive = area2 > 1e-6
    fn_u = fn / np.maximum(area2[:, None], 1e-12)

    qv = np.round(tris.reshape(-1, 3) * 1000.0).astype(np.int64)
    _, inv_v = np.unique(qv, axis=0, return_inverse=True)
    idx = inv_v.ravel().reshape(n_tri, 3)
    s_order = np.argsort(idx, axis=1)
    s_idx = np.take_along_axis(idx, s_order, axis=1)
    s_pos = np.take_along_axis(tris, s_order[:, :, None], axis=1)

    for p0, p1, p2 in [(0, 1, 2), (1, 2, 0), (0, 2, 1)]:
        e = s_idx[:, [p0, p1]]
        ek = e[:, 0].astype(np.int64) * 10000000 + e[:, 1].astype(np.int64)
        eo = np.argsort(ek, kind='mergesort')
        ek_s = ek[eo]
        for step in (1, 2, 3):
            if len(eo) <= step:
                continue
            same_e = ek_s[step:] == ek_s[:-step]
            ta = eo[:-step][same_e]
            tb = eo[step:][same_e]
            cos_n = np.abs(np.einsum('ij,ij->i', fn_u[ta], fn_u[tb]))
            va = s_pos[ta, p0]
            vb = s_pos[ta, p1]
            vca = s_pos[ta, p2]
            vcb = s_pos[tb, p2]
            ab = vb - va
            ab_u = ab / np.maximum(np.linalg.norm(ab, axis=1, keepdims=True), 1e-12)
            pa = (vca - va) - ab_u * np.einsum('ij,ij->i', vca - va, ab_u)[:, None]
            pb = (vcb - va) - ab_u * np.einsum('ij,ij->i', vcb - va, ab_u)[:, None]
            same_side = np.einsum('ij,ij->i', pa, pb) > 0.0
            drop = (cos_n > 0.98) & same_side & alive[ta]
            alive[tb[drop]] = False

    return tris[alive].reshape(-1, 3), nrms[alive].reshape(-1, 3), sts[alive].reshape(-1, 2)


def weld_coincident_xy(pos_list, cell):
    """Snaps vertices (across several arrays) whose horizontal (XY, Z-up) positions agree to within `cell`
    onto their mean position, so neighbouring terrain tiles meet without cracks. Two passes with grids
    offset by half a cell catch pairs that straddle a cell boundary in the first pass."""
    sizes = [len(p) for p in pos_list]
    allp = np.concatenate(pos_list).astype(np.float64)
    for offset in (0.0, 0.5 * cell):
        key = np.floor((allp[:, :2] + offset) / cell).astype(np.int64)
        _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        sums = np.zeros((len(counts), 3))
        np.add.at(sums, inv, allp)
        allp = sums[inv] / counts[inv, None]
    out, start = [], 0
    for n in sizes:
        out.append(allp[start:start + n].astype(np.float32))
        start += n
    return out


def zup_to_yup(pos_zup, nrm_zup, st):
    """Converts Blender Z-up (x, y, z) around (ORIGIN_X, ORIGIN_Y, ORIGIN_Z) to glTF Y-up (x, z, -y),
    and flips USD lower-left V to glTF upper-left (1 - V)."""
    px = pos_zup[:, 0] - ORIGIN_X
    py = pos_zup[:, 2] - ORIGIN_Z
    pz = -(pos_zup[:, 1] - ORIGIN_Y)
    pos_yup = np.stack([px, py, pz], axis=1).astype(np.float32)

    nx = nrm_zup[:, 0]
    ny = nrm_zup[:, 2]
    nz = -nrm_zup[:, 1]
    nrm_yup = np.stack([nx, ny, nz], axis=1).astype(np.float32)
    norms = np.linalg.norm(nrm_yup, axis=1, keepdims=True)
    nrm_yup /= np.maximum(norms, 1e-6)

    uv_gltf = np.stack([st[:, 0], 1.0 - st[:, 1]], axis=1).astype(np.float32)
    return pos_yup, nrm_yup, uv_gltf


def index_triangles(pos, nrm, uv):
    """Deduplicates (pos, nrm, uv) into indexed uint32 triangles."""
    n = len(pos)
    if n == 0:
        return None
    # Quantize slightly to weld identical corner vertices
    key = np.hstack([
        np.rint(pos * 2000.0).astype(np.int32),
        np.rint(nrm * 500.0).astype(np.int32),
        np.rint(uv * 4096.0).astype(np.int32),
    ])
    key_view = np.ascontiguousarray(key).view(np.dtype((np.void, key.dtype.itemsize * key.shape[1]))).ravel()
    _, unique_idx, inv_idx = np.unique(key_view, return_index=True, return_inverse=True)
    u_pos = pos[unique_idx].astype(np.float32)
    u_nrm = nrm[unique_idx].astype(np.float32)
    u_uv = uv[unique_idx].astype(np.float32)
    indices = inv_idx.astype(np.uint32)
    return u_pos, u_nrm, u_uv, indices


def quat_to_rotmat(q):
    """q is (N, 4) with (w, x, y, z) from USD quath[]. Returns (N, 3, 3) rotation matrices."""
    q = q.astype(np.float32)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norm, 1e-6)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((len(q), 3, 3), dtype=np.float32)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def parse_point_instancer(usd_path):
    """Returns dict: proto_name -> (positions_m (N,3), scales_m (N,3), quats (N,4))."""
    stream, proc = open_usd_lines(usd_path)
    instancers = {}
    cur_name = None
    cur_ori = None
    cur_pos = None
    cur_sc = None
    for line in stream:
        s = line.strip()
        if s.startswith('def PointInstancer "'):
            if cur_name and cur_pos is not None:
                instancers[cur_name] = (cur_pos, cur_sc, cur_ori)
            cur_name = s.split('"')[1]
            cur_ori = None
            cur_pos = None
            cur_sc = None
        elif cur_name:
            if s.startswith('quath[] orientations ='):
                cur_ori = parse_float_array(s, 4)
            elif s.startswith('point3f[] positions ='):
                cur_pos = parse_float_array(s, 3) * 0.01
            elif s.startswith('float3[] scales ='):
                cur_sc = parse_float_array(s, 3) * 0.01
    if cur_name and cur_pos is not None:
        instancers[cur_name] = (cur_pos, cur_sc, cur_ori)
    stream.close()
    if proc is not None:
        proc.wait()
    return instancers


def main():
    t0 = time.time()
    log('Preparing Jungle Ruins RGBA leaf textures...')
    prepare_tree_leaf_textures()

    # Buckets of triangles per material: mat_name -> list of (pos_yup, nrm_yup, uv_gltf)
    mat_buckets = {m: [] for m in MATERIALS}

    def add_to_bucket(mat_name, pos_zup, nrm_zup, st, weld_grid=0.0):
        if mat_name not in mat_buckets or len(pos_zup) == 0:
            return
        if weld_grid > 0.0:
            pos_zup, nrm_zup, st = decimate_triangles(pos_zup, nrm_zup, st, weld_grid)
        pos_zup, nrm_zup, st = dedupe_triangles(pos_zup, nrm_zup, st)
        if len(pos_zup) == 0:
            return
        p_y, n_y, uv_g = zup_to_yup(pos_zup, nrm_zup, st)
        mat_buckets[mat_name].append((p_y, n_y, uv_g))

    # 1. Pyramid (all 19 subsets, full detail)
    log('1/6 Parsing Pyramid.usd ...')
    pyr_meshes = parse_usd_meshes(os.path.join(USD_DIR, 'elements', 'Pyramid', 'Pyramid.usd'))
    for m in pyr_meshes:
        tr = m['translate'].astype(np.float32)
        sc = m['scale'].astype(np.float32)
        for mat_name, face_idx in m['subsets']:
            res = build_triangles_from_mesh(m, face_idx)
            if res is None:
                continue
            pos, nrm, st = res
            pos = pos * sc + tr
            add_to_bucket(mat_name, pos, nrm, st, weld_grid=0.025)

    # 2. River
    log('2/6 Parsing River.usd ...')
    riv_meshes = parse_usd_meshes(os.path.join(USD_DIR, 'elements', 'River', 'River.usd'))
    for m in riv_meshes:
        tr = m['translate'].astype(np.float32)
        sc = m['scale'].astype(np.float32)
        res = build_triangles_from_mesh(m)
        if res is None:
            continue
        pos, nrm, st = res
        pos = pos * sc + tr
        # Keep river within 155 m of scene center
        tri_pos = pos.reshape(-1, 3, 3)
        center_xy = tri_pos.mean(axis=1)[:, :2]
        keep = (np.abs(center_xy[:, 0] - ORIGIN_X) < 155.0) & (np.abs(center_xy[:, 1] - ORIGIN_Y) < 155.0)
        if np.any(keep):
            pos = tri_pos[keep].reshape(-1, 3)
            nrm = nrm.reshape(-1, 3, 3)[keep].reshape(-1, 3)
            st = st.reshape(-1, 3, 2)[keep].reshape(-1, 2)
            add_to_bucket('River', pos, nrm, st, weld_grid=0.5)

    # 3. Terrain_Cinematic (clip to 155m around clearing center)
    log('3/6 Parsing Terrain_Cinematic.usd ...')
    ter_meshes = parse_usd_meshes(os.path.join(USD_DIR, 'elements', 'Terrain', 'Terrain_Cinematic.usd'))
    ter_tiles = []
    for m in ter_meshes:
        tr = m['translate'].astype(np.float32)
        sc = m['scale'].astype(np.float32)
        mat_name = m['default_mat']
        if not mat_name or mat_name not in MATERIALS:
            continue
        res = build_triangles_from_mesh(m)
        if res is None:
            continue
        pos, nrm, st = res
        pos = pos * sc + tr
        tri_pos = pos.reshape(-1, 3, 3)
        tri_nrm = nrm.reshape(-1, 3, 3)
        tri_st = st.reshape(-1, 3, 2)
        c_xy = tri_pos.mean(axis=1)[:, :2]
        in_bounds = (np.abs(c_xy[:, 0] - ORIGIN_X) < 155.0) & (np.abs(c_xy[:, 1] - ORIGIN_Y) < 155.0)
        if not np.any(in_bounds):
            continue
        ter_tiles.append((mat_name, tri_pos[in_bounds].reshape(-1, 3), tri_nrm[in_bounds].reshape(-1, 3),
                          tri_st[in_bounds].reshape(-1, 2)))
    # Neighbouring source tiles share border vertices in XY but disagree on height by up to ~27 cm (and
    # by ~0.2 mm in XY), which leaves thin slits through which the sky shows. Weld them to a common position.
    if ter_tiles:
        welded = weld_coincident_xy([t[1] for t in ter_tiles], cell=0.05)
        ter_tiles = [(t[0], w, t[2], t[3]) for t, w in zip(ter_tiles, welded)]
    for mat_name, pos, nrm, st in ter_tiles:
        # Keep the terrain at full source resolution: vertex-clustering only part of it (or each tile
        # separately) snaps shared border vertices differently on each side and tears cracks.
        add_to_bucket(mat_name, pos, nrm, st, weld_grid=0.0)

    # Build a subdivided water surface mesh across the river valley basin around the Pyramid & Banyan clearing
    # at Z-up elevation z = -0.85 m wherever the underlying terrain riverbed drops below the water plane.
    if ter_tiles:
        water_z = -0.85
        all_ter = np.vstack([t[1] for t in ter_tiles])
        cell_size = 2.5
        gx = np.arange(ORIGIN_X - 152.5, ORIGIN_X + 152.5, cell_size, dtype=np.float32)
        gy = np.arange(ORIGIN_Y - 152.5, ORIGIN_Y + 152.5, cell_size, dtype=np.float32)
        nx, ny = len(gx), len(gy)
        # Mark grid cells whose terrain height is below water_z + 0.65m (with 1-cell dilation so the water
        # edge tucks cleanly beneath the sloped riverbank without gaps).
        sub_pts = all_ter[all_ter[:, 2] < water_z + 0.65]
        ix = np.clip(((sub_pts[:, 0] - gx[0]) / cell_size).astype(np.int32), 0, nx - 1)
        iy = np.clip(((sub_pts[:, 1] - gy[0]) / cell_size).astype(np.int32), 0, ny - 1)
        wet = np.zeros((nx, ny), dtype=bool)
        wet[ix, iy] = True
        wet_dil = wet.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                wet_dil |= np.roll(np.roll(wet, dx, axis=0), dy, axis=1)
        wx_idx, wy_idx = np.nonzero(wet_dil)
        if len(wx_idx) > 0:
            x0 = gx[wx_idx]
            y0 = gy[wy_idx]
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            z0 = np.full_like(x0, water_z)
            # Counter-clockwise triangles in Z-up (+Z normal -> +Y normal in Y-up glTF)
            v00 = np.stack([x0, y0, z0], axis=1)
            v10 = np.stack([x1, y0, z0], axis=1)
            v11 = np.stack([x1, y1, z0], axis=1)
            v01 = np.stack([x0, y1, z0], axis=1)
            r_pos = np.stack([v00, v10, v11, v00, v11, v01], axis=1).reshape(-1, 3)
            r_nrm = np.zeros_like(r_pos)
            r_nrm[:, 2] = 1.0
            r_st = (r_pos[:, :2] - np.array([ORIGIN_X, ORIGIN_Y], dtype=np.float32)) * 0.05
            add_to_bucket('River', r_pos, r_nrm, r_st, weld_grid=0.0)
            log(f'  Generated {len(r_pos) // 3} River valley water triangles at z={water_z}m')

    # 4. Hero Banyan Tree (placed overlooking the riverbank clearing at Z-up (585.0, 782.0, 1.5))
    log('4/6 Parsing Banyan.usd ...')
    ban_meshes = parse_usd_meshes(os.path.join(USD_DIR, 'elements', 'Banyan', 'Banyan.usd'))
    banyan_Clearing_pos = np.array([585.0, 782.0, 1.5], dtype=np.float32)
    for m in ban_meshes:
        for mat_name, face_idx in m['subsets']:
            res = build_triangles_from_mesh(m, face_idx)
            if res is None:
                continue
            pos, nrm, st = res
            grid = 0.24 if 'Leaves' in mat_name else 0.14
            pos_d, nrm_d, st_d = decimate_triangles(pos, nrm, st, grid)
            add_to_bucket(mat_name, pos_d * 0.75 + banyan_Clearing_pos, nrm_d, st_d, weld_grid=0.0)

    # 5. Instanced Trees & Foliage around the Pyramid & River clearing
    def scatter_element(elem_folder, classes_file, pi_file, max_dist, max_instances, proto_weld_grid, center_xy=(555.0, 785.0)):
        classes_path = os.path.join(USD_DIR, 'elements', elem_folder, classes_file)
        pi_path = os.path.join(USD_DIR, 'elements', elem_folder, pi_file)
        proto_meshes = parse_usd_meshes(classes_path)
        instancers = parse_point_instancer(pi_path)

        # Pre-simplify each prototype's material subsets in local space
        proto_dict = {}
        for pm in proto_meshes:
            pname = pm['name']
            subs = []
            if pm['subsets']:
                for mat_name, face_idx in pm['subsets']:
                    if mat_name not in MATERIALS:
                        continue
                    res = build_triangles_from_mesh(pm, face_idx)
                    if res is None:
                        continue
                    p, n, st = decimate_triangles(*res, proto_weld_grid)
                    if len(p) > 0:
                        subs.append((mat_name, p, n, st))
            elif pm['default_mat'] and pm['default_mat'] in MATERIALS:
                res = build_triangles_from_mesh(pm)
                if res is not None:
                    p, n, st = decimate_triangles(*res, proto_weld_grid)
                    if len(p) > 0:
                        subs.append((pm['default_mat'], p, n, st))
            if subs:
                proto_dict[pname] = subs

        if not proto_dict:
            return
        fallback_subs = next(iter(proto_dict.values()))
        total_added = 0
        c_xy = np.array(center_xy, dtype=np.float32)
        active = []
        for pname, (pos_all, sc_all, ori_all) in instancers.items():
            if len(pos_all) == 0:
                continue
            d = np.linalg.norm(pos_all[:, :2] - c_xy, axis=1)
            # Keep scattered trees/shrubs/grass on the riverbanks above the deep river channel (z >= -1.05m).
            mask = np.nonzero((d < max_dist) & (pos_all[:, 2] >= -1.05))[0]
            if len(mask) > 0:
                active.append((pname, pos_all, sc_all, ori_all, d, mask))

        for pname, pos_all, sc_all, ori_all, d, mask in active:
            subs = proto_dict.get(pname)
            if subs is None:
                for k, v in proto_dict.items():
                    if pname.endswith(k) or k.endswith(pname):
                        subs = v
                        break
            if subs is None:
                subs = fallback_subs
            per_proto_max = max(1, max_instances // max(1, len(active)))
            if len(mask) > per_proto_max:
                order = np.argsort(d[mask])
                half = per_proto_max // 2
                chosen = np.concatenate([
                    mask[order[:half]],
                    mask[order[half::max(1, (len(mask) - half) // max(1, per_proto_max - half))]][:per_proto_max - half]
                ])
                mask = chosen
            pos_sel = pos_all[mask]
            sc_sel = sc_all[mask] if sc_all is not None and len(sc_all) == len(pos_all) else np.ones_like(pos_sel)
            ori_sel = ori_all[mask] if ori_all is not None and len(ori_all) == len(pos_all) else np.tile([1, 0, 0, 0], (len(mask), 1))
            rot_sel = quat_to_rotmat(ori_sel)

            for mat_name, p_loc, n_loc, st_loc in subs:
                p_scaled = p_loc[None, :, :] * sc_sel[:, None, :]
                p_world = np.einsum('mvi,mji->mvj', p_scaled, rot_sel) + pos_sel[:, None, :]
                n_world = np.einsum('vi,mji->mvj', n_loc, rot_sel)
                st_world = np.broadcast_to(st_loc[None, :, :], (len(mask), len(st_loc), 2))
                add_to_bucket(
                    mat_name,
                    p_world.reshape(-1, 3),
                    n_world.reshape(-1, 3),
                    st_world.reshape(-1, 2),
                    weld_grid=0.0,
                )
            total_added += len(mask)
        log(f'  Scattered {total_added:4d} instances from {elem_folder}')

    log('5/6 Scattering jungle trees & foliage ...')
    scatter_element('QueenForest', 'queenforest_classes.usda', 'PI_S_QueenForest.usd', max_dist=135.0, max_instances=48, proto_weld_grid=0.18)
    scatter_element('RiverForest', 'riverforest_classes.usda', 'PI_S_RiverForest.usd', max_dist=125.0, max_instances=40, proto_weld_grid=0.22)
    scatter_element('RiverSapling', 'riversapling_classes.usda', 'PI_RiverSapling.usd', max_dist=105.0, max_instances=75, proto_weld_grid=0.10)
    scatter_element('Shrub', 'shrub_classes.usda', 'PI_Shrub.usd', max_dist=95.0, max_instances=200, proto_weld_grid=0.06)
    scatter_element('Pyramid_Grass_B', 'grass_B_classes.usda', 'PI_Pyramid_GrassB.usd', max_dist=75.0, max_instances=320, proto_weld_grid=0.05)
    scatter_element('Grass_A', 'grass_A_classes.usda', 'PI_Grass_A.usd', max_dist=75.0, max_instances=220, proto_weld_grid=0.05)

    # 6. Serialize glTF 2.0 + binary buffer
    log('6/6 Writing JungleRuins_Scene.gltf + JungleRuins_Scene.bin ...')
    bin_path = os.path.join(JUNGLE_DIR, 'JungleRuins_Scene.bin')
    gltf_path = os.path.join(JUNGLE_DIR, 'JungleRuins_Scene.gltf')

    gltf_textures = []
    gltf_images = []
    tex_index_by_relpath = {}

    def get_tex_idx(rel_path):
        if not rel_path or not os.path.exists(os.path.join(JUNGLE_DIR, rel_path)):
            return None
        if rel_path not in tex_index_by_relpath:
            idx = len(gltf_images)
            gltf_images.append({'uri': rel_path})
            gltf_textures.append({'source': idx, 'sampler': 0})
            tex_index_by_relpath[rel_path] = idx
        return tex_index_by_relpath[rel_path]

    gltf_materials = []
    gltf_meshes = []
    gltf_nodes = []
    gltf_buffer_views = []
    gltf_accessors = []

    total_verts = 0
    total_tris = 0

    with open(bin_path, 'wb') as bf:
        for mat_name, chunks in mat_buckets.items():
            if not chunks:
                continue
            pos = np.concatenate([c[0] for c in chunks], axis=0)
            nrm = np.concatenate([c[1] for c in chunks], axis=0)
            uv = np.concatenate([c[2] for c in chunks], axis=0)
            pos, nrm, uv = dedupe_triangles(pos, nrm, uv)
            indexed = index_triangles(pos, nrm, uv)
            if indexed is None:
                continue
            u_pos, u_nrm, u_uv, indices = indexed
            v_cnt = len(u_pos)
            i_cnt = len(indices)
            if v_cnt == 0 or i_cnt == 0:
                continue

            total_verts += v_cnt
            total_tris += i_cnt // 3

            # Material entry
            bc_rel, nm_rel, orm_rel, dbl, alpha_mode, base_factor, rough_factor = MATERIALS[mat_name]
            pbr = {
                'baseColorFactor': base_factor,
                'metallicFactor': 0.0,
                'roughnessFactor': rough_factor,
            }
            bc_idx = get_tex_idx(bc_rel)
            if bc_idx is not None:
                pbr['baseColorTexture'] = {'index': bc_idx}
            orm_idx = get_tex_idx(orm_rel)
            if orm_idx is not None:
                pbr['metallicRoughnessTexture'] = {'index': orm_idx}

            mat_obj = {
                'name': mat_name,
                'pbrMetallicRoughness': pbr,
                'doubleSided': dbl,
                'alphaMode': alpha_mode,
            }
            if alpha_mode == 'MASK':
                mat_obj['alphaCutoff'] = 0.5
            nm_idx = get_tex_idx(nm_rel)
            if nm_idx is not None:
                mat_obj['normalTexture'] = {'index': nm_idx}

            mat_idx = len(gltf_materials)
            gltf_materials.append(mat_obj)

            # Write buffer views (aligned to 4 bytes)
            def write_blob(arr, target):
                off = bf.tell()
                data = arr.tobytes()
                bf.write(data)
                pad = (4 - (len(data) % 4)) % 4
                if pad:
                    bf.write(b'\0' * pad)
                bv_idx = len(gltf_buffer_views)
                bv = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data)}
                if target is not None:
                    bv['target'] = target
                gltf_buffer_views.append(bv)
                return bv_idx

            bv_pos = write_blob(u_pos, 34962)
            bv_nrm = write_blob(u_nrm, 34962)
            bv_uv = write_blob(u_uv, 34962)
            bv_idx = write_blob(indices, 34963)

            acc_pos = len(gltf_accessors)
            gltf_accessors.append({
                'bufferView': bv_pos,
                'componentType': 5126,
                'count': v_cnt,
                'type': 'VEC3',
                'min': u_pos.min(axis=0).tolist(),
                'max': u_pos.max(axis=0).tolist(),
            })
            acc_nrm = len(gltf_accessors)
            gltf_accessors.append({
                'bufferView': bv_nrm,
                'componentType': 5126,
                'count': v_cnt,
                'type': 'VEC3',
            })
            acc_uv = len(gltf_accessors)
            gltf_accessors.append({
                'bufferView': bv_uv,
                'componentType': 5126,
                'count': v_cnt,
                'type': 'VEC2',
            })
            acc_idx = len(gltf_accessors)
            gltf_accessors.append({
                'bufferView': bv_idx,
                'componentType': 5125,
                'count': i_cnt,
                'type': 'SCALAR',
            })

            mesh_idx = len(gltf_meshes)
            gltf_meshes.append({
                'name': mat_name,
                'primitives': [{
                    'attributes': {
                        'POSITION': acc_pos,
                        'NORMAL': acc_nrm,
                        'TEXCOORD_0': acc_uv,
                    },
                    'indices': acc_idx,
                    'material': mat_idx,
                    'mode': 4,
                }],
            })
            gltf_nodes.append({'name': mat_name, 'mesh': mesh_idx})

        total_bytes = bf.tell()

    gltf_doc = {
        'asset': {'version': '2.0', 'generator': 'Bistro build_jungle_gltf.py'},
        'scene': 0,
        'scenes': [{'name': 'JungleRuins', 'nodes': list(range(len(gltf_nodes)))}],
        'nodes': gltf_nodes,
        'meshes': gltf_meshes,
        'materials': gltf_materials,
        'textures': gltf_textures,
        'images': gltf_images,
        'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}],
        'bufferViews': gltf_buffer_views,
        'accessors': gltf_accessors,
        'buffers': [{'uri': 'JungleRuins_Scene.bin', 'byteLength': total_bytes}],
    }
    with open(gltf_path, 'w', encoding='utf-8') as f:
        json.dump(gltf_doc, f, indent=2)

    log(f'Built {gltf_path}: {total_verts:,} vertices, {total_tris:,} triangles, '
        f'{len(gltf_materials)} materials in {time.time() - t0:.1f}s ({total_bytes / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
