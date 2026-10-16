#!/usr/bin/env python3
"""Checks the camera path (camera_path.cpp) against the obstacle map written by the converter.

Reports the minimum horizontal clearance along the path and writes an annotated map image.
Usage: tools/check_path.py [out.png]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_map import control_points, path_samples  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGIN = (-62.46, -87.94)
CELL = 0.25


NUM_BANDS = 16
_band_points = {}


def load_bands():
    img = np.asarray(Image.open(os.path.join(ROOT, 'build_assets', 'obstacles.pgm'))) > 0
    return img.reshape(NUM_BANDS, img.shape[0] // NUM_BANDS, img.shape[1])


def band_points(bands, b):
    if b not in _band_points:
        ys, xs = np.nonzero(bands[b])
        _band_points[b] = np.stack([xs * CELL + ORIGIN[0] + CELL / 2, ys * CELL + ORIGIN[1] + CELL / 2], axis=1)
    return _band_points[b]


def clearance_3d(bands, p, vertical_margin=1.0):
    """Horizontal clearance from geometry within +-vertical_margin of the point's height."""
    lo = max(0, int(np.floor(p[1] - vertical_margin)))
    hi = min(NUM_BANDS - 1, int(np.floor(p[1] + vertical_margin)))
    best = 1e9
    for b in range(lo, hi + 1):
        pts = band_points(bands, b)
        if len(pts):
            best = min(best, float(np.sqrt(((pts - (p[0], p[2])) ** 2).sum(axis=1)).min()))
    return best


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'build_assets', 'path_check.png')
    bands = load_bands()
    grid = bands.any(axis=0)
    h, w = grid.shape
    pts = control_points()
    samples = path_samples(pts, per_segment=48)
    worst = []
    for i, s in enumerate(samples):
        worst.append((clearance_3d(bands, s), i, s))
    worst_sorted = sorted(worst, key=lambda x: x[0])
    print('minimum clearance %.2f m' % worst_sorted[0][0])
    seg_len = 48
    for d, i, s in worst_sorted[:15]:
        print('  %.2f m at segment %d (t=%.2f)  pos (%.1f, %.1f, %.1f)' % (d, i // seg_len, (i % seg_len) / seg_len, *s))

    scale = 3
    im = Image.fromarray(np.where(grid, 200, 30).astype(np.uint8)).convert('RGB').resize((w * scale, h * scale),
                                                                                       Image.NEAREST)
    dr = ImageDraw.Draw(im)
    to_px = lambda x, z: ((x - ORIGIN[0]) / CELL * scale, (z - ORIGIN[1]) / CELL * scale)
    for a, b in zip(worst[:-1], worst[1:]):
        col = (255, 60, 60) if min(a[0], b[0]) < 1.0 else ((255, 200, 0) if min(a[0], b[0]) < 1.6 else (60, 255, 60))
        dr.line([to_px(a[2][0], a[2][2]), to_px(b[2][0], b[2][2])], fill=col, width=3)
    for i, p in enumerate(pts):
        x, y = to_px(p[0], p[2])
        dr.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(0, 160, 255), width=2)
        dr.text((x + 7, y - 7), str(i), fill=(0, 200, 255))
    for gx in range(-60, 120, 10):
        x, _ = to_px(gx, 0)
        dr.text((x + 2, 2), str(gx), fill=(120, 120, 255))
    for gz in range(-80, 100, 10):
        _, y = to_px(0, gz)
        dr.text((2, y + 2), str(gz), fill=(120, 120, 255))
    im.save(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
