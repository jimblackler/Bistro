#!/usr/bin/env python3
"""Relaxes a rough list of camera control points toward positions with more clearance from geometry.

Each point is moved (horizontally, within a small radius) to maximise distance from obstacles in the
converter's obstacle map, with a penalty for moving. Prints C++ for camera_path.cpp.
Usage: tools/relax_path.py
"""
import os

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_path import clearance_3d, load_bands  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGIN = (-62.46, -87.94)
CELL = 0.25

# Rough route (x, y, z) in metres. See README for the overall shape of the fly-through.
ROUTE = [
    # Plaza: sweep from the north side of the plaza down toward the archway and street.
    (61.0, 5.5, 30.0),
    (50.0, 4.3, 31.0),
    (40.0, 3.8, 28.5),
    (28.0, 3.4, 23.5),
    (16.0, 3.1, 18.5),
    (5.0, 3.0, 14.5),
    (-5.0, 3.0, 11.5),
    (-12.5, 3.1, 4.5),
    # Low pass up the long street.
    (-14.0, 3.3, -6.0),
    (-5.0, 3.5, -16.0),
    (4.0, 3.6, -27.0),
    (12.5, 3.7, -37.5),
    (18.5, 3.8, -46.5),
    # Smooth street-level U-turn at the north end (staying low so building facades frame the sky).
    (22.5, 4.0, -54.0),
    (22.0, 4.2, -59.5),
    (17.5, 4.5, -58.5),
    (15.5, 4.8, -51.5),
    # Return pass down the long street at balcony height.
    (12.5, 5.2, -41.5),
    (4.5, 5.4, -29.5),
    (-4.5, 5.2, -18.5),
    (-14.5, 4.6, -7.5),
    # Round the corner and run east past the bistro terrace toward the plaza.
    (-12.5, 3.8, 2.0),
    (-8.5, 4.2, 10.5),
    (3.0, 5.0, 15.5),
    (16.0, 5.6, 20.0),
    (28.0, 6.2, 24.0),
    (40.0, 6.8, 27.5),
    (53.0, 7.2, 27.5),
    # Sweep around the plaza tree and back to the start.
    (66.0, 7.8, 21.5),
    (79.0, 8.0, 23.5),
    (83.5, 7.5, 31.0),
    (74.5, 6.5, 27.5),
]


def main():
    bands = load_bands()
    out = []
    for (x, y, z) in ROUTE:
        def clearance(px, pz):
            return clearance_3d(bands, (px, y, pz))

        best = (clearance(x, z), x, z)
        base = best[0]
        for dx in np.arange(-3.0, 3.01, 0.5):
            for dz in np.arange(-3.0, 3.01, 0.5):
                c = clearance(x + dx, z + dz)
                score = min(c, 4.0) - 0.25 * np.hypot(dx, dz)
                if score > min(best[0], 4.0) - 0.25 * np.hypot(best[1] - x, best[2] - z):
                    best = (c, x + dx, z + dz)
        out.append((best[1], y, best[2]))
        print('  (%.1f, %.1f, %.1f) clearance %.2f -> (%.1f, %.1f) clearance %.2f' % (x, y, z, base, best[1], best[2],
                                                                                    best[0]))
    print()
    for p in out:
        print('    {%.1ff, %.1ff, %.1ff},' % p)


if __name__ == '__main__':
    main()
