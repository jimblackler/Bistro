#!/usr/bin/env python3
"""Overlays the camera path (from environments/<env>/environment.json) onto a top-down screenshot of the scene.

Usage:
  1. adb shell am start -n com.android.demo.bistro/.MainActivity --es env <env> --es camera "EX,EY,EZ,TX,TY,TZ" --ez hud false
     (e.g. "20,190,5,20,0,4.99" for a view straight down), then grab a screenshot.
  2. tools/path_map.py screenshot.png out.png EX EY EZ TX TY TZ [--env <env>] [--loop <index>]
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOV_Y = math.radians(52.0)


def default_env_id():
    env_root = os.path.join(ROOT, 'environments')
    entries = []
    if os.path.isdir(env_root):
        for name in sorted(os.listdir(env_root)):
            path = os.path.join(env_root, name, 'environment.json')
            if os.path.isfile(path):
                with open(path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                entries.append((cfg.get('order', 0), cfg.get('id', name)))
    entries.sort()
    return entries[0][1] if entries else 'bistro'


def control_points(env_id=None, loop_index=0):
    if env_id is None:
        env_id = default_env_id()
    cfg_path = os.path.join(ROOT, 'environments', env_id, 'environment.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    loops = cfg.get('cameraLoops', [])
    if not loops:
        return []
    keyframes = loops[min(loop_index, len(loops) - 1)].get('keyframes', [])
    return [tuple(float(v) for v in kf[0]) for kf in keyframes]


def catmull_rom(p0, p1, p2, p3, t):
    def knot(ti, a, b):
        return ti + math.sqrt(max(np.linalg.norm(b - a), 1e-4))
    t0 = 0.0
    t1 = knot(t0, p0, p1)
    t2 = knot(t1, p1, p2)
    t3 = knot(t2, p2, p3)
    u = t1 + (t2 - t1) * t
    a1 = p0 * ((t1 - u) / (t1 - t0)) + p1 * ((u - t0) / (t1 - t0))
    a2 = p1 * ((t2 - u) / (t2 - t1)) + p2 * ((u - t1) / (t2 - t1))
    a3 = p2 * ((t3 - u) / (t3 - t2)) + p3 * ((u - t2) / (t3 - t2))
    b1 = a1 * ((t2 - u) / (t2 - t0)) + a2 * ((u - t0) / (t2 - t0))
    b2 = a2 * ((t3 - u) / (t3 - t1)) + a3 * ((u - t1) / (t3 - t1))
    return b1 * ((t2 - u) / (t2 - t1)) + b2 * ((u - t1) / (t2 - t1))


def path_samples(points, per_segment=32):
    pts = [np.array(p) for p in points]
    n = len(pts)
    out = []
    for s in range(n):
        for i in range(per_segment):
            out.append(catmull_rom(pts[(s - 1) % n], pts[s], pts[(s + 1) % n], pts[(s + 2) % n], i / per_segment))
    out.append(out[0])
    return np.array(out)


def project(points, eye, target, w, h):
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, [0, 1, 0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    rel = points - eye
    x, y, z = rel @ s, rel @ u, rel @ f
    fy = 1.0 / math.tan(FOV_Y / 2)
    fx = fy / (w / h)
    ndc_x = x * fx / z
    ndc_y = -y * fy / z
    return np.stack([(ndc_x * 0.5 + 0.5) * w, (ndc_y * 0.5 + 0.5) * h], axis=1)


def main():
    args = sys.argv[1:]
    env_id = None
    loop_idx = 0
    if '--env' in args:
        idx = args.index('--env')
        env_id = args[idx + 1]
        del args[idx:idx + 2]
    if '--loop' in args:
        idx = args.index('--loop')
        loop_idx = int(args[idx + 1])
        del args[idx:idx + 2]
    src, out = args[0], args[1]
    eye = np.array([float(v) for v in args[2:5]])
    target = np.array([float(v) for v in args[5:8]])
    im = Image.open(src).convert('RGB')
    w, h = im.size
    d = ImageDraw.Draw(im)
    pts = control_points(env_id=env_id, loop_index=loop_idx)
    samples = path_samples(pts)
    proj = project(samples, eye, target, w, h)
    d.line([tuple(p) for p in proj], fill=(255, 40, 40), width=4)
    cp = project(np.array(pts), eye, target, w, h)
    for i, p in enumerate(cp):
        d.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7], fill=(255, 255, 0))
        d.text((p[0] + 9, p[1] - 6), str(i), fill=(255, 255, 255))
    for gx in range(-60, 120, 10):
        line = project(np.array([[gx, 0, z] for z in range(-90, 100, 5)], dtype=float), eye, target, w, h)
        d.line([tuple(p) for p in line], fill=(0, 120, 255), width=1)
        d.text(tuple(line[0] + [2, 2]), str(gx), fill=(0, 200, 255))
    for gz in range(-90, 100, 10):
        line = project(np.array([[x, 0, gz] for x in range(-60, 120, 5)], dtype=float), eye, target, w, h)
        d.line([tuple(p) for p in line], fill=(0, 120, 255), width=1)
        d.text(tuple(line[0] + [2, 2]), str(gz), fill=(0, 200, 255))
    im.save(out)


if __name__ == '__main__':
    main()
