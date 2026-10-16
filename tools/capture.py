#!/usr/bin/env python3
"""Captures screenshots of the running app at given fly-through times (for reviewing the camera path).

Usage: tools/capture.py OUT_DIR t1 t2 ...      (times in seconds)
       tools/capture.py OUT_DIR --camera "x,y,z,tx,ty,tz"
"""
import os
import subprocess
import sys
import time

PKG = 'com.android.demo.bistro/.MainActivity'


def shot(path, extras):
    subprocess.run(['adb', 'shell', 'am', 'start', '-n', PKG] + extras, check=True, capture_output=True)
    time.sleep(1.5)
    with open(path, 'wb') as f:
        f.write(subprocess.run(['adb', 'exec-out', 'screencap', '-p'], check=True, capture_output=True).stdout)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    args = sys.argv[2:]
    if args and args[0] == '--camera':
        shot(os.path.join(out, 'camera.png'), ['--es', 'camera', args[1], '--ez', 'hud', 'false'])
        return
    for t in args:
        shot(os.path.join(out, 't%06.1f.png' % float(t)),
             ['--ef', 'startTime', t, '--ez', 'freeze', 'true', '--ez', 'hud', 'false'])
        print('captured t=%s' % t, flush=True)


if __name__ == '__main__':
    main()
