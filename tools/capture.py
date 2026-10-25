#!/usr/bin/env python3
"""Captures screenshots of the running app at given fly-through times (for reviewing the camera path).

Usage: tools/capture.py OUT_DIR [--env <env>] [--wait <sec>] t1 t2 ...      (times in seconds)
       tools/capture.py OUT_DIR [--env <env>] --camera "x,y,z,tx,ty,tz"
       tools/capture.py OUT_DIR --start-screen [--env <env>]
"""
import os
import subprocess
import sys
import time

PKG = 'com.android.demo.bistro/.MainActivity'


def shot(path, extras, wait_sec=1.8):
    subprocess.run(['adb', 'shell', 'am', 'start', '-n', PKG] + extras, check=True, capture_output=True)
    time.sleep(wait_sec)
    with open(path, 'wb') as f:
        f.write(subprocess.run(['adb', 'exec-out', 'screencap', '-p'], check=True, capture_output=True).stdout)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    args = sys.argv[2:]
    env_extra = []
    wait_sec = 1.8
    if '--env' in args:
        idx = args.index('--env')
        env_extra = ['--es', 'env', args[idx + 1]]
        del args[idx:idx + 2]
    if '--wait' in args:
        idx = args.index('--wait')
        wait_sec = float(args[idx + 1])
        del args[idx:idx + 2]
    if args and args[0] == '--start-screen':
        shot(os.path.join(out, 'start_screen.png'), env_extra + ['--ez', 'startScreen', 'true'], wait_sec=wait_sec)
        print('captured start_screen.png', flush=True)
        return
    if args and args[0] == '--camera':
        shot(os.path.join(out, 'camera.png'), env_extra + ['--es', 'camera', args[1], '--ez', 'hud', 'false'],
             wait_sec=wait_sec)
        return
    for i, t in enumerate(args):
        # Allow extra wait on the first frame in case the environment is streaming into VRAM.
        w = max(wait_sec, 4.0) if i == 0 else wait_sec
        shot(os.path.join(out, 't%06.1f.png' % float(t)),
             env_extra + ['--ef', 'startTime', t, '--ez', 'freeze', 'true', '--ez', 'hud', 'false'],
             wait_sec=w)
        print('captured t=%s' % t, flush=True)


if __name__ == '__main__':
    main()
