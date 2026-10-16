#!/bin/bash
# Usage: tools/bench.sh <time> [prop=value ...]
# Restarts the app frozen at the given fly-through time with debug properties applied, and prints the
# GPU timing reported by the renderer.
T=$1; shift
for kv in "$@"; do adb shell setprop "debug.bistro.${kv%%=*}" "${kv#*=}"; done
adb shell am force-stop com.android.demo.bistro
adb logcat -c
adb shell am start -n com.android.demo.bistro/.MainActivity --ef startTime "$T" --ez freeze true --ez hud false > /dev/null
sleep 14
adb logcat -d -s BistroThread | grep stats | tail -1 | sed 's/.*stats: //'
for kv in "$@"; do adb shell setprop "debug.bistro.${kv%%=*}" "''"; done
