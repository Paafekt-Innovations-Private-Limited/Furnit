#!/bin/bash
# Remove SHARP split .pte from device models_cpu (external + internal when possible).
# Prevents mixed old Part4b + new Part1–4a (ExecuTorch error 18 / InvalidArgument on Part4b forward).
#
# Usage: ./clear_device_models_cpu.sh
# Requires: adb, unlocked device; internal clear needs debuggable app (run-as).
set -euo pipefail

PKG="com.furnit.android"
EXT_BASE="/sdcard/Android/data/${PKG}/files/models_cpu"

command -v adb &>/dev/null || { echo "Error: adb not found"; exit 1; }
adb devices | grep -q "device$" || { echo "Error: No device"; exit 1; }

echo "Clearing external: $EXT_BASE (all .pte — use before v2-only deploy)"
adb shell "mkdir -p $EXT_BASE && rm -f $EXT_BASE/*.pte" && echo "  OK (external)"

if adb shell "run-as $PKG true" 2>/dev/null; then
  echo "Clearing internal models_cpu (run-as $PKG)…"
  adb shell "run-as $PKG sh -c 'rm -f files/models_cpu/*.pte'" && echo "  OK (internal)"
else
  echo "  Skip internal: app not debuggable or run-as failed."
  echo "  → Uninstall app OR Settings → Apps → Furnit → Storage → Clear data to drop internal copies."
fi

echo "Done. Next: push a full matching set, e.g. ./push_sharp_executorch_cpu_models.sh /path/to/export"
