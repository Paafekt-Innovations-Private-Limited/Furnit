#!/usr/bin/env bash
# One-time (or repeat-safe) copy from legacy files/models/sharp_split*.pte into
# files/models_cpu/ and files/models_vulkan/ based on filename (*vulkan* → vulkan dir).
#
# Usage: cd android && ./migrate_legacy_models_to_cpu_vulkan.sh
set -euo pipefail
command -v adb >/dev/null || { echo "adb not found"; exit 1; }
adb devices | grep -q "device$" || { echo "No device"; exit 1; }

BASE="/sdcard/Android/data/com.furnit.android/files/models"
CPU="/sdcard/Android/data/com.furnit.android/files/models_cpu"
VK="/sdcard/Android/data/com.furnit.android/files/models_vulkan"

adb shell "mkdir -p '$CPU' '$VK'"

echo "Migrating sharp_split*.pte from $BASE → models_cpu | models_vulkan …"
adb shell "for f in $BASE/sharp_split*.pte; do [ -f \"\$f\" ] || continue; b=\$(basename \"\$f\"); case \"\$b\" in *vulkan*) cp \"\$f\" \"$VK/\" ;; *) cp \"\$f\" \"$CPU/\" ;; esac; done"

echo "models_cpu:"
adb shell "ls -1 $CPU 2>/dev/null | wc -l; ls -1 $CPU 2>/dev/null"
echo "models_vulkan:"
adb shell "ls -1 $VK 2>/dev/null | wc -l; ls -1 $VK 2>/dev/null"
echo "Done."
