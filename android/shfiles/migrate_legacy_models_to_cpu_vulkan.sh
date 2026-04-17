#!/usr/bin/env bash
# One-time (or repeat-safe) copy from legacy files/models/sharp_split*.pte into
# files/models_cpu/ and files/models_cpuvulkan_hybrid/ based on filename (*vulkan* → hybrid dir).
#
# The app also migrates legacy files/models_vulkan → models_cpuvulkan_hybrid on launch.
#
# Usage: cd android && ./migrate_legacy_models_to_cpu_vulkan.sh
set -euo pipefail
command -v adb >/dev/null || { echo "adb not found"; exit 1; }
adb devices | grep -q "device$" || { echo "No device"; exit 1; }

BASE="/sdcard/Android/data/com.furnit.android/files/models"
CPU="/sdcard/Android/data/com.furnit.android/files/models_cpu"
HYBRID="/sdcard/Android/data/com.furnit.android/files/models_cpuvulkan_hybrid"

adb shell "mkdir -p '$CPU' '$HYBRID'"

echo "Migrating sharp_split*.pte from $BASE → models_cpu | models_cpuvulkan_hybrid …"
adb shell "for f in $BASE/sharp_split*.pte; do [ -f \"\$f\" ] || continue; b=\$(basename \"\$f\"); case \"\$b\" in *vulkan*) cp \"\$f\" \"$HYBRID/\" ;; *) cp \"\$f\" \"$CPU/\" ;; esac; done"

echo "models_cpu:"
adb shell "ls -1 $CPU 2>/dev/null | wc -l; ls -1 $CPU 2>/dev/null"
echo "models_cpuvulkan_hybrid:"
adb shell "ls -1 $HYBRID 2>/dev/null | wc -l; ls -1 $HYBRID 2>/dev/null"
echo "Done."
