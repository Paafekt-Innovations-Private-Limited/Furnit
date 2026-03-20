#!/bin/bash
# Push Vulkan AAR-compat .pte models (from export_sharp_executorch_split4.py --vulkan-aar-compat) to device.
# Usage: ./push_sharp_vulkan_aar_compat.sh [DIR]
# Default DIR: sharp_vulkan_aar_compat
set -euo pipefail

DIR="${1:-sharp_vulkan_aar_compat}"
DEST="/sdcard/Android/data/com.furnit.android/files/models_vulkan"

[ -d "$DIR" ] || { echo "Error: $DIR not found. Run export with --vulkan-aar-compat --output-dir $DIR"; exit 1; }
command -v adb &>/dev/null || { echo "Error: adb not found"; exit 1; }
adb devices | grep -q "device$" || { echo "Error: No device"; exit 1; }

adb shell "mkdir -p $DEST"

for f in sharp_split_part1_vulkan_fp16.pte sharp_split_part2_vulkan_fp16.pte sharp_split_part3_vulkan_fp16.pte \
         sharp_split_part4.pte \
         sharp_split_part4a_chunk_512_vulkan.pte sharp_split_part4a_chunk_65_vulkan.pte sharp_split_part4b_vulkan.pte; do
  [ -f "$DIR/$f" ] && adb push "$DIR/$f" "$DEST/$f" || echo "Skip (not found): $f"
done

echo "Done. Models in $DEST"
