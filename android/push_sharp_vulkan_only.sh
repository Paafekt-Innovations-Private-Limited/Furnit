#!/bin/bash
# Push SHARP .pte from sharp_vulkan_only/ (or backup folder) to device.
# Put **hybrid** INT8 sidecars here too: sharp_split_part1_int8.pte + sharp_split_part2_int8.pte
# (same directory as Vulkan Part3/4 — no separate models_cpu push required).
#
# Usage:
#   ./push_sharp_vulkan_only.sh
#   ./push_sharp_vulkan_only.sh /path/to/folder
#
# Destination: models_vulkan (etVulkan APK)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:-$SCRIPT_DIR/sharp_vulkan_only}"
DEST="/sdcard/Android/data/com.furnit.android/files/models_vulkan"

if ! command -v adb &> /dev/null; then
  echo "Error: adb not found in PATH"
  exit 1
fi

if ! adb devices | grep -q "device$"; then
  echo "Error: No device connected. Connect device and enable USB debugging."
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: Model dir not found: $MODEL_DIR"
  echo "Run ./export_sharp_vulkan_only.sh first."
  exit 1
fi

echo "Pushing Vulkan-only SHARP models to device..."
echo "Source:      $MODEL_DIR"
echo "Destination: $DEST"
echo ""

adb shell "mkdir -p $DEST"

for artifact in "$MODEL_DIR"/sharp_split_part*.pte "$MODEL_DIR"/sharp_split_part*.pte.manifest.json; do
  if [ -f "$artifact" ]; then
    file="$(basename "$artifact")"
    size="$(ls -lh "$artifact" | awk '{print $5}')"
    echo "Pushing $file ($size)..."
    adb push "$artifact" "$DEST/$file"
  fi
done

# Part1 warmup needs part1_test_patch_f32.bin (from export_sharp_executorch_split4.py --part1-only)
if [ -f "$MODEL_DIR/part1_test_patch_f32.bin" ]; then
  echo "Pushing part1_test_patch_f32.bin (for Part1 warmup)..."
  adb push "$MODEL_DIR/part1_test_patch_f32.bin" "$DEST/part1_test_patch_f32.bin"
fi

echo ""
echo "Done! Vulkan-only models pushed to $DEST"
