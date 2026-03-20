#!/bin/bash
# Push Vulkan-only SHARP .pte files from sharp_vulkan_only/ to device.
#
# Usage:
#   ./push_sharp_vulkan_only.sh
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

for pte in "$MODEL_DIR"/sharp_split_part*.pte; do
  if [ -f "$pte" ]; then
    file="$(basename "$pte")"
    size="$(ls -lh "$pte" | awk '{print $5}')"
    echo "Pushing $file ($size)..."
    adb push "$pte" "$DEST/$file"
  fi
done

echo ""
echo "Done! Vulkan-only models pushed."
