#!/bin/bash
# Push Native .pt SHARP split model files (.ptl) to an Android device.
#
# Export first:
#   python export_sharp_torchscript_split.py
#
# Usage:
#   ./push_sharp_torchscript_split.sh [MODEL_DIR]
#
# MODEL_DIR defaults to: executorch_models (where export script writes)
#
# The app looks for models here on-device:
#   /sdcard/Android/data/com.furnit.android/files/models
#
# Split mode avoids OOM: each part ~500-800MB, loaded one at a time.
# Full 2.5GB model crashes on load.
set -euo pipefail

MODEL_DIR="${1:-executorch_models}"
DEST="/sdcard/Android/data/com.furnit.android/files/models"

if ! command -v adb &> /dev/null; then
  echo "Error: adb not found in PATH"
  exit 1
fi

if ! adb devices | grep -q "device$"; then
  echo "Error: No device connected. Connect device and enable USB debugging."
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: MODEL_DIR not found: $MODEL_DIR"
  echo "Run: python export_sharp_torchscript_split.py"
  exit 1
fi

echo "Pushing Native .pt SHARP split models to device..."
echo "Source:      $MODEL_DIR"
echo "Destination: $DEST"
echo ""

adb shell "mkdir -p $DEST"

shopt -s nullglob
FILES=(
  "$MODEL_DIR"/sharp_scripted_part1.ptl
  "$MODEL_DIR"/sharp_scripted_part2.ptl
  "$MODEL_DIR"/sharp_scripted_part3.ptl
  "$MODEL_DIR"/sharp_scripted_part4.ptl
)

MISSING=0
for path in "${FILES[@]}"; do
  if [ ! -f "$path" ]; then
    echo "Missing: $(basename "$path")"
    MISSING=1
  fi
done
if [ "$MISSING" -eq 1 ]; then
  echo "Error: Export split models first: python export_sharp_torchscript_split.py"
  exit 1
fi

for path in "${FILES[@]}"; do
  file="$(basename "$path")"
  size="$(ls -lh "$path" | awk '{print $5}')"
  echo "Pushing $file ($size)..."
  adb push "$path" "$DEST/$file" 2>/dev/null || adb push "$path" "$DEST/$file"
done

echo ""
echo "Done! Native .pt split models pushed."
echo "In the app: Settings > Inference Backend = Native .pt"
