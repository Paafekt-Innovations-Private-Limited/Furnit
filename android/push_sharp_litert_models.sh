#!/bin/bash
# Push SHARP LiteRT (TFLite) model files to an Android device.
#
# Usage:
#   ./push_sharp_litert_models.sh [MODEL_DIR]
#
# MODEL_DIR defaults to: sharp_litert_models
#
# The app looks for models here on-device:
#   /sdcard/Android/data/com.furnit.android/files/models
#   /data/local/tmp/furnit
#
# Notes:
# - Split LiteRT is preferred when present:
#     sharp_part{1..4}_fp16.tflite
# - Single model fallback (very large) is NOT recommended:
#     vit_gaussian_fp16.tflite
set -euo pipefail

MODEL_DIR="${1:-sharp_litert_models}"
DEST1="/sdcard/Android/data/com.furnit.android/files/models"
DEST2="/data/local/tmp/furnit"

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
  exit 1
fi

echo "Pushing SHARP LiteRT model files to device..."
echo "Source:         $MODEL_DIR"
echo "Destination 1:  $DEST1"
echo "Destination 2:  $DEST2"
echo ""

adb shell "mkdir -p $DEST1"
adb shell "mkdir -p $DEST2"

shopt -s nullglob
FILES=(
  "$MODEL_DIR"/sharp_part*_fp16.tflite
  "$MODEL_DIR"/vit_gaussian_fp16.tflite
)

if [ ${#FILES[@]} -eq 0 ]; then
  echo "Error: No LiteRT model files found in $MODEL_DIR"
  echo "Expected: sharp_part1_fp16.tflite ... sharp_part4_fp16.tflite"
  exit 1
fi

for path in "${FILES[@]}"; do
  if [ -f "$path" ]; then
    file="$(basename "$path")"
    size="$(ls -lh "$path" | awk '{print $5}')"
    echo "Pushing $file ($size)..."
    adb push "$path" "$DEST1/$file" >/dev/null
    adb push "$path" "$DEST2/$file" >/dev/null
  fi
done

echo ""
echo "Done! LiteRT model files pushed successfully."
