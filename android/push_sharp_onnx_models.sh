#!/bin/bash
# Push SHARP ONNX model files to an Android device.
#
# Usage:
#   ./push_sharp_onnx_models.sh [MODEL_DIR]
#
# MODEL_DIR defaults to: sharp_onnx_models
#
# The app looks for models here on-device:
#   /sdcard/Android/data/com.furnit.android/files/models
#
# Notes:
# - Split ONNX (sharp_part{1..4}.onnx + .data) is preferred when present.
# - Regular ONNX (sharp_fp32_aligned.onnx + .data) is used as fallback.
set -euo pipefail

MODEL_DIR="${1:-sharp_onnx_models}"
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
  exit 1
fi

echo "Pushing SHARP ONNX model files to device..."
echo "Source:      $MODEL_DIR"
echo "Destination: $DEST"
echo ""

adb shell "mkdir -p $DEST"

shopt -s nullglob
FILES=(
  "$MODEL_DIR"/sharp_part*.onnx
  "$MODEL_DIR"/sharp_part*.onnx.data
  "$MODEL_DIR"/sharp_fp32_aligned.onnx
  "$MODEL_DIR"/sharp_fp32_aligned.onnx.data
  "$MODEL_DIR"/sharp_mixed_fp16.onnx
  "$MODEL_DIR"/sharp_mixed_fp16.onnx.data
)

if [ ${#FILES[@]} -eq 0 ]; then
  echo "Error: No SHARP ONNX model files found in $MODEL_DIR"
  echo "Expected split files like: sharp_part1.onnx / sharp_part1.onnx.data"
  exit 1
fi

for path in "${FILES[@]}"; do
  if [ -f "$path" ]; then
    file="$(basename "$path")"
    size="$(ls -lh "$path" | awk '{print $5}')"
    echo "Pushing $file ($size)..."
    adb push "$path" "$DEST/$file" >/dev/null
  fi
done

echo ""
echo "Done! ONNX model files pushed successfully."
echo "In the app: Settings > Developer > Inference Backend = ONNX (default)"

