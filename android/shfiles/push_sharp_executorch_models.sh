#!/bin/bash
# Push SHARP ExecuTorch split model files to an Android device.
#
# Usage:
#   ./push_sharp_executorch_models.sh [MODEL_DIR]
#
# MODEL_DIR defaults to: executorch_models
#
# The app copies from external to internal storage on first run for faster mmap.
# Export with XNNPACK (default) for CPU-optimized kernels:
#   python export_sharp_executorch_split4.py --weights /path/to/sharp.pt
#
# Notes:
# - XNNPACK backend: fast CPU kernels, stable across Android devices.
# - Models are copied to internal storage on first load.
set -euo pipefail

MODEL_DIR="${1:-executorch_models}"
# Portable / CPU-oriented split models → use with etCpu APK (models_cpu).
DEST="/sdcard/Android/data/com.furnit.android/files/models_cpu"

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
  echo "Export first: python export_sharp_executorch_split4.py --weights /path/to/sharp.pt"
  exit 1
fi

echo "Pushing SHARP ExecuTorch (XNNPACK) model files to device..."
echo "Source:      $MODEL_DIR"
echo "Destination: $DEST"
echo ""

adb shell "mkdir -p $DEST"

shopt -s nullglob
FILES=(
  "$MODEL_DIR"/sharp_split_part1.pte
  "$MODEL_DIR"/sharp_split_part2.pte
  "$MODEL_DIR"/sharp_split_part3.pte
  "$MODEL_DIR"/sharp_split_part4.pte
)

if [ ${#FILES[@]} -eq 0 ]; then
  echo "Error: No SHARP ExecuTorch files found in $MODEL_DIR"
  echo "Expected: sharp_split_part1.pte .. sharp_split_part4.pte"
  echo "Export first: python export_sharp_executorch_split4.py --weights /path/to/sharp.pt"
  exit 1
fi

for path in "${FILES[@]}"; do
  if [ -f "$path" ]; then
    file="$(basename "$path")"
    size="$(ls -lh "$path" | awk '{print $5}')"
    echo "Pushing $file ($size) -> $DEST..."
    adb push "$path" "$DEST/$file"
  fi
done

# Also push to /data/local/tmp/furnit/ (EXTRA_SEARCH_DIRS) so split models are found
# even if external storage path differs on device
EXTRA_DIR="/data/local/tmp/furnit"
if adb shell "mkdir -p $EXTRA_DIR" 2>/dev/null; then
  echo ""
  echo "Pushing to $EXTRA_DIR (fallback search dir)..."
  for path in "${FILES[@]}"; do
    if [ -f "$path" ]; then
      file="$(basename "$path")"
      echo "  $file..."
      adb push "$path" "$EXTRA_DIR/$file" >/dev/null 2>&1 || true
    fi
  done
fi

echo ""
echo "Done! ExecuTorch model files pushed successfully."
echo "In the app: Settings > Developer > Inference Backend = ExecuTorch"
echo "The app syncs models_cpu → internal storage on first load (see logcat: ExecuTorch model roots)."
echo ""
echo "Verify export used XNNPACK (not Vulkan):"
echo "  adb logcat | grep -i xnnpack"
echo "  (You should see XNNPACK-related logs, NOT Vulkan backend.)"
