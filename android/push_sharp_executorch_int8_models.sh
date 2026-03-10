#!/bin/bash
# Push SHARP ExecuTorch INT8 model files to an Android device.
#
# Expects:
#   - executorch_int8_models/ : sharp_split_part1_int8.pte, part2_int8.pte, part3_int8.pte
#   - executorch_models/       : sharp_split_part4a_chunk_512.pte, part4a_chunk_65.pte, sharp_split_part4b.pte
#   - (optional)               : sharp_split_part4b_int8.pte (INT8 decoder; C++ full pipeline prefers this when present)
#
# Usage:
#   ./push_sharp_executorch_int8_models.sh
#
# Or with custom dirs:
#   ./push_sharp_executorch_int8_models.sh [INT8_DIR] [CHUNKED_PART4_DIR]
#
# Destination: /sdcard/Android/data/com.furnit.android/files/models/
#
# If adb push says "target is not a directory":
#   1. Install and run the Furnit app once (open it) so Android creates the path.
#   2. Then run this script again, or: adb shell mkdir -p "/sdcard/Android/data/com.furnit.android/files/models" && adb push ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INT8_DIR="${1:-$SCRIPT_DIR/executorch_int8_models}"
CHUNKED_DIR="${2:-$SCRIPT_DIR/executorch_models}"
DEST="/sdcard/Android/data/com.furnit.android/files/models"

if ! command -v adb &> /dev/null; then
  echo "Error: adb not found in PATH"
  exit 1
fi

if ! adb devices | grep -q "device$"; then
  echo "Error: No device connected. Connect device and enable USB debugging."
  exit 1
fi

echo "Pushing SHARP ExecuTorch INT8 model files to device..."
echo "  INT8 parts (1-3):    $INT8_DIR"
echo "  Chunked Part 4:     $CHUNKED_DIR"
echo "  Destination:        $DEST"
echo ""

# Create destination (if app was never run, parent path may not exist — run the app once then retry)
adb shell "mkdir -p $DEST" || true

FILES_TO_PUSH=()

# INT8 encoder parts
for f in "$INT8_DIR"/sharp_split_part1_int8.pte "$INT8_DIR"/sharp_split_part2_int8.pte "$INT8_DIR"/sharp_split_part3_int8.pte; do
  if [ -f "$f" ] && [ -s "$f" ]; then
    FILES_TO_PUSH+=("$f")
  else
    echo "Warning: Missing or empty $f"
  fi
done

# Chunked Part 4 (FP32)
for f in "$CHUNKED_DIR"/sharp_split_part4a_chunk_512.pte "$CHUNKED_DIR"/sharp_split_part4a_chunk_65.pte "$CHUNKED_DIR"/sharp_split_part4b.pte; do
  if [ -f "$f" ] && [ -s "$f" ]; then
    FILES_TO_PUSH+=("$f")
  else
    echo "Warning: Missing or empty $f"
  fi
done

# Optional: Part1/Part2 batch=4 (C++ patch batching: 35 → ~11 launches)
for f in "$INT8_DIR"/sharp_split_part1_b4_int8.pte "$INT8_DIR"/sharp_split_part2_b4_int8.pte; do
  if [ -f "$f" ] && [ -s "$f" ]; then
    FILES_TO_PUSH+=("$f")
  fi
done

# Optional: INT8 Part 4b (C++ full pipeline prefers this when present)
for dir in "$CHUNKED_DIR" "$INT8_DIR"; do
  f="$dir/sharp_split_part4b_int8.pte"
  if [ -f "$f" ] && [ -s "$f" ]; then
    FILES_TO_PUSH+=("$f")
    break
  fi
done

# Part4b 16-tile models (optional; for tiled path when Settings > Part4b tiled = ON)
TILE_FILES=(
  sharp_split_part4b_tile_full.pte
  sharp_split_part4b_tile_00.pte sharp_split_part4b_tile_01.pte sharp_split_part4b_tile_02.pte sharp_split_part4b_tile_03.pte
  sharp_split_part4b_tile_04.pte sharp_split_part4b_tile_05.pte sharp_split_part4b_tile_06.pte sharp_split_part4b_tile_07.pte
  sharp_split_part4b_tile_08.pte sharp_split_part4b_tile_09.pte sharp_split_part4b_tile_10.pte sharp_split_part4b_tile_11.pte
  sharp_split_part4b_tile_12.pte sharp_split_part4b_tile_13.pte sharp_split_part4b_tile_14.pte sharp_split_part4b_tile_15.pte
)
for f in "${TILE_FILES[@]}"; do
  for dir in "$INT8_DIR" "$CHUNKED_DIR"; do
    if [ -f "$dir/$f" ] && [ -s "$dir/$f" ]; then
      FILES_TO_PUSH+=("$dir/$f")
      break
    fi
  done
done

if [ ${#FILES_TO_PUSH[@]} -eq 0 ]; then
  echo "Error: No valid .pte files found."
  echo "Export INT8 parts:  python export_sharp_executorch_int8_split4.py"
  echo "Export chunked P4:  python export_sharp_executorch_split4.py --chunked-part4"
  exit 1
fi

for path in "${FILES_TO_PUSH[@]}"; do
  file="$(basename "$path")"
  size="$(ls -lh "$path" | awk '{print $5}')"
  echo "Pushing $file ($size) -> $DEST..."
  adb push "$path" "$DEST/$file"
done

echo ""
echo "Done! ExecuTorch INT8 model files pushed."
echo "In the app: Settings > Developer > Inference Backend = ExecuTorch INT8"
echo ""
echo "For 16-tile Part4b (faster):"
echo "  - Ensure tile .pte files were pushed (17 files: tile_full + tile_00..15)."
echo "  - Settings > Developer > Part4b tiled (experimental) = ON, Stable mode = OFF."
echo "  - Reinstall the app if logcat shows 'libsharp_executorch_tiles.so not found'."
echo ""
