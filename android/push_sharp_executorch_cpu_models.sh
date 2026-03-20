#!/bin/bash
# Push SHARP ExecuTorch CPU (portable) .pte to device for "CPU ExecuTorch INT8".
# Usage: ./push_sharp_executorch_cpu_models.sh [MODEL_DIR]
# MODEL_DIR default: executorch_models
set -euo pipefail

MODEL_DIR="${1:-executorch_models}"
# etCpu APK reads ExecuTorch .pte from models_cpu (internal + external app storage).
DEST="/sdcard/Android/data/com.furnit.android/files/models_cpu"

[ -d "$MODEL_DIR" ] || { echo "Error: $MODEL_DIR not found"; exit 1; }
command -v adb &>/dev/null || { echo "Error: adb not found"; exit 1; }
adb devices | grep -q "device$" || { echo "Error: No device"; exit 1; }

adb shell "mkdir -p $DEST"

# Working set: Part4b_int8/fp16/.pte single; tile_00..15 + tile_full + tile_b4. Missing files skipped.
for f in sharp_split_part1_int8.pte sharp_split_part2_int8.pte sharp_split_part3_int8.pte \
         sharp_split_part4a_chunk_512.pte sharp_split_part4a_chunk_65.pte \
         sharp_split_part4b_int8.pte sharp_split_part4b_fp16.pte sharp_split_part4b.pte \
         sharp_split_part1_b4_int8.pte sharp_split_part2_b4_int8.pte \
         sharp_split_part4b_tile_00.pte sharp_split_part4b_tile_01.pte sharp_split_part4b_tile_02.pte \
         sharp_split_part4b_tile_03.pte sharp_split_part4b_tile_04.pte sharp_split_part4b_tile_05.pte \
         sharp_split_part4b_tile_06.pte sharp_split_part4b_tile_07.pte sharp_split_part4b_tile_08.pte \
         sharp_split_part4b_tile_09.pte sharp_split_part4b_tile_10.pte sharp_split_part4b_tile_11.pte \
         sharp_split_part4b_tile_12.pte sharp_split_part4b_tile_13.pte sharp_split_part4b_tile_14.pte \
         sharp_split_part4b_tile_15.pte sharp_split_part4b_tile_full.pte sharp_split_part4b_tile_b4.pte \
         sharp_split_part1.pte sharp_split_part1_fp16.pte sharp_split_part2.pte sharp_split_part2_fp16.pte \
         sharp_split_part3.pte sharp_split_part3_fp16.pte \
         sharp_split_part1_b4.pte sharp_split_part2_b4.pte; do
  [ -f "$MODEL_DIR/$f" ] && adb push "$MODEL_DIR/$f" "$DEST/$f"
done

echo "Done. $DEST"
