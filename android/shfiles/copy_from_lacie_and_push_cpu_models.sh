#!/bin/bash
# Copy a **consistent** CPU SHARP split set from LaCie (or any folder) into **android/models_cpu/**,
# then push to device **models_cpu** (same name as the app uses on internal/external storage).
#
# Usage: ./copy_from_lacie_and_push_cpu_models.sh [SOURCE_DIR]
# Default SOURCE_DIR: /Volumes/LaCie/march10th2026/v2
# Alt (full INT8 set + single Part4b + all tiles): /Volumes/LaCie/BUModelsMar17th2026/executorch_int8_models
set -euo pipefail

cd "$(dirname "$0")"

LACIE="${1:-/Volumes/LaCie/march10th2026/v2}"
LOCAL_DIR="models_cpu"

# Must all exist on SOURCE_DIR (same export lineage).
REQUIRED=(
  sharp_split_part1_int8.pte
  sharp_split_part2_int8.pte
  sharp_split_part3_int8.pte
  sharp_split_part4a_chunk_512.pte
  sharp_split_part4a_chunk_65.pte
)

# Copied if present (batch-4 encoders speed Part1+2; pipeline works without them).
OPTIONAL=(
  sharp_split_part1_b4_int8.pte
  sharp_split_part2_b4_int8.pte
  sharp_split_part4b_int8.pte
  sharp_split_part4b_fp16.pte
  sharp_split_part4b.pte
  sharp_split_part4b_tile_00.pte
  sharp_split_part4b_tile_01.pte
  sharp_split_part4b_tile_02.pte
  sharp_split_part4b_tile_03.pte
  sharp_split_part4b_tile_04.pte
  sharp_split_part4b_tile_05.pte
  sharp_split_part4b_tile_06.pte
  sharp_split_part4b_tile_07.pte
  sharp_split_part4b_tile_08.pte
  sharp_split_part4b_tile_09.pte
  sharp_split_part4b_tile_10.pte
  sharp_split_part4b_tile_11.pte
  sharp_split_part4b_tile_12.pte
  sharp_split_part4b_tile_13.pte
  sharp_split_part4b_tile_14.pte
  sharp_split_part4b_tile_15.pte
  sharp_split_part4b_tile_full.pte
  sharp_split_part4b_tile_b4.pte
  sharp_split_part1.pte
  sharp_split_part1_fp16.pte
  sharp_split_part2.pte
  sharp_split_part2_fp16.pte
  sharp_split_part3.pte
  sharp_split_part3_fp16.pte
  sharp_split_part1_b4.pte
  sharp_split_part2_b4.pte
)

[ -d "$LACIE" ] || { echo "Error: source dir not found: $LACIE"; exit 1; }

for f in "${REQUIRED[@]}"; do
  [ -f "$LACIE/$f" ] || { echo "Error: missing required file: $LACIE/$f"; exit 1; }
done

if [[ ! -f "$LACIE/sharp_split_part4b_int8.pte" && ! -f "$LACIE/sharp_split_part4b.pte" && ! -f "$LACIE/sharp_split_part4b_tile_b4.pte" && ! -f "$LACIE/sharp_split_part4b_tile_00.pte" && ! -f "$LACIE/sharp_split_part4b_tile_full.pte" ]]; then
  echo "Error: need Part4b: single (part4b_int8 / part4b) and/or tiled (tile_b4 / tile_00 / tile_full) in $LACIE"
  exit 1
fi

mkdir -p "$LOCAL_DIR"
echo "Source: $LACIE"
echo "Staging: $LOCAL_DIR"

for f in "${REQUIRED[@]}"; do
  cp "$LACIE/$f" "$LOCAL_DIR/$f"
  echo "  copied $f"
done

for f in "${OPTIONAL[@]}"; do
  if [[ -f "$LACIE/$f" ]]; then
    cp "$LACIE/$f" "$LOCAL_DIR/$f"
    echo "  copied $f (optional)"
  fi
done

echo "Pushing to device…"
./push_sharp_executorch_cpu_models.sh "$LOCAL_DIR"
echo "Done. Open app and run inference; if internal cache was stale, cold-start or clear app data once."
