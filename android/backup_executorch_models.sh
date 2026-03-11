#!/usr/bin/env bash
# Backup main ExecuTorch SHARP pipeline models to a date+timestamp folder.
# Usage: ./backup_executorch_models.sh [destination_root]
#   destination_root defaults to EXECUTORCH_BACKUP_ROOT or /Volumes/LaCie
# Example: ./backup_executorch_models.sh /Volumes/ExtremeFSSD

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/executorch_int8_models"
BACKUP_ROOT="${1:-${EXECUTORCH_BACKUP_ROOT:-/Volumes/LaCie}}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
BACKUP_DIR="${BACKUP_ROOT}/Furnit_ExecuTorch_models_${TIMESTAMP}"

# Main pipeline models only (Part1–3 INT8, Part4a chunks, Part4b single + tiled 16)
MAIN_MODELS=(
  "sharp_split_part1_int8.pte"
  "sharp_split_part2_int8.pte"
  "sharp_split_part3_int8.pte"
  "sharp_split_part4a_chunk_512.pte"
  "sharp_split_part4a_chunk_65.pte"
  "sharp_split_part4b.pte"
  "sharp_split_part4b_int8.pte"
  "sharp_split_part4b_tile_full.pte"
  "sharp_split_part4b_tile_00.pte"
  "sharp_split_part4b_tile_01.pte"
  "sharp_split_part4b_tile_02.pte"
  "sharp_split_part4b_tile_03.pte"
  "sharp_split_part4b_tile_04.pte"
  "sharp_split_part4b_tile_05.pte"
  "sharp_split_part4b_tile_06.pte"
  "sharp_split_part4b_tile_07.pte"
  "sharp_split_part4b_tile_08.pte"
  "sharp_split_part4b_tile_09.pte"
  "sharp_split_part4b_tile_10.pte"
  "sharp_split_part4b_tile_11.pte"
  "sharp_split_part4b_tile_12.pte"
  "sharp_split_part4b_tile_13.pte"
  "sharp_split_part4b_tile_14.pte"
  "sharp_split_part4b_tile_15.pte"
)

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "Models dir not found: $MODELS_DIR"
  exit 1
fi

if [[ ! -d "$BACKUP_ROOT" ]]; then
  echo "Backup root not found or not mounted: $BACKUP_ROOT"
  echo "Usage: $0 [destination_root]   e.g. $0 /Volumes/ExtremeFSSD"
  exit 1
fi

mkdir -p "$BACKUP_DIR"
echo "Backing up main ExecuTorch models to: $BACKUP_DIR"

COPIED=0
SKIPPED=0
for name in "${MAIN_MODELS[@]}"; do
  src="${MODELS_DIR}/${name}"
  if [[ -f "$src" ]]; then
    if [[ -s "$src" ]]; then
      cp -p "$src" "$BACKUP_DIR/"
      echo "  $name"
      ((COPIED++)) || true
    else
      echo "  (skip empty) $name"
      ((SKIPPED++)) || true
    fi
  else
    echo "  (missing) $name"
    ((SKIPPED++)) || true
  fi
done

echo "Done: $COPIED copied, $SKIPPED skipped/missing. Backup at: $BACKUP_DIR"
