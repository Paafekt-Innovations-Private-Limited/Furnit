#!/bin/bash
# Backup SHARP ExecuTorch INT8 model folders to a new date-named directory.
#
# Copies:
#   executorch_int8_models/  (Part1–3 INT8 .pte)
#   executorch_models/        (Part4 chunked: part4a_chunk_512, part4a_chunk_65, part4b)
#
# Usage:
#   ./backup_executorch_int8_models.sh
#   ./backup_executorch_int8_models.sh /Volumes/LaCie
#   ./backup_executorch_int8_models.sh /Volumes/LaCie Furnit_int8_backup_20260226
#
# Default backup root: /Volumes/LaCie
# Default folder name: Furnit_executorch_int8_backup_YYYYMMDD_HHMM (unique per run)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INT8_DIR="$SCRIPT_DIR/executorch_int8_models"
CHUNKED_DIR="$SCRIPT_DIR/executorch_models"
BACKUP_ROOT="${1:-/Volumes/LaCie}"
BACKUP_NAME="${2:-Furnit_executorch_int8_backup_$(date +%Y%m%d_%H%M)}"
BACKUP_DIR="$BACKUP_ROOT/$BACKUP_NAME"

if [[ ! -d "$INT8_DIR" ]]; then
  echo "Error: INT8 dir not found: $INT8_DIR"
  echo "Export first: python export_sharp_executorch_int8_split4.py"
  exit 1
fi
if [[ ! -d "$CHUNKED_DIR" ]]; then
  echo "Error: Chunked Part4 dir not found: $CHUNKED_DIR"
  echo "Export first: python export_sharp_executorch_split4.py --chunked-part4"
  exit 1
fi
if [[ ! -d "$BACKUP_ROOT" ]]; then
  echo "Error: Backup root not found or not mounted: $BACKUP_ROOT"
  exit 1
fi

mkdir -p "$BACKUP_DIR"
echo "Backing up ExecuTorch INT8 models..."
echo "  From: $SCRIPT_DIR"
echo "  To:   $BACKUP_DIR"
echo ""

cp -R "$INT8_DIR" "$BACKUP_DIR/"
cp -R "$CHUNKED_DIR" "$BACKUP_DIR/"

echo "Backup done."
echo "  $BACKUP_DIR/executorch_int8_models/"
echo "  $BACKUP_DIR/executorch_models/"
du -sh "$BACKUP_DIR"/executorch_int8_models "$BACKUP_DIR"/executorch_models 2>/dev/null || true
