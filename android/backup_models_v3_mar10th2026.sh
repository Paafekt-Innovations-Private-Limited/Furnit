#!/usr/bin/env bash
# Backup SHARP/ExecuTorch models into mar10th2026/v3 on external hard disk.
# Usage: ./backup_models_v3_mar10th2026.sh [backup_root]
#   backup_root defaults to /Volumes/LaCie (or set EXTERNAL_BACKUP_ROOT)
# Creates: <backup_root>/mar10th2026/v3/ with executorch_int8_models and executorch_models.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_ROOT="${1:-${EXTERNAL_BACKUP_ROOT:-/Volumes/LaCie}}"
DATE_FOLDER="mar10th2026"
V3_FOLDER="v3"
DEST_DIR="${BACKUP_ROOT}/${DATE_FOLDER}/${V3_FOLDER}"

# Model folders to include (relative to android/)
MODEL_DIRS=(
  "executorch_int8_models"
  "executorch_models"
)

if [[ ! -d "$BACKUP_ROOT" ]]; then
  echo "External drive not found: $BACKUP_ROOT"
  echo "Usage: $0 [backup_root]   e.g. $0 /Volumes/LaCie  or  $0 /Volumes/ExtremeFSSD"
  exit 1
fi

mkdir -p "$DEST_DIR"
echo "Backup destination: $DEST_DIR"

for dir in "${MODEL_DIRS[@]}"; do
  SRC="${SCRIPT_DIR}/${dir}"
  if [[ -d "$SRC" ]]; then
    echo "Copying $dir ..."
    cp -R "$SRC" "${DEST_DIR}/"
    echo "  done: $dir"
  else
    echo "  (skip, not found) $dir"
  fi
done

echo "Backup complete: $DEST_DIR"
