#!/usr/bin/env bash
# Backup large (gitignored) Furnit files to external HD in a date-named folder.
# Usage:
#   BACKUP_ROOT=/Volumes/YourExternalHD ./scripts/backup_large_files_to_external.sh
#   ./scripts/backup_large_files_to_external.sh /Volumes/YourExternalHD
#
# Creates: $BACKUP_ROOT/Furnit_large_backup/YYYY-MM-DD/ with copies of:
#   android/app/libs/
#   android/app/src/main/cpp/executorch_lib*/
#   android/app/src/main/jniLibs/
#   android/logcat_crash.txt
#   android/sharp_vulkan_only/ (ExecuTorch Vulkan .pte)
#   android/sharp_portable_latent0/, sharp_vulkan_only_latent0/
#   android/executorch_models*/, sharp_litert_models/, models_cpu/

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_ROOT="${BACKUP_ROOT:-$1}"
DATE_FOLDER="$(date +%Y-%m-%d)"
DEST_DIR="Furnit_large_backup/${DATE_FOLDER}"

if [ -z "$BACKUP_ROOT" ]; then
  echo "Usage: BACKUP_ROOT=/Volumes/YourExternalHD $0"
  echo "   or: $0 /Volumes/YourExternalHD"
  exit 1
fi

FULL_DEST="${BACKUP_ROOT}/${DEST_DIR}"
mkdir -p "$FULL_DEST"
cd "$PROJECT_ROOT"

echo "Backup destination: $FULL_DEST"

copy_if_exists() {
  local src="$1"
  local dest_subdir="${2:-.}"
  if [ -e "$src" ]; then
    mkdir -p "$FULL_DEST/$dest_subdir"
    cp -R "$src" "$FULL_DEST/$dest_subdir/"
    echo "  Copied: $src"
  else
    echo "  Skip (missing): $src"
  fi
}

copy_if_exists "android/app/libs" "android/app"
copy_if_exists "android/app/src/main/cpp/executorch_lib" "android/app/src/main/cpp"
copy_if_exists "android/app/src/main/jniLibs" "android/app/src/main"
if [ -f "android/logcat_crash.txt" ]; then
  mkdir -p "$FULL_DEST/android"
  cp "android/logcat_crash.txt" "$FULL_DEST/android/"
  echo "  Copied: android/logcat_crash.txt"
else
  echo "  Skip (missing): android/logcat_crash.txt"
fi

# ExecuTorch / SHARP model trees (gitignored; large)
copy_if_exists "android/sharp_vulkan_only" "android"
copy_if_exists "android/sharp_portable_latent0" "android"
copy_if_exists "android/sharp_vulkan_only_latent0" "android"
copy_if_exists "android/executorch_models" "android"
copy_if_exists "android/executorch_fp16_models" "android"
copy_if_exists "android/executorch_models_vulkan" "android"
copy_if_exists "android/sharp_litert_models" "android"
copy_if_exists "android/models_cpu" "android"

echo "Done. Backup is in $FULL_DEST"
