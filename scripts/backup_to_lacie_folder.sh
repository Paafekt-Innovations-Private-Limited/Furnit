#!/usr/bin/env bash
# One-shot backup of gitignored models + native libs to a named folder on LaCie (or any volume).
# Usage:
#   ./scripts/backup_to_lacie_folder.sh
#   BACKUP_TOP=/Volumes/LaCie/MyBackup ./scripts/backup_to_lacie_folder.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_TOP="${BACKUP_TOP:-/Volumes/LaCie/BackUp21stMarch2026GPU}"
DEST="$BACKUP_TOP/Furnit"

if [ ! -d "$BACKUP_TOP" ]; then
  echo "ERROR: Backup volume not mounted: $BACKUP_TOP"
  exit 1
fi

mkdir -p "$DEST/android"
cd "$PROJECT_ROOT"

GIT_REV="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DATE_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat > "$DEST/README_BACKUP.txt" << EOF
Furnit backup (large / gitignored artifacts)
Date (UTC): $DATE_ISO
Git rev:    $GIT_REV
Source:     $PROJECT_ROOT

Restore: copy android/* subdirs back into your Furnit/android/ tree as needed.
EOF

echo "Backing up to $DEST"

rsync_archive_if_exists() {
  local name="$1"
  if [ -e "android/$name" ]; then
    rsync -a "android/$name" "$DEST/android/"
    echo "  rsync: android/$name"
  else
    echo "  skip:  android/$name (missing)"
  fi
}

rsync_archive_if_exists sharp_vulkan_only
rsync_archive_if_exists sharp_portable_latent0
rsync_archive_if_exists sharp_vulkan_only_latent0
rsync_archive_if_exists sharp_litert_models
rsync_archive_if_exists executorch_models
rsync_archive_if_exists executorch_fp16_models
rsync_archive_if_exists executorch_models_vulkan
rsync_archive_if_exists models_cpu

for lib in android/app/libs android/app/src/main/jniLibs; do
  if [ -e "$lib" ]; then
    mkdir -p "$DEST/$(dirname "$lib")"
    rsync -a "$lib" "$DEST/$(dirname "$lib")/"
    echo "  rsync: $lib"
  fi
done

for cpp in android/app/src/main/cpp/executorch_lib android/app/src/main/cpp/executorch_lib_etVulkan android/app/src/main/cpp/executorch_lib_etCpu; do
  if [ -e "$cpp" ]; then
    mkdir -p "$DEST/android/app/src/main/cpp"
    rsync -a "$cpp" "$DEST/android/app/src/main/cpp/"
    echo "  rsync: $cpp"
  fi
done

echo "Done. $DEST/README_BACKUP.txt"
