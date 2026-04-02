#!/usr/bin/env bash
# Standalone: push SHARP hybrid .pte to a phone. No Furnit repo / Gradle required.
# Expects: adb on PATH, USB debugging on, device authorized.
#
# Multiple adb devices: physical phone is preferred over emulator. Override:
#   export ANDROID_SERIAL=<serial>   # from adb devices
#
# Layout (what you unzip from Google Drive):
#   FurnitFriend/
#     push_furnit_sharp_models.sh
#     adb_common.sh
#     models_cpuvulkan_hybrid/   ← all .pte files here
#     furnit.apk                 ← optional; use install_furnit_apk.sh
#
# Usage:
#   chmod +x push_furnit_sharp_models.sh   # if your OS did not preserve +x
#   ./push_furnit_sharp_models.sh
#   ./push_furnit_sharp_models.sh /path/to/models_cpuvulkan_hybrid
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=adb_common.sh
source "$SCRIPT_DIR/adb_common.sh"

MODEL_DIR="${1:-$SCRIPT_DIR/models_cpuvulkan_hybrid}"
DEST="/sdcard/Android/data/com.furnit.android/files/models_cpuvulkan_hybrid"

is_skipped_full_vulkan_part12() {
  local file_name="$1"
  [[ "$file_name" =~ ^sharp_split_part[12]_vulkan_(fp16|fp32)\.pte$ ]] ||
    [[ "$file_name" =~ ^sharp_split_part[12]_vulkan_(fp16|fp32)\.pte\.manifest\.json$ ]]
}

if ! command -v adb &> /dev/null; then
  echo "Error: adb not found. Install Android Platform Tools and add adb to PATH:"
  echo "  https://developer.android.com/tools/releases/platform-tools"
  exit 1
fi

SERIAL="$(pick_physical_serial)" || exit 1
echo "Using device: $SERIAL"
ADB=(adb -s "$SERIAL")

if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: Model folder not found: $MODEL_DIR"
  echo "Put the hybrid .pte folder next to this script as: $(basename "$SCRIPT_DIR")/models_cpuvulkan_hybrid/"
  echo "Or run:  $0 /full/path/to/models_cpuvulkan_hybrid"
  exit 1
fi

echo "Pushing Furnit SHARP hybrid models..."
echo "  Source:      $MODEL_DIR"
echo "  Destination: $DEST"
echo ""

"${ADB[@]}" shell "mkdir -p $DEST"

shopt -s nullglob
artifacts=(
  "$MODEL_DIR"/sharp_split_part*.pte
  "$MODEL_DIR"/sharp_split_part*.pte.manifest.json
)
shopt -u nullglob

count=0
for artifact_path in "${artifacts[@]}"; do
  if [ ! -f "$artifact_path" ]; then
    continue
  fi
  file_name="$(basename "$artifact_path")"
  if is_skipped_full_vulkan_part12 "$file_name"; then
    echo "Skipping (not used in hybrid): $file_name"
    continue
  fi
  readable_size="$(ls -lh "$artifact_path" | awk '{print $5}')"
  echo "Pushing $file_name ($readable_size)..."
  "${ADB[@]}" push "$artifact_path" "$DEST/$file_name"
  count=$((count + 1))
done

if [ -f "$MODEL_DIR/part1_test_patch_f32.bin" ]; then
  echo "Pushing part1_test_patch_f32.bin..."
  "${ADB[@]}" push "$MODEL_DIR/part1_test_patch_f32.bin" "$DEST/part1_test_patch_f32.bin"
fi

if [ "$count" -eq 0 ]; then
  echo ""
  echo "Warning: No sharp_split_part*.pte files were pushed. Check that models_cpuvulkan_hybrid is complete."
  exit 1
fi

echo ""
echo "Done. Models are on the device at: $DEST"
echo "Open the Furnit app and use SHARP / create room from photo as usual."
