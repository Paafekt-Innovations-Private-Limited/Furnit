#!/bin/bash
# Push **CPU + Vulkan hybrid** SHARP .pte to the device for Android Studio installs
# (INT8 Part1+2 + Vulkan Part3–4 in one folder).
#
# Skips standalone full-Vulkan Part1/Part2 (`sharp_split_part1_vulkan_*.pte`, `sharp_split_part2_vulkan_*.pte`)
# and their manifests — those are large and not used when hybrid INT8 sidecars are present.
#
# Multiple adb targets: prefers a **physical** device over `emulator-*`. Override:
#   export ANDROID_SERIAL=<serial>   # adb devices
#
# Default source: android/models_cpuvulkan_hybrid (populate via populate_models_cpuvulkan_hybrid_from_backups.sh
# or copy the hybrid set from sharp_vulkan_only + INT8 without the part1/2 vulkan-only .pte).
#
# Usage:
#   ./push_sharp_cpuvulkan_hybrid_androidstudio.sh
#   ./push_sharp_cpuvulkan_hybrid_androidstudio.sh /path/to/folder
#
# Destination: models_cpuvulkan_hybrid (etVulkan APK; legacy models_vulkan is auto-migrated by the app)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=friend_tester_bundle/adb_common.sh
source "$SCRIPT_DIR/friend_tester_bundle/adb_common.sh"

MODEL_DIR="${1:-$SCRIPT_DIR/models_cpuvulkan_hybrid}"
DEST="/sdcard/Android/data/com.furnit.android/files/models_cpuvulkan_hybrid"

# Full-Vulkan Part1/2 only — hybrid path uses sharp_split_part1_int8.pte + sharp_split_part2_int8.pte on CPU.
is_skipped_full_vulkan_part12() {
  local file_name="$1"
  [[ "$file_name" =~ ^sharp_split_part[12]_vulkan_(fp16|fp32)\.pte$ ]] ||
    [[ "$file_name" =~ ^sharp_split_part[12]_vulkan_(fp16|fp32)\.pte\.manifest\.json$ ]]
}

if ! command -v adb &> /dev/null; then
  echo "Error: adb not found in PATH"
  exit 1
fi

SERIAL="$(pick_physical_serial)" || exit 1
echo "Using device: $SERIAL"
ADB=(adb -s "$SERIAL")

if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: Model dir not found: $MODEL_DIR"
  echo "Create android/models_cpuvulkan_hybrid/ with hybrid .pte, or run ./populate_models_cpuvulkan_hybrid_from_backups.sh"
  exit 1
fi

echo "Pushing Android Studio CPU+Vulkan hybrid SHARP models (skipping full-Vulkan Part1/Part2 .pte)..."
echo "Source:      $MODEL_DIR"
echo "Destination: $DEST"
echo ""

"${ADB[@]}" shell "mkdir -p $DEST"

shopt -s nullglob
artifacts=(
  "$MODEL_DIR"/sharp_split_part*.pte
  "$MODEL_DIR"/sharp_split_part*.pte.manifest.json
)
shopt -u nullglob

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
done

# Part1 warmup needs part1_test_patch_f32.bin (from export_sharp_executorch_split4.py --part1-only)
if [ -f "$MODEL_DIR/part1_test_patch_f32.bin" ]; then
  echo "Pushing part1_test_patch_f32.bin (for Part1 warmup)..."
  "${ADB[@]}" push "$MODEL_DIR/part1_test_patch_f32.bin" "$DEST/part1_test_patch_f32.bin"
fi

echo ""
echo "Done! Android Studio hybrid models pushed to $DEST"
