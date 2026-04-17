#!/bin/bash
# Wipe device models_cpu (external + internal when possible), then push the March 2026 v2 set.
#
# Required (6): part1–3 int8, part4a 512+65, part4b_tile_b4.
# Optional (if present under SRC): single Part4b for **Stable Part4b (single)** — less tile fog, more RAM:
#   sharp_split_part4b_int8.pte | sharp_split_part4b_fp16.pte | sharp_split_part4b.pte
# Any combination may exist; native prefers INT8 → FP16 → FP32 when Stable is ON.
#
# Usage:
#   ./deploy_sharp_v2_to_models_cpu.sh
#   ./deploy_sharp_v2_to_models_cpu.sh /path/to/v2
#   ./deploy_sharp_v2_to_models_cpu.sh "$(pwd)/models_cpu"   # local staging (see models_cpu/README.md)
#   SKIP_CLEAR=1 ./deploy_sharp_v2_to_models_cpu.sh   # push only (device already clean)
set -euo pipefail

cd "$(dirname "$0")"

# Default LaCie v2; use "$(pwd)/models_cpu" after copy_from_lacie or manual staging.
SRC="${1:-/Volumes/LaCie/march10th2026/v2}"
PKG="com.furnit.android"
DEST="/sdcard/Android/data/${PKG}/files/models_cpu"

V2_FILES=(
  sharp_split_part1_int8.pte
  sharp_split_part2_int8.pte
  sharp_split_part3_int8.pte
  sharp_split_part4a_chunk_512.pte
  sharp_split_part4a_chunk_65.pte
  sharp_split_part4b_tile_b4.pte
)

# Pushed only if the file exists in SRC (not required for v2-minimal deploy).
OPTIONAL_SINGLE_PART4B=(
  sharp_split_part4b_int8.pte
  sharp_split_part4b_fp16.pte
  sharp_split_part4b.pte
)

command -v adb &>/dev/null || { echo "Error: adb not found"; exit 1; }
adb devices | grep -q "device$" || { echo "Error: No device"; exit 1; }
[ -d "$SRC" ] || { echo "Error: source dir not found: $SRC"; exit 1; }

for f in "${V2_FILES[@]}"; do
  [[ -f "$SRC/$f" ]] || { echo "Error: missing $SRC/$f"; exit 1; }
done

if [[ "${SKIP_CLEAR:-}" == "1" ]]; then
  echo "SKIP_CLEAR=1 — not clearing device models_cpu."
else
  ./clear_device_models_cpu.sh
fi

adb shell "mkdir -p $DEST"
for f in "${V2_FILES[@]}"; do
  echo "Push $f"
  adb push "$SRC/$f" "$DEST/$f"
done

optional_pushed=0
for f in "${OPTIONAL_SINGLE_PART4B[@]}"; do
  if [[ -f "$SRC/$f" ]]; then
    echo "Push optional single Part4b: $f"
    adb push "$SRC/$f" "$DEST/$f"
    optional_pushed=$((optional_pushed + 1))
  fi
done

echo ""
echo "Deployed ${#V2_FILES[@]} required files to $DEST"
if (( optional_pushed > 0 )); then
  echo "Also pushed $optional_pushed optional single Part4b file(s). Keep Settings → Stable Part4b (single) ON."
else
  echo "No optional single Part4b in $SRC — only tiled path (tile_b4). Add sharp_split_part4b_int8.pte (or fp16/.pte) for single-decoder quality."
fi
echo "Open the app (or kill + reopen) so syncExternalSharpSplitPteToInternal copies to internal and prunes stale .pte."
