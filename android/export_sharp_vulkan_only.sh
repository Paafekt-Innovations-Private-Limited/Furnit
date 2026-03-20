#!/bin/bash
# Export SHARP ExecuTorch Vulkan-only models to a FRESH folder (no XNNPACK mixing).
#
# Output: sharp_vulkan_only/ — contains ONLY Vulkan .pte files. No XNNPACK, no INT8.
# Use this folder for push; app loads Vulkan-only and avoids XNNPACK SIGSEGV.
#
# Prereqs: Python env with torch, executorch, sharp source and weights.
#
# Usage:
#   cd android
#   ./export_sharp_vulkan_only.sh [--push]
#
# Optional: --push  Push exported .pte files to device after export.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# NEW folder — never mix with executorch_models (which may have XNNPACK)
OUTPUT_DIR="${SCRIPT_DIR}/sharp_vulkan_only"
WEIGHTS="${WEIGHTS:-${SCRIPT_DIR}/sharp_litert_models/sharp_2572gikvuh.pt}"
SHARP_SRC="${SHARP_SRC:-${SCRIPT_DIR}/third_party/ml-sharp/src}"

if [ ! -d "$SHARP_SRC" ]; then
  echo "ERROR: SHARP source not found at $SHARP_SRC"
  echo "Override with: export SHARP_SRC=/path/to/ml-sharp/src"
  exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
  echo "ERROR: Weights not found at $WEIGHTS"
  echo "Override with: export WEIGHTS=/path/to/sharp.pt"
  exit 1
fi

# Vulkan FP16 uses view_convert_buffer_float_half shader — prebuilt AAR lacks it (Error 0x20).
# Use vulkan-aar-compat (FP32) so .pte only uses shaders in executorch-android-vulkan 1.1.0 AAR.
# Pin Python to match Maven: pip install 'executorch==1.1.0' (newer pip + old AAR → Error 0x20 on forward).
VULKAN_AAR_COMPAT="${VULKAN_AAR_COMPAT:-true}"
DTYPE="${DTYPE:-fp16}"
PATCH_BATCH="${PATCH_BATCH:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-1536}"
UNIFY_FP16="${UNIFY_FP16:-}"

echo "=============================================="
echo "Export SHARP ExecuTorch — VULKAN ONLY (no XNNPACK)"
echo "  Output: $OUTPUT_DIR"
echo "  Backend: Vulkan (GPU)  dtype=${DTYPE}  image_size=${IMAGE_SIZE}  patch_batch=${PATCH_BATCH}  vulkan_aar_compat=${VULKAN_AAR_COMPAT}"
echo "  Chunked Part 4: part4a_chunk_512_vulkan, part4a_chunk_65_vulkan, part4b_vulkan,"
echo "                  part4b_tile_00 / part4b_tile_b2 / part4b_tile_b4 / part4b_tile_full"
echo "                  plus split tile_00 + tile_b2 Vulkan-safe stage_a/raw_heads + portable init_base/compose"
echo "=============================================="
echo ""

# Remove old output to avoid mixing
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

VULKAN_ARGS=(
  --backend vulkan
  --chunked-part4
  --dtype "${DTYPE}"
  --image-size "${IMAGE_SIZE}"
  --patch-batch-size "${PATCH_BATCH}"
  --sharp-src "${SHARP_SRC}"
  --weights "${WEIGHTS}"
  --output-dir "${OUTPUT_DIR}"
)
[ "$VULKAN_AAR_COMPAT" = "true" ] && VULKAN_ARGS+=(--vulkan-aar-compat)
VULKAN_ARGS+=(--vulkan-safe-part4b-tile)
# Optional (see EXECUTORCH_VULKAN_EXAMPLE_ALIGNMENT.md): ETRecord dir, small texture limits, bundled .bpte, run Vulkan test per part
[ -n "${ETRECORD_DIR:-}" ] && VULKAN_ARGS+=(-r "${ETRECORD_DIR}")
[ "${SMALL_TEXTURE_LIMITS:-0}" = "1" ] && VULKAN_ARGS+=(--small-texture-limits)
[ "${BUNDLED:-0}" = "1" ] && VULKAN_ARGS+=(-b)
[ "${RUN_TEST:-0}" = "1" ] && VULKAN_ARGS+=(-t)
if [ -n "$UNIFY_FP16" ]; then
  VULKAN_ARGS+=(--unify-fp16)
elif [ "$DTYPE" = "fp16" ] && [ "$VULKAN_AAR_COMPAT" != "true" ]; then
  VULKAN_ARGS+=(--unify-fp16)
fi

python3 "${SCRIPT_DIR}/export_sharp_executorch_split4.py" "${VULKAN_ARGS[@]}"

echo ""
echo "Export done. Vulkan-only models in: $OUTPUT_DIR"
echo "To push to device:"
echo "  ./push_sharp_vulkan_only.sh"
echo ""

PUSH=false
for arg in "$@"; do
  if [ "$arg" = "--push" ]; then
    PUSH=true
    break
  fi
done

if [ "$PUSH" = true ]; then
  echo "Pushing to device..."
  "${SCRIPT_DIR}/push_sharp_vulkan_only.sh"
fi
