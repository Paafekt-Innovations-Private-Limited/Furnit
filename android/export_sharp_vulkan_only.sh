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

# Vulkan FP16 + batch=2 = 95% success (avoids INT8 staging crash)
DTYPE="${DTYPE:-fp16}"
PATCH_BATCH="${PATCH_BATCH:-2}"

echo "=============================================="
echo "Export SHARP ExecuTorch — VULKAN ONLY (no XNNPACK)"
echo "  Output: $OUTPUT_DIR"
echo "  Backend: Vulkan (GPU)  dtype=${DTYPE}  patch_batch=${PATCH_BATCH}"
echo "  Chunked Part 4: part4a_chunk_512_vulkan, part4a_chunk_65_vulkan, part4b_vulkan"
echo "=============================================="
echo ""

# Remove old output to avoid mixing
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

python3 "${SCRIPT_DIR}/export_sharp_executorch_split4.py" \
  --backend vulkan \
  --chunked-part4 \
  --dtype "${DTYPE}" \
  --patch-batch-size "${PATCH_BATCH}" \
  --sharp-src "${SHARP_SRC}" \
  --weights "${WEIGHTS}" \
  --output-dir "${OUTPUT_DIR}"

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
