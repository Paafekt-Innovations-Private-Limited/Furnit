#!/bin/bash
# Export SHARP ExecuTorch split models for FULL VULKAN with memory optimizations.
#
# Uses skills learnt for memory optimization:
#  - Backend: Vulkan (GPU) for speed
#  - Chunked Part 4: 4a_chunk_512 + 4a_chunk_65 + 4b to avoid single 755MB Part 4 (reduces LMK risk)
#
# Prereqs: Python env with torch, executorch, sharp source and weights.
#
# Usage:
#   cd android
#   ./export_sharp_executorch_vulkan_full.sh [--push]
#
# Optional: --push  Push exported .pte files to device after export.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/executorch_models"
WEIGHTS="${WEIGHTS:-${SCRIPT_DIR}/sharp_litert_models/sharp_2572gikvuh.pt}"

# Default: sharp source next to script
SHARP_SRC="${SCRIPT_DIR}/third_party/ml-sharp/src"

if [ ! -d "$SHARP_SRC" ]; then
  echo "Warning: SHARP source not found at $SHARP_SRC"
  echo "Override with: export SHARP_SRC=/path/to/ml-sharp/src"
fi

if [ ! -f "$WEIGHTS" ]; then
  echo "Warning: Weights not found at $WEIGHTS"
  echo "Override with: export WEIGHTS=/path/to/sharp.pt"
fi

# FP16 + B2 = 95% Vulkan success (avoids INT8 staging crash). Example:
#   WEIGHTS=/path/to/sharp.pt DTYPE=fp16 PATCH_BATCH=2 ./export_sharp_executorch_vulkan_full.sh
DTYPE="${DTYPE:-fp32}"
PATCH_BATCH="${PATCH_BATCH:-1}"
[ "$DTYPE" = "fp16" ] && PATCH_BATCH="${PATCH_BATCH:-2}"

echo "=============================================="
echo "Export SHARP ExecuTorch — FULL VULKAN + memory optimization"
echo "  Backend: Vulkan (GPU)  dtype=${DTYPE}  patch_batch=${PATCH_BATCH}"
echo "  Chunked Part 4: part4a_chunk_512, part4a_chunk_65, part4b (lower peak RAM)"
echo "=============================================="
echo ""

python3 "${SCRIPT_DIR}/export_sharp_executorch_split4.py" \
  --backend vulkan \
  --chunked-part4 \
  --dtype "${DTYPE}" \
  --patch-batch-size "${PATCH_BATCH}" \
  --sharp-src "${SHARP_SRC}" \
  --weights "${WEIGHTS}" \
  --output-dir "${OUTPUT_DIR}"

echo ""
echo "Export done. To push to device:"
echo "  ./push_sharp_executorch_models.sh ${OUTPUT_DIR}"
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
  "${SCRIPT_DIR}/push_sharp_executorch_models.sh" "${OUTPUT_DIR}"
fi
