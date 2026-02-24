#!/bin/bash
# Export SHARP ExecuTorch split models for FULL VULKAN with memory optimizations.
#
# Uses skills learnt for memory optimization:
#  - Backend: Vulkan (GPU) for speed
#  - Part1+Part2 combined: one .pte so Vulkan AOT memory planning can share token buffer with Part1/Part2 intermediates
#  - Chunked Part 4: 4a_chunk_512 + 4a_chunk_65 + 4b to avoid single 755MB Part 4 (reduces LMK risk)
#  - Runtime: GC+sleep between parts; uses combined Part1+2 when present, else two-phase with tokens in RAM
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
WEIGHTS="${SCRIPT_DIR}/sharp_litert_models/sharp_2572gikvuh.pt"

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

echo "=============================================="
echo "Export SHARP ExecuTorch — FULL VULKAN + memory optimization"
echo "  Backend: Vulkan (GPU)"
echo "  Chunked Part 4: part4a_chunk_512, part4a_chunk_65, part4b (lower peak RAM)"
echo "=============================================="
echo ""

python3 "${SCRIPT_DIR}/export_sharp_executorch_split4.py" \
  --backend vulkan \
  --combined-part1-part2 \
  --chunked-part4 \
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
