#!/usr/bin/env bash
# Generate missing portable single-decoder Part4b + matching Part4a chunks for etCpu.
# Your LaCie folder may have tile_00 + part4a but NO sharp_split_part4b.pte — this creates the three
# non-_vulkan files (overwrite-safe: writes into OUTPUT_DIR).
#
# Requires: same Python env as full export (torch, executorch), weights at default --weights path.
# Portable Part4b compile can take a long time and a lot of RAM.
#
# Usage:
#   cd android
#   ./export_sharp_executorch_chunked_part4_portable_only.sh
#   ./export_sharp_executorch_chunked_part4_portable_only.sh /path/to/custom_out

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="${1:-${SCRIPT_DIR}/models_cpu}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
python3 export_sharp_executorch_split4.py \
  --backend portable \
  --chunked-part4-only \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "Done. New/overwritten (portable, no _vulkan):"
echo "  sharp_split_part4a_chunk_512.pte"
echo "  sharp_split_part4a_chunk_65.pte"
echo "  sharp_split_part4b.pte   <-- single decoder for Stable mode + etCpu"
echo ""
echo "Verify Part4b has no Vulkan delegate:"
echo "  python3 inspect_pte_delegates.py \"$OUTPUT_DIR/sharp_split_part4b.pte\""
echo ""
echo "Push to device (etCpu → models_cpu):"
echo "  adb shell mkdir -p /sdcard/Android/data/com.furnit.android/files/models_cpu"
echo "  adb push \"$OUTPUT_DIR/sharp_split_part4a_chunk_512.pte\" /sdcard/Android/data/com.furnit.android/files/models_cpu/"
echo "  adb push \"$OUTPUT_DIR/sharp_split_part4a_chunk_65.pte\"  /sdcard/Android/data/com.furnit.android/files/models_cpu/"
echo "  adb push \"$OUTPUT_DIR/sharp_split_part4b.pte\"           /sdcard/Android/data/com.furnit.android/files/models_cpu/"
