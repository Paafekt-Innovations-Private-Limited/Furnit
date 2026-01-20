#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: docker_convert_entrypoint.sh <input.onnx> <output.tflite> [tmpdir]"
  exit 2
fi

INPUT_ONNX="$1"
OUTPUT_TFLITE="$2"
TMPDIR="${3:-/workspace/tmp_conv}"

mkdir -p "$TMPDIR"
echo "Converting ONNX: $INPUT_ONNX -> TFLite: $OUTPUT_TFLITE (tmp: $TMPDIR)"

python /workspace/scripts/docker_convert_onnx_to_tflite.py --input "$INPUT_ONNX" --output "$OUTPUT_TFLITE" --tmpdir "$TMPDIR"

echo "Conversion finished"
