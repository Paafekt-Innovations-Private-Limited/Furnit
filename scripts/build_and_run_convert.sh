#!/usr/bin/env bash
set -euo pipefail

# Build the conversion docker image and run conversion mounting repository root
# Usage: ./build_and_run_convert.sh <input.onnx> <output.tflite>

IMG_NAME=furnit/onnx-convert:latest

docker build -t "$IMG_NAME" -f scripts/Dockerfile.convert .

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <input.onnx> <output.tflite>"
  exit 2
fi

INPUT_ONNX="$1"
OUTPUT_TFLITE="$2"

mkdir -p $(dirname "$OUTPUT_TFLITE")

docker run --rm -v "$(pwd)":/workspace -w /workspace "$IMG_NAME" \
  /usr/local/bin/docker_convert_entrypoint.sh "$INPUT_ONNX" "$OUTPUT_TFLITE" /workspace/tmp_conv

echo "Output written: $OUTPUT_TFLITE"
