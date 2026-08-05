#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
DEST_DIR="$REPO_ROOT/Furnit/Models/RTMDet"

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /absolute/or/relative/path/to/rtmdet-ins-m-raw-fp16.tflite" >&2
  exit 64
fi

SRC_PATH=$1

if [ ! -e "$SRC_PATH" ]; then
  echo "Missing source path: $SRC_PATH" >&2
  exit 66
fi

SRC_BASENAME=$(basename -- "$SRC_PATH")
case "$SRC_BASENAME" in
  *.tflite) ;;
  *)
    echo "Expected a .tflite path, got: $SRC_BASENAME" >&2
    exit 65
    ;;
esac

mkdir -p "$DEST_DIR"
DEST_PATH="$DEST_DIR/rtmdet-ins-m-raw-fp16.tflite"

cp "$SRC_PATH" "$DEST_PATH"

echo "Installed:"
echo "  $DEST_PATH"
echo
echo "Next:"
echo "1. Confirm the file is tracked with Git LFS."
echo "2. The canonical path/name is already mapped to the RTMDetModel On-Demand Resources tag."
echo "3. Build for a physical iOS device; RTMDet requires the LiteRT Metal delegate."
