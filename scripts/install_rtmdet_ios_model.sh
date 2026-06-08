#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
DEST_DIR="$REPO_ROOT/Furnit/Models/RTMDet"

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /absolute/or/relative/path/to/rtmdet-ins-m.mlpackage" >&2
  exit 64
fi

SRC_PATH=$1

if [ ! -e "$SRC_PATH" ]; then
  echo "Missing source path: $SRC_PATH" >&2
  exit 66
fi

SRC_BASENAME=$(basename -- "$SRC_PATH")

case "$SRC_BASENAME" in
  *.mlpackage|*.mlmodelc)
    ;;
  *)
    echo "Expected a .mlpackage or .mlmodelc path, got: $SRC_BASENAME" >&2
    exit 65
    ;;
esac

mkdir -p "$DEST_DIR"
DEST_PATH="$DEST_DIR/$SRC_BASENAME"

rm -rf "$DEST_PATH"
cp -R "$SRC_PATH" "$DEST_PATH"

echo "Installed:"
echo "  $DEST_PATH"
echo
echo "Next:"
echo "1. Open Xcode."
echo "2. Ensure $SRC_BASENAME is included in the Furnit app target resources."
echo "3. Run the app, open Settings -> Image scan, and switch backend to RTMDet-Ins-m."
