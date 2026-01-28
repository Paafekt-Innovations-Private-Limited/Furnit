#!/usr/bin/env bash
set -euo pipefail

# Copies the ONNX model into the Android app assets folder.
# Usage: ./scripts/copy_model_to_assets.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_SRC="$ROOT_DIR/android/yoloe-11l-seg-pf.onnx"
ASSETS_DIR="$ROOT_DIR/android/app/src/main/assets"

if [ ! -f "$MODEL_SRC" ]; then
  echo "Model not found: $MODEL_SRC"
  echo "Place the ONNX model at $MODEL_SRC or update the script." >&2
  exit 2
fi

mkdir -p "$ASSETS_DIR"
cp -v "$MODEL_SRC" "$ASSETS_DIR/"
echo "Copied $(basename "$MODEL_SRC") to $ASSETS_DIR"
#!/usr/bin/env bash
# Simple helper to copy the exported ONNX model into Android app assets
set -euo pipefail

SRC="android/yoloe-11l-seg-pf.onnx"
DST_DIR="android/app/src/main/assets"

if [ ! -f "$SRC" ]; then
  echo "Source ONNX model not found at $SRC"
  exit 2
fi

mkdir -p "$DST_DIR"
cp -v "$SRC" "$DST_DIR/"
echo "Copied $SRC -> $DST_DIR/"
