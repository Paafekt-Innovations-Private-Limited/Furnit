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
