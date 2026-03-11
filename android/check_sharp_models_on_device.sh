#!/bin/bash
# Check which SHARP ExecuTorch model files exist on the device (internal + external).
# Run this instead of pushing blindly — tells you exactly what's there and what's missing.
#
# Usage:
#   ./check_sharp_models_on_device.sh
#
# With a specific device (same as push script):
#   ADB_SERIAL=adb-53181JEBF16055-Y6PQA5._adb-tls-connect._tcp ./check_sharp_models_on_device.sh
#
# Requires: adb, device with Furnit app installed (for run-as internal path).
set -euo pipefail

if [ -n "${ADB_SERIAL:-}" ]; then
  ADB_CMD="adb -s $ADB_SERIAL"
else
  ADB_CMD="adb"
fi

if ! command -v adb &> /dev/null; then
  echo "Error: adb not found in PATH"
  exit 1
fi

if [ -n "${ADB_SERIAL:-}" ]; then
  if ! $ADB_CMD get-state &> /dev/null; then
    echo "Error: Device $ADB_SERIAL not found. Run: adb devices -l"
    exit 1
  fi
  echo "Using device: $ADB_SERIAL"
elif [ "$($ADB_CMD devices | grep -c 'device$' || true)" -gt 1 ]; then
  echo "Error: More than one device connected. Set ADB_SERIAL, e.g.:"
  echo "  ADB_SERIAL=YOUR_DEVICE_SERIAL ./check_sharp_models_on_device.sh"
  exit 1
elif ! $ADB_CMD devices | grep -q "device$"; then
  echo "Error: No device connected."
  exit 1
fi

PACKAGE=com.furnit.android
INTERNAL_DIR="/data/data/$PACKAGE/files/models"
EXTERNAL_DIR="/storage/emulated/0/Android/data/$PACKAGE/files/models"

echo ""
echo "=== Internal (app filesDir) ==="
INTERNAL_LIST="$($ADB_CMD shell "run-as $PACKAGE ls -la files/models/ 2>/dev/null" || true)"
if [ -z "$INTERNAL_LIST" ]; then
  echo "(empty or not readable)"
else
  echo "$INTERNAL_LIST"
fi

echo ""
echo "=== External (getExternalFilesDir) ==="
EXTERNAL_LIST="$($ADB_CMD shell "ls -la $EXTERNAL_DIR/ 2>/dev/null" || true)"
if [ -z "$EXTERNAL_LIST" ] || echo "$EXTERNAL_LIST" | grep -q "No such file"; then
  echo "(dir missing or empty)"
else
  echo "$EXTERNAL_LIST"
fi

echo ""
echo "=== INT8 Part4b (sharp_split_part4b_int8.pte) ==="
HAS_INT8_INTERNAL="$($ADB_CMD shell "run-as $PACKAGE sh -c 'test -f files/models/sharp_split_part4b_int8.pte && echo yes || echo no' 2>/dev/null" | tr -d '\r')"
HAS_INT8_EXTERNAL="$($ADB_CMD shell "test -f $EXTERNAL_DIR/sharp_split_part4b_int8.pte && echo yes || echo no" 2>/dev/null | tr -d '\r')"
echo "  Internal: ${HAS_INT8_INTERNAL:-no}"
echo "  External: ${HAS_INT8_EXTERNAL:-no}"

if [ "$HAS_INT8_INTERNAL" = "no" ] && [ "$HAS_INT8_EXTERNAL" = "no" ]; then
  echo ""
  echo "  -> INT8 Part4b not on device. C++ will use FP32 Part4b. To use INT8 decoder once:"
  echo "     ./push_sharp_executorch_int8_models.sh   (with sharp_split_part4b_int8.pte in executorch_models/ or executorch_int8_models/)"
fi
if [ "$HAS_INT8_INTERNAL" = "yes" ] || [ "$HAS_INT8_EXTERNAL" = "yes" ]; then
  echo ""
  echo "  -> INT8 Part4b present; full pipeline will use it."
fi
echo ""
