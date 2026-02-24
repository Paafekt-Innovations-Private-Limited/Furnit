#!/bin/bash
# Monitor memory and logs during Native .pt SHARP inference.
# Run in a separate terminal while the app runs generation:
#
#   ./monitor_native_pt_inference.sh
#
# Or log to file:
#   ./monitor_native_pt_inference.sh 2>&1 | tee native_pt_inference.log
#
# To check memory only (every 5s):
#   watch -n5 'adb shell dumpsys meminfo com.furnit.android | head -30'
set -e

echo "Monitoring Native .pt inference. Press Ctrl+C to stop."
echo ""

adb logcat -c
adb logcat | grep -E "SharpService|NativePtSharp|Part [0-9]" --line-buffered
