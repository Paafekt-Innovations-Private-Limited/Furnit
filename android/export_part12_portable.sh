#!/bin/bash
# Export Part1+Part2 as portable (CPU) FP16 to fix Vulkan "ptr" crash.
# Run from android/. Then: adb push executorch_models/sharp_split_part1.pte executorch_models/sharp_split_part2.pte /sdcard/Android/data/com.furnit.android/files/models/
set -e
cd "$(dirname "$0")"
python3 export_sharp_executorch_split4.py --part12-only-portable --output-dir executorch_models
echo "Push to device: adb push executorch_models/sharp_split_part1.pte executorch_models/sharp_split_part2.pte /sdcard/Android/data/com.furnit.android/files/models/"
