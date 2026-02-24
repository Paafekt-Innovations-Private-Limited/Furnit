#!/usr/bin/env bash
# Enable ExecuTorch Vulkan profiling on device so logcat shows where time is spent.
# Use before running the app; clear with: adb shell setprop debug.executorch.vulkan.enable_profiling 0
set -e
adb shell setprop debug.executorch.vulkan.enable_profiling 1
echo "Vulkan profiling enabled. Run the app and capture logcat; look for gaps between GPU dispatches (CPU fallback stalls)."
echo "To disable: adb shell setprop debug.executorch.vulkan.enable_profiling 0"
