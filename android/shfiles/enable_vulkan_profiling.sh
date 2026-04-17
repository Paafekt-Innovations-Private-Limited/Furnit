#!/usr/bin/env bash
# Enable ExecuTorch Vulkan profiling on device so logcat shows where time is spent.
# Use before running the app; clear with: adb shell setprop debug.executorch.vulkan.enable_profiling 0
# For ETDump + Inspector (per-op timing), see android/docs/EXECUTORCH_VULKAN_PROFILING.md
set -e
adb shell setprop debug.executorch.vulkan.enable_profiling 1
echo "Vulkan profiling enabled. Run the app and capture logcat; look for gaps between GPU dispatches (CPU fallback stalls)."
echo "To disable: adb shell setprop debug.executorch.vulkan.enable_profiling 0"
echo "For ETDump/Inspector: see android/docs/EXECUTORCH_VULKAN_PROFILING.md"
