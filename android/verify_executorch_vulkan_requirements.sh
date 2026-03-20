#!/usr/bin/env bash
# Verify ExecuTorch and Vulkan requirements for this project.
# Run from android/: ./verify_executorch_vulkan_requirements.sh [path/to/part1.pte]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ExecuTorch & Vulkan requirements check ==="
echo ""

# 1. ExecuTorch AAR version from build.gradle
echo "1. ExecuTorch Android (AAR) version:"
if grep -q "executorch-android-vulkan" app/build.gradle 2>/dev/null; then
    grep "executorch-android-vulkan" app/build.gradle | sed 's/^/   /'
else
    echo "   (app/build.gradle not found or no executorch-android-vulkan line)"
fi
echo "   Expected: 1.1.0 (current stable 1.1). See https://github.com/pytorch/executorch/releases"
echo ""

# 2. Vulkan requirements summary
echo "2. Vulkan (device) requirements:"
echo "   - Vulkan API: 1.1 or above (required by ExecuTorch Vulkan backend)"
echo "   - Recommended extensions for FP16/INT8:"
echo "     VK_KHR_16bit_storage, VK_KHR_8bit_storage, VK_KHR_shader_float16_int8"
echo "   - Verify on device: install \"Vulkan Hardware Capability Viewer\" and check API version + extensions"
echo ""

# 3. Optional: verify Part1 .pte has Vulkan delegate
if [ -n "$1" ] && [ -f "$1" ]; then
    echo "3. Delegate verification for: $1"
    if command -v python3 &>/dev/null; then
        python3 inspect_pte_delegates.py "$1" 2>/dev/null || true
    else
        python inspect_pte_delegates.py "$1" 2>/dev/null || true
    fi
else
    echo "3. Delegate verification: (optional) run with a Part1 .pte path to verify Vulkan in .pte:"
    echo "   ./verify_executorch_vulkan_requirements.sh executorch_models/sharp_split_part1_vulkan_fp16.pte"
fi
echo ""

echo "Full checklist: android/docs/VULKAN_CHECKLIST.md"
