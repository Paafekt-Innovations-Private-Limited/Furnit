#!/bin/bash
# Build app so Java and native ExecuTorch come from the SAME AAR (executorch-android:1.1.0).
# Use this to fix UnsatisfiedLinkError initHybrid when running Part1-only test.
# No Vulkan; Part1 runs on portable/CPU. After this, run Part1-only test in Settings.
#
# Usage: cd android && ./build_part1_test_no_vulkan.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_LIB_CPU="${SCRIPT_DIR}/app/src/main/cpp/executorch_lib_etCpu"

echo "Cleaning and building etCpu flavor (ExecuTorch XNNPACK AAR only)..."
./gradlew clean

# Remove any local .so so extract overwrites with AAR content
if [ -d "$EXECUTORCH_LIB_CPU" ]; then
  rm -f "${EXECUTORCH_LIB_CPU}"/libexecutorch*.so "${EXECUTORCH_LIB_CPU}"/libextension_*.so 2>/dev/null || true
  echo "Cleared executorch_lib_etCpu (so extract uses XNNPACK AAR)"
fi

./gradlew :app:extractExecutorchSoFromAar
./gradlew :app:assembleEtCpuDebug

echo ""
echo "Done. Install the etCpu debug APK from app/build/outputs/apk/etCpu/debug/ and run Settings → Developer → Part1 only test."
echo "If initHybrid is still missing, see android/docs/EXECUTORCH_JNI_MISMATCH.md"
