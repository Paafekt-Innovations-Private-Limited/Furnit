#!/usr/bin/env bash
# Small etVulkan debug APK **without** bundled .pte (installs reliably on phones; models via adb push).
# Use when assemble_friend_apk_with_models.sh produces an APK > ~2 GiB and the device says
# "There was a problem with the package" / invalid app file (many installers use 32-bit sizes).
#
# After install, push models:
#   ./push_sharp_vulkan_only.sh
#   # → Android/data/com.furnit.android/files/models_cpuvulkan_hybrid/
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VARIANT="${VARIANT:-etVulkanDebug}"
cap_first() {
  local s="$1"
  echo "$(echo "${s:0:1}" | tr '[:lower:]' '[:upper:]')${s:1}"
}
TASK_VARIANT="$(cap_first "$VARIANT")"

echo "Building :app:assemble${TASK_VARIANT} with NO bundled models (-PskipExecutorchAssets=true) …"
./gradlew ":app:assemble${TASK_VARIANT}" -PskipExecutorchAssets=true "$@"

echo ""
echo "Install: adb install -r app/build/outputs/apk/etVulkan/debug/app-etVulkan-arm64-v8a-debug.apk"
echo "Then:    ./push_sharp_vulkan_only.sh"
