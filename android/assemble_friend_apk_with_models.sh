#!/usr/bin/env bash
# Build an APK with SHARP .pte bundled under assets/models_cpu + models_vulkan.
# Copies a timestamped duplicate to android/friend-apk-dist/ so Android Studio Run keeps using
# app/build/outputs/apk/... only (no friend vs Studio overwrite confusion).
# Local day-to-day: keep skipExecutorchAssets=true in gradle.properties and adb push models instead.
#
# Usage (from repo android/):
#   ./assemble_friend_apk_with_models.sh
#       → assemble with -PskipExecutorchAssets=false AND -PbundleSharpVulkanHybridApk=true
#         (full sharp_vulkan_only/*.pte exceeds Zip32 ~4GiB — see MODELS_APK_BUNDLE_TESTING.md)
#   ./assemble_friend_apk_with_models.sh etVulkanRelease
#   VARIANT=etCpuDebug ./assemble_friend_apk_with_models.sh
#
# Extra Gradle flags after -- (e.g. force fp32 or add chunk_65):
#   ./assemble_friend_apk_with_models.sh -- -PbundleSharpVulkanPrecision=fp32 -PbundleSharpVulkanIncludeChunk65=true
# To try copying every *.pte (will FAIL if total > ~4GiB Zip32 APK limit):
#   ./assemble_friend_apk_with_models.sh -- -PbundleSharpVulkanHybridApk=false
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Bash 3.2–safe: capitalize first character (etVulkanDebug → EtVulkanDebug)
cap_first() {
  local s="$1"
  local c rest
  c=$(echo "${s:0:1}" | tr '[:lower:]' '[:upper:]')
  rest="${s:1}"
  echo "${c}${rest}"
}

VARIANT="${VARIANT:-etVulkanDebug}"
GRADLE_EXTRA=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    shift
    GRADLE_EXTRA=("$@")
    break
  fi
  VARIANT="$1"
  shift
done

TASK_VARIANT="$(cap_first "$VARIANT")"
echo "Building :app:assemble${TASK_VARIANT} with bundled models (hybrid/minimal Vulkan set; Zip32-safe) …"
./gradlew ":app:assemble${TASK_VARIANT}" \
  -PskipExecutorchAssets=false \
  -PbundleSharpVulkanHybridApk=true \
  "${GRADLE_EXTRA[@]}"

# Friend APKs: copy to android/friend-apk-dist/ (timestamped) so Android Studio outputs stay the canonical debug path.
DIST_DIR="$SCRIPT_DIR/friend-apk-dist"
mkdir -p "$DIST_DIR"
# etVulkanDebug → flavor etVulkan + debug; etCpuRelease → etCpu + release (case-insensitive suffix strip)
BUILD_TYPE="debug"
echo "$VARIANT" | grep -qiE 'release$' && BUILD_TYPE="release"
FLAVOR="$(echo "$VARIANT" | sed -E 's/(Debug|DEBUG|debug|Release|RELEASE|release)$//')"
OUT_DIR="$SCRIPT_DIR/app/build/outputs/apk/$FLAVOR/$BUILD_TYPE"
TS="$(date +%Y%m%d-%H%M%S)"
COPIED=0
if [[ -d "$OUT_DIR" ]]; then
  shopt -s nullglob
  for apk in "$OUT_DIR"/*.apk; do
    base="$(basename "$apk")"
    dest="$DIST_DIR/friend-${TS}-${base}"
    cp -f "$apk" "$dest"
    echo "Friend copy: $dest"
    COPIED=$((COPIED + 1))
  done
  shopt -u nullglob
fi
if [[ "$COPIED" -eq 0 ]]; then
  echo "WARN: No .apk found under $OUT_DIR — check variant name (current: $VARIANT → $FLAVOR/$BUILD_TYPE)."
fi

echo ""
echo "Done. Studio/default Gradle output (unchanged): $SCRIPT_DIR/app/build/outputs/apk/"
echo "Friend timestamped copies: $DIST_DIR/"
