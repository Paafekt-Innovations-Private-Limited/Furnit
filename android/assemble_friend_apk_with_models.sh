#!/usr/bin/env bash
# Build a friend APK with SHARP ExecuTorch .pte for ML room creation.
#
# Default flow (-PbundleSharpVulkanHybridApk=true, same as below):
#   • If android/models_cpuvulkan_hybrid/*.pte exists (non-empty): ONLY that folder is copied → APK
#     assets/models_cpuvulkan_hybrid (no models_cpu .pte; sharp_vulkan_only is not merged).
#     Populate: ./populate_models_cpuvulkan_hybrid_from_backups.sh — see models_cpuvulkan_hybrid/README.md.
#   • If that folder is empty: falls back to minimal sharp_vulkan_only + INT8 Part1+2 from
#     executorch_int8_models → still only models_cpuvulkan_hybrid in the APK for SHARP (no models_cpu).
#
# YOLOE / other detectors stay in src/main/assets as today (unchanged by this script).
#
# Copies a timestamped duplicate to android/friend-apk-dist/ so Android Studio Run keeps using
# app/build/outputs/apk/... only (no friend vs Studio overwrite confusion).
# Local day-to-day: keep skipExecutorchAssets=true in gradle.properties and adb push models instead.
#
# Usage (from repo android/):
#   ./assemble_friend_apk_with_models.sh
#   ./assemble_friend_apk_with_models.sh -- -PbundleSharpVulkanHybridApk=true
#       → -PskipExecutorchAssets=false -PbundleSharpVulkanHybridApk=true (+ any extra flags after --)
#   ./assemble_friend_apk_with_models.sh etVulkanRelease
#   VARIANT=etCpuDebug ./assemble_friend_apk_with_models.sh
#
# Extra Gradle flags after -- (e.g. force fp32, chunk_65, disable staged dir):
#   ./assemble_friend_apk_with_models.sh -- -PbundleSharpVulkanPrecision=fp32 -PbundleSharpVulkanIncludeChunk65=true
#   ./assemble_friend_apk_with_models.sh -- -PuseModelsCpuVulkanHybridStagingForFriendApk=false
# To try copying every *.pte from sharp_vulkan_only (will FAIL if total > ~4GiB Zip32 APK limit):
#   ./assemble_friend_apk_with_models.sh -- -PbundleSharpVulkanHybridApk=false
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Many on-device installers use signed 32-bit sizes (~2 GiB). Bundled fp32 Part1–3 + Part4 often exceeds that.
# See android/friend-apk-dist/README.md and android/docs/MODELS_APK_BUNDLE_TESTING.md.
apk_byte_size() {
  local f="$1"
  if stat -f%z "$f" >/dev/null 2>&1; then stat -f%z "$f"
  else stat -c%s "$f"
  fi
}

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
HYBRID_STAGING="$SCRIPT_DIR/models_cpuvulkan_hybrid"
shopt -s nullglob
hybrid_ptes=("$HYBRID_STAGING"/*.pte)
shopt -u nullglob
if [[ ${#hybrid_ptes[@]} -gt 0 ]]; then
  echo "SHARP hybrid: ${#hybrid_ptes[@]} .pte in models_cpuvulkan_hybrid/ → friend APK uses that tree only (no models_cpu ExecuTorch .pte)."
else
  echo "Note: models_cpuvulkan_hybrid/ has no .pte — Gradle will build hybrid from sharp_vulkan_only + executorch_int8_models (see app/build.gradle). For staged-only layout run: ./populate_models_cpuvulkan_hybrid_from_backups.sh"
fi
echo ""

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
    sz="$(apk_byte_size "$dest")"
    mb=$((sz / 1024 / 1024))
    echo "  Size: ${sz} bytes (~${mb} MiB)"
    # Integer.MAX_VALUE — common ~2 GiB sideload / PackageInstaller pitfall
    if [[ "$sz" -gt 2000000000 ]]; then
      echo ""
      echo "  *** WARNING: APK is larger than ~2 GiB. Many phones report \"problem with the app file\" or"
      echo "      \"package appears to be invalid\" when installing from Files / chat / Bluetooth."
      echo "      This is usually NOT a corrupt build — the full Vulkan bundle is too large for those installers."
      echo "      Try:  adb install -r \"$dest\""
      echo "      Or:   ./assemble_friend_apk_shell_only.sh  then  ./push_sharp_vulkan_only.sh"
      echo ""
    fi
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
