#!/usr/bin/env bash
# Build a small friend/test APK **without** bundling SHARP ExecuTorch .pte (avoids ~2–3+ GiB installs that
# fail on many phones). Same workflow as assemble_friend_apk_with_models.sh, but forces
# -PskipExecutorchAssets=true. Your friend installs the lightweight APK, copies a hybrid model folder from
# Google Drive to the laptop, then adb push (see end-of-run instructions and friend-apk-dist/README.md).
#
# Does **not** change gradle.properties — Android Studio Run/Debug keeps using your usual local setup
# (default skipExecutorchAssets=true). This script only passes -P for this Gradle invocation.
#
# Gradle flavors are **etVulkan** or **etCpu** only. "CPU+Vulkan hybrid" SHARP uses the **etVulkan** APK and
# models under files/models_cpuvulkan_hybrid/. Alias:
#   etCpuVulkanDebug  →  etVulkanDebug  (same as assemble_friend_apk_with_models.sh etVulkanDebug)
#
# Usage (from repo android/):
#   ./assemble_friend_apk_without_models.sh
#   ./assemble_friend_apk_without_models.sh etVulkanDebug
#   ./assemble_friend_apk_without_models.sh etCpuVulkanDebug
#   VARIANT=etCpuDebug ./assemble_friend_apk_without_models.sh
#   ./assemble_friend_apk_without_models.sh -- -PsomeFlag=value
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

die() {
  echo "Error: $*" >&2
  exit 1
}

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

while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    shift
    GRADLE_EXTRA=("$@")
    break
  fi
  VARIANT="$1"
  shift
done

# Hybrid CPU Part1+2 + Vulkan Part3–4: single product flavor "etVulkan" (not a separate "etCpuVulkan" flavor).
case "$VARIANT" in
  etCpuVulkanDebug|EtCpuVulkanDebug|etCpuVulkanRelease|EtCpuVulkanRelease)
    echo "Note: $VARIANT is an alias for etVulkanDebug / etVulkanRelease (CPU+Vulkan hybrid uses etVulkan flavor)."
    if echo "$VARIANT" | grep -qiE 'release$'; then
      VARIANT="etVulkanRelease"
    else
      VARIANT="etVulkanDebug"
    fi
    ;;
esac

if echo "$VARIANT" | grep -qiE 'release$'; then
  echo "WARNING: $VARIANT builds a release APK. This project may emit release-unsigned.apk (not phone-installable)."
  echo "Prefer etVulkanDebug or etCpuDebug for friend installs unless release signing is configured."
  echo ""
fi

TASK_VARIANT="$(cap_first "$VARIANT")"
echo "Building :app:assemble${TASK_VARIANT} with NO bundled SHARP .pte (-PskipExecutorchAssets=true) …"
if [[ ${#GRADLE_EXTRA[@]} -gt 0 ]]; then
  ./gradlew ":app:assemble${TASK_VARIANT}" \
    -PskipExecutorchAssets=true \
    "${GRADLE_EXTRA[@]}"
else
  ./gradlew ":app:assemble${TASK_VARIANT}" \
    -PskipExecutorchAssets=true
fi

DIST_DIR="$SCRIPT_DIR/friend-apk-dist"
mkdir -p "$DIST_DIR"
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
    if [[ "$base" == *"-unsigned.apk" ]]; then
      die "build produced unsigned APK ($base). Use *Debug for friend installs, or add release signing."
    fi
    dest="$DIST_DIR/friend-nomodels-${TS}-${base}"
    if cp -f "$apk" "$dest"; then
      echo "Friend copy: $dest"
      sz="$(apk_byte_size "$dest")"
    else
      echo "WARNING: failed to copy friend APK to $dest (likely low disk space)."
      sz="$(apk_byte_size "$apk")"
      dest="$apk"
    fi
    mb=$((sz / 1024 / 1024))
    echo "  Size: ${sz} bytes (~${mb} MiB)"
    COPIED=$((COPIED + 1))
  done
  shopt -u nullglob
fi
if [[ "$COPIED" -eq 0 ]]; then
  echo "WARN: No .apk found under $OUT_DIR — check variant name (current: $VARIANT → $FLAVOR/$BUILD_TYPE)."
fi

HYBRID_DEST="/sdcard/Android/data/com.furnit.android/files/models_cpuvulkan_hybrid"
CPU_DEST="/sdcard/Android/data/com.furnit.android/files/models_cpu"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Install (from laptop; replace APK path if you use friend-apk-dist copy):"
echo "  adb install -r $OUT_DIR/app-${FLAVOR}-arm64-v8a-${BUILD_TYPE}.apk"
echo ""
if [[ "$FLAVOR" == "etVulkan" ]]; then
  echo "SHARP models (etVulkan / CPU+Vulkan hybrid): push the folder from Google Drive — same file set as"
  echo "  android/models_cpuvulkan_hybrid/ on your machine (see models_cpuvulkan_hybrid/README.md)."
  echo "  One-shot from repo (laptop has the folder locally):"
  echo "    cd $SCRIPT_DIR && ./push_sharp_cpuvulkan_hybrid_androidstudio.sh /path/to/downloaded/hybrid_folder"
  echo "  Or manually:"
  echo "    adb shell mkdir -p $HYBRID_DEST"
  echo "    adb push /path/to/downloaded/hybrid_folder/*.pte $HYBRID_DEST/"
  echo "    # Include any .pte.manifest.json and part1_test_patch_f32.bin if present in your export."
  echo ""
  echo "Destination on device: $HYBRID_DEST"
elif [[ "$FLAVOR" == "etCpu" ]]; then
  echo "SHARP models (etCpu / XNNPACK): push CPU portable .pte to:"
  echo "    $CPU_DEST"
  echo "  Helper: ./push_sharp_executorch_cpu_models.sh [dir]"
  echo ""
fi
echo "YOLO / Furniture Fit assets (NCNN/TFLite) stay inside the APK — no separate Drive bundle for those."
echo "Android Studio output unchanged: $SCRIPT_DIR/app/build/outputs/apk/"
echo "Friend copy prefix: friend-nomodels-* → $DIST_DIR/"
echo "Tester without repo: zip APK + models_cpuvulkan_hybrid/ + $SCRIPT_DIR/friend_tester_bundle/ (see PACK_FOR_TESTER.md there)."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
