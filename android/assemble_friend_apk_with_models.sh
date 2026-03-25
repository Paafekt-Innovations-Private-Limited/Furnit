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
#   ./assemble_friend_apk_with_models.sh etVulkanDebug
#   VARIANT=etCpuDebug ./assemble_friend_apk_with_models.sh
#
# Extra Gradle flags after -- (e.g. disable staged dir):
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

die() {
  echo "Error: $*" >&2
  exit 1
}

# Bash 3.2–safe: capitalize first character (etVulkanDebug → EtVulkanDebug)
cap_first() {
  local s="$1"
  local c rest
  c=$(echo "${s:0:1}" | tr '[:lower:]' '[:upper:]')
  rest="${s:1}"
  echo "${c}${rest}"
}

validate_hybrid_staging() {
  local dir="$1"
  local total_bytes=0
  local f p
  local has_part1_int8=0
  local has_part2_int8=0
  local has_part3_fp16=0
  local has_part3_fp32=0
  local has_part4a_512=0
  local has_part4a_65=0
  local a_stage_pre=0 a_decoder=0 a_init=0 a_heads=0 a_compose=0
  local b_stage_a=0 b_init=0 b_heads=0 b_compose=0
  local c_stage_pre=0 c_decoder=0 c_init=0 c_heads=0 c_compose=0
  local d_stage_a=0 d_init=0 d_heads=0 d_compose=0
  local e_tile_b2=0 e_tile_b4=0
  local f_tile_00=0 f_tile_full=0
  local extra_files=()
  local strategy_count=0
  local strategy_name=""
  local chosen_part3=""

  shopt -s nullglob
  local hybrid_ptes=("$dir"/*.pte)
  shopt -u nullglob
  if [[ ${#hybrid_ptes[@]} -eq 0 ]]; then
    return 0
  fi

  for p in "${hybrid_ptes[@]}"; do
    f="$(basename "$p")"
    total_bytes=$((total_bytes + $(apk_byte_size "$p")))
    case "$f" in
      sharp_split_part1_int8.pte) has_part1_int8=1 ;;
      sharp_split_part2_int8.pte) has_part2_int8=1 ;;
      sharp_split_part3_vulkan_fp16.pte) has_part3_fp16=1 ;;
      sharp_split_part3_vulkan_fp32.pte) has_part3_fp32=1 ;;
      sharp_split_part4a_chunk_512_vulkan.pte) has_part4a_512=1 ;;
      sharp_split_part4a_chunk_65_vulkan.pte) has_part4a_65=1 ;;
      sharp_split_part4b_tile_00_stage_pre_vulkan.pte) a_stage_pre=1 ;;
      sharp_split_part4b_tile_00_decoder_head.pte) a_decoder=1 ;;
      sharp_split_part4b_tile_00_init_base.pte) a_init=1; b_init=1 ;;
      sharp_split_part4b_tile_00_raw_heads_vulkan.pte) a_heads=1; b_heads=1 ;;
      sharp_split_part4b_tile_00_compose.pte) a_compose=1; b_compose=1 ;;
      sharp_split_part4b_tile_00_stage_a_vulkan.pte) b_stage_a=1 ;;
      sharp_split_part4b_tile_b2_stage_pre_vulkan.pte) c_stage_pre=1 ;;
      sharp_split_part4b_tile_b2_decoder_head.pte) c_decoder=1 ;;
      sharp_split_part4b_tile_b2_init_base.pte) c_init=1; d_init=1 ;;
      sharp_split_part4b_tile_b2_raw_heads_vulkan.pte) c_heads=1; d_heads=1 ;;
      sharp_split_part4b_tile_b2_compose.pte) c_compose=1; d_compose=1 ;;
      sharp_split_part4b_tile_b2_stage_a_vulkan.pte) d_stage_a=1 ;;
      sharp_split_part4b_tile_b2.pte) e_tile_b2=1 ;;
      sharp_split_part4b_tile_b4.pte) e_tile_b4=1 ;;
      sharp_split_part4b_tile_00.pte) f_tile_00=1 ;;
      sharp_split_part4b_tile_full.pte) f_tile_full=1 ;;
      *) extra_files+=("$f") ;;
    esac
  done

  [[ $has_part1_int8 -eq 1 ]] || die "models_cpuvulkan_hybrid missing sharp_split_part1_int8.pte"
  [[ $has_part2_int8 -eq 1 ]] || die "models_cpuvulkan_hybrid missing sharp_split_part2_int8.pte"
  [[ $has_part4a_512 -eq 1 ]] || die "models_cpuvulkan_hybrid missing sharp_split_part4a_chunk_512_vulkan.pte"
  [[ $has_part4a_65 -eq 1 ]] || die "models_cpuvulkan_hybrid missing sharp_split_part4a_chunk_65_vulkan.pte"

  if [[ $has_part3_fp16 -eq 1 && $has_part3_fp32 -eq 1 ]]; then
    die "models_cpuvulkan_hybrid has both sharp_split_part3_vulkan_fp16.pte and _fp32.pte. Keep exactly one."
  fi
  if [[ $has_part3_fp16 -eq 1 ]]; then
    chosen_part3="sharp_split_part3_vulkan_fp16.pte"
  elif [[ $has_part3_fp32 -eq 1 ]]; then
    chosen_part3="sharp_split_part3_vulkan_fp32.pte"
  else
    die "models_cpuvulkan_hybrid missing sharp_split_part3_vulkan_fp16.pte or sharp_split_part3_vulkan_fp32.pte"
  fi

  if [[ $a_stage_pre -eq 1 && $a_decoder -eq 1 && $a_init -eq 1 && $a_heads -eq 1 && $a_compose -eq 1 ]]; then
    strategy_count=$((strategy_count + 1))
    strategy_name="fine-split tile_00"
  fi
  if [[ $b_stage_a -eq 1 && $b_init -eq 1 && $b_heads -eq 1 && $b_compose -eq 1 ]]; then
    strategy_count=$((strategy_count + 1))
    strategy_name="split tile_00"
  fi
  if [[ $c_stage_pre -eq 1 && $c_decoder -eq 1 && $c_init -eq 1 && $c_heads -eq 1 && $c_compose -eq 1 ]]; then
    strategy_count=$((strategy_count + 1))
    strategy_name="fine-split tile_b2"
  fi
  if [[ $d_stage_a -eq 1 && $d_init -eq 1 && $d_heads -eq 1 && $d_compose -eq 1 ]]; then
    strategy_count=$((strategy_count + 1))
    strategy_name="split tile_b2"
  fi
  if [[ $e_tile_b2 -eq 1 || $e_tile_b4 -eq 1 ]]; then
    strategy_count=$((strategy_count + 1))
    strategy_name="legacy batched tile"
  fi
  if [[ $f_tile_00 -eq 1 || $f_tile_full -eq 1 ]]; then
    strategy_count=$((strategy_count + 1))
    strategy_name="legacy sequential tile"
  fi

  [[ $strategy_count -ge 1 ]] || die "models_cpuvulkan_hybrid is missing a complete Part4b strategy"
  [[ $strategy_count -eq 1 ]] || die "models_cpuvulkan_hybrid has multiple Part4b strategies. Keep only one compact strategy."

  if [[ ${#extra_files[@]} -gt 0 ]]; then
    die "models_cpuvulkan_hybrid contains extra .pte not allowed for friend APK staging: ${extra_files[*]}"
  fi

  echo "Validated models_cpuvulkan_hybrid:"
  echo "  Part3:  $chosen_part3"
  echo "  Part4b: $strategy_name"
  echo "  Payload size: $((total_bytes / 1024 / 1024)) MiB"
  echo ""
  if [[ "$total_bytes" -gt 2000000000 ]]; then
    echo "WARNING: staged hybrid payload is > ~2 GiB. Some phones/installers reject the final APK even if Pixel accepts it."
    echo ""
  fi
}

VARIANT="${VARIANT:-etVulkanDebug}"
GRADLE_EXTRA=()
HYBRID_STAGING="$SCRIPT_DIR/models_cpuvulkan_hybrid"
shopt -s nullglob
hybrid_ptes=("$HYBRID_STAGING"/*.pte)
shopt -u nullglob
if [[ ${#hybrid_ptes[@]} -gt 0 ]]; then
  echo "SHARP hybrid: ${#hybrid_ptes[@]} .pte in models_cpuvulkan_hybrid/ → friend APK uses that tree only (no models_cpu ExecuTorch .pte)."
  validate_hybrid_staging "$HYBRID_STAGING"
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

if echo "$VARIANT" | grep -qiE 'release$'; then
  echo "WARNING: $VARIANT builds a release APK. This project currently emits release-unsigned.apk (not phone-installable)."
  echo "Use etVulkanDebug for friend installs unless you add a release signing config."
  echo ""
fi

TASK_VARIANT="$(cap_first "$VARIANT")"
echo "Building :app:assemble${TASK_VARIANT} with bundled models (hybrid/minimal Vulkan set; Zip32-safe) …"
if [[ ${#GRADLE_EXTRA[@]} -gt 0 ]]; then
  ./gradlew ":app:assemble${TASK_VARIANT}" \
    -PskipExecutorchAssets=false \
    -PbundleSharpVulkanHybridApk=true \
    "${GRADLE_EXTRA[@]}"
else
  ./gradlew ":app:assemble${TASK_VARIANT}" \
    -PskipExecutorchAssets=false \
    -PbundleSharpVulkanHybridApk=true
fi

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
OVERSIZED=0
if [[ -d "$OUT_DIR" ]]; then
  shopt -s nullglob
  for apk in "$OUT_DIR"/*.apk; do
    base="$(basename "$apk")"
    if [[ "$base" == *"-unsigned.apk" ]]; then
      die "build produced unsigned APK ($base). Use etVulkanDebug for friend installs, or add release signing."
    fi
    dest="$DIST_DIR/friend-${TS}-${base}"
    if cp -f "$apk" "$dest"; then
      echo "Friend copy: $dest"
      sz="$(apk_byte_size "$dest")"
    else
      echo "WARNING: failed to copy friend APK to $dest (likely low disk space)."
      echo "         Use the built APK directly: $apk"
      sz="$(apk_byte_size "$apk")"
      dest="$apk"
    fi
    mb=$((sz / 1024 / 1024))
    echo "  Size: ${sz} bytes (~${mb} MiB)"
    # Integer.MAX_VALUE — common ~2 GiB sideload / PackageInstaller pitfall
    if [[ "$sz" -gt 2000000000 ]]; then
      echo ""
      echo "  *** ERROR: APK is larger than ~2 GiB. Many phones report \"problem with the app file\" or"
      echo "      \"package appears to be invalid\" when installing from Files / chat / Bluetooth."
      echo "      Use a smaller staged hybrid set, or use:"
      echo "        ./assemble_friend_apk_shell_only.sh"
      echo "        ./push_sharp_cpuvulkan_hybrid_androidstudio.sh"
      echo ""
      OVERSIZED=1
    fi
    COPIED=$((COPIED + 1))
  done
  shopt -u nullglob
fi
if [[ "$COPIED" -eq 0 ]]; then
  echo "WARN: No .apk found under $OUT_DIR — check variant name (current: $VARIANT → $FLAVOR/$BUILD_TYPE)."
fi

if [[ "$OVERSIZED" -eq 1 ]]; then
  echo "WARNING: friend APK built, but it is large enough that some phones may reject it during sideload."
fi

echo ""
echo "Done. Studio/default Gradle output (unchanged): $SCRIPT_DIR/app/build/outputs/apk/"
echo "Friend timestamped copies: $DIST_DIR/"
