#!/usr/bin/env bash
# Copy hybrid tree: INT8 Part1+2 + Vulkan Part3/4a + **fine-split tile_00** Part4b (strategy A, 5 files).
# Defaults: VK from android/sharp_vulkan_only, INT8 from /Volumes/LaCie/Backup21stApr2026/android/executorch_models_cpu_from_lacie.
# Override: VK_SRC= CPU_SRC= DEST=
# Precision policy: prefer Vulkan FP16 Part3 when available; fall back to FP32 only if FP16 is absent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="${DEST:-$REPO_ROOT/models_cpuvulkan_hybrid}"
VK_SRC="${VK_SRC:-$REPO_ROOT/sharp_vulkan_only}"
CPU_SRC="${CPU_SRC:-/Volumes/LaCie/Backup21stApr2026/android/executorch_models_cpu_from_lacie}"

die() { echo "Error: $*" >&2; exit 1; }

[[ -d "$VK_SRC" ]] || die "Vulkan source dir not found: $VK_SRC"
[[ -d "$CPU_SRC" ]] || die "CPU INT8 source dir not found: $CPU_SRC"

mkdir -p "$DEST"

echo "Populating $DEST"
echo "  Vulkan (Part3–4a + fine-split Part4b): $VK_SRC"
echo "  INT8 Part1–2:                         $CPU_SRC"
echo ""

# This staging folder is used directly by assemble_friend_apk_with_models.sh.
# Wipe old SHARP staging files first so stale fp32 / extra Part4b variants do not
# accidentally get bundled into a friend APK.
shopt -s nullglob
stale_stage_files=("$DEST"/sharp_split_part*.pte "$DEST"/sharp_split_part*.pte.manifest.json)
shopt -u nullglob
if [[ ${#stale_stage_files[@]} -gt 0 ]]; then
  rm -f "${stale_stage_files[@]}"
fi

PART3_FILE="sharp_split_part3_vulkan_fp16.pte"
if [[ ! -f "$VK_SRC/$PART3_FILE" ]]; then
  PART3_FILE="sharp_split_part3_vulkan_fp32.pte"
fi
[[ -f "$VK_SRC/$PART3_FILE" ]] || die "missing $VK_SRC/sharp_split_part3_vulkan_fp16.pte and $VK_SRC/sharp_split_part3_vulkan_fp32.pte"
echo "  Using Vulkan Part3:                   $PART3_FILE"
echo ""

for f in \
  "$PART3_FILE" \
  sharp_split_part4a_chunk_512_vulkan.pte \
  sharp_split_part4a_chunk_65_vulkan.pte \
  sharp_split_part4b_tile_00_stage_pre_vulkan.pte \
  sharp_split_part4b_tile_00_decoder_head.pte \
  sharp_split_part4b_tile_00_init_base.pte \
  sharp_split_part4b_tile_00_raw_heads_vulkan.pte \
  sharp_split_part4b_tile_00_compose.pte
do
  [[ -f "$VK_SRC/$f" ]] || die "missing $VK_SRC/$f"
  cp -v "$VK_SRC/$f" "$DEST/$f"
done

for f in sharp_split_part1_int8.pte sharp_split_part2_int8.pte
do
  [[ -f "$CPU_SRC/$f" ]] || die "missing $CPU_SRC/$f"
  cp -v "$CPU_SRC/$f" "$DEST/$f"
done

echo ""
ls -lh "$DEST"
echo "Done. See models_cpuvulkan_hybrid/README.md"
