#!/usr/bin/env bash
# Copy hybrid tree: INT8 Part1+2 + Vulkan Part3/4a + **fine-split tile_00** Part4b (strategy A, 5 files).
# Defaults: VK from android/sharp_vulkan_only, INT8 from LaCie backup. Override: VK_SRC= CPU_SRC= DEST=
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-$SCRIPT_DIR/models_cpuvulkan_hybrid}"
VK_SRC="${VK_SRC:-$SCRIPT_DIR/sharp_vulkan_only}"
CPU_SRC="${CPU_SRC:-/Volumes/LaCie/BackUp21stMarch2026GPU/Furnit/android/models_cpu}"

die() { echo "Error: $*" >&2; exit 1; }

[[ -d "$VK_SRC" ]] || die "Vulkan source dir not found: $VK_SRC"
[[ -d "$CPU_SRC" ]] || die "CPU INT8 source dir not found: $CPU_SRC"

mkdir -p "$DEST"

echo "Populating $DEST"
echo "  Vulkan (Part3–4a + fine-split Part4b): $VK_SRC"
echo "  INT8 Part1–2:                         $CPU_SRC"
echo ""

for f in \
  sharp_split_part3_vulkan_fp32.pte \
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

# Remove stale Part4b if switching strategies
rm -f "$DEST/sharp_split_part4b_tile_b2.pte" "$DEST/sharp_split_part4b_vulkan.pte"

for f in sharp_split_part1_int8.pte sharp_split_part2_int8.pte
do
  [[ -f "$CPU_SRC/$f" ]] || die "missing $CPU_SRC/$f"
  cp -v "$CPU_SRC/$f" "$DEST/$f"
done

echo ""
ls -lh "$DEST"
echo "Done. See models_cpuvulkan_hybrid/README.md"
