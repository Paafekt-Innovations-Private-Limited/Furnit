#!/bin/bash
# Solution 4: Increase swap size to reduce LMK kills during SHARP Part 4.
# REQUIRES ROOT. Run on device via: adb shell su -c 'sh /path/to/script'
#
# Part 4 decoder activation memory can exceed 4GB. Default zram swap (~2-4GB)
# is often insufficient, causing LMK to kill com.furnit.android.
#
# This script increases zram swap to 8GB. Paths vary by device:
#   - Pixel/stock: /dev/block/zram0
#   - Some: /sys/block/zram0/disksize
#
# Usage:
#   adb push scripts/increase_swap_root.sh /data/local/tmp/
#   adb shell su -c 'sh /data/local/tmp/increase_swap_root.sh'
#
# Verify: adb shell cat /proc/swaps

set -e

ZRAM_DISK="/sys/block/zram0/disksize"
ZRAM_DEV="/dev/block/zram0"
SIZE_MB=8192

if [ ! -f "$ZRAM_DISK" ]; then
  echo "zram disksize not found at $ZRAM_DISK"
  echo "Try: ls /sys/block/zram*/disksize"
  exit 1
fi

echo "Current swap:"
cat /proc/swaps 2>/dev/null || true

echo ""
echo "Increasing zram to ${SIZE_MB}MB..."

swapoff "$ZRAM_DEV" 2>/dev/null || true
echo "${SIZE_MB}M" > "$ZRAM_DISK"
mkswap "$ZRAM_DEV"
swapon "$ZRAM_DEV"

echo ""
echo "New swap:"
cat /proc/swaps
