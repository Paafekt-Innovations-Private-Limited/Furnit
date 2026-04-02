#!/usr/bin/env bash
# Install the Furnit APK from this folder (no repo needed). Requires adb.
#
# Put exactly one .apk here, or pass the path (any filename is fine, e.g. furnit.apk):
#   ./install_furnit_apk.sh
#   ./install_furnit_apk.sh /path/to/app-etVulkan-arm64-v8a-debug.apk
#
# Multiple adb devices: physical phone is preferred over emulator. Override:
#   export ANDROID_SERIAL=<serial>   # adb devices
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=adb_common.sh
source "$SCRIPT_DIR/adb_common.sh"

APK="${1:-}"

if [[ -z "$APK" ]]; then
  shopt -s nullglob
  candidates=("$SCRIPT_DIR"/*.apk)
  shopt -u nullglob
  if [[ ${#candidates[@]} -eq 1 ]]; then
    APK="${candidates[0]}"
  elif [[ ${#candidates[@]} -gt 1 ]]; then
    echo "Several .apk files in this folder. Pass the one to install:"
    printf '  %s\n' "${candidates[@]}"
    echo "Usage: $0 /path/to/furnit.apk"
    exit 1
  fi
fi

if [[ -z "$APK" || ! -f "$APK" ]]; then
  echo "No APK found. Copy the lightweight Furnit .apk into this folder, then run:"
  echo "  $0"
  exit 1
fi

if ! command -v adb &> /dev/null; then
  echo "Error: adb not found. Install Android Platform Tools: https://developer.android.com/tools/releases/platform-tools"
  exit 1
fi

SERIAL="$(pick_physical_serial)" || exit 1
echo "Using device: $SERIAL"
ADB=(adb -s "$SERIAL")

echo "Installing: $APK"
"${ADB[@]}" install -r "$APK"
echo "Done. Next: ./push_furnit_sharp_models.sh (if models are not bundled in the APK)."
