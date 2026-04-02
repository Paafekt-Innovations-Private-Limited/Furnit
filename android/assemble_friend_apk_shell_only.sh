#!/usr/bin/env bash
# Deprecated name — use assemble_friend_apk_without_models.sh (same behavior).
# Small etVulkan debug APK without bundled .pte; models via adb push.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/assemble_friend_apk_without_models.sh" "$@"
