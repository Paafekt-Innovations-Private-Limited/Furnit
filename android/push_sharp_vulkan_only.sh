#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEW_SCRIPT_PATH="$SCRIPT_DIR/push_sharp_cpuvulkan_hybrid_androidstudio.sh"

echo "Deprecated: use ./push_sharp_cpuvulkan_hybrid_androidstudio.sh"
exec "$NEW_SCRIPT_PATH" "$@"
