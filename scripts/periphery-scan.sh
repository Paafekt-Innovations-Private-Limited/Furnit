#!/usr/bin/env bash
# Run Periphery against the Furnit iOS target. Requires: brew install peripheryapp/periphery/periphery
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if ! command -v periphery >/dev/null 2>&1; then
  echo "Periphery not found. Install: brew install peripheryapp/periphery/periphery" >&2
  exit 1
fi
OUT="${1:-periphery-report.txt}"
periphery scan --disable-update-check 2>&1 | tee "$OUT"
echo ""
echo "Report written to: $OUT"
