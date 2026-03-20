#!/bin/bash
# 1) Re-run SHARP ExecuTorch Vulkan export with full logging.
# 2) Build a ready-to-paste help prompt that includes log path and log tail.
#
# Usage: cd android && ./reexport_and_build_help_prompt.sh
#
# Output:
#   - export_log_vulkan_YYYYMMDD_HHMMSS.txt (full export log)
#   - help_request_prompt.md (paste this when asking for help)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="${SCRIPT_DIR}/help_request_prompt.md"
LOG_TAIL_LINES=400

echo "Running SHARP ExecuTorch Vulkan export (full log to export_log_vulkan_*.txt)..."
"${SCRIPT_DIR}/export_sharp_executorch_with_log.sh"
EXIT_CODE=$?

# Log path was written at start of export
LOG_FILE=""
if [ -f "${SCRIPT_DIR}/export_log_latest_path.txt" ]; then
  LOG_FILE=$(cat "${SCRIPT_DIR}/export_log_latest_path.txt")
fi

# Build help prompt
{
  echo "# Help request: SHARP ExecuTorch Vulkan export / Android runtime"
  echo ""
  echo "## Context"
  echo "- **Project:** Furnit Android app. SHARP model (4-part split) exported to ExecuTorch .pte with **Vulkan** backend for Part1/Part2/Part3/Part4."
  echo "- **Goal:** Verify export is correct; app sometimes fails with BackendFailed (error 32), device-lost, or mem-pressure/OOM during Part1+2."
  echo "- **Export:** Vulkan FP16, chunked Part4 (4a_512, 4a_65, 4b), patch_batch=2 for Part1/Part2."
  echo ""
  echo "## Exact export command"
  echo '```'
  echo "cd android"
  echo "python3 export_sharp_executorch_split4.py \\"
  echo "  --backend vulkan \\"
  echo "  --chunked-part4 \\"
  echo "  --dtype fp16 \\"
  echo "  --patch-batch-size 2 \\"
  echo "  --sharp-src third_party/ml-sharp/src \\"
  echo "  --weights sharp_litert_models/sharp_2572gikvuh.pt \\"
  echo "  --output-dir executorch_models"
  echo '```'
  echo ""
  echo "## What I need help with"
  echo "- Confirm from the export log below: did export complete for all parts (Part1, Part2, Part3, Part4, Part4a chunk 512/65, Part4b)? Any errors or warnings?"
  echo "- If export succeeded but the app fails on device (BackendFailed 32, VK_DEVICE_LOST, or OOM): is that likely an export/runtime mismatch or device/driver/memory?"
  echo ""
  echo "## Export log"
  if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
    echo "Full log file: \`$LOG_FILE\`"
    echo ""
    echo "Last ${LOG_TAIL_LINES} lines:"
    echo '```'
    tail -n "$LOG_TAIL_LINES" "$LOG_FILE"
    echo '```'
  else
    echo "(Log file not found. Check android/export_log_vulkan_*.txt)"
  fi
  echo ""
  echo "---"
  echo "If the log above is truncated, attach the full file: \`$LOG_FILE\`"
} > "$PROMPT_FILE"

echo ""
echo "Export exit code: $EXIT_CODE"
echo "Help prompt written to: $PROMPT_FILE"
echo "Paste the contents of $PROMPT_FILE when asking for help (or attach the full log: $LOG_FILE)."

exit "$EXIT_CODE"
