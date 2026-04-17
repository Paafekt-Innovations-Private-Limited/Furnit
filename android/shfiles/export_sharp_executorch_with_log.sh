#!/bin/bash
# Run SHARP ExecuTorch export WITH VULKAN backend and write full logs to a file.
# Usage: cd android && ./export_sharp_executorch_with_log.sh
# Log file: android/export_log_vulkan_YYYYMMDD_HHMMSS.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${SCRIPT_DIR}/export_log_vulkan_$(date +%Y%m%d_%H%M%S).txt"
export EXPORT_BACKEND=vulkan

OUTPUT_DIR="${SCRIPT_DIR}/executorch_models"
WEIGHTS="${WEIGHTS:-${SCRIPT_DIR}/sharp_litert_models/sharp_2572gikvuh.pt}"
SHARP_SRC="${SHARP_SRC:-${SCRIPT_DIR}/third_party/ml-sharp/src}"
DTYPE="${DTYPE:-fp16}"
PATCH_BATCH="${PATCH_BATCH:-2}"

# So we can find the log after run (script stdout goes to log)
echo "$LOG_FILE" > "${SCRIPT_DIR}/export_log_latest_path.txt"
# Append all output to log file (no process substitution for sandbox compatibility)
exec >> "$LOG_FILE" 2>&1

echo "=============================================="
echo "SHARP ExecuTorch export — BACKEND: VULKAN"
echo "  Log file: ${LOG_FILE}"
echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="

echo ""
echo "=== Host / paths ==="
echo "Hostname: $(hostname 2>/dev/null || echo 'n/a')"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
echo "REPO_ROOT: ${REPO_ROOT}"
echo "PWD at start: $(pwd)"

echo ""
echo "=== Git (repo root) ==="
(cd "$REPO_ROOT" && git status && git branch -v && git log -1 --oneline 2>/dev/null) || true

echo ""
echo "=== Git submodules (repo root) ==="
(cd "$REPO_ROOT" && git submodule status 2>/dev/null) || echo "(no submodules or not a git repo)"
if [ -d "$REPO_ROOT/.git/modules" ]; then
  echo "Git modules dir contents:"
  ls -la "$REPO_ROOT/.git/modules" 2>/dev/null || true
fi

echo ""
echo "=== Nested repos (third_party) ==="
for d in "$SCRIPT_DIR/third_party"/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  if [ -d "$d/.git" ]; then
    echo "  $name: git repo"
    (cd "$d" && git status -sb && git log -1 --oneline 2>/dev/null) | sed 's/^/    /'
  else
    echo "  $name: not a git repo"
  fi
done

echo ""
echo "=== Export env (VULKAN) ==="
echo "EXPORT_BACKEND=${EXPORT_BACKEND:-vulkan}"
echo "SHARP_SRC=${SHARP_SRC}"
echo "WEIGHTS=${WEIGHTS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DTYPE=${DTYPE}"
echo "PATCH_BATCH=${PATCH_BATCH}"

echo ""
echo "=== SHARP_SRC contents (top-level) ==="
if [ -d "$SHARP_SRC" ]; then
  ls -la "$SHARP_SRC" 2>/dev/null || true
else
  echo "  (directory not found)"
fi

echo ""
echo "=== Weights file ==="
if [ -f "$WEIGHTS" ]; then
  ls -la "$WEIGHTS"
else
  echo "  (file not found)"
fi

echo ""
echo "=== Python ==="
python3 --version
echo "which python3: $(which python3)"
echo "sys.path (first 5):"
python3 -c "import sys; print('\n'.join(sys.path[:5]))" 2>/dev/null || true

echo ""
echo "=== Pip list (torch, executorch, sympy) ==="
pip3 list 2>/dev/null | grep -iE "torch|executorch|sympy" || pip3 list 2>/dev/null | head -30

echo ""
echo "=== ExecuTorch (if importable) ==="
python3 -c "
import sys
try:
  import torch; print('torch:', torch.__version__)
except Exception as e: print('torch:', e)
try:
  import executorch; print('executorch:', getattr(executorch, '__version__', 'no __version__'))
except Exception as e: print('executorch:', e)
" 2>/dev/null || true

echo ""
echo "=============================================="
echo "RUNNING EXPORT — BACKEND: VULKAN (--backend vulkan)"
echo "  dtype=${DTYPE}  patch_batch=${PATCH_BATCH}"
echo "  Chunked Part 4: part4a_chunk_512, part4a_chunk_65, part4b"
echo "=============================================="
echo ""
echo "=== Exact Python command (VULKAN) ==="
echo "python3 export_sharp_executorch_split4.py --backend vulkan --chunked-part4 --dtype ${DTYPE} --patch-batch-size ${PATCH_BATCH} --sharp-src ${SHARP_SRC} --weights ${WEIGHTS} --output-dir ${OUTPUT_DIR}"
echo ""

python3 "${SCRIPT_DIR}/export_sharp_executorch_split4.py" \
  --backend vulkan \
  --chunked-part4 \
  --dtype "${DTYPE}" \
  --patch-batch-size "${PATCH_BATCH}" \
  --sharp-src "${SHARP_SRC}" \
  --weights "${WEIGHTS}" \
  --output-dir "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Export finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC') — exit code ${EXIT_CODE}"
echo "Full log written to: ${LOG_FILE}"
echo "=============================================="

exit $EXIT_CODE
