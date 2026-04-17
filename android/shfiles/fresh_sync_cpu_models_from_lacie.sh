#!/bin/bash
# One-shot: clear device models_cpu, push v2 six-file set + any optional single Part4b in the same folder
# (sharp_split_part4b_int8.pte / fp16 / .pte) for Stable Part4b (single) — clearer than tile_b4 alone.
# Local staging folder: **android/models_cpu/** — see models_cpu/README.md
# Larger bundle: ./copy_from_lacie_and_push_cpu_models.sh /path
#
# Usage:
#   ./fresh_sync_cpu_models_from_lacie.sh
#   ./fresh_sync_cpu_models_from_lacie.sh /Volumes/LaCie/march10th2026/v2
#   ./fresh_sync_cpu_models_from_lacie.sh "$(pwd)/models_cpu"
#   SKIP_CLEAR=1 ./fresh_sync_cpu_models_from_lacie.sh
set -euo pipefail

cd "$(dirname "$0")"
SRC="${1:-/Volumes/LaCie/march10th2026/v2}"

./deploy_sharp_v2_to_models_cpu.sh "$SRC"
echo "fresh_sync: finished."
