#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPTH_PRO_DIR="${REPO_ROOT}/android/third_party/ml-depth-pro"
CHECKPOINT_DIR="${DEPTH_PRO_DIR}/checkpoints"

if [[ ! -d "${DEPTH_PRO_DIR}" ]]; then
  echo "DEPTH_PRO REPO NOT FOUND AT ${DEPTH_PRO_DIR}" >&2
  echo "CLONE IT FIRST: git clone https://github.com/apple/ml-depth-pro ${DEPTH_PRO_DIR}" >&2
  exit 1
fi

mkdir -p "${CHECKPOINT_DIR}"
cd "${DEPTH_PRO_DIR}"

echo "DEPTH_PRO CHECKPOINT DOWNLOAD START"
echo "DEPTH_PRO CHECKPOINT DIR ${CHECKPOINT_DIR}"
bash ./get_pretrained_models.sh
echo "DEPTH_PRO CHECKPOINT DOWNLOAD COMPLETE"
