#!/usr/bin/env bash
# Clone ExecuTorch (tag aligned with executorch-android AAR 1.1.0) and copy minimal
# headers into android/third_party/executorch so CMake can build libsharp_executorch_full
# and libsharp_executorch_tiles.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANDROID_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY="${ANDROID_DIR}/third_party"
EXECUTORCH_DEST="${THIRD_PARTY}/executorch"
# Tag matching org.pytorch:executorch-android:1.1.0
EXECUTORCH_TAG="v1.1.0"
CLONE_DIR="${THIRD_PARTY}/executorch_repo"

if [[ -d "${EXECUTORCH_DEST}" && -f "${EXECUTORCH_DEST}/extension/module/module.h" ]]; then
  echo "Headers already present at ${EXECUTORCH_DEST}"
  exit 0
fi

mkdir -p "${THIRD_PARTY}"
if [[ ! -d "${CLONE_DIR}" ]]; then
  echo "Cloning ExecuTorch ${EXECUTORCH_TAG} into ${CLONE_DIR} ..."
  git clone --depth 1 --branch "${EXECUTORCH_TAG}" https://github.com/pytorch/executorch.git "${CLONE_DIR}"
fi

echo "Copying minimal include tree to ${EXECUTORCH_DEST} ..."
mkdir -p "${EXECUTORCH_DEST}"
cp -R "${CLONE_DIR}/extension" "${EXECUTORCH_DEST}/"
mkdir -p "${EXECUTORCH_DEST}/runtime"
cp -R "${CLONE_DIR}/runtime/core" "${EXECUTORCH_DEST}/runtime/"
cp -R "${CLONE_DIR}/runtime/executor" "${EXECUTORCH_DEST}/runtime/"
cp -R "${CLONE_DIR}/runtime/platform" "${EXECUTORCH_DEST}/runtime/"

if [[ ! -f "${EXECUTORCH_DEST}/extension/module/module.h" ]]; then
  echo "ERROR: module.h not found after copy"
  exit 1
fi
echo "Done. Rebuild the app to build libsharp_executorch_full and libsharp_executorch_tiles."
