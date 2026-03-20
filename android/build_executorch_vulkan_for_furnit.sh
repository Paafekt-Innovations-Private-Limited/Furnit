#!/usr/bin/env bash
# Build ExecuTorch for Android with full Vulkan kernels and copy lib(s) into Furnit.
# Use this when the prebuilt executorch-android-vulkan AAR causes Part1 crash or ~16s/patch.
#
# Prerequisites:
#   - ANDROID_NDK  (e.g. export ANDROID_NDK=$ANDROID_HOME/ndk/25.2.9519653); NDK r25c recommended.
#   - ANDROID_SDK or ANDROID_HOME  (for ExecuTorch's AAR step; e.g. export ANDROID_SDK=$ANDROID_HOME).
#   - Vulkan SDK with glslc on PATH (e.g. source Vulkan-SDK/*/setup-env.sh).
#   - ExecuTorch source (clone below or set EXECUTORCH_SOURCE_DIR).
#
# Usage:
#   cd android
#   ./build_executorch_vulkan_for_furnit.sh
#
# Then build Furnit with the local lib:
#   ./gradlew assembleDebug -PexecutorchUseLocalLib

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FURNIT_LIB="${FURNIT_LIB:-${SCRIPT_DIR}/app/src/main/cpp/executorch_lib}"
EXECUTORCH_SOURCE_DIR="${EXECUTORCH_SOURCE_DIR:-}"
ANDROID_ABIS="${ANDROID_ABIS:-arm64-v8a}"

echo "Furnit executorch_lib dir: ${FURNIT_LIB}"

if [[ -z "${ANDROID_NDK}" ]]; then
  echo "Error: ANDROID_NDK is not set. Set it to your NDK root (e.g. NDK r25c)."
  exit 1
fi

if ! command -v glslc &>/dev/null; then
  echo "Error: glslc not found. Install Vulkan SDK and add it to PATH (e.g. source Vulkan-SDK/*/setup-env.sh)."
  exit 1
fi
echo "glslc: $(which glslc) ($(glslc --version 2>&1 | head -1))"

# Clone ExecuTorch if not provided
if [[ -z "${EXECUTORCH_SOURCE_DIR}" ]]; then
  EXECUTORCH_SOURCE_DIR="${SCRIPT_DIR}/../executorch"
  if [[ ! -d "${EXECUTORCH_SOURCE_DIR}/.git" ]]; then
    echo "Cloning ExecuTorch release/1.1 into ${EXECUTORCH_SOURCE_DIR}..."
    mkdir -p "$(dirname "${EXECUTORCH_SOURCE_DIR}")"
    git clone -b release/1.1 https://github.com/pytorch/executorch.git "${EXECUTORCH_SOURCE_DIR}"
    echo "Initializing submodules (json, gflags, flatbuffers, vulkan deps, etc.)..."
    (cd "${EXECUTORCH_SOURCE_DIR}" && git submodule update --init --recursive)
  else
    echo "Using existing ExecuTorch at ${EXECUTORCH_SOURCE_DIR}"
    if [[ ! -d "${EXECUTORCH_SOURCE_DIR}/third-party/json" ]] || [[ ! -f "${EXECUTORCH_SOURCE_DIR}/third-party/json/CMakeLists.txt" ]]; then
      echo "Initializing submodules..."
      (cd "${EXECUTORCH_SOURCE_DIR}" && git submodule update --init --recursive)
    fi
  fi
fi

if [[ ! -f "${EXECUTORCH_SOURCE_DIR}/scripts/build_android_library.sh" ]]; then
  echo "Error: ${EXECUTORCH_SOURCE_DIR}/scripts/build_android_library.sh not found."
  exit 1
fi

# Build with Vulkan enabled (full GLSL shader set); arm64-v8a only to save time
export EXECUTORCH_BUILD_VULKAN=ON
export ANDROID_ABIS=arm64-v8a

echo "Building ExecuTorch Android with EXECUTORCH_BUILD_VULKAN=ON..."
(cd "${EXECUTORCH_SOURCE_DIR}" && sh scripts/build_android_library.sh)

# Copy built lib(s) into Furnit
BUILT_SO="${EXECUTORCH_SOURCE_DIR}/cmake-out-android-so/arm64-v8a/libexecutorch.so"
CMAKE_OUT="${EXECUTORCH_SOURCE_DIR}/cmake-out-android-arm64-v8a"

if [[ ! -f "${BUILT_SO}" ]]; then
  echo "Error: Build did not produce ${BUILT_SO}"
  exit 1
fi

mkdir -p "${FURNIT_LIB}"
cp -f "${BUILT_SO}" "${FURNIT_LIB}/libexecutorch.so"
echo "Copied libexecutorch.so to ${FURNIT_LIB}/"

# If split build produced libexecutorch_core.so, copy it too (Furnit CMake links both when present)
for core in "${CMAKE_OUT}/extension/android/libexecutorch_core.so" \
            "${CMAKE_OUT}/lib/libexecutorch_core.so" \
            "${EXECUTORCH_SOURCE_DIR}/cmake-out-android-so/arm64-v8a/libexecutorch_core.so"; do
  if [[ -f "${core}" ]]; then
    cp -f "${core}" "${FURNIT_LIB}/libexecutorch_core.so"
    echo "Copied libexecutorch_core.so to ${FURNIT_LIB}/"
    break
  fi
done

echo ""
echo "Done. Build Furnit with the local ExecuTorch lib (so Gradle does not overwrite it):"
echo "  cd ${SCRIPT_DIR}"
echo "  ./gradlew assembleDebug -PexecutorchUseLocalLib"
echo ""
echo "Then install and test: ./gradlew installDebug"
