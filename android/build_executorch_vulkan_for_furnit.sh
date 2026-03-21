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
# etVulkan flavor uses executorch_lib_etVulkan
FURNIT_LIB="${FURNIT_LIB:-${SCRIPT_DIR}/app/src/main/cpp/executorch_lib_etVulkan}"
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

# Build with Vulkan enabled (full GLSL shader set incl. view_convert_buffer_float_half for FP16); arm64-v8a only
export EXECUTORCH_BUILD_VULKAN=ON
export ANDROID_ABIS=arm64-v8a
# Enable event tracer and ETDump so the app can record Part4b Vulkan profiling (see android/docs/EXECUTORCH_VULKAN_PROFILING.md)
export EXECUTORCH_ENABLE_EVENT_TRACER=ON
export EXECUTORCH_BUILD_DEVTOOLS=ON

# Optional: FP16 shader flags (fixes Error 0x20 / view_convert_buffer_float_half). Pass to ExecuTorch cmake.
# If ExecuTorch's build_android_library.sh doesn't support these, build may fail — then try without.
EXTRA_VK_FLAGS="${EXTRA_VK_FLAGS:--DEXECUTORCH_VULKAN_FP16_ENABLED=ON -DEXECUTORCH_VULKAN_INCLUDE_ALL_SHADERS=ON}"
echo "Extra Vulkan CMake flags: ${EXTRA_VK_FLAGS}"

echo "Building ExecuTorch Android with EXECUTORCH_BUILD_VULKAN=ON..."
# Inject FP16 shader flags into ExecuTorch cmake (fixes view_convert_buffer_float_half / Error 0x20).
# If cmake rejects unknown options, set EXTRA_VK_FLAGS="" and re-run.
BUILD_SCRIPT="${EXECUTORCH_SOURCE_DIR}/scripts/build_android_library.sh"
PATCHED="${EXECUTORCH_SOURCE_DIR}/scripts/build_android_library.furnit_patched"
# Inject FP16 + event tracer/devtools into cmake line
PROFILING_FLAGS="-DEXECUTORCH_ENABLE_EVENT_TRACER=ON -DEXECUTORCH_BUILD_DEVTOOLS=ON"
sed "s|-DCMAKE_BUILD_TYPE=\"\${EXECUTORCH_CMAKE_BUILD_TYPE}\" |-DCMAKE_BUILD_TYPE=\"\${EXECUTORCH_CMAKE_BUILD_TYPE}\" ${EXTRA_VK_FLAGS} ${PROFILING_FLAGS} |" "${BUILD_SCRIPT}" > "${PATCHED}"
chmod +x "${PATCHED}"
if (cd "${EXECUTORCH_SOURCE_DIR}" && sh scripts/build_android_library.furnit_patched); then
  :
else
  echo "Build with extra flags failed. Retrying without (EXECUTORCH_BUILD_VULKAN=ON only)..."
  (cd "${EXECUTORCH_SOURCE_DIR}" && sh scripts/build_android_library.sh)
fi
rm -f "${PATCHED}"

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
echo "Done. Build Furnit etVulkan with the local lib (so Gradle does not overwrite it):"
echo "  cd ${SCRIPT_DIR}"
echo "  ./gradlew assembleEtVulkanDebug -PexecutorchUseLocalLib"
echo ""
echo "Then install: ./gradlew installEtVulkanDebug"
