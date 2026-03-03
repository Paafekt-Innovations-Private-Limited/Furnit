#!/bin/bash
# Build ExecuTorch Android AAR with Vulkan (including view_convert_buffer_float_half for Part4b).
# Run this from the directory that contains your executorch clone (sibling to Furnit).
#
# Prereqs (you already did: clone executorch, brew install vulkan-volk):
#   1. glslc (Vulkan shader compiler). If missing:  brew install shaderc
#   2. Android NDK (e.g. via Android Studio SDK Manager). Set ANDROID_NDK.
#   3. Android SDK. Set ANDROID_SDK or ANDROID_HOME.
#
# Usage:
#   export ANDROID_NDK=/path/to/ndk
#   export ANDROID_SDK=/path/to/sdk
#   ./scripts/build_executorch_vulkan_aar.sh
#
# The script expects executorch cloned at ../executorch (sibling of Furnit) or
# pass EXECUTORCH_DIR=/path/to/executorch.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FURNIT_ANDROID="$(cd "$SCRIPT_DIR/.." && pwd)"
# Default: executorch sibling of Furnit (e.g. tries01/executorch)
EXECUTORCH_DIR="${EXECUTORCH_DIR:-$(cd "$FURNIT_ANDROID/../.." && pwd)/executorch}"

if [ ! -d "$EXECUTORCH_DIR" ] || [ ! -f "$EXECUTORCH_DIR/scripts/build_android_library.sh" ]; then
  echo "Error: ExecuTorch not found at $EXECUTORCH_DIR"
  echo "Clone: git clone -b release/1.1 https://github.com/pytorch/executorch.git"
  echo "Or set EXECUTORCH_DIR=/path/to/executorch"
  exit 1
fi

if ! command -v glslc &>/dev/null; then
  echo "Error: glslc not found. Install Vulkan shader compiler:"
  echo "  brew install shaderc"
  exit 1
fi

# Default NDK to Furnit's version if unset or path missing
if [ -z "$ANDROID_NDK" ] || [ ! -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]; then
  DEFAULT_NDK="$HOME/Library/Android/sdk/ndk/26.3.11579264"
  if [ -f "$DEFAULT_NDK/build/cmake/android.toolchain.cmake" ]; then
    export ANDROID_NDK="$DEFAULT_NDK"
    echo "Using ANDROID_NDK=$ANDROID_NDK"
  elif [ -n "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK is set but toolchain not found: $ANDROID_NDK/build/cmake/android.toolchain.cmake"
    echo "Installed NDKs: ls \$HOME/Library/Android/sdk/ndk/"
    exit 1
  else
    echo "Error: ANDROID_NDK not set and default not found. Example:"
    echo "  export ANDROID_NDK=\$HOME/Library/Android/sdk/ndk/26.3.11579264"
    exit 1
  fi
fi

if [ -z "$ANDROID_SDK" ] && [ -z "$ANDROID_HOME" ]; then
  echo "Error: Set ANDROID_SDK or ANDROID_HOME. Example:"
  echo "  export ANDROID_SDK=\$HOME/Library/Android/sdk"
  exit 1
fi

export ANDROID_SDK="${ANDROID_SDK:-$ANDROID_HOME}"
export EXECUTORCH_BUILD_VULKAN=ON
# Build only arm64-v8a for phones (faster)
export ANDROID_ABIS="arm64-v8a"

echo "Building ExecuTorch Android Vulkan AAR..."
echo "  EXECUTORCH_DIR=$EXECUTORCH_DIR"
echo "  ANDROID_NDK=$ANDROID_NDK"
echo "  ANDROID_SDK=$ANDROID_SDK"
echo "  EXECUTORCH_BUILD_VULKAN=$EXECUTORCH_BUILD_VULKAN"
echo ""

mkdir -p "$FURNIT_ANDROID/app/libs"
export BUILD_AAR_DIR="$FURNIT_ANDROID/app/libs"
cd "$EXECUTORCH_DIR"
# Ensure third-party deps (json, gflags, flatbuffers, vulkan, xnnpack, etc.) are present
if [ ! -f "$EXECUTORCH_DIR/third-party/json/CMakeLists.txt" ]; then
  echo "Initializing git submodules (json, gflags, flatbuffers, vulkan, xnnpack...)..."
  git submodule update --init --recursive
fi
sh scripts/build_android_library.sh

echo ""
echo "Done. AAR should be at: $FURNIT_ANDROID/app/libs/executorch.aar"
echo "Next: in Furnit android/app/build.gradle use: implementation(files(\"libs/executorch.aar\")) and add soloader/fbjni if needed."
echo "Then re-export Part4b Vulkan and push models."
