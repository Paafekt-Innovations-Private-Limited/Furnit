# Build ExecuTorch Android with Full Vulkan Kernels

When the prebuilt **executorch-android-vulkan** AAR causes slow Part1 (~16s per patch) and/or a crash during or after Part1 forward, the AAR’s limited shader set or driver behavior is the cause. Building ExecuTorch from source with **full Vulkan kernels** (all GLSL shaders compiled in) is the fix.

## Quick fix: use the Furnit script

From the **android** directory:

```bash
export ANDROID_NDK=/path/to/ndk   # e.g. $ANDROID_HOME/ndk/25.2.9519653; NDK r25c recommended
export ANDROID_SDK=$ANDROID_HOME  # or ANDROID_HOME (for ExecuTorch AAR step)
source /path/to/Vulkan-SDK/*/setup-env.sh   # so glslc is on PATH

cd /path/to/Furnit/android
chmod +x build_executorch_vulkan_for_furnit.sh
./build_executorch_vulkan_for_furnit.sh
```

The script clones ExecuTorch (if needed), builds with `EXECUTORCH_BUILD_VULKAN=ON`, and copies `libexecutorch.so` (and `libexecutorch_core.so` if present) into `app/src/main/cpp/executorch_lib/`. Then build Furnit:

```bash
./gradlew assembleDebug -PexecutorchUseLocalLib
./gradlew installDebug
```

To use an existing ExecuTorch checkout: `EXECUTORCH_SOURCE_DIR=/path/to/executorch ./build_executorch_vulkan_for_furnit.sh`.

## Prerequisites

- **Android SDK** and **Android NDK**  
  Set `ANDROID_HOME` (SDK) and `ANDROID_NDK` (NDK root containing `build/cmake/android.toolchain.cmake`).  
  Prefer **NDK r25c**; NDK r28c has known issues with some Vulkan GLSL kernels (`GL_EXT_integer_dot_product` etc.).
- **Vulkan SDK** (for host)  
  The Vulkan backend compiles GLSL shaders at build time. Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#android) and ensure `glslc` is on your `PATH`:
  ```bash
  glslc --version
  ```
  If needed, run `source setup-env.sh` in your Vulkan SDK directory.
- **Python 3.10–3.13** and **conda** or **venv** (for ExecuTorch Python/export if you re-export .pte).

## Manual steps (if not using the script)

### 1. Clone and prepare ExecuTorch

```bash
git clone -b release/1.1 https://github.com/pytorch/executorch.git
cd executorch
# Optional: conda create -yn executorch python=3.10 && conda activate executorch
```

### 2. Build Android library with Vulkan enabled

From the ExecuTorch repo root:

```bash
export ANDROID_HOME=/path/to/android/sdk
export ANDROID_NDK=/path/to/android/ndk
export EXECUTORCH_BUILD_VULKAN=ON
# Optional: build only arm64-v8a
export ANDROID_ABIS=arm64-v8a

sh scripts/build_android_library.sh
```

This runs CMake with `-DEXECUTORCH_BUILD_VULKAN=ON`, compiling the full Vulkan backend (including the in-tree GLSL shader library). The script copies the built `.so` to `cmake-out-android-so/arm64-v8a/libexecutorch.so` (and builds an AAR under `extension/android/`). If the AAR step fails, set `ANDROID_SDK` or `ANDROID_HOME` to your Android SDK path.

### 3. Copy the built lib into Furnit

Replace the app’s ExecuTorch lib with the one you just built (so it includes full Vulkan kernels):

```bash
EXECUTORCH_BUILD=./cmake-out-android-so/arm64-v8a
FURNIT_LIB=/path/to/Furnit/android/app/src/main/cpp/executorch_lib

cp -f "$EXECUTORCH_BUILD/libexecutorch.so" "$FURNIT_LIB/libexecutorch.so"
```

If your ExecuTorch build produced a split (e.g. `libexecutorch_core.so`), copy that too; Furnit’s CMake will link both when present (see [EXECUTORCH_LINK_FIX.md](EXECUTORCH_LINK_FIX.md)).

### 4. Build Furnit without overwriting your lib

So that the Gradle extract task does not overwrite `executorch_lib/libexecutorch.so` with the Maven AAR:

```bash
cd /path/to/Furnit/android
./gradlew assembleDebug -PexecutorchUseLocalLib
```

`-PexecutorchUseLocalLib` makes `extractExecutorchSoFromAar` skip extraction when `executorch_lib/libexecutorch.so` already exists.

### 5. Install and test

```bash
./gradlew installDebug
```

Run a capture with **ExecuTorch INT8 (Vulkan)**. Part1 should be much faster and the pipeline should proceed without the previous crash (assuming the crash was due to missing/limited shaders in the prebuilt AAR).

## Shader registration

No extra “shader pre-registration” is needed in the app. The Vulkan backend uses static initialization to register the backend and its shaders when the library is loaded. Linking the ExecuTorch-built `libexecutorch.so` (with Vulkan compiled in) is enough; the app’s JNI layer does not need to call any Vulkan registration API.

## References

- [Vulkan Backend — ExecuTorch 1.0](https://docs.pytorch.org/executorch/1.0/android-vulkan.html)
- [Using ExecuTorch on Android](https://docs.pytorch.org/executorch/stable/using-executorch-android.html) — “Using Vulkan Backend” and `EXECUTORCH_BUILD_VULKAN=ON`
- [Building from Source — ExecuTorch 1.1](https://docs.pytorch.org/executorch/stable/using-executorch-building-from-source.html)
- [EXECUTORCH_VULKAN_REGISTRATION.md](EXECUTORCH_VULKAN_REGISTRATION.md) — registration, missing shader crash, and AAR-compat export
