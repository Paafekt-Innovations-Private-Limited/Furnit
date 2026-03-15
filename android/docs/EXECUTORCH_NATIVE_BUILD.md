# Building libsharp_executorch_full and libsharp_executorch_tiles

The C++ ExecuTorch INT8 pipeline (`libsharp_executorch_full.so`) and the 16-tile Part4b library (`libsharp_executorch_tiles.so`) are built by CMake when two conditions are met.

## 1. libexecutorch.so (auto-extracted)

**Done by Gradle.** The `extractExecutorchSoFromAar` task copies `libexecutorch.so` from the `org.pytorch:executorch-android` AAR into `app/src/main/cpp/executorch_lib/` before CMake runs. No manual step needed if the dependency resolves.

## 2. ExecuTorch C++ headers (required)

CMake expects ExecuTorch headers at **`android/third_party/executorch/`** (extension and runtime). If that directory is missing, the native ExecuTorch targets are skipped and the app falls back to the Kotlin pipeline.

**Option A – clone and copy (recommended):**

```bash
cd android
./scripts/setup_executorch_headers.sh
```

This clones the ExecuTorch repo (tag aligned with the AAR) and copies the minimal include tree into `third_party/executorch/`.

**Option B – manual:**

```bash
cd android
git clone --depth 1 --branch v1.1.0 https://github.com/pytorch/executorch.git third_party/executorch
```

Then build the app; CMake will find `third_party/executorch/extension/module/module.h` and build both native libs.

## Verify

After a successful build, the APK should contain:

- `libsharp_executorch_full.so`
- `libsharp_executorch_tiles.so`
- `libexecutorch.so` or `libexecutorch_jni.so`

Logcat will show:

```
Native full INT8 pipeline loaded (C++ ExecuTorch INT8 option in Settings)
```

when the library is present. If you see "library libsharp_executorch_full.so not found", the native build was skipped (missing headers or .so).
