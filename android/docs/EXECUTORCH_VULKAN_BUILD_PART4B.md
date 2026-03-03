# Building ExecuTorch Vulkan with Part4b FP16 shader (view_convert_buffer_float_half)

The prebuilt `org.pytorch:executorch-android-vulkan:1.1.0` AAR does not include the shader `view_convert_buffer_float_half`, so Part4b Vulkan fails at runtime. This doc gives links and steps to build a Vulkan AAR that includes it.

---

## Links you need

| What | Link |
|------|------|
| **Vulkan SDK** (for `glslc`) | https://vulkan.lunarg.com/sdk/home — use 1.4.321.0 or newer. |
| **ExecuTorch repo** | https://github.com/pytorch/executorch |
| **Android: use on Android** | https://pytorch.org/executorch/stable/using-executorch-android.html |
| **Building from source** | https://pytorch.org/executorch/stable/using-executorch-building-from-source.html (optional reference) |
| **Vulkan backend overview** | https://pytorch.org/executorch/stable/backends/vulkan/vulkan-overview.html |

---

## What’s already done in Furnit

In **this repo**, the file  
`android/third_party/executorch/backends/vulkan/runtime/graph/ops/glsl/view_convert_buffer.yaml`  
was updated to add:

- `[float, half]` → shader `view_convert_buffer_float_half`
- `[half, float]` → shader `view_convert_buffer_half_float`

So the **source** for the missing shader is fixed in Furnit’s `third_party/executorch`. You only need to build an Android Vulkan AAR from a tree that has this change (and has `glslc` so the build can generate SPIR-V).

---

## Option A: Build in upstream ExecuTorch (recommended)

Do this on a machine where you can install the Vulkan SDK and Android NDK.

### 1. Install Vulkan SDK

- **macOS:** https://vulkan.lunarg.com/sdk/home#mac — download and install, or e.g. `brew install vulkan-sdk`.  
  Then ensure `glslc` is on PATH (often via the SDK’s “Setup Environment” script or by adding its `bin` to PATH).
- **Linux:** same SDK page; or use your distro’s Vulkan packages if they provide `glslc` 1.4+.

Check:

```bash
glslc --version
```

### 2. Clone ExecuTorch and apply the YAML fix

```bash
git clone -b release/1.1 https://github.com/pytorch/executorch.git
cd executorch
```

Apply the same YAML change as in Furnit. Edit  
`backends/vulkan/runtime/graph/ops/glsl/view_convert_buffer.yaml`  
and add these two lines to the `combos` list (same indentation as the existing `parameter_values` entries):

```yaml
        - parameter_values: [float, half]
        - parameter_values: [half, float]
```

So the `combos` section includes at least:

- existing entries: `[int32, float]`, `[int32, half]`, `[uint8, float]`, `[uint8, half]`, `[uint8, int32]`, `[float, int32]`
- plus: `[float, half]`, `[half, float]`

### 3. Set Android SDK/NDK and build Vulkan AAR

ExecuTorch’s Android doc: https://pytorch.org/executorch/stable/using-executorch-android.html  

From the **ExecuTorch repo root**:

```bash
export ANDROID_HOME=/path/to/your/Android/sdk
export ANDROID_NDK=/path/to/your/Android/sdk/ndk/<version>
export EXECUTORCH_BUILD_VULKAN=ON
sh scripts/build_android_library.sh
```

Use your real paths. NDK version used in their CI is r28c; similar versions should work. The script builds the native lib (including Vulkan and the new shaders, because CMake runs `gen_vulkan_spv.py` when `EXECUTORCH_BUILD_VULKAN=ON`) and packages an AAR.

### 4. Use the built AAR in Furnit

- Locate the built AAR (script output or under the executorch build dir; see script or `build`-style dirs).
- In Furnit’s `android/app/build.gradle`, replace the Vulkan dependency with the file, e.g.:

  ```gradle
  implementation(files("libs/executorch-vulkan.aar"))   // or path you copied to
  ```

  and add if not already present:

  ```gradle
  implementation("com.facebook.soloader:soloader:0.10.5")
  implementation("com.facebook.fbjni:fbjni:0.7.0")
  ```

- Copy the AAR into `android/app/libs/` (or the path you used in `files(...)`).
- Rebuild the Furnit app.

### 5. Export Part4b for Vulkan and push

From Furnit repo:

```bash
cd android
PYTHONPATH=third_party/executorch:$PYTHONPATH python export_sharp_executorch_int8_split4.py --chunked-part4 --part4b-fp16 --part4b-backend vulkan
./push_sharp_executorch_int8_models.sh executorch_int8_models executorch_int8_models
```

If the app still had the old Part4b on internal storage, remove it so the new one is used:

```bash
adb shell run-as com.furnit.android rm -f files/models/sharp_split_part4b_fp16.pte
```

Then run SHARP again; Part4b should run on Vulkan without the “view_convert_buffer_float_half” error.

---

## Option B: Regenerate SPIR-V only (Furnit tree), then build elsewhere

If you prefer to regenerate shaders once inside Furnit (same YAML is already there) and then build ExecuTorch Android elsewhere:

### 1. Install Vulkan SDK

Same as Option A step 1; ensure `glslc` is on PATH.

### 2. Run the shader generator in Furnit

From **Furnit repo root**:

```bash
EXECUTORCH_ROOT="$(pwd)/android/third_party/executorch"
GLSL_PATH="${EXECUTORCH_ROOT}/backends/vulkan/runtime/graph/ops/glsl"
OUT_PATH="${EXECUTORCH_ROOT}/backends/vulkan/runtime/graph/ops/glsl/generated"
TMP_PATH="/tmp/et_vulkan_shaders"
mkdir -p "$OUT_PATH" "$TMP_PATH"

python3 "${EXECUTORCH_ROOT}/backends/vulkan/runtime/gen_vulkan_spv.py" \
  -i "$GLSL_PATH" \
  -c "$(which glslc)" \
  -t "$TMP_PATH" \
  -o "$OUT_PATH" \
  --optimize
```

This writes `spv.cpp` / `spv.h` in `OUT_PATH` including `view_convert_buffer_float_half` and `view_convert_buffer_half_float`. You would then need to point an ExecuTorch Android/Vulkan build at this `third_party/executorch` tree (or copy the generated files into an executorch clone and build there). Option A is usually simpler.

---

## Summary

- **Links:** Vulkan SDK (LunarG), ExecuTorch repo, Android + “Building from Source” docs above.
- **Fix in Furnit:** `view_convert_buffer.yaml` already has `[float, half]` and `[half, float]`.
- **You do:** Install Vulkan SDK → clone ExecuTorch → apply same YAML change → set `EXECUTORCH_BUILD_VULKAN=ON` → run `scripts/build_android_library.sh` → use the resulting AAR in Furnit → re-export Part4b Vulkan and push.
