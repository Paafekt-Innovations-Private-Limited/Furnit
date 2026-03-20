# ExecuTorch UnsatisfiedLinkError: initHybrid — JNI/version mismatch

## What the crash means

```text
java.lang.UnsatisfiedLinkError: No implementation found for
com.facebook.jni.HybridData org.pytorch.executorch.Module.initHybrid(java.lang.String, int, int)
```

This is **not** an export or model failure. The app dies **before** any model load: the Java `Module` class is from one ExecuTorch build, but the native `.so` that gets loaded does **not** export the matching JNI symbol `initHybrid(String, int, int)`.

So: **Java ↔ native ExecuTorch version/source mismatch** (or missing JNI lib), not a bad `.pte`.

---

## Where things come from in this project

| What | Source |
|------|--------|
| **Java/Kotlin bindings** (`org.pytorch.executorch.Module`) | Always from Maven: `implementation 'org.pytorch:executorch-android:1.1.0'` |
| **Native `.so` (default)** | Extracted by `extractExecutorchSoFromAar`: from **executorch-android-vulkan:1.1.0** AAR when using Vulkan, else from **executorch-android:1.1.0** |
| **Native `.so` (local build)** | When `-PexecutorchUseLocalLib`: **not** overwritten; uses whatever is in `executorch_lib_etVulkan/` or `executorch_lib_etCpu/` for the flavor you build (e.g. from `build_executorch_vulkan_for_furnit.sh`) |
| **What gets packaged as `libexecutorch_jni.so`** | The **AAR** supplies `libexecutorch_jni.so` (JNI bridge with `initHybrid`). We do **not** copy our runtime over it (that caused UnsatisfiedLinkError). Java/Kotlin explicitly loads it with `System.loadLibrary("executorch_jni")` before `Module.load()` (Part1OnlyTest, ExecutorchInt8Sharp). |

So we can mix:

- Java from **executorch-android:1.1.0**
- .so from **executorch-android-vulkan:1.1.0** (different AAR; might have different JNI)
- or .so from **local build** (different version/API than 1.1.0 AAR)

Any of those can cause `initHybrid` to be missing in the loaded lib.

---

## What to do: use one source for Java and .so

### Option A — Same AAR for both (no Vulkan, best for Part1-only test)

Use **only** `executorch-android:1.1.0` for both Java and native. Do **not** extract from the Vulkan AAR and do **not** use local libs.

1. Clean and remove local executorch libs so the extract task can replace them:
   ```bash
   cd android
   ./gradlew clean
   rm -f app/src/main/cpp/executorch_lib_etCpu/libexecutorch*.so app/src/main/cpp/executorch_lib_etCpu/libextension_*.so
   ```
2. Build **without** Vulkan and **without** local lib (so .so is taken from the same AAR as the Java): use the **etCpu** Gradle flavor:
   ```bash
   ./gradlew :app:extractExecutorchSoFromAar
   ./gradlew :app:assembleEtCpuDebug
   ```
   That way the packaged `libexecutorch_jni.so` is from **executorch-android:1.1.0**, matching the Java `Module` class. Part1-only test (portable `.pte`) should then run without `UnsatisfiedLinkError`.

3. Re-run Part1-only test in the app (Settings → Developer → Part1 only test → Run).

### Option B — Vulkan: same version, no local mix

If you need Vulkan:

- Use **executorch-android-vulkan:1.1.0** for **both** Java and .so: build the **etVulkan** flavor (`:app:assembleEtVulkanDebug`) so `implementation` and CMake’s `executorch_lib_etVulkan/` stay in sync. Avoid `executorchUseLocalLib` unless you know the local `.so` matches the AAR JNI. Then check that the Vulkan AAR’s `.so` actually contains the JNI (e.g. `initHybrid`). If the published Vulkan AAR omits JNI, Option A is the only way to get a working `Module.load()` until you have a single build that includes both Vulkan and JNI.

### Option C — Local ExecuTorch build

If you use `-PexecutorchUseLocalLib` and your own `.so` in `executorch_lib_etVulkan/` or `executorch_lib_etCpu/` (match the flavor you build):

- The **Java** side is still from Maven 1.1.0. The local `.so` must be from an ExecuTorch build that has the **same** JNI API as the 1.1.0 AAR (same `Module.initHybrid(String, int, int)` etc.). Otherwise you get `UnsatisfiedLinkError`. Prefer building the full Android AAR from that same ExecuTorch tree and using it for both Java and native, instead of mixing Maven Java + local .so.

---

## Checks you can run

### 1. See what native libs are in the APK

```bash
unzip -l app/build/outputs/apk/etCpu/debug/app-etCpu-arm64-v8a-debug.apk | grep -E "executorch|fbjni|\.so"
```

Confirm you have something like `lib/arm64-v8a/libexecutorch_jni.so` (and optionally `libexecutorch.so`). If `libfbjni.so` is required by ExecuTorch, it should come from the same AAR.

### 2. See if your packaged .so has the JNI symbol

On the **same** `.so` that is packed as `libexecutorch_jni.so` (e.g. from `app/src/main/cpp/executorch_lib_etCpu/libexecutorch.so` or from the AAR):

```bash
nm -D app/src/main/cpp/executorch_lib_etCpu/libexecutorch.so | grep initHybrid
# or after extracting from AAR:
unzip -p app/build/outputs/apk/etCpu/debug/app-etCpu-arm64-v8a-debug.apk "lib/arm64-v8a/libexecutorch_jni.so" > /tmp/executorch_jni.so
nm -D /tmp/executorch_jni.so | grep initHybrid
```

If `initHybrid` does not appear, that .so does not implement the JNI the Java `Module` expects → use Option A (same AAR) or fix the local build (Option C).

### 3. Gradle dependencies

- **ExecuTorch:** `etCpuImplementation 'org.pytorch:executorch-android:1.1.0'` and `etVulkanImplementation 'org.pytorch:executorch-android-vulkan:1.1.0'`
- **Extract configs:** `executorchVulkanExtract` / `executorchXnnpackExtract` resolve the AARs for Gradle.
- **How .so is included:** `extractExecutorchSoFromAar` copies into `executorch_lib_etVulkan/` and `executorch_lib_etCpu/` for CMake (per flavor). The AAR’s `libexecutorch_jni.so` is used as-is (we do not overwrite it). App code calls `System.loadLibrary("executorch_jni")` before `Module.load()`.

---

## Summary

- **Blocker:** JNI/API mismatch (Java vs packaged .so), not model export.
- **Safest fix for Part1-only test:** Option A — build **etCpu** (`:app:assembleEtCpuDebug`), no local lib, so Java and .so both come from `executorch-android:1.1.0`.
- Then re-run the Part1-only test; only after that passes should you worry about export or Vulkan.
