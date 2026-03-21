# Bundling SHARP / ExecuTorch `.pte` in the test APK

For **friend testing** (sideload), Gradle can copy all `*.pte` from `android/sharp_vulkan_only/` into `app/src/main/assets/models/` before packaging. The app then extracts any `sharp_split*.pte` from assets on first run (`ExecutorchInt8Sharp.ensureModelsFromAssets()`).

## Backup first

Large trees should be backed up off-repo before you rely on Gradle copies or clean builds.

Example (LaCie):

` /Volumes/LaCie/BackupMar22nd2026/Furnit_Android_models_Mar22_2026/`

Contains:

- `sharp_vulkan_only/` — full Vulkan export tree  
- `app_src_main_assets/` — prior `app/src/main/assets/` snapshot  

## APK size

A full `sharp_vulkan_only` bundle is on the order of **~5+ GB** of `.pte` alone, plus the app (native libs, ONNX, etc.). Expect a **multi‑GB** arm64 debug/release APK. This is **not** suitable for Play Store as a single monolithic APK; it is intended for **direct sharing / sideload testing** only.

## Gradle properties

| Property | Effect |
|----------|--------|
| `-PskipExecutorchAssets` | Disables **all** model copy into assets (smallest APK; use adb push to `files/models_vulkan` or `models_cpu`). |
| `-PskipSharpVulkanOnlyInAssets` | Still copies from `../executorch_int8_models` and `../executorch_models` if present, but **omits** `android/sharp_vulkan_only` (faster dev builds). |
| `-PincludePart4bTilesInAssets` | Also copies 16 tile `.pte` from the int8/chunked dirs when those dirs exist (redundant if tiles already live under `sharp_vulkan_only`). |

Default when `sharp_vulkan_only` exists: **Vulkan `.pte` are included** unless you pass `-PskipSharpVulkanOnlyInAssets` or `-PskipExecutorchAssets`.

## Git

`*.pte` is listed in `.gitignore`; copied files under `assets/models/` are not meant to be committed.

## Prod

Use **`-PskipExecutorchAssets`** for production builds (or omit `sharp_vulkan_only` on the build machine) and deliver models via **adb**, OTA, or Play Asset Delivery as appropriate.
