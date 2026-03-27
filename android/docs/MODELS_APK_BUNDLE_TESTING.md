# Bundling SHARP / ExecuTorch `.pte` in the test APK

For **friend testing** (sideload), Gradle copies `*.pte` from `android/sharp_vulkan_only/` into **`app/build/bundled-pte-assets/models_cpuvulkan_hybrid/`**, and CPU splits from `executorch_int8_models` / `executorch_models` into **`models_cpu/`**. That tree is merged as an extra asset source — **not** under `src/main/`, so repo `.gitignore` rules for `*.pte` cannot strip them from the APK. The app copies bundled `sharp_split*.pte` from `assets/models_cpu` and `assets/models_cpuvulkan_hybrid` into internal storage during **`ExecutorchInt8Sharp.initialize()`** and **`hydrateBundledAndExternalModels()`** (also when opening the SHARP flow). **Default dev workflow:** `skipExecutorchAssets=true` in `gradle.properties` (fast Android Studio Run; use **adb push** for models). **Friend APK with models inside:** run **`android/assemble_friend_apk_with_models.sh`** or `./gradlew :app:assembleEtVulkanDebug -PskipExecutorchAssets=false`. On first launch, if internal **`models_cpu`** / **`models_cpuvulkan_hybrid`** have no `sharp_split*.pte`, the app copies from bundled assets (see `FurnitApplication` / `hydrateBundledAndExternalModels`).

## Backup first

Large trees should be backed up off-repo before you rely on Gradle copies or clean builds.

Example (LaCie):

` /Volumes/LaCie/BackupMar22nd2026/Furnit_Android_models_Mar22_2026/`

Contains:

- `sharp_vulkan_only/` — full Vulkan export tree  
- `app_src_main_assets/` — prior `app/src/main/assets/` snapshot  

## APK size and the **~4 GiB Zip32 limit**

A full `sharp_vulkan_only` tree is often **~6–7+ GB** of `.pte` alone. **Installable APKs are Zip32 archives**: offsets in the central directory are **32‑bit**, so the **entire APK must stay under ~4 GiB** (you may see `Zip32 cannot place CD entry … MAX=4294967295`). **Zip64 APKs are not a fix** — the Android platform does not treat them as normal installable APKs. That is a **platform limit**, not a mistake: your disk folder is supposed to hold many variants (FP16/FP32, tiles, chunk sizes); the **APK must only carry the subset the app actually loads**.

**Gradle default (when `skipExecutorchAssets=false`):** copy a **minimal Vulkan file list** from `sharp_vulkan_only`, not the whole folder — so Android Studio builds stay under Zip32 unless you opt in with `-PincludeAllSharpVulkanPteInAssets=true`.

So you **cannot** ship “copy every `*.pte`” + `executorch_models` duplicates in one sideload APK. For friend testing you must either:

- **Bundle a minimal Vulkan set** (Gradle flags below), **or**
- **`-PskipExecutorchAssets=true`** and **`adb push`** models to `files/models_cpuvulkan_hybrid/` (or use Play Asset Delivery, etc.).

This is **not** suitable for Play Store as a single monolithic multi‑GB APK; it is for **curated test builds or external delivery** only.

### ~2 GiB sideload / “invalid package” (not a corrupt APK)

After bundling **fp32 Vulkan Part1–3** plus Part4a/4b, the debug APK is often **~3–3.7 GiB** (still under the Zip32 **~4 GiB** archive limit, so Gradle can build it). Many **phone installers** still fail around **~2 GiB** because parts of the stack treat sizes as **signed 32‑bit** (`Integer.MAX_VALUE`). Symptoms: *There was a problem with this app*, *package appears to be invalid*, etc.

**Mitigations**

- Install with **`adb install -r`** from a PC/Mac.
- Prefer **`assemble_friend_apk_without_models.sh`** (alias: `assemble_friend_apk_shell_only.sh`) + **`push_sharp_cpuvulkan_hybrid_androidstudio.sh`** so the installable APK stays small.
- If you must stay under ~2 GiB in one APK, you need **smaller exports** (e.g. **FP16** Part1–3 if your device supports them) and/or fewer bundled modules — there is no Gradle flag to bypass the platform installer limit.

## Gradle properties

| Property | Effect |
|----------|--------|
| `-PskipExecutorchAssets` | Disables **all** model copy into assets (smallest APK; use adb push to `files/models_cpuvulkan_hybrid` or `models_cpu`). |
| `-PskipSharpVulkanOnlyInAssets` | Still copies from `../executorch_int8_models` and `../executorch_models` if present, but **omits** `android/sharp_vulkan_only` (faster dev builds). |
| `-PincludePart4bTilesInAssets` | Also copies 16 tile `.pte` from the int8/chunked dirs when those dirs exist (redundant if tiles already live under `sharp_vulkan_only`). |
| `-PskipExecutorchChunkedDirInAssets=true` | Skips **`../executorch_models`** (CPU-named `part4a_chunk_*.pte`, `sharp_split_part4b.pte`, etc.). Use when **etVulkan** loads only Vulkan `.pte` from `sharp_vulkan_only`. |
| `-PskipExecutorchInt8DirInAssets=true` | Skips **`../executorch_int8_models`** (CPU INT8 Part1–3). Not needed for **full Vulkan hybrid** (`sharp_executorch_full_vulkan.cpp` uses `*_vulkan_*.pte` for Part1–3). |
| *(default when bundling)* | If `skipExecutorchAssets=false` and `sharp_vulkan_only` exists, **minimal Vulkan list** is used automatically (unless `includeAll…` below). |
| `-PincludeAllSharpVulkanPteInAssets=true` | **Expert / avoid:** copy **every** `*.pte` from `sharp_vulkan_only` — usually **>4 GiB** and **`package*` fails**. |
| `-PbundleSharpVulkanHybridApk=true` | Friend APK: skips int8 + `executorch_models`; bundles **core Vulkan Part1–3 + Part4a** plus **Part4b splits / tiles** (single `part4b_vulkan`, portable fallbacks, split `tile_b2`, fine-split / split `tile_00`, legacy `tile_b2` / `tile_b4` / `tile_00` / `tile_full`). Gradle **only copies files that exist**. Build **fails** if **no** complete Part4b strategy is present after copy. |
| `-PbundleSharpVulkanFriendIncludeLegacyTileGrid=true` | Also try to bundle **`sharp_split_part4b_tile_01.pte` … `tile_15.pte`** (large; can push **Zip32 ~4 GiB** APK over the limit if many exist). Default **false**. |
| `-PbundleSharpVulkanHybridFriendBundleAllPart4bSlices=true` | If `sharp_split_part4b_vulkan.pte` **exists**, still bundle **all** extra Part4b tile/split names below (for debugging or forced tiled routing). Default **false** — when the single Vulkan decoder is present, extras are **skipped** so the friend APK does not balloon past Zip32. |

**Hybrid friend Part4b logic (important):** If **`sharp_split_part4b_vulkan.pte`** is in `sharp_vulkan_only/`, the friend build bundles **only** the core 6–7 files (Part1–3 + Part4a + that decoder). If you rely on **split/tiled** Part4b **and** that file is missing from the folder, Gradle adds the split/tile filename list (except legacy `tile_01…15` unless `-PbundleSharpVulkanFriendIncludeLegacyTileGrid=true`). If you **need** both single + many slices in one APK, use `-PbundleSharpVulkanHybridFriendBundleAllPart4bSlices=true` and accept Zip32 / size risk or use **adb push**.
| `-PbundleSharpVulkanOnlyMinimal=true` | Same minimal Vulkan list as above, but **does not** skip int8/chunked dirs unless you also set skip flags. |
| `-PbundleSharpVulkanMinimalIncludePart4Monolith=true` | Adds **`sharp_split_part4.pte`** (~755 MB) to the minimal set — only if you still need the old monolithic Part4 export; normal Vulkan hybrid does **not**. |
| `-PbundleSharpVulkanPrecision=fp16` or `fp32` | Used with **minimal** / **hybrid** mode; default **`fp16`** (smaller). If `sharp_split_part1_vulkan_fp16.pte` is **missing** but **fp32** exists, Gradle **auto-picks fp32** for Part1–3 so friend APKs are not empty for those layers. Native (etVulkan) resolves **fp32 before fp16** at runtime. |
| `-PbundleSharpVulkanIncludeChunk65=true` / `false` | Include `sharp_split_part4a_chunk_65_vulkan.pte` (~+600 MB). **Default `true`** — [ExecutorchInt8Sharp] requires `hasPart4a65()` for the Vulkan full pipeline; omitting it causes *Missing models: part4a_65_vulkan*. Set **`false`** only for custom builds that change Kotlin/C++. |

**Example (Vulkan hybrid friend APK — no CPU Part3/4 `.pte`, fits Zip32):**

```bash
./gradlew :app:assembleEtVulkanDebug \
  -PskipExecutorchAssets=false \
  -PbundleSharpVulkanHybridApk=true \
  -PbundleSharpVulkanPrecision=fp32
```

Same as minimal + skip CPU dirs, if you prefer explicit flags:

```bash
./gradlew :app:assembleEtVulkanDebug \
  -PskipExecutorchAssets=false \
  -PbundleSharpVulkanOnlyMinimal=true \
  -PskipExecutorchChunkedDirInAssets=true \
  -PskipExecutorchInt8DirInAssets=true \
  -PbundleSharpVulkanPrecision=fp16
```

### Android Studio

1. Open **`android/gradle.properties`**.
2. Set **`skipExecutorchAssets=false`** → **Sync Now**.
3. **Build → Build Bundle(s) / APK(s) → Build APK(s)** (e.g. `etVulkanDebug`).  
   You should see a log line: *using minimal sharp_vulkan_only list*.
4. **Optional — smallest APK for friends** (Vulkan weights only, no CPU duplicate `.pte`): **File → Settings → Build, Execution, Deployment → Gradle** → **Gradle projects** → add to **Command-line options**:  
   `-PbundleSharpVulkanHybridApk=true`  
   (or run **`android/assemble_friend_apk_with_models.sh`** from a terminal).  
5. **Android Studio** still installs from **`android/app/build/outputs/apk/<flavor>/<buildType>/`** (unchanged).
6. **Friend sideload:** run **`android/assemble_friend_apk_with_models.sh`** — after the build it **copies** APK(s) to **`android/friend-apk-dist/`** with a **`friend-YYYYMMDD-HHMMSS-...`** name so friend artifacts never replace Studio’s latest debug APK. See **`android/friend-apk-dist/README.md`**.

## Git

`*.pte` is listed in `.gitignore`; copied files under `build/bundled-pte-assets/models_cpu|models_cpuvulkan_hybrid` are not meant to be committed. Demo GLBs (if any) live under `app/src/main/assets/bundled_rooms/`.

## Prod

Use **`-PskipExecutorchAssets`** for production builds (or omit `sharp_vulkan_only` on the build machine) and deliver models via **adb**, OTA, or Play Asset Delivery as appropriate.

## Troubleshooting: `compress*DebugAssets` / Java heap space

Bundling many large `.pte` files makes `:app:compressEtVulkanDebugAssets` memory-hungry if Gradle tries to **compress** them. The app module adds **`pte`** to **`androidResources.noCompress`** in `app/build.gradle` so `.pte` stay **uncompressed** in the APK (less heap, faster mmap load).

If you still hit **Java heap space**:

1. Raise **`org.gradle.jvmargs`** in `android/gradle.properties` (repo default is **8g** for bundled-model builds).
2. Close other heavy apps; run **`./gradlew --stop`** then rebuild.
3. Temporarily omit the huge tree: **`-PskipSharpVulkanOnlyInAssets=true`** and rely on `executorch_models` / int8 copies only.

## Troubleshooting: `package*Debug` / Zip32 cannot place CD entry

The APK grew past the **Zip32 ~4 GiB** limit. Use **`-PbundleSharpVulkanHybridApk=true`** (or minimal + skip int8 + skip chunked), prefer **`fp16`** for size, and omit chunk_65 unless you need it. If you still exceed 4 GiB, stop bundling assets and use **adb push** instead.

**Still failing after switching to minimal?** An older build may have copied **all** `sharp_vulkan_only/*.pte` into **`app/build/bundled-pte-assets/`**; Gradle’s Copy task **does not remove** files that are no longer part of the copy spec, so huge stale `.pte` can remain until you clean. Fix: **Build → Clean Project** in Android Studio, or delete `android/app/build/bundled-pte-assets/`. Current `app/build.gradle` wipes that folder at the start of each `copyExecutorchModelsIntoAssets` run so this should not recur.

### Friend APK shows “Missing models” but the APK is ~1–2 GB

That size is **normal** for a Vulkan hybrid bundle: six to seven large `.pte` files (Part1–3 Vulkan, Part4a 512 + 65, Part4b Vulkan) often total **~1.5–2 GB** uncompressed in the APK.

If inference still says models are missing:

1. **Clear scoped external models** on the test device: delete or empty  
   `Android/data/com.furnit.android/files/models_cpuvulkan_hybrid/`  
   Old **adb push** leftovers (one or two `.pte` files) used to trigger an over-aggressive **prune** that removed the copies hydrated from the APK. Current app versions **skip** that prune unless external looks like a **full** push (see `ExecutorchInt8Sharp.syncExternalSharpSplitPteToInternal`).
2. **Clear app data** or reinstall the friend APK so internal `files/models_cpuvulkan_hybrid` is re-copied from assets.
3. Confirm **logcat** after launch: `hydrateBundledAndExternalModels` should report a non-zero `models_cpuvulkan_hybrid` `sharp_split*.pte` count.
