# Friend / sideload APK copies (separate from Android Studio)

**Android Studio “Run”** still installs from the normal Gradle output:

`app/build/outputs/apk/<flavor>/<buildType>/`

**Friend builds** should use **`../assemble_friend_apk_with_models.sh`**. With **`android/models_cpuvulkan_hybrid/*.pte`** populated, SHARP ExecuTorch ships **only** under APK `assets/models_cpuvulkan_hybrid/` (no `models_cpu` `.pte` from the copy task). Populate that folder via `../populate_models_cpuvulkan_hybrid_from_backups.sh` (see `../models_cpuvulkan_hybrid/README.md`).

After a successful build, that script **copies** the built APK(s) **here** with a timestamp prefix so:

- Studio debug runs do **not** get confused with friend artifacts.
- Repeated friend builds **do not overwrite** each other (new timestamp each run).

Example files after a run:

- `friend-20260322-143022-app-etVulkan-arm64-v8a-debug.apk`

`*.apk` files are gitignored (see repo `.gitignore`). This folder is only on your machine.

## “There is a problem with this app” / invalid package (large friend APK)

A **full** Vulkan hybrid bundle (Part1–3 fp32 + Part4a 512/65 + Part4b + native libs) often produces an APK **above ~2–3 GiB**.

Many **on-device installers** (Files app, some OEM “package installers”, Bluetooth share) use **32-bit file sizes** internally. APKs **larger than about 2 GiB** (`Integer.MAX_VALUE` bytes) are then rejected with a **generic** error — the file is not necessarily corrupt.

**What to do**

1. Install from a computer (often works better for huge APKs):  
   `adb install -r /path/to/friend-…-app-etVulkan-arm64-v8a-debug.apk`
2. Or ship a **small APK** and push models separately (most reliable):  
   `android/assemble_friend_apk_without_models.sh`  
   (alias: `assemble_friend_apk_shell_only.sh`)  
   then push the hybrid folder from your machine or from a Google Drive zip using  
   `android/push_sharp_cpuvulkan_hybrid_androidstudio.sh`  
   See **Friend without models (Google Drive + adb)** below.
3. This project’s APKs are **arm64-v8a only** — very old 32-bit-only phones cannot install them.

**Verify the ZIP** (optional): `unzip -t your.apk` should end with “No errors detected”.

## Friend without models (Google Drive + adb)

For testers who cannot install a multi-gigabyte APK:

1. **You** build the lightweight APK (no bundled `.pte`):
   ```bash
   cd android
   ./assemble_friend_apk_without_models.sh etVulkanDebug
   # Shorthand for the same flavor: ./assemble_friend_apk_without_models.sh etCpuVulkanDebug
   ```
2. Share **`friend-apk-dist/friend-nomodels-*-app-etVulkan-arm64-v8a-debug.apk`** (or install from `app/build/outputs/apk/etVulkan/debug/`).
3. Upload **`android/models_cpuvulkan_hybrid/`** (complete hybrid set — see `models_cpuvulkan_hybrid/README.md`) to Google Drive as a zip or folder.
4. **Tester** downloads, unzips on the laptop, enables USB debugging, then:
   - **No repo / no checkout:** zip **`android/friend_tester_bundle/`** (scripts + `README_FRIEND.md`) together with the **APK** and **`models_cpuvulkan_hybrid/`**. They follow **`README_FRIEND.md`** — only `adb` required.
   - **Or** from a full clone:  
   ```bash
   adb install -r friend-nomodels-…-app-etVulkan-arm64-v8a-debug.apk
   ./push_sharp_cpuvulkan_hybrid_androidstudio.sh /path/to/downloaded/models_cpuvulkan_hybrid
   ```
   Destination on phone:  
   `/sdcard/Android/data/com.furnit.android/files/models_cpuvulkan_hybrid/`

YOLO / NCNN / TFLite for Furniture Fit stay in the APK; only ExecuTorch `.pte` need this push.

**`etCpu`** flavor instead: `./assemble_friend_apk_without_models.sh etCpuDebug` and `./push_sharp_executorch_cpu_models.sh` → `files/models_cpu/`.
