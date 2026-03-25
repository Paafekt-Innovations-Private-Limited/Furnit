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
   `android/assemble_friend_apk_shell_only.sh`  
   then `android/push_sharp_cpuvulkan_hybrid_androidstudio.sh`
3. This project’s APKs are **arm64-v8a only** — very old 32-bit-only phones cannot install them.

**Verify the ZIP** (optional): `unzip -t your.apk` should end with “No errors detected”.
