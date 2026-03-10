# Running the app from Android Studio

## "Installation failed" / "Unknown failure" / "Exception occurred"

If the app **won’t install** (Error code UNKNOWN, "Exception occurred"), the APK is often **too large** because it embeds ExecuTorch .pte models (~1–2 GB). Some devices or ADB versions fail on huge APKs.

**Fix: build a small APK and push models yourself**

1. In Android Studio: **File → Settings → Build, Execution, Deployment → Compiler**. In **Command-line Options**, add:
   ```text
   -PskipExecutorchAssets
   ```
   (Or run from terminal: `./gradlew :app:assembleDebug -PskipExecutorchAssets`.)
2. **Sync** and **Build → Rebuild Project**.
3. **Run** the app — install should succeed (APK will be much smaller).
4. Push models to the device once (with one device connected):
   ```bash
   cd android && ./push_sharp_executorch_int8_models.sh
   ```
   Or push to a specific device: `adb -s <device_id> push ...` (see script).

After that, run from Android Studio as usual. The app reads models from internal/external storage.

**Other checks:** Free space on device/emulator (several GB). Uninstall the existing Furnit app and try Install again.

---

## App not starting? (launcher / ABI)

The project builds **only arm64-v8a** (smaller APK, real devices). It will **not** run on a default **x86/x86_64 emulator** — the APK has no matching ABI, so the app may not install or may not start.

### Fix: use an ARM64 device or emulator

1. **Physical device**  
   Connect an Android phone (almost all recent phones are arm64-v8a). Enable USB debugging and choose the device in the Run dropdown.

2. **Emulator**  
   Create an AVD that uses an **ARM64** system image:
   - **Device Manager** → **Create Device** → pick a device (e.g. Pixel 6).
   - **System image**: choose a **Google APIs** or **Google Play** image that is **ARM 64-bit** (e.g. "Tiramisu" or "UpsideDownCake" **ARM 64**), not x86_64.
   - Finish and select this AVD when you Run.

If your only image is x86_64, download an ARM64 image: **Create Device** → **System image** → **Other Images** → pick a release with "ARM 64" in the ABI column → **Download**.

### Run configuration

- **Module:** `app`
- **Launch option:** Default (launches the **LAUNCHER** activity = **LoginActivity**).

If you don’t see the app in the Run dropdown, use **Run → Edit Configurations**, add **Android App**, set module to **app**, then **Apply** and **Run**.

### After a clean clone

1. **File → Sync Project with Gradle Files**
2. **Build → Make Project**
3. Select an **arm64-v8a** device or emulator and click **Run**
