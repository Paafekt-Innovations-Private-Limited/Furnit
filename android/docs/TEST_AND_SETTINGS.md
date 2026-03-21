# Test in Android Studio & Settings

## 1. Open and run in Android Studio

1. **Open the project**  
   Open the **android** folder (e.g. `Furnit/android` or your worktree’s `ftf/android`).  
   Use **File → Open** and select the `android` directory so Gradle sync runs correctly.

2. **Sync and build**  
   Let Gradle sync finish. Build once: **Build → Make Project** (or run `./gradlew assembleDebug` in the `android` folder).

3. **Select device**  
   In the device dropdown, pick a connected phone/emulator (USB debugging enabled for a physical device).

4. **Run**  
   Click **Run** (green play) or **Run → Run 'app'**. The app installs and launches.

---

## 2. Settings to use for SHARP Vulkan room creation

For the current known-good Vulkan room-creation recipe, see
[`EXECUTORCH_VULKAN_KNOWN_GOOD_FLOW.md`](EXECUTORCH_VULKAN_KNOWN_GOOD_FLOW.md).

In the app: **Profile → Settings** (or the gear icon), then scroll to the **Developer** section.

| Setting | What to set | Why |
|--------|-------------|-----|
| **Inference Backend** | **ExecuTorch INT8 (Vulkan)** | Uses the active Vulkan SHARP path. |
| **Max Gaussians** | **All** | Matches the validated reference run. |
| **Use true 1280x1280** | **OFF** | Keep the currently validated `1536` hybrid split pipeline. |
| **Prefer Vulkan FP16 models** | **ON** | Use FP16 Vulkan exports when present. |
| **Prefer single Part4b** | **OFF** | The current known-good Vulkan path is the fine-split `tile_00` route, not the old single-decoder path. |
| **Record ETDump on next room creation** | **OFF** | Leave profiling off for normal runs. |

Optional:

- **Swap tile NDC X/Y**: Leave **OFF** unless you’ve been told to enable it for a specific device.

---

## 3. Quick checklist

- [ ] Project opened as `android` folder in Android Studio  
- [ ] Gradle sync and build succeeded  
- [ ] Device selected and app installed via Run  
- [ ] **Settings → Inference Backend** = **ExecuTorch INT8 (Vulkan)**  
- [ ] **Max Gaussians** = **All**  
- [ ] **Use true 1280x1280** = **OFF**  
- [ ] **Prefer Vulkan FP16 models** = **ON**  
- [ ] **Prefer single Part4b** = **OFF**  
- [ ] Models pushed to device (e.g. `./push_sharp_executorch_int8_models.sh`), or already present in app storage  

After that, use the SHARP / “New room” flow in the app; inference will use the above backend and options.
