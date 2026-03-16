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

## 2. Settings to use for SHARP (ExecuTorch INT8)

In the app: **Profile → Settings** (or the gear icon), then scroll to the **Developer** section.

| Setting | What to set | Why |
|--------|-------------|-----|
| **Inference Backend** | **ExecuTorch INT8** | Uses the native C++ INT8 pipeline (Part1–Part4b) and the `.pte` models you pushed. |
| **C++ ExecuTorch INT8** (Kotlin vs C++) | **C++** | Runs the full pipeline in native code; usually faster and lower overhead. |
| **Part1/Part2 batch=4** | **ON** | Uses batch-4 models for Part1/Part2 (faster). Turn **OFF** if the app crashes right after “Part1+Part2 ready”. |
| **Stable (single Part4b)** vs **Split / tiled Part4b** | **Stable (single Part4b)** | Single Part4b run; simpler and often more stable. Use Split/tiled only if you need tiled decoding. |

Optional:

- **Max Gaussians**: **All** (default), or **300k** / **500k** to cap splat count.
- **Swap tile NDC X/Y**: Leave **OFF** unless you’ve been told to enable it for a specific device.

There is **no separate “Vulkan” toggle**. Vulkan is used automatically when the loaded `.pte` files were exported with `--backend vulkan`. If you only have XNNPACK-exported models, the app uses CPU (XNNPACK); push Vulkan-exported `.pte` files to use the GPU.

---

## 3. Quick checklist

- [ ] Project opened as `android` folder in Android Studio  
- [ ] Gradle sync and build succeeded  
- [ ] Device selected and app installed via Run  
- [ ] **Settings → Inference Backend** = **ExecuTorch INT8**  
- [ ] **C++** selected for ExecuTorch INT8  
- [ ] **Part1/Part2 batch=4** ON (turn off if it crashes)  
- [ ] **Stable (single Part4b)** selected  
- [ ] Models pushed to device (e.g. `./push_sharp_executorch_int8_models.sh`), or already present in app storage  

After that, use the SHARP / “New room” flow in the app; inference will use the above backend and options.
