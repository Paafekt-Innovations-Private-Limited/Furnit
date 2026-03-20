# Model Setup Guide

This guide explains how to set up the AI room generation models after checking out the Furnit Android codebase.

## Prerequisites

- Android device connected via USB
- ADB installed and working (`adb devices` shows your device)
- Model files (obtained from team Drive or backup)

## Model Files Required

### Split ONNX Models (Required for AI Room Generation)

| File | Size | Description |
|------|------|-------------|
| `sharp_part1.onnx` | ~439KB | Part 1 graph |
| `sharp_part1.onnx.data` | ~902MB | Part 1 weights |
| `sharp_part2.onnx` | ~354KB | Part 2 graph |
| `sharp_part2.onnx.data` | ~256MB | Part 2 weights |
| `sharp_part3.onnx` | ~387KB | Part 3 graph |
| `sharp_part3.onnx.data` | ~546MB | Part 3 weights |
| `sharp_part4.onnx` | ~4.9MB | Part 4 graph |
| `sharp_part4.onnx.data` | ~790MB | Part 4 weights |

**Total: ~2.5GB**

### YOLOE Segmentation Model (No ADB push needed)

The YOLOE segmentation ONNX is packaged in the app’s `assets/` and is copied to cache on first use. Your friend does **not** need to `adb push` any YOLOE files.

### Optional: Other Backends (Currently Disabled)

The repo contains wrappers for NCNN, ExecuTorch, and LiteRT, but the app is configured to run **ONNX-only** by default. That means you do **not** need to push any NCNN/ExecuTorch/LiteRT model files unless you explicitly enable those backends in code.

#### Optional: LiteRT (TFLite) Models

LiteRT requires `.tflite` model files on the device. There are two options:

1. **Split LiteRT (recommended; more memory-stable)**:
   - `sharp_part1_fp16.tflite`
   - `sharp_part2_fp16.tflite`
   - `sharp_part3_fp16.tflite`
   - `sharp_part4_fp16.tflite`

2. **Single LiteRT (NOT recommended; often killed by Android LMK)**:
   - `vit_gaussian_fp16.tflite` (can be ~1.2GB+)

The repo includes export scripts (requires the SHARP `.pt` checkpoint and SHARP source repo):

```bash
cd android
python export_sharp_litert_split.py --weights /path/to/sharp_checkpoint.pt
./push_sharp_litert_models.sh sharp_litert_models
```

#### Optional: ExecuTorch Models (XNNPACK or Vulkan backend)

**Memory-optimized single .pte (recommended to avoid Part 4 OOM / LMK):**

A single full model with **greedy memory planning** + FP16 + XNNPACK reuses buffers across the graph, keeping peak activation memory much lower than the 4-part split (which cannot reuse across parts). Export and push:

```bash
cd android
python export_sharp_executorch_all.py --variant memory_optimized
# Then push sharp_full_memory_optimized.pte to device (see push script below)
```

The app prefers `sharp_full_memory_optimized.pte` when present. See `android/docs/MEMORY_OPTIMIZATION.md` for the full prioritized plan (chunked attention, INT8, etc.).

**Split models (alternative):** Backend is chosen at export time. Portable (CPU fallback) models cause 10+ minute inference.

| File | Size | Description |
|------|------|-------------|
| `sharp_split_part1.pte` | ~582MB | Patch Encoder A (blocks 0-11) |
| `sharp_split_part2.pte` | ~577MB | Patch Encoder B (blocks 12-23) |
| `sharp_split_part3.pte` | ~582MB | Image Encoder A |
| `sharp_split_part4.pte` | ~755MB | Image Encoder B + Decoder + Gaussians |

**Total: ~2.5GB**

Export options for split (pick one):
```bash
# XNNPACK (CPU optimized, 1-2 min inference) — recommended
python export_sharp_executorch_split4.py --weights /path/to/sharp.pt --backend xnnpack --output-dir executorch_models

# Vulkan GPU (20-60 sec inference) — fastest, requires Vulkan support
python export_sharp_executorch_split4.py --weights /path/to/sharp.pt --backend vulkan --output-dir executorch_models

# DO NOT use portable unless debugging — 10+ min inference (CPU scalar fallback)
python export_sharp_executorch_split4.py --weights /path/to/sharp.pt --backend portable --output-dir executorch_models
```

```bash
./push_sharp_executorch_models.sh executorch_models
```

**INT8 split + native C++ full pipeline (etCpu):** push to `models_cpu` and avoid mixing Part4b with older parts:

```bash
cd android
./clear_device_models_cpu.sh   # optional but recommended after changing exports
./push_sharp_executorch_cpu_models.sh /path/to/export_dir
# LaCie one-shot: ./fresh_sync_cpu_models_from_lacie.sh
```

See **`docs/EXECUTORCH_CPU_MODELS_SYNC.md`**.

The script pushes models to:
1. `/sdcard/Android/data/com.furnit.android/files/models/` (external app storage)
2. `/data/local/tmp/furnit/` (fallback search dir, so split mode is found even if external path differs)

Verify export: `ls -lh executorch_models/` — expect part1–3 ~500–600MB each, part4 ~700–800MB.

#### Optional: Native .pt (TorchScript / LibTorch) split models

The full 2.5GB `.ptl` model crashes on load (OOM). **Split mode is preferred** — each part ~500–800MB, loaded one at a time during inference.

| File | Size | Description |
|------|------|-------------|
| `sharp_scripted_part1.ptl` | ~500–600MB | Patch Encoder A |
| `sharp_scripted_part2.ptl` | ~500–600MB | Patch Encoder B |
| `sharp_scripted_part3.ptl` | ~500–600MB | Image Encoder A |
| `sharp_scripted_part4.ptl` | ~700–800MB | Image Encoder B + Decoder + Gaussians |

Export and push:
```bash
cd android
python export_sharp_torchscript_split.py
./push_sharp_torchscript_split.sh
```

#### Optional: NCNN Models

| File | Description |
|------|-------------|
| `sharp.ncnn.param` | NCNN graph definition |
| `sharp.ncnn.bin` | NCNN weights |

## Setup Steps

### Step 1: Install the App

```bash
cd android
./gradlew installDebug
```

Note: This repo is configured to build **ONNX-only by default**. To also build the native NCNN libraries, run:
```bash
cd android
./gradlew installDebug -Pfurnit.enableNative=true
```

### Step 2: Get Model Files

Option A: Download from team Google Drive (ask team lead for link)

Option B: Copy from a colleague who has them:
```bash
# On colleague's machine - export models
mkdir -p ~/furnit_models_backup
adb pull /storage/emulated/0/Android/data/com.furnit.android/files/models/ ~/furnit_models_backup/
# Share the ~/furnit_models_backup folder
```

### Step 3: Push Models to Device

```bash
# Create models directory on device (if needed)
adb shell mkdir -p /storage/emulated/0/Android/data/com.furnit.android/files/models/

# Push all model files
adb push sharp_part1.onnx /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part1.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part2.onnx /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part2.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part3.onnx /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part3.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part4.onnx /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part4.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
```

Or push all at once from a folder:
```bash
adb push /path/to/models/* /storage/emulated/0/Android/data/com.furnit.android/files/models/
```

Or use the helper script (pushes only SHARP ONNX files):
```bash
cd android
./push_sharp_onnx_models.sh /path/to/models
```

### Step 4: Verify Installation

```bash
adb shell ls -lh /storage/emulated/0/Android/data/com.furnit.android/files/models/
```

You should see all 8 files listed.

### Step 5: Test

1. Open the Furnit app
2. Go to "Create Room" > "AI Room"
3. Select a photo
4. Wait for generation (takes 1-2 minutes on most devices)

## Troubleshooting

### "Model not available" error

1. Check if models exist:
   ```bash
   adb shell ls /storage/emulated/0/Android/data/com.furnit.android/files/models/
   ```

2. If empty, the app package folder may not exist yet. Run the app once first, then push models.

3. Check if NCNN toggle is ON in Settings > Developer. Turn it OFF if you don't have NCNN models.

### Models push fails with "Permission denied"

Run the app at least once to create the app data folder, then try again.

### Generation is very slow

- Split ONNX is slower than NCNN
- First run may be slower due to model loading
- Close other apps to free memory

### Native .pt split: debugging stalls / LMK

If Part 1 completes but Part 2/3/4 stall or the app is killed:

1. **Monitor logs** (run in separate terminal during generation):
   ```bash
   cd android
   ./monitor_native_pt_inference.sh 2>&1 | tee inference.log
   ```
   Or: `adb logcat | grep -E "NativePtSharp|Part [0-9]"`

2. **Check memory** while inference runs:
   ```bash
   adb shell dumpsys meminfo com.furnit.android
   ```
   Or every 5s: `watch -n5 'adb shell dumpsys meminfo com.furnit.android | head -35'`

3. **Reduce LMK pressure**: Close other apps, avoid switching away during generation.

4. **Logs to look for**:
   - `Part 2: before load` / `load done` — if missing, Part 2 load is hanging
   - `Part 2: first forward` — first forward can take 10–60s (backend init)
   - `Part 2: patch 0/35` … `patch 34/35` — per-patch progress
   - `JVM: X/Y MB, SysAvail: Z MB` — low SysAvail (<200MB) increases LMK risk

## Settings

In the app: **Settings > Developer**

| Setting | Description |
|---------|-------------|
| Inference Backend | ONNX is supported; others are disabled by default |
| Debug Mode | Show additional logs |

## File Locations on Device

| Type | Path |
|------|------|
| Models | `/storage/emulated/0/Android/data/com.furnit.android/files/models/` |
| Generated Rooms | `/data/user/0/com.furnit.android/files/sharp_rooms/` |

## Need Help?

- Check logs: `adb logcat | grep -i sharp`
- See [AI_ROOM_GENERATION.md](AI_ROOM_GENERATION.md) for technical details
