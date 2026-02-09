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
