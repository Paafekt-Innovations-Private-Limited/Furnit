# AI Room Generation - Technical Documentation

## Overview

The Furnit Android app generates 3D Gaussian Splat rooms from a single 2D image using the **SHARP (Single-image House-scale Avatar Reconstruction Pipeline)** model. This document describes the architecture, model backends, and deployment process.

## Architecture

```
┌─────────────────┐
│   User Image    │
│  (1536x1536)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SharpService   │  ← Singleton, manages backend selection
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Backend Selection               │
│  (Based on settings & availability)     │
└────────┬──────────┬──────────┬─────────┘
         │          │          │
         ▼          ▼          ▼
┌─────────┐  ┌────────────┐  ┌─────────┐
│  NCNN   │  │ Split ONNX │  │  ONNX   │
│ (Fast)  │  │ (Memory    │  │ (Large) │
│         │  │  Efficient)│  │         │
└────┬────┘  └─────┬──────┘  └────┬────┘
     │             │              │
     └─────────────┼──────────────┘
                   ▼
         ┌─────────────────┐
         │  Gaussian PLY   │
         │  Output File    │
         └─────────────────┘
```

## Model Backends

Only **ONNX Runtime** is required to work and to be deployed to devices. The repo contains wrappers for NCNN, ExecuTorch, and LiteRT, but those are **disabled by default** to avoid build/runtime churn. If you enable them, you are responsible for providing their model files and validating them on-device.

### 1. NCNN Backend (Optional - Disabled by default)

**Files Required:**
- `sharp.ncnn.param` - Model graph definition
- `sharp.ncnn.bin` - Model weights

**Advantages:**
- Optimized for mobile CPUs (ARM NEON, Vulkan GPU)
- Lowest memory footprint
- Fastest inference

**Location on device:**
```
/storage/emulated/0/Android/data/com.furnit.android/files/models/
```

**Deployment:**
```bash
adb push sharp.ncnn.param /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp.ncnn.bin /storage/emulated/0/Android/data/com.furnit.android/files/models/
```

### 2. Split ONNX Backend (Memory Efficient, Recommended)

**Files Required (4 parts, ~2.5GB total):**
| File | Size | Description |
|------|------|-------------|
| `sharp_part1.onnx` | 439KB | Part 1 graph |
| `sharp_part1.onnx.data` | 902MB | Part 1 weights |
| `sharp_part2.onnx` | 354KB | Part 2 graph |
| `sharp_part2.onnx.data` | 256MB | Part 2 weights |
| `sharp_part3.onnx` | 387KB | Part 3 graph |
| `sharp_part3.onnx.data` | 546MB | Part 3 weights |
| `sharp_part4.onnx` | 4.9MB | Part 4 graph |
| `sharp_part4.onnx.data` | 790MB | Part 4 weights |

**How it works:**
1. Load Part 1, run inference, save intermediate tensors to disk, unload
2. Load Part 2, load intermediates, run, save outputs, unload
3. Load Part 3, load intermediates, run, save outputs, unload
4. Load Part 4, load intermediates, run, get final Gaussian output

**Advantages:**
- Runs on devices with 4-6GB RAM
- Only ~600MB loaded at a time
- Memory-mapped intermediate tensors

**Deployment:**
```bash
adb push sharp_part1.onnx /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part1.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part2.onnx /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part2.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part3.onnx /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part3.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part4.onnx /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_part4.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
```

### 3. Regular ONNX Backend (Fallback)

**Files Required:**
- `sharp_fp32_aligned.onnx` (~2.5GB single file)

**Note:** May cause OOM on devices with less than 8GB RAM.

## Backend Selection Logic

```kotlin
// In SharpService.kt

// "inference_backend" preference is normalized to a supported backend.
// With default config, non-ONNX backends are forced to ONNX.
//
// When ONNX is selected:
//   if (Split ONNX models exist) → Use Split ONNX
//   else if (Regular ONNX exists) → Use Regular ONNX
//   else → Fail
```

## Settings

**Location:** Settings → Developer → "Inference Backend"

| Setting | Behavior |
|---------|----------|
| ONNX (default) | Uses Split ONNX if present, else Regular ONNX |
| NCNN / ExecuTorch / LiteRT | Present in repo but disabled by default |

## Output Format

The model outputs a **3D Gaussian Splat PLY file** with the following properties per vertex:

| Property | Count | Description |
|----------|-------|-------------|
| x, y, z | 3 | Position |
| nx, ny, nz | 3 | Normals (unused, zeros) |
| f_dc_0/1/2 | 3 | Spherical harmonics DC (color) |
| f_rest_0-44 | 45 | Higher-order SH (zeros) |
| opacity | 1 | Opacity (logit transformed) |
| scale_0/1/2 | 3 | Scale (log transformed) |
| rot_0/1/2/3 | 4 | Rotation quaternion |

**Total:** 62 floats × 4 bytes = **248 bytes per Gaussian**

## Performance Optimizations

### Implemented:
1. **Multi-threaded inference** - Uses up to 4 CPU cores
2. **Memory-mapped tensors** - Intermediate data stored on disk
3. **Batched PLY writes** - 500 vertices per I/O operation
4. **Buffered output streams** - 256KB buffer for file writes
5. **Session reuse** - OrtEnvironment singleton across parts

### Configuration (SplitOnnxSharp.kt):
```kotlin
setIntraOpNumThreads(min(numCores, 4))  // Parallel MatMul
setInterOpNumThreads(1)                  // Sequential ops (memory-safe)
setMemoryPatternOptimization(true)       // Buffer reuse
setCPUArenaAllocator(false)              // Avoid pre-allocation
```

## File Locations

| Type | Path |
|------|------|
| Models | `/storage/emulated/0/Android/data/com.furnit.android/files/models/` |
| Generated Rooms | `/data/user/0/com.furnit.android/files/sharp_rooms/` |
| Temp Files | `/data/user/0/com.furnit.android/cache/sharp_temp/` |

## Backup Models

```bash
# Pull all models from device
mkdir -p ~/furnit_models_backup
adb pull /storage/emulated/0/Android/data/com.furnit.android/files/models/ ~/furnit_models_backup/

# Restore models to device
adb push ~/furnit_models_backup/models/* /storage/emulated/0/Android/data/com.furnit.android/files/models/
```

## Troubleshooting

### "SHARP model not available"
- Check if model files exist on device
- Verify file permissions (`chmod 666`)
- Check logcat for specific errors: `adb logcat | grep -i sharp`

### Out of Memory
- Use Split ONNX backend (default)
- Close other apps
- Restart device to clear memory

### Slow Generation
- Enable NCNN if models available
- Check CPU thermal throttling
- Verify not running in background

## Code Structure

```
services/
├── SharpService.kt      # Main entry point, backend selection
├── NcnnSharp.kt         # NCNN backend implementation
├── SplitOnnxSharp.kt    # Split ONNX backend (4-part)
└── OnnxSharp.kt         # Regular ONNX backend
```

## Model Conversion

### To create Split ONNX from full ONNX:
```python
# Use ONNX graph surgery to split at intermediate nodes
# Each part should be ~600MB for mobile compatibility
```

### To create NCNN from ONNX:
```bash
# Using ncnn tools
onnx2ncnn sharp.onnx sharp.ncnn.param sharp.ncnn.bin
# Then optimize
ncnnoptimize sharp.ncnn.param sharp.ncnn.bin sharp_opt.ncnn.param sharp_opt.ncnn.bin 65536
```

## References

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [NCNN](https://github.com/Tencent/ncnn)
- [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/)
