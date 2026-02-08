# SHARP Model Android Implementation

## Overview

This document describes the implementation of the SHARP (Single-image 3D Gaussian Splat) model on Android for generating 3D room reconstructions from a single photo.

## Archit111ask question about problem statement. will get steps.
1ecture

### Model Split Strategy

The SHARP model (~1.4GB total) was split into 4 parts for memory-efficient mobile inference:

| Part | Component | Size | Purpose |
|------|-----------|------|---------|
| Part 1 | ViT Encoder | ~350MB | Image feature extraction (ViT-Large backbone) |
| Part 2 | Decoder A | ~400MB | First half of depth/splat decoder |
| Part 3 | Decoder B | ~400MB | Second half of decoder |
| Part 4 | Gaussian Head | ~250MB | Final 14-channel Gaussian parameter output |

### Why Split?

- **Memory constraints**: Full model requires ~2GB RAM during inference
- **Android heap limits**: Most devices have 512MB-1GB heap limit
- **Streaming approach**: Each part runs sequentially, intermediate tensors saved to disk
- **Tensor exchange**: Parts communicate via serialized float tensors in app's cache directory

## Inference Pipeline

```
Image (1200x1600)
    → Resize to 1536x1536
    → Part 1: Encoder (ViT features)
    → Save ~200 intermediate tensors to disk
    → Part 2: Decoder A
    → Save intermediate tensors
    → Part 3: Decoder B
    → Save intermediate tensors
    → Part 4: Gaussian Head
    → Output: [B, 14, H, W] Gaussian parameters
    → Post-process to PLY file
```

## Timing Breakdown (Pixel 8 Pro)

| Step | Duration | Notes |
|------|----------|-------|
| Part 1 (Encoder) | ~2 min 13 sec | ViT-Large is compute-heavy |
| Part 2 (Decoder A) | ~46 sec | Moderate |
| Part 3 (Decoder B) | ~7 sec | Fast |
| Part 4 (GaussianHead) | ~2 min 15 sec | Upsampling + convolutions |
| **Total** | **~5 min 20 sec** | End-to-end inference |

### Bottlenecks Identified

1. **Part 1 (Encoder)** - 42% of total time
   - ViT-Large with 24 transformer blocks
   - Self-attention is O(n²) on sequence length
   - 1536x1536 input = 1024 patches (32x32 grid)

2. **Part 4 (Gaussian Head)** - 42% of total time
   - Multiple bilinear upsampling operations (2x each)
   - Large intermediate feature maps at full resolution
   - Final convolution to 14 output channels

3. **Disk I/O for tensor exchange** - ~10% overhead
   - ~200 tensors saved between Part 1 and Part 2
   - Largest tensor: ~80MB (encoder features)
   - Mitigated by using app's cache directory (fast storage)

## Output Format

### Gaussian Parameters (14 channels)

The model outputs 14 parameters per pixel:

| Index | Parameter | Range | Notes |
|-------|-----------|-------|-------|
| 0-2 | Position (x, y, z) | Raw | 3D location in camera space |
| 3-5 | Scale (sx, sy, sz) | Raw | Gaussian ellipsoid scales |
| 6-9 | Rotation (qw, qx, qy, qz) | Normalized | Quaternion orientation |
| 10 | Opacity | [0, 1] | Alpha transparency |
| 11-13 | Color (r, g, b) | [0, 1] | RGB color |

### PLY File Format

Standard 3D Gaussian Splatting PLY format (62 floats per vertex):

```
ply
format binary_little_endian 1.0
element vertex N
property float x, y, z           # Position
property float nx, ny, nz        # Normals (unused, set to 0)
property float f_dc_0..2         # SH DC coefficients (color)
property float f_rest_0..44      # Higher-order SH (set to 0)
property float opacity           # Logit-transformed opacity
property float scale_0..2        # Log-transformed scales
property float rot_0..3          # Normalized quaternion
end_header
[binary vertex data]
```

## Viewer Implementation

### WebGL Rendering (SparkJS)

The 3D Gaussian splats are rendered using:
- **THREE.js** for scene management
- **SparkJS** (sparkjs.dev) for optimized Gaussian splatting
- **OrbitControls** for touch interaction

### Camera Positioning

```javascript
// Auto-frame using bounding box
const box = new THREE.Box3().setFromObject(splatMesh);
const center = box.getCenter(new THREE.Vector3());
const size = box.getSize(new THREE.Vector3());

// Camera positioned on negative Z, looking into room
const cameraDistance = (maxDim / 2) / Math.tan(fov / 2) * 1.5;
camera.position.set(center.x, center.y, center.z - cameraDistance);
controls.target.copy(center);
```

### Orientation Handling

- **Portrait photos**: No rotation needed for ONNX PLY output
- **Landscape photos**: No rotation needed
- PLY from ONNX has correct Y-up orientation by default
- Camera positioned on negative Z axis looking into positive Z (into room)

## Backend Selection

The app supports multiple inference backends:

| Backend | Status | Notes |
|---------|--------|-------|
| Split ONNX | **Active** | Memory-efficient 4-part inference |
| Full ONNX | Fallback | Requires mmap, may OOM |
| NCNN | Disabled | Implementation incomplete |

Selection is controlled by Settings toggle:
- NCNN switch OFF → Uses Split ONNX (recommended)
- NCNN switch ON → Attempts NCNN (not recommended)

## Files Structure

```
app/src/main/java/com/furnit/android/
├── services/
│   ├── SharpService.kt         # Main service, backend selection
│   ├── SplitOnnxSharp.kt       # 4-part ONNX inference
│   ├── OnnxSharp.kt            # Full ONNX (fallback)
│   └── NcnnSharp.kt            # NCNN (disabled)
├── SharpRoomActivity.kt        # WebGL viewer

External storage models:
/Android/data/com.furnit.android/files/models/
├── sharp_part1.onnx            # ~350MB
├── sharp_part2.onnx            # ~400MB
├── sharp_part3.onnx            # ~400MB
└── sharp_part4.onnx            # ~250MB
```

## Performance Optimizations Applied

1. **ONNX Runtime settings**
   - 4 intra-op threads (half of available cores)
   - Graph optimizations enabled
   - Memory pattern optimization

2. **Tensor management**
   - Immediate release after use
   - GC hints between parts
   - Cache directory cleanup on completion

3. **Image preprocessing**
   - Bilinear resize to 1536x1536
   - RGB normalization in-place
   - Direct float buffer creation

## Future Optimization Opportunities

1. **Quantization**: INT8/FP16 could reduce Part 1 time by 40-50%
2. **GPU acceleration**: NNAPI delegate for supported operations
3. **Model pruning**: Remove unused decoder pathways
4. **Streaming output**: Write PLY while Part 4 runs
5. **Batch processing**: Process multiple images in pipeline

## Comparison with iOS

| Aspect | iOS | Android |
|--------|-----|---------|
| Model format | CoreML | ONNX (split) |
| Inference time | ~2 min | ~5 min 20 sec |
| Memory usage | ~1.5GB peak | ~800MB peak |
| GPU acceleration | ANE | CPU only |

The iOS implementation benefits from Apple Neural Engine (ANE) acceleration, achieving ~2.5x faster inference.

## Known Limitations

1. **Processing time**: 5+ minutes may cause user abandonment
2. **Battery drain**: Heavy CPU usage during inference
3. **Storage**: ~1.4GB for model files
4. **Memory**: Requires ~4GB RAM device for stable operation
5. **Thermal**: Extended inference may trigger thermal throttling

## Version History

- **v1.0**: Initial Split ONNX implementation
- **v1.1**: Fixed orientation handling (removed unnecessary rotations)
- **v1.2**: Removed joystick, improved camera positioning
