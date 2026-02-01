# NCNN SHARP Model - Component Status

## Problem Summary

The full SHARP model cannot run in NCNN due to a **channel dimension mismatch** caused by PNNX's tracing:

```
Original PyTorch:
  - Extract 35 patches: [35, 3, 384, 384] (batch dim = 35)
  - Conv2d processes each: [35, 1024, 24, 24]

PNNX/NCNN Conversion:
  - Extract 35 patches: [3, 384, 384] each
  - Concat along channels: [105, 384, 384]  ← WRONG!
  - Conv2d expects: [3, 384, 384]  ← MISMATCH!
```

NCNN doesn't support true batch dimensions. PNNX converts `torch.cat(patches, dim=0)` as channel concatenation, not batch stacking.

## Working Components

| Component | Input | Output | Size | Status |
|-----------|-------|--------|------|--------|
| `sharp_single_patch_embed.ncnn` | [3, 384, 384] | [576, 1024] | 1.5 MB | ✅ Works |
| `sharp_single_patch_encoder.ncnn` | [576, 1024] | [577, 1024] | 579 MB | ✅ Works |
| `sharp_image_encoder.ncnn` | [3, 384, 384] | [577, 1024] | 580 MB | ✅ Works |
| `sharp_decoder.ncnn` | 5 feature maps | [256, 768, 768] | 35 MB | ⚠️ Untested |

### Embedding Files
- `patch_cls_token.bin` (4 KB) - [1, 1024] CLS token for patch encoder
- `patch_pos_embed.bin` (2.3 MB) - [577, 1024] positional embeddings
- `image_cls_token.bin` (4 KB) - [1, 1024] CLS token for image encoder
- `image_pos_embed.bin` (2.3 MB) - [577, 1024] positional embeddings

## Non-Working Components

| Component | Issue |
|-----------|-------|
| `sharp_full.ncnn` | Channel mismatch at conv_106 (expects 3 channels, receives 105) |
| `sharp_pyramid_v2.ncnn` | Same issue - crashes on inference |
| `sharp_vit_blocks.ncnn` | Batch=35 not supported in NCNN |

## Required Pipeline

To run SHARP with NCNN, the following pipeline is needed:

```
1. Image Input [1536×1536×3]
   ↓
2. Extract 35 patches (C++ code)
   - Scale 1.0x: 25 patches (5×5 grid, stride 288)
   - Scale 0.5x: 9 patches (3×3 grid, stride 192)
   - Scale 0.25x: 1 patch (full 384×384)
   ↓
3. For each of 35 patches:
   a. Run patch_embed → [576, 1024]
   b. Add CLS token → [577, 1024]
   c. Add pos_embed → [577, 1024]
   d. Run patch_encoder → [577, 1024]
   ↓
4. Run image_encoder on lowest-res patch → [577, 1024]
   ↓
5. Merge features back to spatial maps (COMPLEX)
   - Reshape to [24, 24, 1024] per patch
   - Merge overlapping regions
   - Upsample to 5 feature map sizes
   ↓
6. Run decoder → [256, 768, 768]
   ↓
7. Run gaussian head → Gaussian parameters
```

## Missing Implementation: Merge Logic

The merge step (step 5) is complex and requires:

1. **Patch layout understanding:**
   - 1.0x scale: 5×5 grid with 288px stride → 96×96 final size
   - 0.5x scale: 3×3 grid with 192px stride → 48×48 final size
   - 0.25x scale: 1×1 → 24×24 final size

2. **Operations:**
   - Reshape ViT output [577, 1024] → [24, 24, 1024]
   - Remove CLS token (index 0)
   - Merge overlapping patches with padding=3
   - Apply upsample convolutions

3. **Upsample modules (need to export):**
   - upsample_latent0: 1024→256, 4× upsample
   - upsample_latent1: 1024→256, 2× upsample
   - upsample0: 1024→512, 2× upsample
   - upsample1: 1024→1024, 2× upsample
   - upsample2: 1024→1024, 2× upsample
   - upsample_lowres + fuse_lowres

## Performance Estimate

With serial patch processing:
- Patch embed × 35: ~0.5 seconds each = ~17.5 seconds
- ViT encoder × 35: ~1 second each = ~35 seconds
- Image encoder: ~1 second
- Merge + Decoder: ~2 seconds
- **Total: ~55-60 seconds** per image

This is significantly slower than the ONNX split model (~30-45 seconds) due to:
- No batch parallelism
- Overhead of 35 separate inference calls

## Options to Proceed

### Option A: Complete Component-Based Implementation
- Implement merge logic in C++
- Export remaining upsample weights
- Accept ~60s inference time
- **Effort: HIGH** (1-2 days)

### Option B: Fix NCNN Param File
- Manually edit param to process patches correctly
- Requires deep understanding of NCNN layer format
- Error-prone
- **Effort: VERY HIGH** (uncertain timeline)

### Option C: Alternative Runtime
- Use ONNX Runtime (user declined)
- Use MNN or other mobile framework
- **Effort: MEDIUM** (need to investigate)

### Option D: Server-Side Processing
- Send image to server for SHARP processing
- Return Gaussians to device
- **Effort: LOW** (but adds latency, requires connectivity)

## Recommendation

Given the complexity, the most practical path forward is:

1. **Short-term**: Continue using ONNX split model (working)
2. **Medium-term**: Implement Option A (component-based NCNN) if ONNX performance is insufficient
3. **Long-term**: Monitor PNNX updates for better batch handling

## Files Location

All NCNN models are in:
```
/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models/
```

To deploy working components:
```bash
adb push sharp_single_patch_embed.ncnn.param /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push sharp_single_patch_embed.ncnn.bin /storage/emulated/0/Android/data/com.furnit.android/files/models/
# ... repeat for other components
```
