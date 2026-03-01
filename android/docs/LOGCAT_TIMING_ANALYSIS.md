# Logcat timing analysis – ExecuTorch INT8 SHARP (1600×1200, stretch-to-square)

**Device:** 53181JEBF16055  
**Run:** 2025-03-01 13:51:57 – 13:54:44 (~2 min 47 s total)

---

## Pipeline timing breakdown

| Phase | Duration | % of total | Notes |
|-------|----------|------------|--------|
| **Part4b forward** | **114,203 ms** (~1 min 54 s) | **68.5%** | Single large decoder; main bottleneck |
| Part1+2 (1× patches, 5×5) | 30,959 ms | 18.6% | 25 patches @ ~1.24 s/patch |
| Part1+2 (0.5× patches, 3×3) | 11,445 ms | 6.9% | 9 patches @ ~1.27 s/patch |
| Part4a (chunked decoder) | 3,408 ms | 2.0% | 512 + 65 tokens |
| writePly | 3,048 ms | 1.8% | 1,179,648 Gaussians → ~279 MB PLY |
| Part3 (image encoder) | 1,347 ms | 0.8% | Single forward |
| Part1+2 (0.25×, 1 patch) | 1,278 ms | 0.8% | 35th patch |
| Part1+2 load | 38 ms | — | MMAP load |
| Stretch 1600×1200→1536×1536 | ~20 ms | — | Negligible |
| **TOTAL** | **166,772 ms** (~2 min 47 s) | 100% | |

---

## Findings

### 1. Part4b dominates (68.5%)
- Single forward on 7 inputs (tokens, img, latent0, latent1, x0Feat, x1Feat, x2Feat).
- ~114 s on this device; likely CPU-bound INT8 or limited GPU/NNAPI use.
- **Optimization levers:** ExecuTorch backend (Vulkan/GPU), quantization (e.g. keep position/rotation FP16), or smaller model/variant for Part4b.

### 2. Part1+2 patch encoder is second (26.3% combined)
- 35 patches total: 25 (1×) + 9 (0.5×) + 1 (0.25×).
- ~1.24–1.27 s per patch; load cost (38 ms) is negligible.
- **Optimization levers:** Batch patches if the model supports it, or offload to GPU/NNAPI; otherwise reduce patch count or resolution only if quality is acceptable.

### 3. Part4a and Part3 are modest
- Part4a: ~3.4 s for 577 chunks; Part3: ~1.3 s. Not the main targets after Part4b and Part1+2.

### 4. writePly (~3 s) is acceptable
- 1.18M Gaussians, 62 floats/vertex, buffered I/O; ~3 s is reasonable for this size.

### 5. Viewer / WebView
- PLY size: **292,554,236 bytes** (~279 MB) for 1.18M Gaussians.
- `autoFrameRoom()` runs 6 attempts; `CAMERA_FRAME fallback=1` uses fallback dims (9.29×6.94×4.73); camera ends at `pos=0, 0.014, -0.615`, target `z=-2.1`, distance ~1.49.
- Room loads and frames; no errors in the snippet.

---

## Suggested priority

1. **Part4b** – Profile backend (CPU vs Vulkan/NNAPI); consider FP16 for position/scale/rotation if INT8 causes artifacts; explore smaller or split Part4b if available.
2. **Part1+2** – Explore batching or GPU/NNAPI for patch encoder; validate quality before reducing patches.
3. **Part4a / Part3 / writePly** – Lower priority unless Part4b and Part1+2 are already optimized.

---

## Quick reference (from logcat)

```
[TIMING] Part4b forward: 114203ms
[TIMING] writePly: 3048ms
[TIMING] TOTAL pipeline: 166772ms (166.772s)
Gaussians=1179648  room=9.286518 x 6.9414244 x 4.7297034 m
```
