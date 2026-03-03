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

## Run 2 (2025-03-01 15:04 – after chunked INT8 push)

| Phase | Duration | % of total |
|-------|----------|------------|
| Part4b forward | 134,298 ms (~2 min 14 s) | 72.3% |
| Part1+2 (1×) | 30,716 ms | 16.5% |
| Part1+2 (0.5×) | 11,446 ms | 6.2% |
| Part4a | 3,362 ms | 1.8% |
| writePly | 2,438 ms | 1.3% |
| Part3 | 1,349 ms | 0.7% |
| Part1+2 (0.25×) | 1,335 ms | 0.7% |
| **TOTAL** | **185,771 ms** (~3 min 6 s) | 100% |

**Note:** Log showed "Part4b: using sharp_split_part4b_fp16.pte (FP16)" – so the device had an existing `sharp_split_part4b_fp16.pte` (e.g. from a previous push). We only pushed `sharp_split_part4b.pte` (FP32); if both exist, the app prefers FP16. Part4b was ~20 s slower than Run 1 (134 s vs 114 s); could be FP16 build, device load, or thermal. Log line was updated to show actual filename + (FP16)/(FP32) so it's unambiguous.

---

## Quick reference (from logcat)

**Run 1:**
```
[TIMING] Part4b forward: 114203ms
[TIMING] writePly: 3048ms
[TIMING] TOTAL pipeline: 166772ms (166.772s)
Gaussians=1179648  room=9.286518 x 6.9414244 x 4.7297034 m
```

**Run 2:**
```
Part4b: using sharp_split_part4b_fp16.pte (FP16)  ← device had FP16 file
[TIMING] Part4b forward: 134298ms
[TIMING] writePly: 2438ms
[TIMING] TOTAL pipeline: 185771ms (185.771s)
```

---

## Run 3 (2025-03-01 16:54 – Part4b FP32 only, after removing part4b_fp16.pte)

| Phase | Duration | % of total |
|-------|----------|------------|
| Part4b forward | 142,307 ms (~2 min 22 s) | 72.4% |
| Part1+2 (1×) | 31,498 ms | 16.0% |
| Part1+2 (0.5×) | 11,929 ms | 6.1% |
| Part4a | 3,717 ms | 1.9% |
| writePly | 3,151 ms | 1.6% |
| Part3 | 1,507 ms | 0.8% |
| Part1+2 (0.25×) | 1,417 ms | 0.7% |
| **TOTAL** | **196,473 ms** (~3 min 16 s) | 100% |

**Note:** Part4b FP32 (`sharp_split_part4b.pte`) confirmed in log. Part4b ~142 s (vs 134 s with old FP16 file, vs 114 s in earliest Run 1). FP32 decoder on CPU is the bottleneck; Vulkan or working FP16 export would be the next lever.

**SharpRoomActivity:** Logs show the viewer opening twice in quick succession (two "Opening SharpRoomActivity with PLY", two "Copied PLY" / "Loading PLY file") with "Brain: stopBrainDetection()" between—suggesting one instance replaced by another or duplicate start; PLY is copied twice.

**Run 3 quick reference:**
```
Part4b: using sharp_split_part4b.pte (FP32)
[TIMING] Part4b forward: 142307ms
[TIMING] writePly: 3151ms
[TIMING] TOTAL pipeline: 196473ms (196.473s)
```
