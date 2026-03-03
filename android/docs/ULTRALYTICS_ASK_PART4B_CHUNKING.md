# Ask Ultralytics: Chunking Part4b (Gaussian decoder)

Part4a is already chunked (two .pte chunks: 512 + 65 tokens). Part4b is a single large forward (~120–140 s on device) and is not chunked. This doc is for asking Ultralytics how to chunk Part4b so we can run it in smaller steps (better progress UX, possibly lower peak memory).

---

## Ultralytics response (summary)

- **Chunking strategy:** Split by **token range**. Split `[1, 577, 1024]` into smaller blocks (e.g. four chunks of ~144 tokens). Each forward produces a subset of the ~1.2M Gaussians; concatenate in memory or **append directly to the PLY file**.
- **ExecuTorch:** Use `torch.split()` or `torch.narrow()` on the token dimension before export. Shared inputs (image features, latents) passed to every chunk. On Android: pre-allocate full output buffer and use `ByteBuffer.put()` per chunk, or **append each chunk's results to the PLY file** to avoid large tensor concatenations.
- **References:** executorch_wrapper pattern for XNNPACK; consider model quantization for memory; Ultralytics Community Forum for deployment issues.

---

## Implementation: two-stage Part4b split (DONE)

Ultralytics suggested "split by token range," but the SHARP decoder reshapes tokens to a spatial feature map `[1,1024,24,24]` in the very first operation — token-range chunking doesn't apply. Instead, we split Part4b at the **monodepth boundary** into two sequential stages:

### Stage 1: Part4bDepthDecoder (`sharp_split_part4b_depth.pte`)
- **Input:** tokens [1,577,1024] + latent0, latent1, x0_feat, x1_feat, x2_feat (6 inputs, no image)
- **Computes:** reshape tokens → upsample all features → fuse → MultiresConvDecoder → disparity head → monodepth
- **Output:** 7 tensors: monodepth + 5 upsampled encoder features + decoder features
- **Expected time:** ~5–15 seconds (all spatial convolutions, fast)

### Stage 2: Part4bGaussGenerator (`sharp_split_part4b_gauss.pte`)
- **Input:** image [1,3,1536,1536] + monodepth + 6 intermediate feature tensors (8 inputs)
- **Computes:** init_model → feature_model (GaussianDensePredictionTransformer) → prediction_head → gaussian_composer
- **Output:** packed Gaussians [1, N, 14]
- **Expected time:** ~100–130 seconds (bulk of computation)

### Export
```bash
python export_sharp_executorch_split4.py --weights sharp.pt --chunked-part4 --chunked-part4b
python export_sharp_executorch_int8_split4.py --weights sharp.pt --chunked-part4 --chunked-part4b
```

### Android runtime
- Detects `_depth.pte` + `_gauss.pte` → uses two-stage pipeline
- Falls back to single `_part4b.pte` if chunked files not present
- Between stages: destroy depth module, GC, update progress to 55%
- Stage 2: progress ramp 55%→90% over estimated 140s
- Intermediate tensors copied before destroying stage 1 module (safe memory cleanup)

### Append-to-PLY (future)
Stage 2 still produces all Gaussians at once. For true spatial chunking (run stage 2 on tiles), the feature_model is all local convolutions (no attention), so spatial tiling with overlap is theoretically feasible but not yet implemented. Current `writePly` already writes in batches of 1024 Gaussians.

---

## Progress UX (done in code)

We already report progress during Part4b by polling every 2s and ramping from 55% to 90% over an estimated duration. The estimate was 80s but Part4b takes ~120–140s, so the bar reached ~91% and then stayed there for 40–60s. We increased the estimated duration to 140s so the ramp spans the full Part4b time and the bar no longer appears stuck at 91%.
