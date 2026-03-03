# Ask Ultralytics: Part4b chunking failed — validate approach

## Context

We tried splitting Part4b (Gaussian decoder) into two sequential ExecuTorch stages on Android. It failed. We need Ultralytics to validate our analysis and suggest a viable alternative.

---

## Paste this into Ultralytics

```
We have a SHARP-style 3D Gaussian splatting pipeline on Android with ExecuTorch. Part4b is the Gaussian decoder: from tokens + image + 5 feature maps → ~1.2M Gaussians [1, N, 14]. On device it takes ~120–140 seconds.

Following your earlier advice to chunk Part4b, we tried splitting it at the monodepth boundary into two sequential .pte modules:

STAGE 1 (Part4bDepthDecoder):
  Input: tokens [1,577,1024] + latent0 [1,1024,96,96] + latent1 [1,1024,96,96] + x0 [1,1024,96,96] + x1 [1,1024,48,48] + x2 [1,1024,24,24]
  Computes: reshape tokens → upsample all features → fuse → MultiresConvDecoder → disparity head → monodepth
  Output: monodepth [1,2,1536,1536] + 5 upsampled encoder features + decoder features (7 tensors total)

STAGE 2 (Part4bGaussGenerator):
  Input: image [1,3,1536,1536] + monodepth + 6 intermediate feature tensors (8 inputs)
  Computes: init_model → feature_model (GaussianDensePredictionTransformer) → prediction_head → gaussian_composer
  Output: [1, N, 14] Gaussians

RESULT: Stage 1 alone took 87 seconds (not the 5-15s we expected). Then the phone crashed (OOM) when copying the intermediate tensors. The intermediates are massive:
  - feat0 (latent0_up): [1, 256, 768, 768] = ~604 MB
  - decoder_features: [1, 256, 768, 768] = ~604 MB
  - monodepth: [1, 2, 1536, 1536] = ~38 MB
  - feat1–feat4: [1,256,384,384] + [1,512,192,192] + [1,1024,96,96] + [1,1024,48,48] = ~273 MB
  Total intermediates: ~1.5 GB — impossible to copy on a mobile device with 6-8 GB RAM.

The upsampling ops (ConvTranspose2d from 24×24/48×48/96×96 → 768×768) and the MultiresConvDecoder are the bulk of the computation, not init_model/feature_model as we assumed.

We also cannot chunk by token range (your earlier suggestion) because the first operation in Part4b reshapes tokens [1,577,1024] into a spatial feature map [1,1024,24,24] — all 577 tokens are consumed immediately.

Questions:
1. Given the actual architecture (reshape tokens → upsample → decoder → monodepth → init_model → feature_model → prediction_head → gaussian_composer), is there a viable split point where intermediate tensors are small enough for mobile? Or is this model fundamentally not chunkable without changing the architecture?

2. The feature_model (GaussianDensePredictionTransformer) uses only local convolutions (Conv2d, ConvTranspose2d, residual blocks) — no attention. Could we tile the spatial computation (e.g. process 384×384 tiles with overlap instead of the full 768×768 or 1536×1536)? What overlap/padding would be needed given 3×3 kernels and the number of decoder levels?

3. Since the upsampling to 768×768 is a major cost, could we export a lower-resolution variant (e.g. upsample to 384×384 instead of 768×768) that produces fewer Gaussians but runs much faster? Is this a model config change or architecture change?

4. If chunking is not viable, what is the recommended approach for a ~140s single forward on mobile? Is the computation bound by CPU (XNNPACK) and would Vulkan/GPU help for these Conv2d-heavy operations? Any ExecuTorch-specific optimizations for large spatial decoders?
```

---

## Our analysis (for reference)

### Why the split failed

The Part4b forward method:
```python
def forward(self, tokens, image, latent0, latent1, x0, x1, x2):
    # (1) Reshape tokens to spatial — consumes all 577 tokens immediately
    x_lowres = reshape(tokens)  # [1,1024,24,24]

    # (2) Upsample all features to high resolution — THIS IS EXPENSIVE
    latent0_up = upsample_latent0(latent0)    # [1,1024,96,96] → [1,256,768,768]
    latent1_up = upsample_latent1(latent1)    # [1,1024,96,96] → [1,256,384,384]
    x0_up = upsample0(x0)                     # [1,1024,96,96] → [1,512,192,192]
    x1_up = upsample1(x1)                     # [1,1024,48,48] → [1,1024,96,96]
    x2_up = upsample2(x2)                     # [1,1024,24,24] → [1,1024,48,48]
    x_lowres_up = upsample_lowres(x_lowres)   # [1,1024,24,24] → some size
    x_fused = fuse_lowres(cat(x2_up, x_lowres_up))

    # (3) MultiresConvDecoder on 5 feature maps — EXPENSIVE at 768×768
    encoder_features = [latent0_up, latent1_up, x0_up, x1_up, x_fused]
    decoder_features = decoder(encoder_features)  # → [1,256,768,768]

    # (4) Disparity head → monodepth
    disparity = head(decoder_features)  # → [1,2,1536,1536]
    monodepth = 1.0 / disparity.clamp(...)

    # --- We split HERE, but intermediates above are ~1.5 GB ---

    # (5) Init model — produces base Gaussians from image + depth
    init_output = init_model(image, monodepth)

    # (6) Feature model — GaussianDensePredictionTransformer
    #     Has its OWN decoder that processes output_features (the 6 large tensors)
    image_features = feature_model(init_output.feature_input, encodings=output_features)

    # (7) Prediction head + Gaussian composer → [1, N, 14]
    delta_values = prediction_head(image_features)
    gaussians = gaussian_composer(delta_values, init_output.base_values, init_output.global_scale)
    return cat([positions, opacities, scales, quaternions, colors], dim=-1)
```

### Key findings
- Steps (2)+(3) are the bottleneck: upsampling to 768×768 and running the decoder at that resolution
- The feature_model at step (6) has its OWN MultiresConvDecoder that ALSO processes the 6 large feature tensors — so even if we could pass them, stage 2 would also be slow
- Any split point after the upsampling produces ~1.5 GB of intermediates
- Any split point before the upsampling means stage 2 would need to redo the upsampling (duplicate weights + computation)

### What works now
- Single Part4b forward: ~120–140 seconds, produces correct output
- Progress bar: ramps from 55%→90% over estimated 140s (no longer stuck at 91%)
- The code falls back to single Part4b when chunked files are not present on device

### Device
- Samsung Galaxy S24 (Snapdragon 8 Gen 3), 8 GB RAM
- ExecuTorch with XNNPACK backend
- Part4b is FP32 (INT8 quantization breaks decoder skip connections)

---

## Ultralytics response

### 1. No viable linear split point
Confirmed: no "clean" architectural split point exists after the upsampling stage. Splitting before upsampling forces stage 2 to re-compute or store the heavy weights, leading to same OOM/latency. **This model is fundamentally difficult to chunk linearly.**

### 2. Spatial tiling (most promising)
Since `GaussianDensePredictionTransformer` uses only local convolutions (Conv2d/ConvTranspose2d), we can tile the 768x768 or 1536x1536 feature maps into **384x384 tiles with 16-32px overlap** per side. Use `torch.nn.functional.unfold` or manual slicing. Keeps peak memory to one tile's activations.

### 3. Lower-resolution export
Reducing input resolution (e.g. to 768x768) is a **model configuration change**. Quadratic scaling means ~4x faster. Fewer Gaussians and lower fidelity, but fastest way to hit latency targets without rewriting the forward logic.

### 4. Hardware + precision optimizations
- **Vulkan/GPU:** For Conv2d-heavy decoders, Vulkan often outperforms XNNPACK (CPU) on Snapdragon 8 Gen 3.
- **AOT memory planning:** Ensure greedy/AOT memory planning is active to reuse buffers between upsampling layers.
- **FP16:** If INT8 fails, try FP16. Halves memory footprint (1.5 GB → 750 MB) and is natively accelerated by GPU/NPU.

### References
- [ExecuTorch Vulkan](https://docs.pytorch.org/executorch/stable/build-run-vulkan.html)
- [Ultralytics ExecuTorch integration](https://docs.ultralytics.com/integrations/executorch/)
- [Ultralytics Discord](https://discord.com/invite/ultralytics) for architectural changes

---

## Action items and results

### 1. FP16 Part4b -- DONE
Fixed dtype mismatch in initializer.py (torch.ones/tensor/linspace/arange/empty without dtype caused FP32 contamination).
- Fixed: `disparity_factor = torch.ones(...)` → `torch.reciprocal(disparity.clamp(...))`
- Fixed: initializer.py tensor creation with `dtype=depth.dtype`
- Fixed: composer.py `delta_factor` tensor with `dtype=dtype`
- Export: `python export_sharp_executorch_int8_split4.py --chunked-part4 --part4b-fp16`
- Result: `sharp_split_part4b_fp16.pte` (178 MB with XNNPACK, 89 MB with Vulkan)
- Pushed to device; app prefers FP16 when available

### 2. Vulkan backend -- DONE (per PyTorch ExecuTorch docs)
- Export: use `VulkanPartitioner` in export script (see https://docs.pytorch.org/executorch/stable/android-vulkan.html).
- Android: add **executorch-android-vulkan** dependency (org.pytorch:executorch-android-vulkan:1.1.0) so Vulkan .pte runs on GPU. Without it, Vulkan-built Part4b fails with "Resource not found".
- Export command: `PYTHONPATH=third_party/executorch:$PYTHONPATH python export_sharp_executorch_int8_split4.py --chunked-part4 --part4b-fp16 --part4b-backend vulkan`
- Result: `sharp_split_part4b_fp16.pte` 89 MB (Vulkan FP16). Push to device; app prefers FP16 when present and now has Vulkan runtime.

### 3. Lower resolution -- NOT VIABLE (current SHARP)
Added `--image-size` flag to export script. A full export at `--image-size 768` was attempted and **fails**: Part3 (image encoder) uses a fixed `patch_embed` expecting input height 384 (1536/4). At 768, the downscaled input is 192, so the model asserts. **768×768 would require a different ViT / architecture or retraining**; not supported by the current SHARP checkpoint.
- App remains at IMAGE_SIZE=1536 only.

### 4. Spatial tiling -- FUTURE
Most complex but best long-term. Feature model uses only local convolutions, so 384x384 tiles with 16-32px overlap should work.
