# Ask Ultralytics: Part4b bottleneck and backend / precision

Copy the block below into Ultralytics to get advice on ExecuTorch backend and FP16 for Part4b.

---

## Paste this into Ultralytics

```
We're running a SHARP-style 3D Gaussian splatting pipeline on Android with ExecuTorch INT8. The full pipeline has Part1+2 (patch encoder), Part3 (image encoder), Part4a (chunked token decoder), and Part4b (final Gaussian decoder). On device, one Part4b forward takes ~114 seconds and is 68.5% of total inference time (~167 s total). Part4b is a single ExecuTorch forward with 7 inputs: tokens, image features, and several latent/feature maps; output is ~1.2M Gaussians (positions, scales, rotations, SH, etc.).

Questions:
1. ExecuTorch backend: What's the recommended way to get Part4b running on Vulkan or GPU (or NNAPI) on Android instead of CPU? We're using INT8 .pte models; do we need a separate Vulkan build or delegate configuration for ExecuTorch to use GPU for this subgraph?
2. Precision: We've seen advice that INT8 on position/scale/rotation decoder heads can cause severe artifacts. Is it recommended to export Part4b (or only its position/scale/rotation heads) in FP16 while keeping the rest INT8, and what's the best practice for mixed INT8/FP16 in ExecuTorch on Android?
3. Any other deployment practices for reducing Part4b latency or improving quality (e.g. operator fusion, chunking the decoder, or preferred quantization schemes for Gaussian splatting decoders)?
```

---

## Optional: add this if they need model context

```
Context: Part4b consumes combined token sequence (590k floats), image tokens, and merged 1x/0.5x/spatial features. Output is a large tensor of Gaussian parameters (we write to PLY). We're already using MMAP load, 2 threads for Part4b to avoid OOM, and no warm-up on the main path.
```
