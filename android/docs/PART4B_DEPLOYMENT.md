# Part4b deployment: backend, precision, and latency (Ultralytics-aligned)

This doc applies Ultralytics/ExecuTorch deployment practices to the SHARP Part4b decoder to reduce the ~114 s CPU bottleneck and improve quality. References: [ExecuTorch integration](https://www.ultralytics.com/blog/deploy-ultralytics-yolo-models-using-the-executorch-integration), [Making YOLO faster](https://www.ultralytics.com/blog/how-to-make-yolo-models-fast-on-your-favorite-chip), [Best practices](https://docs.ultralytics.com/guides/model-deployment-practices/).

---

## 1. ExecuTorch backend & GPU/NPU acceleration

Part4b does **not** automatically run on GPU. You must **partition the model at export time** for the target delegate. If the delegate doesn’t support an op, ExecuTorch falls back to XNNPACK (CPU), which explains ~114 s latency.

- **Vulkan (Android GPU):** Export Part4b with the **Vulkan partitioner** so the `.pte` contains delegate metadata. Use `export_sharp_executorch_split4.py --chunked-part4 --backend vulkan` (or the INT8 chunked export with `--part4b-backend vulkan`) to produce `sharp_split_part4b.pte` (or `sharp_split_part4b_fp16.pte`) built for Vulkan.
- **NNAPI (NPU/DSP):** Use an ExecuTorch build that includes the NNAPI delegate and partition Part4b for NNAPI at export if your pipeline supports it.
- **No separate app build required:** The same AAR can run a Vulkan-partitioned `.pte`; the runtime picks the backend from the program. Ensure the device has Vulkan drivers and the ExecuTorch Vulkan delegate is linked (e.g. `executorch-android` with Vulkan support or a custom build that adds the Vulkan delegate).

**Export commands (chunked Part4, Part4b for GPU):**

```bash
cd android
# Part4b with Vulkan partition (FP32, chunked pipeline)
python export_sharp_executorch_split4.py --weights /path/to/sharp.pt --chunked-part4 --backend vulkan --output-dir executorch_models
# Then push part1/2/3/4a_chunk_512/4a_chunk_65/4b .pte to device
```

---

## 2. Mixed precision (INT8 vs FP16)

Quantizing **positions, scales, and rotations** to INT8 often causes grid-like artifacts or collapsed Gaussians. Ultralytics recommends keeping **decoder heads (Part4b) in FP16**.

- **Recommended:** Part1–Part3 and Part4a chunks: **INT8** (feature extraction). Part4b: **FP16** (or FP32) so coordinate outputs stay numerically stable.
- **Per-channel quantization** for weights and **dynamic quantization** for activations where possible; keep **coordinate outputs in FP16**.
- **Runtime:** The app supports **optional Part4b FP16**: if `sharp_split_part4b_fp16.pte` is present in the models directory, ExecuTorch INT8 pipeline uses it instead of `sharp_split_part4b.pte`, giving mixed INT8 (Parts 1–4a) + FP16 (Part4b) without code changes.

**Exporting Part4b as FP16 (chunked INT8 pipeline):**

```bash
cd android
python export_sharp_executorch_int8_split4.py --chunked-part4 --part4b-fp16 --part4b-backend vulkan
# Or for CPU-only: --part4b-backend xnnpack
```

This produces:

- `sharp_split_part1_int8.pte`, `sharp_split_part2_int8.pte`, `sharp_split_part3_int8.pte`
- `sharp_split_part4a_chunk_512.pte`, `sharp_split_part4a_chunk_65.pte` (INT8)
- `sharp_split_part4b_fp16.pte` (FP16, Vulkan or XNNPACK)

Push all of these; the app will prefer `sharp_split_part4b_fp16.pte` when available.

---

## 3. Operator fusion

Fusion (e.g. Conv+BN+ReLU into one kernel) reduces memory bandwidth and often the main cause of long CPU runs.

- **Export:** Conv+BN is already fused in our export scripts (`fuse_conv_bn()` before export). For ExecuTorch, fusion is applied during `to_edge` / backend lowering; XNNPACK and Vulkan partitioners enable further fusion. No extra flag needed beyond using the partitioner.
- **Best practice:** Ensure **operation fusion is enabled** at export (we do this via the standard ExecuTorch flow with XnnpackPartitioner / VulkanPartitioner).

---

## 4. Static (AOT) memory planning

ExecuTorch uses **Ahead-of-Time (AOT) memory planning**. Dynamic allocations during Part4b forward are expensive on mobile.

- **Export:** We use **greedy memory planning** for Part4b (`use_greedy_memory_planning=True` in `export_pte`), with `alloc_graph_input=False` and `alloc_graph_output=False` so I/O buffers are caller-managed and the plan stays minimal.
- **Runtime:** Avoid creating large temporary tensors inside the inference loop; feed pre-allocated buffers where the API allows. Our pipeline already passes fixed-shape inputs and a single output tensor.

---

## 5. Chunking Part4b (if needed for VRAM/cache)

If Part4b produces ~1.2M Gaussians in one forward and hits **VRAM or GPU cache limits**, consider **chunking the decoder** (e.g. by spatial tiles or token groups) so each chunk fits in cache. This would require **model/export changes** (e.g. splitting the decoder into multiple .pte calls or a tiled decoder). Currently we do not chunk Part4b; the first step is to use Vulkan + FP16 and measure. If OOM or thrashing persists on target devices, chunking is the next lever.

---

## 6. Warm-up

Ultralytics recommends **warm-up runs** before measuring latency so initial setup (JIT, delegate init, etc.) doesn’t skew timings. We have an optional `WARMUP_PART4B` in `ExecutorchInt8Sharp`; set it to `true` for benchmarking. Leave it `false` for normal use so the first inference isn’t doubled.

---

## 7. Checklist

| Item | Action |
|------|--------|
| Part4b on GPU | Export Part4b with `--backend vulkan` (or `--part4b-backend vulkan` in chunked INT8 export). Push the resulting .pte. |
| Mixed precision | Export Part4b as FP16; push `sharp_split_part4b_fp16.pte`. App will use it when present. |
| Operator fusion | Already applied via Conv+BN fuse and ExecuTorch partitioners. |
| AOT memory | Part4b exported with greedy memory planning; no change needed. |
| Chunking Part4b | Only if Vulkan+FP16 still OOM or thrashing; requires model/export redesign. |
| Warm-up | Use `WARMUP_PART4B=true` only for latency measurements. |

---

## References

- [Deploy Ultralytics YOLO with ExecuTorch](https://www.ultralytics.com/blog/deploy-ultralytics-yolo-models-using-the-executorch-integration)
- [Making YOLO models faster on your chip](https://www.ultralytics.com/blog/how-to-make-yolo-models-fast-on-your-favorite-chip)
- [Best practices for model deployment](https://docs.ultralytics.com/guides/model-deployment-practices/)
- [ExecuTorch Vulkan delegate](https://docs.pytorch.org/executorch/stable/native-delegates-executorch-vulkan-delegate.html) (PyTorch docs)
