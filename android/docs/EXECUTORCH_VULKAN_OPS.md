# ExecuTorch Vulkan: Missing Attention Ops

## What “attention ops not implemented” means

When we export the SHARP model with **Vulkan** (`--backend vulkan`), the ExecuTorch **Vulkan partitioner** walks the model graph and, for each operator:

- If the **Vulkan backend has an implementation** for that op → the op is **delegated to the GPU** (Vulkan).
- If not → the op is **left in the portable program** and runs on **CPU** at runtime.

So the same `.pte` file contains:

- Some ops running on **Vulkan (GPU)** (e.g. matmul, linear, conv, many element‑wise ops).
- Other ops running on **portable (CPU)** because the Vulkan backend doesn’t implement them yet.

The SHARP ViT **attention** block uses masking (replace with `-inf`, then `where`, comparisons, etc.). During export you see log lines like:

```text
[Vulkan Partitioner] Due to [no operator implementation], skipping ... aten.where.self(...)
[Vulkan Partitioner] Due to [no operator implementation], skipping ... aten.logical_not.default(...)
[Vulkan Partitioner] Due to [no operator implementation], skipping ... aten.any.dim(...)
[Vulkan Partitioner] Due to [no operator implementation], skipping ... aten.eq.Scalar(...)
[Vulkan Partitioner] Due to [no operator implementation], skipping ... aten.mul.Scalar(...)
```

Those ops are **not** in the Vulkan op registry, so they are not delegated. They stay in the program and run on CPU. That’s why Part1 (and the rest of the pipeline) can still be slow: the attention masking path is largely on CPU.

## Where the Vulkan backend is implemented

The Vulkan backend is **not** part of the Furnit repo. It lives in the **ExecuTorch** project:

- **Repo:** [pytorch/executorch](https://github.com/pytorch/executorch)
- **Vulkan backend path:** [backends/vulkan](https://github.com/pytorch/executorch/tree/main/backends/vulkan)

In the Android app we use the **prebuilt AAR**:

- `org.pytorch:executorch-android-vulkan:1.1.0` (in `android/app/build.gradle`)

So we do **not** ship or build any ExecuTorch or Vulkan C++/shader code in Furnit. All Vulkan op implementations live in the ExecuTorch upstream repo.

## Ops that need Vulkan implementations (for full GPU attention)

From our export logs, the **skipped** ops (used in attention / masking) are:

| Op | Typical use in attention |
|----|---------------------------|
| `aten.where.self` | Apply mask (choose value vs -inf) |
| `aten.logical_not.default` | Invert mask |
| `aten.any.dim` | Reduce mask over dimension |
| `aten.eq.Scalar` | Compare with -inf |
| `aten.mul.Scalar` | Scale (e.g. 1/sqrt(d_k)) |

Adding these to the Vulkan backend would move the attention masking path to the GPU and could reduce per‑patch time.

## Applied in this repo: aten.where.self

**aten.where.self** has a C++ implementation in ExecuTorch (`runtime/graph/ops/impl/Where.cpp`) but was not registered in the Python op registry, so the partitioner skipped it. We:

1. **Registered it in the Python op registry** (clone: `third_party/executorch/backends/vulkan/op_registry.py`).
2. **Added a one-time patch script** for the *installed* executorch (different versions have different `OpFeatures` APIs): run from `android/`:
   ```bash
   python3 patch_executorch_vulkan_where.py
   ```
   Then re-export with `./export_sharp_executorch_vulkan_full.sh`. After the patch, the partitioner delegates **aten.where.self** to Vulkan (attention masking); **logical_not**, **any.dim**, **eq.Scalar**, **mul.Scalar** still run on CPU until implemented upstream.

## How to implement the remaining ops (upstream in ExecuTorch)

Implementing these ops has to be done **in the ExecuTorch repo**, then a new Android build (e.g. a new or custom AAR) would be needed. We cannot add these implementations only inside Furnit.

Rough steps:

1. **Clone ExecuTorch**
   ```bash
   git clone https://github.com/pytorch/executorch.git
   cd executorch
   ```

2. **Vulkan op registry (Python)**  
   In `backends/vulkan/op_registry.py`, new ops are registered with `@update_features(...)` and an `OpFeatures` (dtypes, storage, etc.).  
   You need to register the missing ops (e.g. `aten.where.self`, `aten.logical_not.default`, `aten.any.dim`, `aten.eq.Scalar`, `aten.mul.Scalar`) and map them to the right C++ kernel.

3. **Vulkan runtime (C++)**  
   In `backends/vulkan/runtime/`, each supported op has a C++ implementation (and often a shader). You add:
   - Dispatch from the executor to your new kernel.
   - A Vulkan kernel (and possibly a GLSL/SPIR‑V shader) that does the same math as the PyTorch op.

4. **Build and test**
   - Run ExecuTorch’s Vulkan tests.
   - Export SHARP again with `--backend vulkan` and confirm the partitioner no longer “skips” those ops and that inference uses them on Vulkan.

5. **Android**
   - Build or obtain an `executorch-android-vulkan` AAR that includes your new kernels.
   - In Furnit, update the dependency to that AAR and re‑run the app.

So: “implement it if Vulkan needs it” means **implement these ops in ExecuTorch’s Vulkan backend** (op registry + C++/Vulkan runtime), then use an updated ExecuTorch build in the app. We can’t add that implementation only inside Furnit because we rely on the prebuilt ExecuTorch Vulkan AAR.

## Recommended action plan (from speed/memory analysis)

| Priority | Action |
|----------|--------|
| **P0** | Implement `mul.Scalar` + `eq.Scalar` as Vulkan ops (BinaryScalarOp-style; need C++ + shaders in ExecuTorch). |
| **P0** | Rewrite attention mask to use **additive masking** (add -inf to attention scores instead of boolean mask + where) — avoids `logical_not`, `any.dim`, and `where` entirely. |
| **P1** | Profile with Vulkan profiling to confirm CPU fallback stalls: `adb shell setprop debug.executorch.vulkan.enable_profiling 1`, then capture logcat and inspect timestamps. |
| **P1** | Vulkan memory planning/reuse at export: we pass `skip_memory_planning: False` to `VulkanPartitioner` so Vulkan preprocess runs greedy memory planning. |
| **P2** | Aggressive module destroy/reload between parts: implemented in `ExecutorchSharp.kt` (destroy → GC → `Thread.sleep(500)` before loading next part). |
| **P2** | Check ExecuTorch nightly builds for newly added Vulkan ops. |
| **Optional** | If the model supports it, try smaller patch size (e.g. 256×256 with more patches) to lower peak memory per forward. |

The single most impactful change is **eliminating CPU fallbacks** (P0) — it addresses both speed and memory (round-trips cause sync stalls and duplicate CPU+GPU tensor allocations).

## AOT memory planning and ExecuTorch memory management

ExecuTorch uses **memory planning** at compile time: tensor locations in fixed-size arenas are planned AOT; non-overlapping tensors share memory. This works with Vulkan’s explicit memory API. For full details (greedy vs naive, Vulkan options, runtime mmap) see **[EXECUTORCH_MEMORY_MANAGEMENT.md](EXECUTORCH_MEMORY_MANAGEMENT.md)**.

ExecuTorch’s Vulkan backend uses **ahead-of-time (AOT) memory planning**: at export it analyzes the size and lifetime of all intermediate tensors and makes tensors with **non-overlapping lifetimes share memory**, which minimizes peak usage.

To get that benefit **across** Part1 and Part2 (so the token buffer between them is planned with Part1/Part2 intermediates), we can export **Part1+Part2 as one graph**:

- **Export:** `--combined-part1-part2` produces `sharp_split_part1_part2_combined.pte` (patch → Part1 → Part2 in one program).
- **Runtime:** If that file is present, the app loads it once and runs one forward per patch; the token buffer is an internal intermediate and can share memory via AOT planning. If the combined file is absent, the app falls back to two-phase (Part1 then Part2) with tokens kept in RAM.

The Vulkan full export script uses `--combined-part1-part2`; push the combined .pte with the rest so the app uses the AOT path.

## Vulkan profiling

To confirm where time is spent (e.g. CPU fallback stalls):

```bash
adb shell setprop debug.executorch.vulkan.enable_profiling 1
# Run the app, capture logcat, then clear the property if desired:
adb shell setprop debug.executorch.vulkan.enable_profiling 0
```

Inspect logcat timestamps for large gaps between GPU dispatches; that indicates CPU fallback stalls.

## Learning from ExecuTorch large-model projects

ExecuTorch’s own examples for **large models** (e.g. Llama on Android, Vulkan Llama) use patterns we follow and extend in Furnit:

### What ExecuTorch does for big models

- **Single .pte when possible**  
  Llama demos run one exported model; the Vulkan Llama tutorial uses one `.pte` with the Vulkan partitioner and optional `--vulkan-force-fp16` for lower memory and latency.

- **Quantization**  
  Llama 1B/3B use 4-bit groupwise (SpinQuant / QAT+LoRA) to cut PTE size and RSS (e.g. 1B: ~2.3 GB → ~1.1 GB PTE, ~3.2 GB → ~1.9 GB RSS). We don’t quantize SHARP in the same way, but we reduce peak RAM by splitting and chunking.

- **Vulkan options**  
  Vulkan partitioner supports `force_fp16` (FP32→FP16 inside the backend for speed/memory) and `skip_memory_planning: False` for AOT memory reuse. See [Vulkan Partitioner API](https://github.com/pytorch/executorch/blob/main/backends/vulkan/partitioner/vulkan_partitioner.py) and the [etvk-llama tutorial](https://github.com/pytorch/executorch/blob/main/docs/source/backends/vulkan/tutorials/etvk-llama-tutorial.md).

- **Device capability checks**  
  Llama docs state e.g. 3B unquantized only on devices with enough RAM (e.g. 16 GB). We do the same idea: use combined Part1+2 only when **available system memory ≥ 5.5 GB**, otherwise two-phase to reduce LMK risk.

- **Runner / load once**  
  The LLM C++ runner does `load()` once then `generate()`; they don’t unload between phases. For SHARP we **do** unload (destroy + GC + short sleep) between parts because our pipeline is multi-stage and peak RAM would otherwise exceed device limits.

### How Furnit applies this

| Pattern | In Furnit |
|--------|-----------|
| Prefer single module when safe | Use `sharp_split_part1_part2_combined.pte` when `getAvailSysMemMB() >= 5500`; otherwise two-phase Part1 → Part2. |
| Memory threshold | 5.5 GB threshold so combined is only used when the device has enough free RAM (avoids LMK at ~5.2–5.4 GB RSS). |
| Vulkan memory planning | Export with `skip_memory_planning: False` and optional `force_fp16: True` for lower Vulkan memory/latency. |
| Split when necessary | Part1, Part2, Part3, Part4 (and chunked Part4a_512, Part4a_65, Part4b) with `module.destroy()` + GC + sleep between parts. |
| Chunked execution | Part 4 runs in chunks (512 + 65 tokens then decoder) to keep peak RAM down. |

So we **do** follow the same ideas: single big module when the device can afford it, explicit memory threshold, Vulkan options for planning and FP16, and aggressive split/destroy/chunk when needed to stay under device limits.

## Exact ExecuTorch Vulkan examples (in this repo)

These live under **`android/third_party/executorch/`** (vendored ExecuTorch).

| What | Path | Notes |
|------|------|--------|
| **Vulkan export example** | `examples/vulkan/export.py` | Canonical Vulkan export: uses `VulkanPartitioner(compile_options)` with `force_fp16`, `skip_memory_planning`, `small_texture_limits`. Exports vision models (e.g. MobileNet V2, ResNet) to `*_vulkan.pte`. Run: `python -m examples.vulkan.export -m mv2 -o .` (from executorch root). |
| **Vulkan example README** | `examples/vulkan/README.md` | Usage for basic export, dynamic shapes, bundled `.bpte`, and `--test` (correctness via pybindings). |
| **Llama + Vulkan (export only)** | `examples/models/llama/export_llama_lib.py` | Llama export with Vulkan: `get_vulkan_partitioner(..., vulkan_force_fp16)`. **No LLM runner in Furnit** — the tutorial and C++ runner (`llama_main`) are built from the ExecuTorch repo separately; see `docs/source/backends/vulkan/tutorials/etvk-llama-tutorial.md`. |
| **Vulkan partitioner** | `backends/vulkan/partitioner/vulkan_partitioner.py` | `VulkanPartitioner(compile_options)` API. |
| **Vulkan preprocess (memory planning)** | `backends/vulkan/vulkan_preprocess.py` | Reads `skip_memory_planning`, `force_fp16` from compile spec; runs AOT memory planning when `skip_memory_planning` is False. |

So the **exact** Vulkan example to mirror is `examples/vulkan/export.py`: same `to_edge_transform_and_lower(..., partitioner=[VulkanPartitioner(compile_options)])` and `compile_options` keys we use in Furnit.

**LLM (Llama) ExecuTorch Vulkan:** The *tutorial* and *export* for Llama 3.2 1B/3B with Vulkan exist under `third_party/executorch` (`etvk-llama-tutorial.md`, `export_llama_lib.py`). The Furnit app does **not** ship or run an LLM; it only uses ExecuTorch Vulkan for SHARP room generation. To run Llama on device you would build the ExecuTorch Llama runner (e.g. `llama_main`) from that repo and push the .pte + tokenizer yourself.

## New classes backend (Settings)

The **New classes** inference backend in Settings uses the **ExecuTorch implementation with Vulkan**: same runtime as the main ExecuTorch option (SHARP split .pte files, Vulkan delegate). It is intended for models exported using the GitHub Vulkan approach (`examples/vulkan/export.py` or our `export_sharp_executorch_split4.py` with `--backend vulkan`). If the ExecuTorch Vulkan model files are not present, the app falls back to ONNX for that session.

**RAG-first:** Before changing this path, query `furnit-ml-rag` (chunks `et_vulkan_approach_copy`, `et_new_classes_executorch_vulkan`) and follow `.cursor/skills/executorch-vulkan-approach/SKILL.md`. Do not use Llama/LLM for New classes.

## References

- ExecuTorch Vulkan backend: [github.com/pytorch/executorch/tree/main/backends/vulkan](https://github.com/pytorch/executorch/tree/main/backends/vulkan)
- Op registry: `backends/vulkan/op_registry.py`
- Runtime kernels: `backends/vulkan/runtime/` (BinaryScalarOp.cpp currently only registers `aten.pow.Tensor_Scalar`; `mul.Scalar` / `eq.Scalar` need analogous C++ + shaders)
- Export script (Furnit): `android/export_sharp_executorch_split4.py` (uses `VulkanPartitioner(compile_options)` with memory planning on; no fallback to XNNPACK)
- Llama + Vulkan (in vendored ExecuTorch only; not run by Furnit): `examples/models/llama/`, `docs/source/backends/vulkan/tutorials/etvk-llama-tutorial.md`
