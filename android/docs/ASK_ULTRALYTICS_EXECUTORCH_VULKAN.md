# Ask Ultralytics / ExecuTorch: Part4b Vulkan sizes_ubo failure

This doc captures a **significant runtime limitation** in the ExecuTorch Vulkan backend (tensor dimensionality) and how to ask upstream for a fix or export workarounds. Ultralytics YOLO and other models are optimized for [ExecuTorch deployment](https://docs.ultralytics.com/integrations/executorch/); high-dimensional models like SHARP can hit backend-specific constraints such as the 4D limit.

**Quick links:** [New ExecuTorch issue](https://github.com/pytorch/executorch/issues/new) · [New Ultralytics issue](https://github.com/ultralytics/ultralytics/issues/new) · [Ultralytics Community Forum](https://community.ultralytics.com)

---

## Recommendations for posting

| Path | Purpose |
|------|--------|
| **Option 1 (ExecuTorch)** | **Runtime fix.** The `(sizes_.size() <= 4)` check is a known constraint in the current Vulkan implementation. Upstream can confirm if it’s a hard limit of the descriptor set layout or a candidate for a patch. |
| **Option 2 (Ultralytics)** | **Export workarounds.** Ultralytics can advise whether the model graph can be changed during export (e.g. `reshape` / `flatten`) so it stays within the 4D limit. See [Exporter Reference](https://docs.ultralytics.com/reference/engine/exporter/). |

---

## Quick technical summary

- **Requirement:** ExecuTorch export currently requires `torch>=2.9.0` per the [Exporter Reference](https://docs.ultralytics.com/reference/engine/exporter/).
- **Backend:** XNNPACK is the default for mobile CPUs; the Vulkan backend is still evolving for complex GPU workloads.
- **Fallback:** If Vulkan stays blocked by the 4D limit, use the **XNNPACK backend** for Part4b, which is well optimized for [mobile CPUs](https://www.ultralytics.com/blog/deploy-ultralytics-yolo-models-using-the-executorch-integration).

For more help: [Ultralytics Community Forum](https://community.ultralytics.com).

---

## Vulkan 4D limit — recommended workarounds

The **Vulkan backend dimensionality limit** (4D) is a known constraint in the current ExecuTorch runtime. While Ultralytics YOLO models are highly optimized for [ExecuTorch deployment](https://docs.ultralytics.com/integrations/executorch/), specific high-dimensional segments like Part4b can trigger the `sizes_ubo` failure.

### Recommended workarounds

| Approach | Description |
|----------|-------------|
| **Switch to XNNPACK** | The [XNNPACK backend](https://www.ultralytics.com/blog/deploy-ultralytics-yolo-models-using-the-executorch-integration) is the default for Ultralytics models. It is highly optimized for mobile CPUs and **natively supports higher-rank tensors (>4D)**, avoiding the Vulkan runtime crash. |
| **Export partitioning** | Use a custom partitioner during export so high-dimensional subgraphs stay on CPU. In the export script, ensure Part4b defaults to **XNNPACK** while other compatible parts use Vulkan. |
| **Model graph modification** | Review the [Ultralytics Exporter Reference](https://docs.ultralytics.com/reference/engine/exporter/) to see if operations can be **flattened or reshaped** to stay within 4 dimensions prior to export. |

### Next steps

1. **Report upstream:** Submit the drafted [ExecuTorch issue](https://github.com/pytorch/executorch/issues/new) to request a fix for the `(sizes_.size() <= 4)` check in `Tensor.cpp`.
2. **Ultralytics support:** Post to the [Ultralytics Community Forum](https://community.ultralytics.com) or [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) for specific guidance on reshaping the model graph for edge compatibility.

---

## No ready-made fix

There is **no** ready-made fix or configuration to bypass the 4D tensor limit in the [ExecuTorch](https://docs.ultralytics.com/integrations/executorch/) Vulkan backend. Official support requires upstream changes to the [ExecuTorch source](https://github.com/pytorch/executorch).

### Recommended actions

| Action | Description |
|--------|-------------|
| **Official support** | Submit the drafted issues to the [ExecuTorch GitHub](https://github.com/pytorch/executorch) and [Ultralytics GitHub](https://github.com/ultralytics/ultralytics/issues) repositories. |
| **Manual patching** | If you patch locally, you must: (1) **Update headers:** In `Tensor.h`, extend `UniformData` (e.g. `ivec4 sizes_v`) to support 6D/8D. (2) **Relax checks:** Remove or update `VK_CHECK_COND(sizes_.size() <= 4)` in `Tensor.cpp` in functions like `sizes_ubo()` and `numel_ubo()`. (3) **Update shaders:** Update GLSL files that define UBO layouts (e.g. `layout_declare_ubo`) to match the new dimensions. |
| **Workaround** | Keep high-dimensional segments like **Part4b** on the [XNNPACK CPU backend](https://docs.ultralytics.com/reference/utils/export/executorch/), which natively handles higher-rank tensors. |

For further assistance: [Ultralytics Community Forum](https://community.ultralytics.com).

### Manual patch points

| Location | What to change |
|----------|----------------|
| **Tensor.h** | `UniformData` struct (sizes, strides, dim_order). |
| **Tensor.cpp** | `get_uniform_data()` and dimension validation logic (`sizes_ubo()`, `dim_order_ubo()`, `strides_ubo()`, `logical_limits_ubo()`, `numel_ubo()`). |
| **GLSL shaders** | Any shader using `ivec4` for coordinate indexing (e.g. under `backends/vulkan/runtime/graph/ops/glsl/`, `layout_declare_ubo(B, "ivec4", ...)`). |

### Local patch applied (ExecuTorch clone)

A manual patch was applied in the **ExecuTorch clone** (e.g. `../executorch` or `EXECUTORCH_DIR`) to support up to **8D** tensors for Part4b Vulkan:

- **Tensor.h:** `UniformData` now uses `UniformSizesStrides` (= `utils::ivec<8>`) for `sizes_v`, `dim_order_v`, `strides_v`; `get_uniform_data()` allows `sizes_.size() <= kTensorDimLimit` (8).
- **Tensor.cpp:** New helper `flip_and_unsqueeze_to_ivec8()`; `UniformData` ctor and `update_metadata()` use it; all `VK_CHECK_COND(sizes_.size() <= 4)` relaxed to `kTensorDimLimit`; `calculate_max_ubo_nbytes()` uses ivec8 size for buffer storage; VulkanImage ctor passes 8-element dim_order/strides.
- **GLSL:** Not changed. Part4b’s Vulkan path (e.g. `view_convert_buffer`) uses `BufferMetadata` (already 8D in `indexing.glslh`), not the per-tensor ivec4 UBOs. Other ops that read `sizes_ubo()` as ivec4 are only used on ≤4D tensors.

After patching, rebuild the Vulkan AAR with `android/scripts/build_executorch_vulkan_aar.sh` and redeploy.

**Runtime:** Part4b Vulkan often hits device OOM (~5GB peak). Per Ultralytics/ExecuTorch (XNNPACK default, Vulkan optional), the app prefers Part4b FP32 (XNNPACK) over Part4b FP16 (Vulkan) so generation completes without LMK kill.

**Export (Ultralytics guidance):** Use a custom partitioner so Part4b is CPU/XNNPACK. This repo does per-module export (separate .pte per part): Part4b is exported with `--part4b-backend xnnpack` (default). That uses a single partitioner for the Part4b subgraph (XnnpackPartitioner). For one-model partitioning you would pass `partitioner=[VulkanPartitioner(), XnnpackPartitioner()]` to `torch2executorch`; here we assign backend per split in `export_sharp_executorch_int8_split4.py`. See [Ultralytics ExecuTorch](https://docs.ultralytics.com/integrations/executorch/) and [compiler delegate and partitioner](https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html).

---

## References

| Topic | Link |
|-------|------|
| ExecuTorch deployment (Ultralytics) | https://docs.ultralytics.com/integrations/executorch/ |
| ExecuTorch source (GitHub) | https://github.com/pytorch/executorch |
| Ultralytics issues (GitHub) | https://github.com/ultralytics/ultralytics/issues |
| Export process / Exporter Reference | https://docs.ultralytics.com/reference/engine/exporter/ |
| XNNPACK / ExecuTorch export utils | https://docs.ultralytics.com/reference/utils/export/executorch/ |
| Deploy YOLO with ExecuTorch (XNNPACK, mobile) | https://www.ultralytics.com/blog/deploy-ultralytics-yolo-models-using-the-executorch-integration |
| Ultralytics Community Forum | https://community.ultralytics.com |

---

## Issue drafts (copy-paste)

Use the text below to open a GitHub issue. **Option 1** = ExecuTorch (runtime fix). **Option 2** = Ultralytics (export guidance).

---

## Option 1: ExecuTorch (pytorch/executorch) — request support for >4D tensors in Vulkan

**Where:** https://github.com/pytorch/executorch/issues/new  

**Title:** Vulkan backend: Part4b FP16 fails with `(sizes_.size() <= 4) is false` in Tensor.cpp sizes_ubo

**Body:**

```
**Environment**
- ExecuTorch built from source (release/1.1) with EXECUTORCH_BUILD_VULKAN=ON
- Android arm64-v8a, custom AAR with view_convert_buffer_float_half shader fix
- Model: SHARP Gaussian decoder (Part4b) exported with VulkanPartitioner, FP16

**What happens**
Part4b loads and starts on Vulkan but then fails at runtime:

```
Part4b FP16 failed (Exception raised from sizes_ubo at 
executorch/backends/vulkan/runtime/api/containers/Tensor.cpp:1050: 
(sizes_.size() <= 4) is false! ). Falling back to FP32 Part4b.
```

**Cause**
In `backends/vulkan/runtime/api/containers/Tensor.cpp`, `sizes_ubo()` (and related UBO accessors) enforce `VK_CHECK_COND(sizes_.size() <= 4)`. Our Part4b graph has at least one tensor with more than 4 dimensions, so the check fails.

**Question**
- Is the 4D limit intentional (e.g. Vulkan UBO layout)? 
- Is there a supported way to run graphs with tensors of rank > 4 on the Vulkan backend, or a plan to support it (e.g. larger sizes_v / uniform layout)?
- If not, is this documented somewhere so export pipelines can avoid >4D when targeting Vulkan?

**Relevant code**
- `Tensor.cpp` around 1049–1073: `sizes_ubo()`, `dim_order_ubo()`, `strides_ubo()`, etc. all have `VK_CHECK_COND(sizes_.size() <= 4)`.
- `Tensor.h` around 705: `get_uniform_data()` same check.
```

---

## Option 2: Ultralytics (ultralytics/ultralytics) — SHARP + ExecuTorch Vulkan

**Where:** https://github.com/ultralytics/ultralytics/issues/new  

**Title:** SHARP ExecuTorch Vulkan export: Part4b fails on device (sizes_.size() <= 4) in Vulkan backend

**Body:**

```
We export SHARP (Gaussian scene reconstruction) to ExecuTorch with Vulkan for the decoder (Part4b) to run on Android GPU. Export succeeds with VulkanPartitioner; on device we hit a runtime limit in ExecuTorch’s Vulkan backend:

```
Exception ... Tensor.cpp:1050: (sizes_.size() <= 4) is false!
```

So Part4b has at least one tensor with more than 4 dimensions, and ExecuTorch’s Vulkan runtime currently only supports up to 4D in its sizes UBO.

**Questions**
- Is there an official or recommended way to export SHARP (or the Gaussian decoder / Part4b) for ExecuTorch Vulkan on Android that avoids >4D tensors (e.g. reshape/split in the export script)?
- Do you have any guidance or plans for SHARP + ExecuTorch Vulkan (e.g. which ops to keep on CPU, or a known-good export config)?
- If the decoder must use >4D, should we treat Vulkan as unsupported for this model and stick to XNNPACK/CPU for Part4b?

**Context**
- We use a 4-part split (encoder parts 1–3 INT8, Part4a INT8, Part4b FP16). Part4b is exported with Vulkan for GPU; the rest is XNNPACK. We already fixed the missing `view_convert_buffer_float_half` shader by building ExecuTorch from source; the remaining blocker is this 4D limit in the Vulkan backend.
```

---

## Deployment practices alignment (Ultralytics-style review)

The Android ExecuTorch SHARP implementation aligns with [Ultralytics deployment practices](https://docs.ultralytics.com/guides/model-deployment-practices/):

| Area | Implementation |
|------|----------------|
| **Memory** | `System.gc()` and `Module.destroy()` between chunks; AOT memory planning via export (greedy/Vulkan options). |
| **Preprocessing** | Pixel normalization `1f / 255f` in `preprocess`; matches [ExecuTorch usage](https://docs.ultralytics.com/integrations/executorch/#usage). |
| **Coordinates** | 1:1 coordinate space in `writePly`; no aspect scaling that would mismatch training distribution ([Best Practices](https://docs.ultralytics.com/guides/model-deployment-practices/#check-data-consistency)). |
| **Efficiency** | LUTs for `linearToSrgb` and `lnLut` to reduce CPU overhead during PLY serialization. |

**Recommendations to verify:**

- **XNNPACK:** Export Part1–Part4a (and fallback Part4b) with [XnnpackPartitioner](https://docs.ultralytics.com/reference/utils/export/executorch/#function-ultralyticsutilsexportexecutorchtorch2executorch) for optimal CPU on Android.
- **Metadata:** Ensure `imgsz` (1536) matches any `metadata.yaml` produced at export for input consistency ([Output structure](https://docs.ultralytics.com/integrations/executorch/#output-structure)).

More: [Ultralytics Community Forum](https://community.ultralytics.com).

---

## Export–Inference sync (production refactor)

To keep **Export (Python)** and **Inference (Kotlin)** in sync for production:

### 1. Export (Python) — SHARP, not YOLO

We use **custom SHARP export scripts**, not `YOLO("yolo26n.pt").export(...)`. Each sub-module is exported separately.

| What | Script / behaviour |
|------|--------------------|
| **Part1–Part4a (INT8)** | `export_sharp_executorch_int8_split4.py` — `XnnpackPartitioner` + `XNNPACKQuantizer`; Part4a as two chunks (512 + 65 tokens). |
| **Part4b** | Same script: `--part4b-backend xnnpack` (default) or `vulkan`; optional `--chunked-part4b` → `sharp_split_part4b_depth.pte` + `sharp_split_part4b_gauss.pte`. |
| **AOT memory** | Greedy `MemoryPlanningPass` after `to_edge`; Vulkan export uses `VulkanPartitioner(compile_options)` with `skip_memory_planning: False`. |
| **imgsz** | 1536 (fixed in export and Kotlin `IMAGE_SIZE`). |

Reference: [Ultralytics ExecuTorch export](https://docs.ultralytics.com/reference/utils/export/executorch/).

### 2. Inference (Kotlin)

| What | Implementation |
|------|----------------|
| **Load** | `Module.load(path, Module.LOAD_MODE_MMAP)` for all parts. |
| **Chunked decode** | `runDecoderChunk(name, data, count)` loads → forward → `mod.destroy()` + `System.gc()` to avoid OOM. |
| **Part4b threads** | `part4bThreadCount() = 2` (stability; higher can trigger Vulkan device lost). |
| **Post-process** | LUT-based `linearToSrgb` / `lnLut` in `writePly`; 1:1 coordinates, no aspect scaling. |
| **Metadata** | If you add `metadata.yaml` to `assets/models`, ensure `imgsz: 1536` matches runtime. [Output structure](https://docs.ultralytics.com/integrations/executorch/#output-structure). |

The full Kotlin implementation lives in `ExecutorchInt8Sharp.kt` (patch merging, Part1–Part4b pipeline, Vulkan device-lost fallback to FP32 single Part4b). Do not replace it with a YOLO-style skeleton; the review snippet was illustrative.

### 3. Checklist

- [x] Part1–Part4a exported with **XnnpackPartitioner** (INT8).
- [x] Part4b exported per backend (XNNPACK or Vulkan); chunked Part4a/4b as separate .pte.
- [x] Kotlin: **LOAD_MODE_MMAP**, **destroy()** / **gc()** between chunks, **part4bThreadCount = 2**.
- [ ] Optional: add `metadata.yaml` to assets with `imgsz: 1536` if your tooling expects it.

[ExecuTorch integration guide](https://docs.ultralytics.com/integrations/executorch/).

---

## Chunked Part4b with XNNPACK (recommended on device)

Chunked Part4b was first tried with **Vulkan** (depth + gauss .pte); on many devices that path hits **Vulkan device lost** (GPU OOM/timeout). Exporting the same two stages with **XNNPACK** gives a CPU path with no Vulkan dependency and no device-lost fallback.

### Export

From `android/` (default weights and output dir):

```bash
python3 export_sharp_executorch_int8_split4.py --chunked-part4b --part4b-fp16 --part4b-backend xnnpack
```

This produces (among the full split):

- `executorch_int8_models/sharp_split_part4b_depth.pte` (~149 MB, FP16 XNNPACK)
- `executorch_int8_models/sharp_split_part4b_gauss.pte` (~29 MB, FP16 XNNPACK)

No app changes: the app already uses `sharp_split_part4b_depth.pte` and `sharp_split_part4b_gauss.pte` when present; it does not care whether they were built with Vulkan or XNNPACK.

### Push to device

Replace only the chunked Part4b files (or push the full set):

```bash
cd android
adb push executorch_int8_models/sharp_split_part4b_depth.pte /storage/emulated/0/Android/data/com.furnit.android/files/models/
adb push executorch_int8_models/sharp_split_part4b_gauss.pte /storage/emulated/0/Android/data/com.furnit.android/files/models/
```

If you previously had Vulkan chunked Part4b on device, these XNNPACK builds overwrite them; the next run will use chunked Part4b on CPU (XNNPACK) and should not hit device lost.

---

## Layer-specific (partial) tiling for Part4b (not done)

Tiling does **not** have to be applied to the whole model. As in [Ultralytics SAM / ViT](https://docs.ultralytics.com/reference/models/sam/modules/blocks/), **window_partition** and **window_unpartition** can wrap **specific layers or blocks** (e.g. high-resolution attention) so that only the heavy ops run per-tile; residuals stay outside to keep gradient flow and global context elsewhere.

**Why it wasn’t done for SHARP Part4b:**

- **Codebase:** SHARP is `ml-sharp` (ViT + UNet-style decoders), not Ultralytics. We’d need to (1) locate the memory-heavy blocks in the Part4b depth decoder and Gaussian generator (e.g. in `sharp/models/blocks.py`, `decoders/`, `encoders/monodepth_encoder.py`), (2) inject `window_partition` / `window_unpartition` around those ops in Python, (3) keep shortcut/residual paths outside the partition so the rest of the model stays global, (4) re-export to ExecuTorch and re-validate.
- **Effort:** That’s a non-trivial model-code and export change; we prioritized the 2-stage split and CPU fallback instead.
- **Risk:** Window size and padding must match what the model expects; wrong placement can change behavior or hurt quality.

**Implemented (optional):** A **4×4 tiled Part4b** pipeline (upsample → 16× tile decoder → gauss) is available. Use **16 pieces = 16 files**, all **XNNPACK only** (no Vulkan):

1. **Export (16 pieces, 16 files, XNNPACK only):**
   ```bash
   cd android
   python3 export_sharp_executorch_int8_split4.py --chunked-part4 --chunked-part4b --part4b-fp16 --part4b-tiled --part4b-backend xnnpack
   ```
   Produces `sharp_split_part4b_upsample.pte`, **16 tile files** `sharp_split_part4b_tile_00.pte` … `sharp_split_part4b_tile_15.pte`, and `sharp_split_part4b_gauss_tiled.pte` (all XNNPACK). The app loads one tile file per tile and unloads it after each forward to minimize peak memory.
2. **Push to device** (16-piece: 1 upsample + 16 tile files + 1 gauss):
   ```bash
   cd android
   adb push executorch_int8_models/sharp_split_part4b_upsample.pte /storage/emulated/0/Android/data/com.furnit.android/files/models/
   for i in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15; do
     adb push executorch_int8_models/sharp_split_part4b_tile_$i.pte /storage/emulated/0/Android/data/com.furnit.android/files/models/
   done
   adb push executorch_int8_models/sharp_split_part4b_gauss_tiled.pte /storage/emulated/0/Android/data/com.furnit.android/files/models/
   ```
3. **App:** In **Settings → Developer**, turn on **Part4b tiled (experimental)**. When the setting is on and upsample + all 16 tile files + gauss_tiled exist, the app uses the 16-piece pipeline (load/unload per tile) on CPU (XNNPACK, no Vulkan). If only `sharp_split_part4b_tile_decoder.pte` is present (no 16 tile files), the app falls back to loading that single file and running it 16 times.

**References:** [SAM modules utils (window_partition / window_unpartition)](https://docs.ultralytics.com/reference/models/sam/modules/utils/), [Block forward (local windowing)](https://docs.ultralytics.com/reference/models/sam/modules/blocks/).

---

## Why single Part4b (XNNPACK) works but the tiled split OOMs

Both use **XNNPACK**; the difference is **when** and **how much** memory is allocated.

| Path | What runs | Peak memory |
|------|------------|-------------|
| **Single Part4b** (one .pte) | One graph: upsample **and** depth decoder **and** Gaussian head in a single forward. | The runtime/compiler can do **memory planning across the whole graph**: each upsample output can be consumed by the decoder and freed before the next one is materialized, or buffers can be reused. So peak is lower. |
| **Tiled split** (upsample.pte → 16 tiles → gauss.pte) | **Upsample** runs as a **separate** forward that **returns 5 full‑res tensors at once** (e.g. 256×768×768, 256×384×384, 512×192×192, …). All 5 must exist in memory at the end of that one forward. | The upsample module has **no visibility** into the tile decoder—it just outputs 5 big tensors. So in **one shot** you allocate ~600 MB+ for those outputs. That single allocation can OOM on devices with limited free RAM (~100 MB). |

So the split **moves** the memory cost: the **tile decoder** stage is low‑memory (16 small tiles), but the **upsample** stage is **front‑loaded** and must materialize all 5 full‑resolution feature maps in one forward. The single file never has to hold all 5 at once in the same way because the graph is one piece and the backend can plan/fuse/reuse.

**Practical fix:** Use single Part4b on low‑RAM devices (or push `sharp_split_part4b.pte` for fallback). To make the tiled path work on the same devices, the upsample would need to be restructured (e.g. output one feature at a time and run tiles incrementally, or run at lower resolution and upsample per‑tile).

---

## Part4b tiled (upsample) OOM — Ask Ultralytics

When using the **16-file tiled Part4b** pipeline (upsample → tile decoder → gauss), the app can go **black** at **"Starting Part4b upsample..."**. That indicates failure during either:

- **Load** of `sharp_split_part4b_upsample.pte` (e.g. mmap or alloc failure), or  
- **Forward** of the upsample model, which still produces **full-resolution** encoder features (e.g. 256×768×768) in one run, so **peak memory** can reach ~1–2 GB and trigger OOM on low-RAM devices.

**Runtime behaviour (this repo):**

- Added **granular logging**: after upsample `Module.load` and after `modUpsample.forward` so logcat shows whether the failure is load vs forward.
- **Fallback:** If the tiled path throws (any `Throwable`, including OOM), the app now tries **single Part4b** (`sharp_split_part4b.pte` or FP32 equivalent) when present, so the user gets a result instead of a black screen. Ensure the single Part4b .pte is on device if you want this fallback.

**What to ask Ultralytics / ExecuTorch:**

- For **ExecuTorch on Android**: Is there a recommended pattern to run a **decoder with high-resolution upsampling** (one forward that outputs several large 4D tensors) without OOM—e.g. **chunked execution**, **tiling the upsample stage**, or **preferring XNNPACK over Vulkan** for this stage?
- For **export**: Can the upsample stage be split into smaller subgraphs (e.g. per-feature upsample) so that each forward has lower peak memory, or is the recommended approach to keep the full decoder on CPU (XNNPACK) and avoid the single large upsample on device?

**Where to post:** [Ultralytics Community Forum](https://community.ultralytics.com) or [ExecuTorch GitHub Discussions](https://github.com/pytorch/executorch/discussions), with a short description of the pipeline (Part4b = upsample + tile decoder + gauss) and that the upsample forward is the current memory bottleneck.

---

## After you post

- **ExecuTorch:** If they confirm the limit or add support for >4D, we can either patch our build or wait for an upstream release.
- **Ultralytics:** If they suggest a different export (e.g. reshape to 4D or keep Part4b on CPU), we can adjust the export or docs accordingly.
- **Further debugging:** [Ultralytics Community Forum](https://community.ultralytics.com).
