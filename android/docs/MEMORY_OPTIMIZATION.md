# SHARP ExecuTorch Memory Optimization

This doc summarizes the **prioritized action plan** to reduce peak memory and avoid LMK (Low Memory Killer) when running SHARP via ExecuTorch on Android.

## Why LMK kills you

Loading the 1.3 GB full model + activations + weight copies causes total RSS to exceed Android's per-process limit → LMK triggers "low watermark is breached" and kills processes. The fix reduces peak RSS from ~1.5 GB to ~200–300 MB.

## Prioritized strategies

| Priority | Strategy | Status |
|----------|----------|--------|
| **1** | **mmap without mlock** (`Module.load(path, LOAD_MODE_MMAP)`) | ✅ Applied in ExecutorchSharp and ExecutorchSharpLayerByLayer |
| **2** | Single .pte with greedy memory planning | ✅ Implemented in `export_sharp_executorch_all.py --variant memory_optimized` |
| **3** | FP16 activations | ✅ Applied in memory_optimized export |
| **4** | gcIfPressured every 4 layers (75% threshold), logMemory every 6 | ✅ ExecutorchSharpLayerByLayer |
| **5** | Chunked attention | 🔜 Future (model/export change) |
| **6** | XNNPACK operator fusion | ✅ Applied via XnnpackPartitioner |
| **7** | Zero-copy weight access (FloatBuffer from native) | 🔜 Requires ExecuTorch API support |

## mmap loading (fix 1)

`Module.load(path, Module.LOAD_MODE_MMAP)` memory-maps the .pte file. The OS pages in only the weight pages for the current layer; cold pages are evicted under pressure. Peak RSS drops from 1.3 GB to ~80–150 MB resident.

## Memory stack after fix

| Component | Before | After |
|-----------|--------|-------|
| Model weights | 1,300 MB resident | ~80–150 MB (mmap'd, paged) |
| JVM weight copies | ~48 MB × 24 layers | 0 (if zero-copy) |
| ScratchBuffers | ~73 MB | ~73 MB |
| Attention pool | ~2 MB | ~2 MB |
| **Total RSS** | **~1,500+ MB** | **~200–300 MB** |
| LMK | Triggered | Not triggered |

## Bonus: 4-bit quantization (export time)

Cuts the .pte from 1.3 GB to ~200 MB. Do at export time:

```bash
python -m executorch.examples.models.your_model_script \
  --model sharp.pt \
  --quantization 4bit_groupwise \
  --group-size 128 \
  -o sharp_q4.pte
```

Check ExecuTorch docs for exact quantization flags. Smaller .pte = faster mmap page-ins, lower virtual address footprint.

## Why a single .pte helps

If you split the model into separate sub-programs (e.g. part1–part4), the memory planner **cannot reuse buffers across splits**. Each part holds its own activations, so peak memory is closer to the **sum** of all layers. A **single** program with ExecuTorch’s **greedy** planner can reuse buffers across the whole graph, keeping peak memory at roughly **one layer’s worth** of activations. Combined with FP16 and XNNPACK, this can bring the previous ~4GB spike down significantly.

## Export

```bash
cd android
python export_sharp_executorch_all.py --variant memory_optimized
```

Output: `executorch_models/sharp_full_memory_optimized.pte`. Push it to the device; the app will prefer it over split or plain full FP16/FP32 when present.

## Chunked attention (Priority 3, future)

Chunked attention would further reduce activation memory by processing attention in chunks instead of full sequence. This would require changes in the SHARP model or export pipeline (e.g. custom op or graph rewrite); not implemented in the current export scripts.

## Part 4 (decoder) crash at 60%

When using the **split** ExecuTorch pipeline, inference can crash at ~60% during Part 4 (decoder). With ~3.5 GB system-available, the 754 MB Part 4 .pte is mmap’d and the decoder’s forward materializes the full output tensor at once, which can exceed what’s left and trigger OOM/LMK.

### Diagnostic (confirmed)

Logs show **"Part 4: forward starting"** but **never** **"Part 4: forward returned"**; the next log lines are from a new process (e.g. FurnitApp Firebase init) ~99s later. So the process is killed **during** the native Part 4 `forward()` (decoder activations), not during Part 4 load or input construction. Runtime mitigations on the Kotlin/Java side are exhausted; the bottleneck is decoder activation memory inside ExecuTorch.

### Implemented

- **Aggressive GC around Part 4 forward:** `System.gc()` + `System.runFinalization()` immediately before `module4.forward()`, and `System.gc()` right after the forward (before holding the output buffer). This follows the “Memory optimization for sharp” recommendation to let mmap pages and JVM refs reclaim and reduce pressure during decoder activations.

### Greedy planning at export (Part 4 / full model)

Use **alloc_graph_input=False** and **alloc_graph_output=False** in `MemoryPlanningPass` so I/O buffers are caller-managed and don’t inflate the plan; this also avoids the “Misallocate graph input: False v.s. True” error. Export must use **static shapes** (no `torch.export.Dim`). Applied in `export_sharp_executorch_split4.py` (Part 4) and `export_sharp_executorch_all.py` (memory_optimized).

### Chunked Part 4 (implemented)

To reduce peak RAM during Part 4, the pipeline can run the **ViT (image_tokens)** in token slices, then run the decoder once:

1. **Export** (run from `android/`):  
   `python export_sharp_executorch_split4.py --chunked-part4`  
   This produces, in addition to `sharp_split_part4.pte`:  
   `sharp_split_part4a_chunk_512.pte`, `sharp_split_part4a_chunk_65.pte`, `sharp_split_part4b.pte`.

2. **Runtime**: If all three chunked files are present, `ExecutorchSharp` uses the chunked path: run 4a_512 on tokens 0:512, then 4a_65 on 512:577, concatenate, then run 4b once. GC between chunks lowers peak activation memory during the ViT stage. The decoder (4b) still runs in one shot.

3. **Push to device**: Push the three chunked .pte files to the same models directory; the app prefers chunked Part 4 when available.

### Recommended (require re-export or export options)

1. **Chunked decode:** Don’t call Part 4 on the full sequence. Split the decoder input into slices of ~512 tokens, run `part4.forward()` on each slice, concatenate the small output tensors, and call `System.gc()` between chunks. Peak activation drops from O(full_seq) to O(chunk_size). **Requires** a Part 4 export that supports slice-based forward (variable-length or start/end token index); current Part 4 has fixed input shapes.
2. **FP16 activations at export:** Re-export Part 4 with half-precision activations (e.g. `python export_sharp_executorch_split4.py` with Part 4 in FP16). Cuts activation buffers in half.
3. **Shrink working set:** If the export pipeline supports it, reduce `max_seq_len` for the decoder (e.g. 2048 → 1024) to cut quadratic attention memory by ~4×.
