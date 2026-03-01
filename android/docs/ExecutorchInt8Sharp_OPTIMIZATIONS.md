# ExecutorchInt8Sharp.kt – Internal optimization summary

Based on a full read of the file. Use alongside the chunked review (ExecutorchInt8Sharp_REVIEW_CHUNKS.md) when sending to Ultralytics or other reviewers.

---

## Already done (current code)

- **Preprocess:** Bulk `asIntBuffer().get(destInt)` + reusable `patchIntArray` / `imageIntArray`; scale = 1f/255f.
- **Part4a→4b:** Null refs (`out512`/`out65` = empty array) + `System.gc()` before Part4b load.
- **PLY:** `BufferedOutputStream` 256KB, batch size 1024, `linearToSrgb` LUT, reusable `plyWriteByteArray`.
- **Part4b warm-up:** Optional `WARMUP_PART4B` one forward before real run.

---

## Further optimizations to consider

### 1. **Lanczos resize (resizeWithLanczos3)**  
- Allocates `IntArray(sw*sh)` and `IntArray(targetW*targetH)` every call. For 1536² that’s large.
- **Idea:** Reuse buffers (e.g. instance-level `lanczosSrcPixels` / `lanczosDstPixels` for max supported size) or skip Lanczos when `side == targetSize` (already no resize in that path).
- **Alternative:** Use RenderScript or native code for resize if Lanczos remains a hotspot.

### 2. **reshapeToSpatial loop order**  
- Access pattern is `tokens[tokenIdx*C + c]` and `out[c*(H*W) + outBase]`. If C is large, consider loop order for cache (e.g. iterate c in inner loop with contiguous `out` writes).
- Current order (h, w, c) may be fine; profile before changing.

### 3. **mergeCrop**  
- Double loop over `c` and `dy,dx`; 1024 × (cH×cW) iterations. Could unroll or use a single flattened index if it helps the JIT; measure first.

### 4. **Part4b: avoid full copy for writePly**  
- `finalParams.dataAsFloatArray` copies the whole tensor. If ExecuTorch exposes a FloatBuffer or direct buffer, `writePly` could accept a buffer and read in chunks to avoid the 16M+ float copy. Depends on ExecuTorch API.

### 5. **runDecoderChunk**  
- Loads the module from disk every call (twice: 512 and 65). If both Part4a chunks were loaded once and reused, you’d save one load; trade-off is more resident memory.

### 6. **Progress reporting**  
- `report(..., progressCallback)` and `delay(2000)` on the main inference path. Ensure callbacks are cheap (e.g. post to main thread and return quickly) so they don’t add measurable latency.

### 7. **Constants**  
- `PARAMS_PER_GAUSSIAN` (14) used in hot PLY loop; already a const. `SH_C0` and `1.3f` in `lnLut(max(..., 0.001f))` are fine. No change needed unless you introduce more literals.

### 8. **Part1/Part2 patch loop**  
- Each patch: `Bitmap.createBitmap`, `preprocess`, `mod1.forward`, `mod2.forward`, `patch.recycle()`. Preprocess and buffers are already optimized. Remaining cost is model forward; no obvious Kotlin-side win without changing the model or backend.

### 9. **Build column offsets**  
- `buildColumnOffsets` is called once per pipeline; result is small. No need to optimize unless profiling shows otherwise.

### 10. **PLY quaternion norm**  
- `sqrt(rw*rw+rx*rx+ry*ry+rz*rz)` and reciprocal; four mults + sqrt. Could try `1f/sqrt(...)` in one call if the runtime provides a fast inverse-sqrt; minor.

---

## What to ask Ultralytics / external reviewer

- "For ExecuTorch INT8 on Android, is there a recommended way to avoid the full `dataAsFloatArray` copy when writing large outputs (e.g. 1.2M Gaussians) to file?"
- "Any deployment practices for reducing Part4b peak memory (e.g. thread count, chunking, or backend flags)?"
- "Is Lanczos3 resize worth the cost for INT8 ViT input, or is bilinear + potential calibration sufficient?"
- "Warm-up: one full Part4b forward vs minimal dummy input – which is recommended for stable timings on mobile?"

---

## Chunked review file

See **ExecutorchInt8Sharp_REVIEW_CHUNKS.md** for five ~150-line chunks and copy-paste instructions for Ultralytics (or any 150-line reviewer). Use the suggested prompt there before each chunk, then ask for a cross-chunk summary after chunk 5.
