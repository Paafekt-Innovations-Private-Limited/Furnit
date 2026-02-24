# Furnit ML RAG

Android/ExecuTorch/ONNX/LiteRT/NCNN inference knowledge for the Furnit app. Query this RAG before implementing or changing ML backends.

## Do it all (ExecuTorch Vulkan approach)

1. **Index** (after adding or editing chunks in `index_all.py`):
   ```bash
   python furnit-ml-rag/index_all.py
   ```

2. **Query** (learn the approach before coding):
   ```bash
   # Full stack: SHARP philosophy + maths + Vulkan + ExecuTorch
   python furnit-ml-rag/query.py "SHARP philosophy single image 3D Gaussian learned depth"
   python furnit-ml-rag/query.py "Gaussian bell curve 3D splatting parameters"
   python furnit-ml-rag/query.py "Vulkan Android ML compute shaders SPIR-V"
   python furnit-ml-rag/query.py "ExecuTorch Vulkan SHARP combination"
   python furnit-ml-rag/query.py "ExecuTorch Vulkan export approach VulkanPartitioner compile_options"
   python furnit-ml-rag/query.py "New classes ExecuTorch Vulkan"
   python furnit-ml-rag/query.py "AOT ahead-of-time memory planning Vulkan skip_memory_planning buffer reuse"
   python furnit-ml-rag/query.py "30 sec room space dimension ExecuTorch Vulkan Vedic"
   ```

3. **Implement** using the RAG chunks as source of truth. Key chunks:
   - `sharp_philosophy` — single image → 3D, learned depth, Gaussian representation.
   - `sharp_gaussian_maths` — bell curve, 3D Gaussian params (pos, scale, rotation, opacity, color), splatting.
   - `vulkan_latest_android_ml` — Vulkan 1.1+ for ML on Android, compute shaders, SPIR-V, best practices.
   - `et_vulkan_approach_copy` — export + Android flow (mirror `examples/vulkan/export.py`). No Llama.
   - `et_new_classes_executorch_vulkan` — New classes = ExecuTorch Vulkan.
   - `executorch_vulkan_sharp_combination` — how ExecuTorch + Vulkan + SHARP align (same Gaussian pipeline on device).
   - `aot_executorch_vulkan` — AOT (ahead-of-time) memory planning: Vulkan skip_memory_planning=False, greedy planning, buffer reuse.
   - `executorch_30sec_room_space_philosophy` — create room in ~30 sec; space at our disposal (patch grid); ExecuTorch Vulkan only; Vedic Urdhva-tiryagbhyam for grid.

4. **Skill.** When working on ExecuTorch Vulkan (SHARP, New classes), follow `.cursor/skills/executorch-vulkan-approach/SKILL.md`.

## Usage

```bash
# Query (requires chromadb, sentence-transformers)
python query.py "ExecuTorch Vulkan transformer"
python query.py "ONNX NNAPI"

# Query and include relevant Android log snippets (indexes logs first if source given)
python query.py --log-file furnit.log "ExecutorchSharp Part 4 timing"
python query.py --adb-logcat "SharpService inference"

# Index Android logs into RAG (standalone)
python index_all.py --index-logs --from-file furnit.log
python index_all.py --index-logs --from-adb

# Rebuild vector DB from exported JSON (e.g. after git pull)
python query.py --rebuild

# Export current DB to JSON
python query.py --export
```

## Layout

- `index_all.py` — defines CHUNKS, indexes into ChromaDB, writes `data/rag_export.json`; `--index-logs --from-file PATH` / `--from-adb` to index Android logs into `furnit_android_logs` collection
- `query.py` — query, `--rebuild`, `--export`; `--log-file PATH` or `--adb-logcat` to read Android logs and include relevant snippets in results
- `data/rag_export.json` — full chunks (committed); use `--rebuild` to restore vector DB
- `data/vector_db/` — ChromaDB (gitignored); holds `furnit_ml` and `furnit_android_logs` collections

## Layers

executorch, onnx_runtime, litert, ncnn, android_gpu, android_npu, vit_mobile, quantization, sharp_model, performance, export, android_lessons, python_to_android, pytorch_native, sgemm, vedic_maths, ml_fundamentals, etc.
