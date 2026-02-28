#!/usr/bin/env python3
"""
Furnit ML RAG - Index all Android ML inference knowledge into ChromaDB.

Layers:
  - executorch: ExecuTorch Vulkan/XNNPACK backend, quantization, deployment
  - onnx_runtime: ONNX Runtime NNAPI EP, GPU, optimization
  - litert: LiteRT (TFLite) GPU delegate, NNAPI, CompiledModel API
  - ncnn: NCNN Vulkan, custom layers, threading
  - android_gpu: Vulkan compute, OpenGL ES, Mali GPU specifics
  - android_npu: NNAPI, QNN SDK, NPU acceleration
  - vit_mobile: Vision Transformer optimization on mobile
  - quantization: INT8/FP16 quantization for all backends
  - sharp_model: SHARP model architecture, patch pyramid, encoder/decoder
  - performance: Benchmarks, bottleneck analysis, optimization strategies
  - sgemm: SGEMM (single-precision GEMM), BLAS, NCNN convolution, ARM ACL
  - vedic_maths: Vedic mathematics sutras, mental math, multiplication shortcuts
  - ml_fundamentals: Google ML Crash Course - neural networks, activation, backpropagation
  - beeware_android: BeeWare, Briefcase, Toga, Chaquopy on Android; Python UI; hand-off to native for Sharp
  - paafekt_standards: Paafekt/Furnit UX standards (e.g. landscape room opening camera position)
"""

import json
from pathlib import Path

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("ERROR: pip install chromadb sentence-transformers")
    exit(1)

DB_PATH = str(Path(__file__).parent / "data" / "vector_db")
EXPORT_PATH = str(Path(__file__).parent / "data" / "rag_export.json")

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name="furnit_ml",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

def get_log_collection():
    """Separate collection for Android log chunks (not wiped by main index)."""
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name="furnit_android_logs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

def index_android_logs(log_file_path=None, from_adb=False):
    """Read Android logs from file or adb logcat -d, chunk, and index into furnit_android_logs."""
    import subprocess
    if from_adb:
        try:
            out = subprocess.run(
                ["adb", "logcat", "-d", "-v", "time"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            log_text = out.stdout or ""
            if out.stderr:
                log_text += "\n" + out.stderr
        except FileNotFoundError:
            print("adb not found in PATH; skipping log index")
            return 0
        except subprocess.TimeoutExpired:
            print("adb logcat timed out; skipping")
            return 0
    elif log_file_path:
        path = Path(log_file_path)
        if not path.exists():
            print(f"Log file not found: {path}")
            return 0
        log_text = path.read_text(encoding="utf-8", errors="replace")
    else:
        return 0

    lines = [ln.strip() for ln in log_text.splitlines() if ln.strip()]
    chunk_size = 80
    chunks = []
    for i in range(0, len(lines), chunk_size):
        block = "\n".join(lines[i : i + chunk_size])
        chunks.append(block)

    if not chunks:
        print("No log content to index")
        return 0

    collection = get_log_collection()
    try:
        existing = collection.count()
        if existing > 0:
            collection.delete(ids=[c for c in collection.get()["ids"]])
    except Exception:
        pass

    ids = [f"android_log_{i}" for i in range(len(chunks))]
    metadatas = [{"layer": "android_logs"} for _ in chunks]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    print(f"Indexed {len(chunks)} Android log chunks")
    return len(chunks)

# ============================================================================
# KNOWLEDGE CHUNKS
# ============================================================================

CHUNKS = [
    # ---- EXECUTORCH ----
    {
        "id": "et_overview",
        "layer": "executorch",
        "content": """ExecuTorch is Meta/PyTorch's on-device inference framework for mobile.
Key backends: XNNPACK (CPU SIMD), Vulkan (GPU), CoreML (iOS), QNN (Qualcomm NPU).
Model format: .pte (PyTorch ExecuTorch).
Export: torch.export -> to_edge_transform_and_lower -> save.
Vulkan backend targets Android GPUs via cross-platform Vulkan 1.1 API.
Supports: dynamic shapes, FP32/FP16 inference, 8-bit/4-bit quantized weights.
Validated on: OnePlus 12, Samsung S23/S24+, Pixel 8 Pro."""
    },
    {
        "id": "et_vulkan_delegate",
        "layer": "executorch",
        "content": """ExecuTorch Vulkan delegate exports:
from executorch.backends.vulkan.partitioner import VulkanPartitioner
model = to_edge_transform_and_lower(exported, partitioner=[VulkanPartitioner()])
Vulkan uses GLSL compute shaders compiled to SPIR-V.
Custom shaders go in src/layer/shader/.
For partial delegation: ops not supported by Vulkan fall back to CPU automatically.
Vulkan delegate handles standard transformer ops (Linear, LayerNorm, Softmax, SDPA) natively on GPU.
This is different from NCNN where custom SDPA layers fall back to CPU."""
    },
    {
        "id": "et_vulkan_approach_copy",
        "layer": "executorch",
        "content": """ExecuTorch Vulkan approach to COPY (from GitHub examples/vulkan/export.py). Use for ANY model (vision, SHARP, new classes). NOT Llama-specific.

Export (Python):
1. program = torch.export.export(model, example_inputs, strict=...)
2. compile_options = {}  # optional: force_fp16=True, skip_memory_planning=True, require_dynamic_shapes=True
3. edge_program = to_edge_transform_and_lower(program, partitioner=[VulkanPartitioner(compile_options)])
4. exec_prog = edge_program.to_executorch()
5. save_pte_program(exec_prog, output_name, output_dir) -> .pte file

Android runtime:
- Module.load(path) loads .pte (backend inside the file is fixed at export time)
- forward: EValue.from(tensor) -> module.forward(inputs) -> outputs[0].toTensor().getDataAsFloatArray()
- Same for SHARP, new classes, or any vision model. No LLM/tokenizer.

Key: Backend is chosen at EXPORT via VulkanPartitioner(compile_options). Options: skip_memory_planning=False (AOT memory planning), force_fp16=True (lower memory/latency). Mirror this exact flow for new models."""
    },
    {
        "id": "et_new_classes_executorch_vulkan",
        "layer": "executorch",
        "content": """New classes backend in Furnit = ExecuTorch implementation with Vulkan. Same approach as main ExecuTorch option (SHARP split .pte, Vulkan delegate). NOT Llama, NOT an LLM.

Implementation: Use et_vulkan_approach_copy. Export model (e.g. classifier or SHARP variant with extended classes) with VulkanPartitioner(compile_options). Push .pte to device. SharpService when inference_backend==new_classes uses executorchSharp (same ExecuTorch Vulkan runtime). If no .pte present, fallback to ONNX for that session.

Do not reference or use Llama/LLM for New classes. The only thing to copy from GitHub is the Vulkan EXPORT and RUNTIME pattern (examples/vulkan/export.py), not the Llama model or tokenizer."""
    },
    {
        "id": "executorch_vulkan_sharp_combination",
        "layer": "executorch",
        "content": """ExecuTorch + Vulkan + SHARP combination (final alignment).

SHARP outputs 3D Gaussians (sharp_philosophy, sharp_gaussian_maths). The same pipeline runs on Android via ExecuTorch with Vulkan: export SHARP (split or full) with VulkanPartitioner(compile_options) → .pte; on device Module.load(.pte) then forward(image) → same Gaussian tensor [N,14]. So: one model (SHARP), one output format (Gaussians), one runtime (ExecuTorch) and one GPU path (Vulkan). Philosophy (single image → learned 3D) and maths (Gaussian params) are unchanged; only the execution backend (Vulkan) and the framework (ExecuTorch) change. Query sharp_philosophy + sharp_gaussian_maths + vulkan_latest_android_ml + et_vulkan_approach_copy for the full stack."""
    },
    {
        "id": "aot_executorch_vulkan",
        "layer": "executorch",
        "content": """AOT (Ahead-of-Time) in ExecuTorch and Vulkan.

AOT = decisions made at export/compile time, not at runtime. For ExecuTorch on Android this means: (1) memory layout and buffer reuse are fixed in the .pte; (2) no dynamic allocation for intermediate tensors on device; (3) lower peak RAM and predictable first-inference cost.

Vulkan AOT memory planning: VulkanPartitioner(compile_options) with skip_memory_planning=False. The Vulkan backend's preprocess runs greedy-style memory planning at export: reuse buffers across ops, share token/activation buffers between subgraphs (e.g. Part1+Part2 combined). Result: fewer VkBuffer allocations on device, less GPU memory. Use force_fp16=True in compile_options for lower bandwidth and memory.

Greedy memory planning (CPU/portable path): For large graphs (e.g. SHARP Part 4) use MemoryPlanningPass(memory_planning_algo=greedy, alloc_graph_input=False, alloc_graph_output=False) in ExecutorchBackendConfig when calling edge.to_executorch(). I/O buffers are caller-managed; intermediates are planned and reused. Reduces peak RAM during inference.

Export flow: to_edge_transform_and_lower(..., partitioner=[VulkanPartitioner({"skip_memory_planning": False, "force_fp16": True})]) then to_executorch(). Optional: combine Part1+Part2 into one .pte so AOT can share the token buffer between Part1 and Part2. See export_sharp_executorch_split4.py and et_vulkan_approach_copy."""
    },
    {
        "id": "executorch_30sec_room_space_philosophy",
        "layer": "executorch",
        "content": """ExecuTorch Vulkan: create room in ~30 sec. Philosophy: time dimension may not be fully under our control; space dimension is at our disposal.

Space we control: patch grid (5×5 + 3×3 + 1 = 35 patches), resolution, merge layout. We choose how much of the image to encode (grid density). Vedic maths: Urdhva-tiryagbhyam (vertically and crosswise) — grid rows × cols; Nikhilam for stride/complement. Mental compute: patch index from (i,j) like digit recurrence. Use vedic_mental_compute for algorithm crossover.

To hit ~30 sec: (1) Export with ExecuTorch Vulkan only — VulkanPartitioner(skip_memory_planning=False, force_fp16=True), optional combined Part1+Part2 for AOT buffer sharing (aot_executorch_vulkan). (2) Push Vulkan .pte to device; runtime Module.load uses backend baked in at export. (3) No Llama, no other backends for this path — ExecuTorch option only. (4) Future: reduce patch grid (e.g. 3×3 at 1x) when decoder supports variable merge size to trade space for time further. RAG chunks: et_vulkan_approach_copy, aot_executorch_vulkan, executorch_vulkan_sharp_combination, vulkan_latest_android_ml."""
    },
    {
        "id": "et_quantization",
        "layer": "executorch",
        "content": """ExecuTorch Vulkan quantization support:
- 8-bit weights + 8-bit dynamically quantized activations (W8A8)
- 4-bit weights + 8-bit activations (W4A8) - for LLMs
- Mixed quantization: quantized weights with FP32/FP16 activations
INT8 quantized linear layers run on Vulkan GPU directly.
Llama 3.2 1B with 4-bit quantization on Samsung S24+: >350 tokens/sec prefill.
For vision models: INT8 is typical. Reduces model by 4x vs FP32, ~2x vs FP16.
Export: use PyTorch quantization APIs before ExecuTorch export."""
    },
    {
        "id": "et_sharp_status",
        "layer": "executorch",
        "content": """ExecuTorch SHARP model status in Furnit:
Models available:
- sharp_single_patch_hybrid_standalone.pte (INT8, 275MB, Vulkan+XNNPACK)
- sharp_single_patch_xnnpack.pte (FP32, 1.1GB, XNNPACK only)
- sharp_gaussian_head.pte (7MB)
Code: ExecutorchSharp.kt - complete, well-structured.
Status: DISABLED (BackendConfig.ENABLE_EXECUTORCH=false).
Current limitation: only processes 25 patches (1x scale), skips 0.5x and 0.25x.
The INT8 Vulkan model is the most promising backend because:
1. INT8 = 4x less compute than FP32
2. Vulkan = full graph on GPU (no custom layer CPU fallback like NCNN)
3. 275MB fits easily in GPU memory"""
    },
    {
        "id": "et_sharp_backend_perf",
        "layer": "executorch",
        "content": """ExecuTorch SHARP backend performance (split 4-part model):

Backend is chosen at EXPORT time, not load time. Module.load(path) uses whatever backend is in the .pte.

| Backend | Export flag | Part1 warmup | Full pipeline |
|---------|-------------|--------------|---------------|
| CPU fallback (portable) | --backend portable | 10+ min | 10+ min |
| XNNPACK (CPU SIMD) | --backend xnnpack | 30-90 sec | 1-2 min |
| Vulkan GPU | --backend vulkan | 5-20 sec | 20-60 sec |

NEVER use portable: causes CPU scalar fallback, 637000ms warmup observed.
Export with XNNPACK or Vulkan: python export_sharp_executorch_split4.py --backend xnnpack|vulkan

Warmup removed: dummy forward on CPU fallback = 10 min. With XNNPACK/Vulkan models first forward is acceptable.
preloadAndWarmup now does load+destroy only (no forward). inferSplitMode has no warmup before patch loop."""
    },
    {
        "id": "et_sharp_part4_crash_diag",
        "layer": "executorch",
        "content": """ExecuTorch SHARP Part 4 crash diagnosis:

If Part 4 loads but app restarts during forward (no 'Part 4 done' log), it's likely LMK (low memory killer) or native crash (SIGSEGV/SIGABRT).

DIAGNOSE:
  adb logcat -b crash -d | tail -200
  adb logcat -d | grep -iE "FATAL EXCEPTION|SIGSEGV|SIGABRT|lowmemorykiller|lmkd|Killed process|OutOfMemoryError"

LMK/OOM: look for lmkd, lowmemorykiller, 'Killed process com.furnit.android'
Native: look for signal + tombstone pointer in crash buffer

Part 4 input sizes (~155-170MB total):
  image: 1x3x1536x1536 ~28MB
  imageTokens: 1x577x1024 ~2.3MB
  latent0/latent1/x0Feat: each 1x1024x96x96 ~38MB (3x ~114MB)
  x1Feat: 1x1024x48x48 ~9MB, x2Feat: 1x1024x24x24 ~2.3MB
Plus Part4 module (~755MB) + decoder intermediates.

Mitigation: System.gc() + Thread.sleep(50) before Part4 forward to reduce memory spikes.
RAG = Retrieval-Augmented Generation (context injection for AI assistants)."""
    },
    {
        "id": "et_sharp_part4_memory_solutions",
        "layer": "executorch",
        "content": """ExecuTorch SHARP Part 4 memory solutions (non-Vulkan):

Root cause: Part 4 decoder activation memory ~4GB peak. LMK kills when swap exhausted.

Solution 2 (implemented): gc + runFinalization + 150ms sleep before Part 4. Abort if availMem < 1GB.
Solution 3: Native allocator - ExecuTorch already native; JVM float[] from Tensor.fromBlob. No easy fix.
Solution 4: Increase zram swap to 8GB (root): swapoff zram0; echo 8192M > disksize; mkswap; swapon. Script: android/scripts/increase_swap_root.sh
Solution 5: Stream Part 4 - split decoder into part4a/b/c.pte, run sequentially. Requires re-export.

Best fix: Export Part 4 with Vulkan. GPU absorbs activation memory."""
    },

    # ---- ONNX RUNTIME ----
    {
        "id": "ort_overview",
        "layer": "onnx_runtime",
        "content": """ONNX Runtime on Android: main inference engine for SHARP in Furnit.
Dependency: com.microsoft.onnxruntime:onnxruntime-android:1.18.0
Two modes in Furnit:
1. Regular ONNX: single 2.4GB graph, mmap, 1 thread, NO_OPT. Slow but works.
2. Split ONNX (preferred): 4 parts ~600MB each, all CPU cores, EXTENDED_OPT.
Split ONNX is fastest at ~5 minutes because model handles 35-patch batching internally.
Key: ONNX model contains the patch pyramid logic IN the graph - no serial patch loop."""
    },
    {
        "id": "ort_nnapi_ep",
        "layer": "onnx_runtime",
        "content": """ONNX Runtime NNAPI Execution Provider:
Enables GPU/NPU acceleration via Android Neural Networks API.
Requirements: Android 8.1+ (API 27+). Recommended: Android 9+ (API 28+).
Configuration flags:
- NNAPI_FLAG_CPU_DISABLED: force GPU/NPU only (API 29+)
- NNAPI_FLAG_USE_FP16: half-precision on GPU (faster, slight accuracy loss)
- NNAPI_FLAG_USE_NCHW: NCHW layout (API 29+)
Adding NNAPI EP to ONNX Runtime:
  val sessionOptions = OrtSession.SessionOptions()
  sessionOptions.addNnapi()  // or addNnapi(flags)
Falls back gracefully to CPU for unsupported ops.
NOT currently used in Furnit - all ONNX inference is CPU-only."""
    },
    {
        "id": "ort_optimization",
        "layer": "onnx_runtime",
        "content": """ONNX Runtime optimization levels:
- NO_OPT: no graph optimizations (current regular ONNX mode)
- BASIC_OPT: constant folding, redundant node elimination
- EXTENDED_OPT: complex fusions (current split ONNX mode)
- ALL_OPT: all optimizations (may break external data references)
Split ONNX threading: intraOpNumThreads = all CPU cores, interOpNumThreads = 1.
Memory: mmap for weights (no full 2.4GB load), arena allocator OFF.
Key bottleneck: even with all cores, 5 min is CPU-bound. Adding NNAPI EP
or running on GPU would target the actual bottleneck (matrix multiplications)."""
    },

    # ---- LITERT (TFLITE) ----
    {
        "id": "litert_overview",
        "layer": "litert",
        "content": """LiteRT (formerly TensorFlow Lite) in Furnit:
Dependency: org.tensorflow:tensorflow-lite:2.17.0 + gpu:2.17.0 + gpu-api:2.17.0
Split model: 4 FP16 parts (~290MB each):
- Part 1: patch_embed (ViT blocks 0-11), run 35x
- Part 2: patch_encoder (ViT blocks 12-23), run 35x
- Part 3: Image Encoder A, run 1x
- Part 4: Image Encoder B + Decoder + Gaussians, run 1x
LiteRT processes 35 patches SERIALLY through Part 1 + Part 2.
Part 4: XNNPACK OFF (graph compilation uses ~800MB), only 2 threads.
Current: CPU+XNNPACK only. GPU delegate disabled (SIGSEGV crash history).
NNAPI disabled (ANEURALNETWORKS_BAD_DATA - ops not supported)."""
    },
    {
        "id": "litert_gpu_delegate",
        "layer": "litert",
        "content": """LiteRT GPU Delegate on Android:
Uses OpenGL ES 3.1 compute shaders or OpenCL.
CRITICAL LIMITATION FOR TRANSFORMERS (2025):
- GPU delegate requires batch_size=1
- Transformer attention internally creates multi-head batches
- Reshape/transpose in attention blocks misinterpreted as batch dimensions
- Error: 'Batch size mismatch, expected 1 but got N'
This is why GPU delegate crashes (SIGSEGV) with SHARP ViT model.
The fix: either restructure model to avoid batch-like reshape,
or use ExecuTorch Vulkan which handles transformer attention natively.
GPU delegate works well for CNNs but NOT for ViT/transformers as of 2025."""
    },
    {
        "id": "litert_next",
        "layer": "litert",
        "content": """LiteRT Next (CompiledModel API) - latest 2025/2026:
Version: v2.1.0 (December 2025), requires API 23+.
Key improvement: CompiledModel API replaces manual delegate configuration.
Hardware priority: NPU > GPU > CPU (automatic selection).
Features:
- Async execution with OS sync fences (2x latency reduction)
- Zero-copy buffer interop (AHardwareBuffer, OpenCL, OpenGL)
- Automated hardware selection (no manual delegate setup)
GPU powered by ML Drift library (WebGL, OpenCL).
NPU: Early Access Program with vendor-specific backends.
This is the future path for LiteRT but requires model re-export and API migration."""
    },

    # ---- NCNN ----
    {
        "id": "ncnn_overview",
        "layer": "ncnn",
        "content": """NCNN in Furnit: native C++ inference via JNI.
Built with Vulkan support (ncnn-20260113-android-vulkan).
Two modes:
1. Full model: sharp.ncnn.param/bin (~2.4GB). Hangs at conv_106 (batch dimension issue).
2. Component mode: separate models per stage. Works but 35 serial patches.
Component models: patch_embed (1.5MB), patch_encoder (579MB), image_encoder (580MB), gaussian_head (3.5MB).
Custom layers: SDPA, pnnx.Expression, SafeConv, etc. registered via register_custom_layers().
SDPA custom layer runs on CPU even with Vulkan enabled - this is the main bottleneck."""
    },
    {
        "id": "ncnn_vulkan_limitation",
        "layer": "ncnn",
        "content": """NCNN Vulkan limitation for SHARP:
Custom layers (SDPA, SafeConv, pnnx.Expression) do NOT have Vulkan shader implementations.
When use_vulkan_compute=true, built-in ops (Convolution, InnerProduct, etc.) run on GPU,
but custom layers fall back to CPU. This creates CPU-GPU data transfer overhead.
SDPA is the heaviest op (Q@K^T + softmax + scores@V per head per block x24).
Result: Vulkan=on gave near-zero improvement because SDPA stayed on CPU.
To fix: either write GLSL compute shaders for SDPA (complex), or switch to a framework
that handles SDPA natively on GPU (ExecuTorch Vulkan, ONNX+NNAPI)."""
    },
    {
        "id": "ncnn_batch_issue",
        "layer": "ncnn",
        "content": """NCNN batch dimension problem:
NCNN does not support true batch dimensions.
PNNX converts torch.cat(patches, dim=0) as CHANNEL concatenation, not batch stacking.
Result: full model expects [3, 384, 384] per patch but receives [105, 384, 384] (35*3 channels).
conv_106 (16x16 patch embed) crashes or hangs on this mismatch.
This forced the component-mode workaround: process 35 patches one at a time.
The serial processing is why NCNN takes 23 minutes (77s/patch x 35 patches).
ONNX and TFLite don't have this problem because they support batch dimensions."""
    },

    # ---- VULKAN (LATEST) ----
    {
        "id": "vulkan_latest_android_ml",
        "layer": "android_gpu",
        "content": """Vulkan for ML on Android (latest and greatest).

Vulkan 1.1+ (1.2 optional): cross-platform GPU API; on Android it drives Mali, Adreno, etc. For ML inference: compute shaders (GLSL compiled to SPIR-V) do matrix ops, activations, attention on GPU. No OpenGL render pipeline—pure compute. Key: VkBuffer for GPU memory, VkDescriptorSet for bindings, dispatch (workgroups) for parallelism. ExecuTorch Vulkan backend uses this: partitioner assigns ops to Vulkan; AOT memory planning (skip_memory_planning=False) reuses buffers; force_fp16 reduces bandwidth. Best practice: export with VulkanPartitioner(compile_options), push .pte, Module.load on device—backend baked in at export. Vulkan 1.1 is the target for broad Android support; 1.2 adds optional features."""
    },
    # ---- ANDROID GPU ----
    {
        "id": "gpu_mali_g715",
        "layer": "android_gpu",
        "content": """Mali-G715 GPU (user's device):
4th-gen Valhall architecture, premium mobile GPU.
Features: fp16 packed/storage/unpacked/arithmetic all supported.
Subgroup size: 16. Matrix multiply instructions: yes (2x ML improvement).
Vulkan 1.1 supported. Compute shaders supported.
NCNN detected: [0 Mali-G715] queueC=0[2] queueT=0[2]
This GPU is capable of running transformer attention via Vulkan compute shaders.
The issue is not hardware - it's that no framework is using it for the heavy ops."""
    },
    {
        "id": "gpu_vulkan_compute",
        "layer": "android_gpu",
        "content": """Vulkan compute on Android for ML inference:
Vulkan 1.1 provides compute shaders for GPU-accelerated ML.
GLSL compute shaders compiled to SPIR-V at build time.
Key ops implementable: matrix multiply, softmax, layer norm, GELU.
Memory management: VkBuffer for GPU memory, staging buffers for CPU-GPU transfer.
For transformers: Q@K^T and scores@V are large GEMM ops ideal for GPU.
ExecuTorch implements these as Vulkan compute shaders natively.
NCNN requires custom GLSL shader implementation per layer.
TFLite uses OpenGL ES 3.1 compute shaders (different API, same concept)."""
    },

    # ---- ANDROID NPU ----
    {
        "id": "npu_nnapi",
        "layer": "android_npu",
        "content": """Android NNAPI (Neural Networks API):
Unified interface to CPU, GPU, and NPU accelerators.
Available: Android 8.1+ (API 27+).
Framework integration: TFLite delegate, ONNX Runtime EP.
Issue with SHARP model: ANEURALNETWORKS_BAD_DATA error.
This typically means the model uses ops not supported by the device's NNAPI driver.
FP16 models with certain reshape/transpose patterns may trigger this.
Workaround: use NNAPI_FLAG_USE_FP16 to allow precision reduction,
or identify and partition unsupported ops to CPU.
Mali-G715 GPU is NOT a dedicated NPU. NNAPI on this device
likely routes to GPU compute or CPU fallback, not a dedicated NPU chip."""
    },
    {
        "id": "npu_qnn",
        "layer": "android_npu",
        "content": """Qualcomm QNN SDK for NPU acceleration:
Only available on Snapdragon devices (Hexagon NPU/HTP).
Backends: HTP (NPU), GPU, DSP, CPU.
Integration: TFLite delegate or standalone SDK.
Converters: qnn-pytorch-converter, qnn-onnx-converter, qnn-tflite-converter.
For SHARP: would need Snapdragon device (not Samsung Exynos/Mali).
If user has Pixel (Tensor chip) or Samsung (Exynos): QNN is NOT available.
Check device SoC before attempting QNN integration."""
    },

    # ---- VIT MOBILE ----
    {
        "id": "vit_mobile_challenges",
        "layer": "vit_mobile",
        "content": """Vision Transformer (ViT) on mobile - key challenges (2025):
1. Self-attention is O(n^2) in sequence length (577 tokens = 332,929 attention scores per head)
2. Most ML frameworks have LIMITED support for ViT ops on mobile GPUs
3. TFLite GPU delegate fails on transformer attention (batch size mismatch)
4. NNAPI support is inconsistent across devices for transformer ops
5. Quantization effects differ by core type: helps on efficient cores, hurts on powerful cores
6. Memory bandwidth is the bottleneck, not raw FLOPS
Research finding: CPU inference with XNNPACK is currently the most RELIABLE path
for ViT on Android, but not the fastest. GPU requires framework-specific solutions."""
    },
    {
        "id": "vit_sharp_architecture",
        "layer": "vit_mobile",
        "content": """SHARP model ViT architecture:
Encoder: DINOv2-Large (24 transformer blocks, dim=1024, 16 heads, head_dim=64).
Input: 1536x1536 image -> 35 patches (25@1x + 9@0.5x + 1@0.25x), each 384x384.
Per patch: patch_embed -> CLS + pos_embed -> 24 blocks -> [577, 1024] output.
Per block: LayerNorm -> MHSA (Q/K/V proj, SDPA, output proj) -> residual -> LayerNorm -> MLP (fc1 4096, fc2 1024) -> residual.
Total compute per patch: ~1.1B multiply-adds (attention) + ~1.5B (MLP) = ~2.6B.
Total for 35 patches: ~91B multiply-adds.
This is why it takes 5 minutes even on all CPU cores with ONNX."""
    },

    # ---- QUANTIZATION ----
    {
        "id": "quant_int8_benefits",
        "layer": "quantization",
        "content": """INT8 quantization benefits for SHARP:
FP32 model (NCNN): 579MB encoder, ~77s/patch on CPU.
FP16 model (LiteRT): 290MB per part, ~2x bandwidth reduction.
INT8 model (ExecuTorch): 275MB total, ~4x compute reduction vs FP32.
INT8 on GPU (Vulkan): supported by ExecuTorch with W8A8 quantization.
Expected speedup: 4x from quantization * 3-5x from GPU = 12-20x over FP32 CPU.
That would bring 77s/patch down to ~4-6s/patch.
Tools: PyTorch quantization -> ExecuTorch export -> Vulkan delegate."""
    },
    {
        "id": "quant_fp16_onnx",
        "layer": "quantization",
        "content": """FP16 for ONNX Runtime:
Current ONNX model is FP32 (2.4GB total, 600MB per split part).
FP16 would halve model size and memory bandwidth.
Export: use onnxruntime.transformers.float16 converter or PyTorch export with half().
ONNX Runtime supports FP16 on CPU (with automatic upcasting where needed).
Combined with NNAPI EP, FP16 ONNX could be significantly faster.
Risk: some ops may lose precision. Test output quality after conversion."""
    },

    # ---- SHARP MODEL ----
    {
        "id": "sharp_philosophy",
        "layer": "sharp_model",
        "content": """SHARP (Apple-style) philosophy: single image to 3D.

One RGB image in → 3D scene out, represented as many 3D Gaussians (not a mesh or depth map). The network does not measure or know depth explicitly: a ViT encoder turns the image into features, and a decoder trained on multi-view or 3D data predicts 3D Gaussian parameters from those features. Depth and shape are learned priors—the model has seen many images with 3D supervision and infers plausible 3D from one view. So the pipeline is: 2D appearance (encoder) → learned 3D structure (decoder) → Gaussian representation that can be rendered from new viewpoints."""
    },
    {
        "id": "sharp_gaussian_maths",
        "layer": "sharp_model",
        "content": """Gaussian representation and bell-curve maths for 3D splatting.

Gaussian bell curve: f(x) = exp(-(x-μ)²/(2σ²)); peak at μ, width σ; symmetric falloff (bright center, smooth fade). In 3D splatting the blob's brightness follows this falloff.

One 3D Gaussian = position (x,y,z), scale (sx,sy,sz), rotation (quaternion qw,qx,qy,qz), opacity, color (r,g,b). SHARP output per Gaussian: 14 floats (e.g. pos 3, opacity 1, scales 3, quats 4, colors 3). Scene = N such Gaussians (e.g. ~1.2M); each covers a small region. Rendering = splat each Gaussian onto the screen with bell-curve falloff; alpha-blend. So 'Gaussian representation' = the 3D world as a cloud of soft 3D blobs defined by these parameters."""
    },
    {
        "id": "sharp_pipeline",
        "layer": "sharp_model",
        "content": """SHARP inference pipeline on Android:
1. Input: 1536x1536 RGB image
2. Create pyramid: 1x (1536), 0.5x (768), 0.25x (384)
3. Extract patches: 25 from 1x (5x5 grid, stride 288), 9 from 0.5x (3x3, stride 192), 1 from 0.25x
4. Each patch [3, 384, 384] -> patch_embed -> [576, 1024]
5. Add CLS token + positional embeddings -> [577, 1024]
6. Run through 24 ViT blocks -> [577, 1024]
7. Reshape to spatial [24, 24, 1024], merge into feature maps
8. Run image encoder on 0.25x image
9. Run decoder -> Gaussian parameters
10. Write PLY file
Bottleneck: step 6 (encoder), repeated 35 times serially in component mode."""
    },
    {
        "id": "sharp_backend_comparison",
        "layer": "sharp_model",
        "content": """SHARP backend performance comparison (measured on device):
| Backend | Time | Patch handling | Hardware |
| ONNX Split | ~5 min | Internal batching (single pass) | CPU all cores |
| NCNN Component | ~23 min | 35 serial patches, 77s each | CPU 1-4 threads |
| LiteRT Split | ~5-8 min (est) | 35 serial through Part1+Part2 | CPU XNNPACK |
| ExecuTorch | untested | 25 serial patches (incomplete) | XNNPACK/Vulkan |
Key insight: ONNX is fastest because it processes patches INTERNALLY in the graph.
All other backends call the encoder 35 times separately.
To match ONNX: need single-graph model OR GPU-accelerated per-patch processing."""
    },

    # ---- PERFORMANCE ----
    {
        "id": "perf_bottleneck",
        "layer": "performance",
        "content": """SHARP performance bottleneck analysis:
Per patch (NCNN measured): crop=2ms, embed=175ms, clspos=4ms, encoder=77000ms, reshape=20ms.
Encoder is 99.7% of per-patch time.
Inside encoder (24 blocks): attention SDPA is the heaviest op.
SDPA per block: 16 heads x (Q@K^T [577x577] + softmax + scores@V [577x64]).
Total SDPA FLOPs per patch: ~24 * 16 * 2 * 577^2 * 64 = ~1.03B multiply-adds.
MLP per block: 577*1024*4096 + 577*4096*1024 = ~4.85B per block, ~116B total.
MLP dominates total FLOPs but is handled by NCNN's built-in InnerProduct (NEON optimized).
SDPA is slower per-FLOP because of custom layer overhead and memory access patterns."""
    },
    {
        "id": "perf_target",
        "layer": "performance",
        "content": """Performance target: < 1 minute total inference.
Current best: ONNX Split at ~5 minutes (CPU all cores).
Gap: need 5x improvement.
Achievable paths (ordered by feasibility):
1. ONNX + NNAPI EP: add EP, test. Expected: 2-3 min. Effort: small.
2. ONNX FP16 model: re-export. Expected: 3-4 min. Effort: medium (Python export).
3. ExecuTorch INT8+Vulkan: enable, push model, test. Expected: 2-4 min. Effort: small.
4. LiteRT CompiledModel API: migrate to v2.1, auto hardware. Expected: unknown. Effort: large.
5. Re-export single-graph TFLite with batch support: Expected: 1-2 min with GPU. Effort: large.
6. Custom Vulkan compute pipeline: Expected: <30s. Effort: months.
Realistic near-term target: 2-3 minutes via ExecuTorch INT8 Vulkan or ONNX+NNAPI."""
    },
    {
        "id": "perf_gpu_vs_cpu",
        "layer": "performance",
        "content": """GPU vs CPU for transformer inference on Android:
Mali-G715 theoretical: ~1.5 TFLOPS FP16, ~0.75 TFLOPS FP32.
CPU (4 big cores): ~50-100 GFLOPS FP32 with NEON.
Raw advantage: GPU is 10-15x faster than CPU for parallel workloads.
But: memory transfer overhead, driver overhead, op compatibility reduce this to 3-5x in practice.
For SHARP encoder (2.6B FLOPs per patch):
- CPU: ~26-52 seconds theoretical (actual ~77s due to memory bottleneck)
- GPU: ~5-15 seconds theoretical with Vulkan
- GPU+INT8: ~2-5 seconds theoretical
Memory bandwidth is often the real bottleneck on mobile, not compute.
INT8 helps bandwidth too (4x less data to move)."""
    },
    {
        "id": "perf_litert_gpu_workaround",
        "layer": "performance",
        "content": """Workaround for LiteRT GPU delegate ViT crash:
The crash occurs because GPU delegate misinterprets attention reshape as batch change.
Potential fixes:
1. Restructure model: flatten multi-head attention to avoid batch-like reshapes.
   Before: [B, heads, seq, dim] -> reshape seen as batch change.
   After: [B, seq, heads*dim] -> single batch, GPU delegate happy.
2. Use GPU delegate with op allowlist: only delegate specific ops (Linear, Conv) to GPU.
3. Use LiteRT Next CompiledModel API which has smarter op partitioning.
4. Skip GPU delegate entirely, use ExecuTorch Vulkan which handles this correctly.
Option 4 (ExecuTorch) is the most practical path."""
    },

    # ---- EXPORT PIPELINE ----
    {
        "id": "export_full_pipeline",
        "layer": "export",
        "content": """Exporting FULL SHARP model to ExecuTorch (correct approach):
The key mistake was exporting patch encoder + gaussian head SEPARATELY.
The 7MB gaussian head is NOT the real decoder (387MB).

CORRECT: Export the FULL pipeline as ONE model:
  Input: [1, 3, 1536, 1536] -> Output: [N, 14] Gaussian params

Script: export_sharp_executorch_all.py
  - Loads sharp_2572gikvuh.pt checkpoint
  - Wraps in SharpFullPipeline (predictor + Gaussian param extraction)
  - Exports 3 variants: FP32, FP16, INT8

The full model internally handles:
  1. Sliding pyramid (25+9+1 patches)
  2. Patch encoder ViT (24 blocks)
  3. Image encoder ViT
  4. Multi-scale decoder
  5. Gaussian prediction head
All in ONE forward pass - no serial patch loop needed."""
    },
    {
        "id": "export_fp32",
        "layer": "export",
        "content": """FP32 ExecuTorch export:
torch.export.export(model, (example_input,), strict=False)
edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))
et_program = edge.to_executorch()

Output: sharp_full_fp32.pte (~2.4GB)
Pros: exact same output as PyTorch, no precision loss
Cons: large, slow (same as ONNX FP32)
Use: baseline for validation"""
    },
    {
        "id": "export_fp16",
        "layer": "export",
        "content": """FP16 ExecuTorch export:
model_fp16 = model.half()
example_fp16 = example_input.half()
Then same export flow as FP32.

Output: sharp_full_fp16.pte (~1.2GB)
Pros: half size, better GPU utilization (Mali-G715 supports FP16 natively)
Cons: slight precision loss (usually negligible for vision models)
Use: good balance of speed and quality"""
    },
    {
        "id": "export_int8",
        "layer": "export",
        "content": """INT8 ExecuTorch export:
from torch.ao.quantization import quantize_dynamic
model_int8 = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

Then export with Vulkan delegate:
from executorch.backends.vulkan.partitioner import VulkanPartitioner
edge = edge.to_backend(VulkanPartitioner())

Output: sharp_full_int8.pte (~600MB-800MB)
Pros: 4x less compute, Vulkan GPU acceleration
Cons: some quality loss, needs validation
Use: fastest inference, target for production"""
    },
    {
        "id": "export_test_validation",
        "layer": "export",
        "content": """Testing exported .pte models:
Script: test_sharp_pte.py

1. Python validation (Mac/Linux):
   python test_sharp_pte.py sharp_full_fp32.pte --compare-pytorch --image room.jpg
   Compares per-field (position, scale, rotation, opacity, color) between
   PyTorch reference and ExecuTorch output.

2. Android validation:
   adb push sharp_full_fp32.pte /sdcard/Android/data/com.furnit.android/files/models/
   Select ExecuTorch in Settings, run generation.
   Check logcat for: 'ExecuTorch component-mode SHARP completed'

Key validation criteria:
- Same Gaussian count as PyTorch/ONNX
- Mean absolute difference < 0.01 per field
- No rainbow/blue patches (color channels correct)
- Room bounds match ONNX output"""
    },
    {
        "id": "export_why_separate_head_failed",
        "layer": "export",
        "content": """Why the separate gaussian_head.pte (7MB) produces wrong output:

The ONNX decoder (Part 4) is 387MB and takes 7 inputs:
  image, imageTokens, latent0, latent1, x0Feat, x1Feat, x2Feat

The ExecuTorch gaussian_head.pte is 7MB and takes 1 input:
  merged 1x features only

It's missing:
  - Image tokens from image encoder
  - Block5 (latent0) features
  - Block11 (latent1) features
  - 0.5x scale features
  - 0.25x scale features
  - Original image for skip connections

The 7MB head is a lightweight approximation trained separately.
It cannot produce correct Gaussian parameters. This is NOT fixable
by tweaking color channels - the output is fundamentally wrong.

Solution: export the FULL model (encoder + decoder + head) as one .pte"""
    },

    # ---- PYTHON TO ANDROID CROSS-LEARNING ----
    {
        "id": "py2android_overview",
        "layer": "python_to_android",
        "content": """Python to Android: Cross-Lingual Transfer for ML Inference.

Like Java->ABAP skill anchoring in the ABAP AI Toolkit, we anchor
Android ML concepts to their Python equivalents:

| Python (PyTorch) | Android (ExecuTorch) |
| model = Model(); model.eval() | module = Module.load(path) |
| output = model(input_tensor) | outputs = module.forward(EValue.from(tensor)) |
| torch.export.export(model, (input,)) | Exported -> Edge -> ExecuTorch .pte |
| model.half() | FP16 .pte (1.2GB vs 2.4GB FP32) |
| quantize_dynamic(model, {Linear}) | INT8 via PT2E + XNNPACKQuantizer |
| torch.randn(1,3,1536,1536) | Tensor.fromBlob(floatArray, longArrayOf(1,3,1536,1536)) |
| result.numpy() | outputs[0].toTensor().getDataAsFloatArray() |
| with torch.no_grad(): | No equivalent needed (inference only) |

Python test -> Android deploy flow:
1. test_sharp_pte.py validates on Mac (PyTorch reference comparison)
2. adb push model.pte to device
3. ExecutorchSharp.kt loads and runs same model on ARM"""
    },
    {
        "id": "py2android_export_results",
        "layer": "python_to_android",
        "content": """Actual export results (measured Feb 15 2026):
PyTorch reference: 1,179,648 Gaussians, 19.3s on Mac CPU (M-series).

Exported models:
| Variant | Size | Status |
| sharp_full_fp32.pte | 2.4 GB | Exported successfully |
| sharp_full_fp16.pte | 1.2 GB | Exported successfully |
| sharp_full_int8.pte | N/A | PT2E quantization API incompatible with current ExecuTorch 1.1 |

Key: FP16 is the sweet spot for Android deployment.
- Half the size of FP32 (1.2GB vs 2.4GB)
- Mali-G715 has native FP16 support (1.5 TFLOPS)
- No precision loss visible in Gaussian output

INT8 requires ExecuTorch nightly or version-specific PT2E APIs.
Use FP16 until ExecuTorch stabilizes INT8 export."""
    },
    {
        "id": "py2android_tensor_format",
        "layer": "python_to_android",
        "content": """Tensor format mapping Python <-> Android:

Python (PyTorch):
  input: torch.Tensor [1, 3, 1536, 1536] float32, range [0, 1]
  output: torch.Tensor [N, 14] float32

Android (Kotlin/ExecuTorch):
  input: Tensor.fromBlob(FloatArray, longArrayOf(1, 3, 1536, 1536))
  output: outputs[0].toTensor().getDataAsFloatArray()

Preprocessing Python:
  from PIL import Image
  import torchvision.transforms as T
  transform = T.Compose([T.Resize((1536,1536)), T.ToTensor()])
  input = transform(Image.open(path)).unsqueeze(0)

Preprocessing Android (Kotlin):
  val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 1536, 1536, true)
  val pixels = IntArray(1536*1536)
  scaledBitmap.getPixels(pixels, 0, 1536, 0, 0, 1536, 1536)
  // Extract CHW float array normalized to [0,1]
  for (i in pixels.indices) {
      floatArray[i] = ((pixels[i] shr 16) and 0xFF) / 255f           // R
      floatArray[channelSize + i] = ((pixels[i] shr 8) and 0xFF) / 255f  // G
      floatArray[2*channelSize + i] = (pixels[i] and 0xFF) / 255f        // B
  }

Output format [N, 14] per Gaussian:
  [0-2]: x, y, z position
  [3-5]: sx, sy, sz scale
  [6-9]: qw, qx, qy, qz rotation quaternion
  [10]: opacity
  [11-13]: r, g, b color"""
    },
    {
        "id": "py2android_full_vs_component",
        "layer": "python_to_android",
        "content": """Full model vs Component model on Android:

WRONG (component mode - current):
  Python: patch_encoder processes 35 patches in batch [35, 3, 384, 384]
  Android: 35 separate Module.forward() calls, serial loop
  Result: 45s for encoding, but WRONG decoder (7MB head != 387MB real decoder)

CORRECT (full model - new):
  Python: model(image) -> single forward pass -> [N, 14] Gaussians
  Android: module.forward(EValue.from(imageTensor)) -> same single pass
  Result: model handles everything internally, correct output

The full model .pte contains the entire pipeline:
  sliding pyramid -> patch encoder -> image encoder -> decoder -> Gaussian head
All in ONE forward pass. No serial patch loop. No separate head.

Expected Android performance (full model FP16):
  Load: ~5-10s (1.2GB model)
  Inference: depends on Vulkan delegate support
  If Vulkan works: ~1-3 min (GPU-accelerated)
  If CPU only: ~5-10 min (similar to ONNX)"""
    },

    # ---- ANDROID DEV: EXECUTORCH SPLIT DEBUGGING (2026) ----
    {
        "id": "android_executorch_xnnpack_only",
        "layer": "android_lessons",
        "content": """ExecuTorch SHARP split: Use XNNPACK backend ONLY. No Vulkan or hybrid.

Vulkan on Mali-G715 can deadlock during first forward. Export with:
  python export_sharp_executorch_split4.py --weights sharp.pt --backend xnnpack --output-dir executorch_models

Backend selection: --backend xnnpack (default) or --backend portable.
Script uses XnnpackPartitioner only. No VulkanPartitioner.
Verify: adb logcat | grep -i xnnpack (should see XNNPACK, NOT Vulkan initialized)."""
    },
    {
        "id": "android_internal_storage",
        "layer": "android_lessons",
        "content": """Models on external storage (/sdcard/.../files/models) cause slow first forward.

External = FUSE layer, slow mmap, poor random access.
Internal = /data/data/com.furnit.android/files/models/, direct ext4, fast mmap.

ExecutorchSharp copies split .pte from external to internal on initialize().
First run: copy happens (~2.5GB). Subsequent: load from internal (5-30x faster first forward).
NcnnSharp does the same. Push to external via adb; app migrates to internal."""
    },
    {
        "id": "android_first_forward_stall",
        "layer": "android_lessons",
        "content": """ExecuTorch first module.forward() can take 10-60 seconds on mobile.

Module.load() loads file and parses graph (~243ms). Backend init happens on FIRST forward:
- Kernel setup, weight mmap, threadpool creation, execution graph allocation.
For 582MB ViT encoder: first forward = 10-60s. Subsequent forwards = 2-5s.

Fix 1: Preload + warmup on screen open (before user selects photo).
  preloadAndWarmup(): Module.load -> dummy forward -> destroy().
  Moves init cost to idle time.
Fix 2: Keep Part1 loaded if RAM allows (~582MB). Or warmup and destroy."""
    },
    {
        "id": "android_checkpoint_logs",
        "layer": "android_lessons",
        "content": """Checkpoint logs to pinpoint stall location in ExecuTorch split:

inferStreaming ENTER thread=... split=... full=...
inferStreaming calling inferSplitMode()
inferSplitMode ENTER thread=...
inferSplitMode scale done in Xms
part1 path=...
Part1 Module.load done in Xms
P1 PATCH 0 preprocess start / preprocess done
P1 PATCH 0 forward start  <- If stall here: first forward init (normal, 10-60s)
P1 PATCH 0 forward done in Xms
Part1 warmup done

If stuck before 'forward start': preprocess or tensor creation.
If stuck at 'forward start': first forward backend init (expected 10-60s)."""
    },
    {
        "id": "android_deferred_preprocess",
        "layer": "android_lessons",
        "content": """Defer large preprocess to reduce early memory pressure.

preprocessPatch(1536x1536) allocates ~38MB (IntArray + FloatArray).
Do NOT run at top of inferSplitMode. imageData only needed for Part 3/4.

Correct order: Part 1 (patches) -> Part 2 (tokens) -> THEN preprocessPatch(scaledBitmap) -> Part 3.
Releases memory from Part 1/2 before 38MB allocation. Reduces OOM and GC thrash."""
    },
    {
        "id": "android_export_commands",
        "layer": "android_lessons",
        "content": """Correct ExecuTorch split export and push commands:

cd android
python export_sharp_executorch_split4.py --weights /path/to/sharp.pt --backend xnnpack --output-dir executorch_models
./push_sharp_executorch_models.sh executorch_models

Verify export: ls -lh executorch_models/
  part1-3: ~500-600MB each, part4: ~700-800MB.
Wrong size or Vulkan export = different sizes or hang on Mali."""
    },
    # ---- ANDROID IMPLEMENTATION LESSONS (existing) ----
    {
        "id": "android_fp32_oom",
        "layer": "android_lessons",
        "content": """FP32 full model (2.4GB) crashes on Android with OOM.
2.4GB model + activations + Android runtime = exceeds device RAM.
Solution: use FP16 (1.2GB) or split mode."""
    },
    {
        "id": "android_fp16_input_type",
        "layer": "android_lessons",
        "content": """FP16 .pte model requires Half (float16) input tensor.
Error: 'expected Half but was Float'. Fix: Tensor.allocateHalfBuffer + halfFloatToShort.
Requires executorch-android:1.1.0 AAR."""
    },
    {
        "id": "android_7mb_head_wrong",
        "layer": "android_lessons",
        "content": """7MB gaussian_head.pte produces WRONG output (blue blob).
Root cause: NOT the real decoder. ONNX Part 4 = 387MB, 7 inputs.
ExecuTorch gaussian_head = 7MB, 1 input. Missing 6 inputs. Fix: use full or split model."""
    },
    {
        "id": "android_model_search_order",
        "layer": "android_lessons",
        "content": """ExecutorchSharp model priority: 1) Split (4 parts), 2) Full FP16/FP32, 3) Legacy component.
Search: internal first, then external, then /data/local/tmp/furnit/.
Copies external->internal on init for faster mmap."""
    },

    # ---- PYTORCH NATIVE / TORCHSCRIPT (bleeding-edge roadmap) ----
    {
        "id": "pt_torchscript_cpp",
        "layer": "pytorch_native",
        "content": """PyTorch C++ frontend: Load .pt and run inference in pure C++. No Python at runtime.
TorchScript: torch.jit.trace(model, example_input) -> scripted.pt. Saved and loaded without Python.
Docs: docs.pytorch.org/tutorials/ - C++ frontend tutorial.
Milestone: sharp.pt -> sharp_scripted.pt -> C++ inference engine. First step to custom inference."""
    },
    {
        "id": "pt_torchscript_ir",
        "layer": "pytorch_native",
        "content": """TorchScript internals for parsing model structure:
- torch::jit::Graph: IR representation of model
- torch::jit::Module: C++ loader for scripted models
- torch.jit.trace vs torch.jit.script: trace captures execution, script parses Python.
Search: TorchScript IR, torch::jit::Graph, torch::jit::Module C++.
Critical for graph analysis and custom backend implementation."""
    },
    {
        "id": "pt_custom_backend",
        "layer": "pytorch_native",
        "content": """Custom PyTorch backend / ATen operator implementation:
- ATen: PyTorch tensor library, backend for operators
- torch dispatcher: routes ops to implementations
- Custom kernel: implement backward-compatible op, register with dispatcher
Search: PyTorch custom backend tutorial, ATen operator backend implementation, torch dispatcher.
Teaches how to implement your own kernel execution (e.g. ARM GEMM for Linear)."""
    },

    # ---- TORCHSCRIPT JIT API (docs.pytorch.org/docs/stable/jit.html) ----
    {
        "id": "jit_deprecation",
        "layer": "pytorch_native",
        "content": """TorchScript is deprecated. PyTorch recommends torch.export instead.
Docs: docs.pytorch.org/docs/stable/jit.html
For new projects: use torch.export. For existing .pt/.ptl mobile deployment (PyTorch Mobile, LiteModuleLoader): TorchScript still works."""
    },
    {
        "id": "jit_creating_code",
        "layer": "pytorch_native",
        "content": """TorchScript creating code - key APIs (docs.pytorch.org/docs/stable/jit.html):

torch.jit.script(fn) - Compile Python function to TorchScript. Use for control flow.
torch.jit.trace(model, example_input) - Trace execution, capture graph. Use for models with limited control flow.
torch.jit.trace_module(module, {"forward": example}) - Trace module methods.
torch.jit.save(scripted, "model.pt") - Save for loading in separate process.
torch.jit.load("model.pt") - Load ScriptModule or ScriptFunction.

For SHARP: trace works (single forward pass). trace(model, torch.randn(1,3,1536,1536))."""
    },
    {
        "id": "jit_scriptmodule",
        "layer": "pytorch_native",
        "content": """TorchScript ScriptModule and ScriptFunction (docs.pytorch.org/docs/stable/jit.html):

ScriptModule - Wrapper for C++ torch::jit::Module. Has methods, attributes, parameters.
ScriptFunction - Single function, no attributes. Functionally equivalent to ScriptModule for one forward().

Both are executable without Python. Saved with torch.jit.save(), loaded with torch.jit.load().
C++ API: torch::jit::load("model.pt") returns Module. module.forward({input}) for inference."""
    },
    {
        "id": "jit_optimization",
        "layer": "pytorch_native",
        "content": """TorchScript optimization APIs (docs.pytorch.org/docs/stable/jit.html):

torch.jit.freeze(scripted) - Freeze ScriptModule, inline submodules and attributes as constants. Reduces overhead.
torch.jit.optimize_for_inference(scripted) - Optimization passes for inference (drop training-only ops).
torch.jit.enable_onednn_fusion(enabled) - OneDNN JIT fusion for x86.
torch.jit.set_fusion_strategy() - Control fusion type and specializations.

For mobile: optimize_for_mobile(traced) -> _save_for_lite_interpreter() produces .ptl."""
    },
    {
        "id": "jit_decorators",
        "layer": "pytorch_native",
        "content": """TorchScript decorators (docs.pytorch.org/docs/stable/jit.html):

@torch.jit.ignore - Leave as Python function, compiler skips it.
@torch.jit.unused - Replace with exception (for dead code paths).
@torch.jit.interface - Annotate classes for type refinement.
torch.jit.isinstance(x, T) - Container type refinement in TorchScript.
torch.jit.Attribute - Pass-through for class attribute type hint.
torch.jit.annotate(type, value) - Give type of value to compiler."""
    },
    {
        "id": "jit_fork_wait",
        "layer": "pytorch_native",
        "content": """TorchScript async (docs.pytorch.org/docs/stable/jit.html):

torch.jit.fork(func, *args) - Create async task, return Future.
torch.jit.wait(future) - Block until Future completes, return result.

Use for parallel execution within traced model. Fork launches func; wait collects result."""
    },

    # ---- VEDIC MATHS ----
    {
        "id": "vedic_overview",
        "layer": "vedic_maths",
        "content": """Vedic Mathematics: ancient Indian system of mental calculation from the Vedas.
16 sutras (aphorisms) and 13 sub-sutras. Enables fast mental arithmetic without calculators.
Bharati Krishna Tirthaji (1884-1960) reconstructed the system. Applications: multiplication,
division, squaring, square roots, cube roots, algebra, simultaneous equations.
Key sutras: Nikhilam Navatashcaramam Dashatah, Urdhva-tiryagbhyam, Ekadhikena Purvena,
Antyayordashake'pi, Sopaantyadvayamantyam. Used in competitive exams, mental math contests."""
    },
    {
        "id": "vedic_nikhilam",
        "layer": "vedic_maths",
        "content": """Nikhilam Navatashcaramam Dashatah: All from 9 and the last from 10.
For multiplication of numbers near a base (10, 100, 1000):
97 x 98: Base 100. Deficiencies: -3, -2. Right side: (-3)x(-2)=6. Left: 97-2 or 98-3 = 95. Answer: 9506.
Works for subtraction too: 1000 - 357 = 6, 4, 3 (9-3, 9-5, 10-7). Answer: 643.
Also: 9's complement for digit sums. Vertically: subtract from base. Crosswise: add deficiencies."""
    },
    {
        "id": "vedic_urdhva",
        "layer": "vedic_maths",
        "content": """Urdhva-tiryagbhyam: Vertically and Crosswise. Universal multiplication formula.
For 2-digit: axb = (a1b1) | (a1b2+a2b1) | (a2b2). Right to left: units, tens, hundreds.
23 x 45: 2x4=8, 2x5+3x4=22 (carry 2), 3x5=15. Result: 8|22|15 -> 8+2|2+1|5 -> 1035.
Extends to 3+ digits. Polynomial multiplication: same pattern. Used in digital circuits,
residue number systems, error-correcting codes (Reed-Solomon). Basis for many fast multipliers."""
    },
    {
        "id": "vedic_ekadhikena",
        "layer": "vedic_maths",
        "content": """Ekadhikena Purvena: By one more than the previous one.
Squaring numbers ending in 5: 25² = 2x3 | 25 = 625. 75² = 7x8 | 25 = 5625.
Rule: (n5)² = n(n+1) | 25. Works because (10n+5)² = 100n²+100n+25 = 100n(n+1)+25.
Division by 9: 1/9 = 0.111..., 2/9 = 0.222..., 13/9 = 1.444... (recurring quotient).
Repeating decimals: numerator digits recur; add carry from previous."""
    },
    {
        "id": "vedic_squaring",
        "layer": "vedic_maths",
        "content": """Vedic squaring techniques:
1. Yavadunam: Whatever the extent of its deficiency. For numbers near base: (a-d)² = a² - 2ad + d².
   96²: Base 100, d=4. 100-2x4=92 (left), 4²=16 (right). 9216.
2. Duplex (D): D(a)=a², D(ab)=2ab, D(abc)=2ac+b². For 2-digit: D(ab)=2ab.
3. Anurupyena: Proportionally. Adjust base for numbers not near 10^n.
4. Sopaantyadvayamantyam: Ultimate and twice the penultimate. For squaring numbers ending in 5."""
    },
    {
        "id": "vedic_division",
        "layer": "vedic_maths",
        "content": """Vedic division methods:
1. Nikhilam: For divisors near base. 1/19: 19's complement from 20 is 1. Multiply by 1, add carry.
   Result: 0.052631578947368421 (recurring).
2. Paravartya: Transpose and apply. For divisors like 9, 99.
3. Dhvajanka: Flag method. Write divisor with 'flag' digit. Standard long division shortcut.
4. Straight division (Dhwajanka variant): Combines flag with recurring pattern.
Used for recurring decimals, divisibility checks, ratio calculations."""
    },
    {
        "id": "vedic_mental_compute",
        "layer": "vedic_maths",
        "content": """Vedic maths for mental computation and algorithms:
- Crossover to computer arithmetic: Urdhva-tiryagbhyam used in hardware multipliers (low latency).
- Karatsuba-like: Split multiplication into smaller products, combine. Similar to divide-and-conquer.
- Digit recurrence: Division by convergence (Newton-Raphson style in some sutras).
- Polynomial evaluation: Horner's method has Vedic analogues (Vinculum for negative digits).
- Coding: Implementing sutras in Python/JS for education apps, mental math trainers.
- Not a replacement for floating-point libraries; useful for integer arithmetic, pedagogy."""
    },

    # ---- SGEMM (Single-precision GEMM) ----
    {
        "id": "sgemm_overview",
        "layer": "sgemm",
        "content": """SGEMM (Single-precision General Matrix Multiply): C = alpha*A*B + beta*C.
Core operation in ML inference. All linear layers, attention (Q@K^T, scores@V), MLP fc layers.
BLAS level 3. Optimized via NEON, ARM matrix units, GPU compute shaders.
Formula: C[m,n] = alpha * sum_k A[m,k]*B[k,n] + beta*C[m,n]
M,N,K dimensions. Row-major: CblasRowMajor, ldA=K, ldB=N, ldC=N."""
    },
    {
        "id": "sgemm_blas_cblas",
        "layer": "sgemm",
        "content": """BLAS cblas_sgemm for SGEMM on iOS/macOS:
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
Transposed A (A^T*B): CblasTrans for A. Transposed B (A*B^T): CblasTrans for B.
Example - FurnitureFitView: blas_sgemm_rowmajor_transA for C = planes^T * B (no matrix copy).
Accelerate framework on iOS provides optimized SGEMM via vDSP/BLAS.
For batched inference: loop over batch or use cblas_sgemm_batch."""
    },
    {
        "id": "sgemm_ncnn",
        "layer": "sgemm",
        "content": """NCNN use_sgemm_convolution:
opt.use_sgemm_convolution = true  // GEMM for conv - major speedup
Convolution im2col + GEMM. SGEMM replaces im2col+gemm with optimized path.
In Furnit: sharp_ncnn_components.h and sharp_ncnn.cpp set use_sgemm_convolution = true.
InnerProduct (fully connected) also uses GEMM. SGEMM is the workhorse for conv and fc."""
    },
    {
        "id": "sgemm_arm_acl",
        "layer": "sgemm",
        "content": """ARM Compute Library SGEMM:
ACL provides NEONNEMMKernel, NEONGEMMMatrixMultiplyKernel for ARM CPUs.
Uses NEON SIMD, SVE, matrix acceleration instructions.
NEGEMM, NEGEMMLowpMatrixMultiplyCore for INT8 GEMM.
For transformers: NEGEMM is used for Q/K/V projections and MLP layers.
Integration: load ACL, create NEGEMM kernel, configure dimensions, run.
Docs: ARM Compute Library GEMM, NEON GEMM tutorial."""
    },
    {
        "id": "sgemm_transformer_ops",
        "layer": "sgemm",
        "content": """SGEMM in transformer inference - where it's used:
1. Q/K/V projections: Linear(dim, dim) = [B,seq,dim] @ [dim,dim] - 3 SGEMMs per block
2. Attention Q@K^T: [B,heads,seq,dim] @ [B,heads,dim,seq] - SGEMM
3. Attention scores@V: [B,heads,seq,seq] @ [B,heads,seq,dim] - SGEMM
4. Output projection: Linear after attention - SGEMM
5. MLP: fc1 [seq,1024]->[1024,4096], fc2 [seq,4096]->[4096,1024] - 2 SGEMMs per block
SHARP 24 blocks: 24*(3 + 2 + 2 + 2) = 216 large SGEMMs per patch. Dominates compute.
Optimize: NEON/SVE, tiling, FP16, INT8 quantization."""
    },
    {
        "id": "native_pt_merge_arraycopy",
        "layer": "sgemm",
        "content": """Native .pt / LiteRT merge optimization: replace nested for-for-for with System.arraycopy.
mergeOnePatchInto copies patch regions to merged output. Per row the source and destination
are contiguous. Use System.arraycopy(patch, srcOff, output, dstOff, copyW) per row instead of
for(dx) element-by-element. Reduces ~331K individual assignments to ~2.5K arraycopy calls.
Same pattern in LiteRTSharp.mergeSinglePatchInPlace. For reshape (tokens to spatial):
destination is strided (c*hw apart) so arraycopy doesn't apply; keep 2 nested loops."""
    },
    {
        "id": "sgemm_mobile_optimization",
        "layer": "sgemm",
        "content": """SGEMM optimization on mobile:
1. Use FP16: half precision, 2x throughput on Mali-G715, ARM matrix units
2. INT8: 4x less data, use NEONNEMMKernel low-precision
3. Tiling: fit L1/L2 cache, reduce memory bandwidth bottleneck
4. Batch size 1: mobile inference, avoid batching overhead
5. NCNN: use_sgemm_convolution = true for conv layers
6. GPU: Vulkan compute shaders, ExecuTorch implements GEMM on GPU
Memory bandwidth is often the bottleneck, not raw FLOPS. Smaller tiles help."""
    },

    # ---- ARM COMPUTE LIBRARY ----
    {
        "id": "acl_overview",
        "layer": "arm_compute",
        "content": """ARM Compute Library (ACL): Optimized primitives for ARM CPUs.
GEMM, convolution, normalization, activation, transformer ops.
Uses NEON, SVE, matrix acceleration. How Apple-level performance is achieved on ARM.
Docs: learn.arm.com - PyTorch digit classification, Android deployment.
Search: ARM Compute Library GEMM tutorial, ACL transformer inference, ACL Android example."""
    },
    {
        "id": "acl_android",
        "layer": "arm_compute",
        "content": """ARM Compute Library Android deployment:
1. Export model 2. Load in Android 3. Preprocess 4. Run inference 5. Optimize with hardware acceleration.
ACL provides optimized implementations using NEON, SVE, matrix acceleration.
Backbone of high-performance ARM inference. Use for custom C++ engine."""
    },

    # ---- PIPELINE ARCHITECTURE ----
    {
        "id": "sharp_native_pipeline",
        "layer": "pytorch_native",
        "content": """SHARP native inference pipeline (Meta/Apple style):

sharp.pt -> torch.jit.trace() -> sharp_scripted.pt -> C++ inference engine -> ARM Compute Library kernels -> Android runtime -> kernel scheduler tuning

Alternative (no ExecuTorch): Python export only -> TorchScript -> Native C++ -> ACL kernels -> Android kernel-optimized scheduling -> direct mmap weights. Full FP32. Maximum performance."""
    },
    {
        "id": "executorch_backend_arch",
        "layer": "pytorch_native",
        "content": """ExecuTorch backend architecture (instructive even if not using ExecuTorch):
1. Export PyTorch model graph 2. Compile and optimize operators 3. Execute via lightweight runtime.
Docs: pytorch.org/executorch - export-to-executorch-tutorial.
Explains graph lowering. Can replicate parts manually for custom engine."""
    },
    {
        "id": "onnx_transformer_opt",
        "layer": "transformer_opt",
        "content": """ONNX Runtime transformer optimization (techniques applicable elsewhere):
- Operator fusion (combine ops to reduce overhead)
- Attention kernel optimization
- Graph rewriting
Docs: onnxruntime.ai/docs/performance/transformers-optimization.
Even without ONNX Runtime, explains optimization techniques for ViT/SHARP."""
    },

    # ---- ANDROID ML ARCHITECTURE ----
    {
        "id": "android_ml_deploy",
        "layer": "android_ml_arch",
        "content": """Android ML deployment model (ARM official):
1. Export model 2. Load model in Android 3. Preprocess input 4. Run inference 5. Optimize with hardware acceleration.
Learn: learn.arm.com - PyTorch digit classification, intro-android.
System-level view for Furnit SHARP deployment."""
    },
    {
        "id": "nnapi_hardware",
        "layer": "android_ml_arch",
        "content": """Android NNAPI hardware acceleration:
NNAPI provides: CPU, GPU, NPU acceleration. Unified interface.
Enable hardware-accelerated inference across CPUs, GPUs, NPUs.
Docs: developer.arm.com - improve PyTorch app performance with Android NNAPI support.
Integration path for PyTorch/TorchScript apps on Android."""
    },

    # ---- TRANSFORMER KERNELS ----
    {
        "id": "transformer_attention_kernel",
        "layer": "transformer_kernels",
        "content": """Transformer attention kernel implementation (where most SHARP performance lies):
- Self-attention: Q@K^T, softmax, scores@V per head
- Search: transformer attention implementation C++, flash attention implementation tutorial
- ViT: 577 tokens, 16 heads, heavy GEMM. Optimize with ACL or custom NEON.
Flash attention: memory-efficient, tiled computation. Applicable to SHARP encoder."""
    },

    # ---- GOOGLE ML CRASH COURSE: NEURAL NETWORKS ----
    # Source: https://developers.google.com/machine-learning/crash-course/neural-networks
    {
        "id": "nn_intro",
        "layer": "ml_fundamentals",
        "content": """Neural networks (Google ML Crash Course - Neural networks intro):
Neural networks automatically identify nonlinear patterns in data, eliminating manual feature cross experimentation.
Nonlinear means you cannot accurately predict with a model of the form b + w1*x1 + w2*x2 (decision surface is not a line).
Feature crosses (e.g. x3 = x1*x2) can represent nonlinear relationships in a linear model, but require manual experimentation.
Neural networks learn optimal feature crosses automatically during training to minimize loss.
Key components: nodes, hidden layers, activation functions. Training uses backpropagation.
Source: developers.google.com/machine-learning/crash-course/neural-networks"""
    },
    {
        "id": "nn_nodes_hidden_layers",
        "layer": "ml_fundamentals",
        "content": """Neural network nodes and hidden layers (Google ML Crash Course):
Linear model: y' = b + w1*x1 + w2*x2. Input nodes (blue), output node (green). Parameters: weights and bias.
Hidden layers: additional layers between input and output. Neurons in hidden layer computed as linear combo of previous layer + weights + bias.
Parameter count: 4 neurons in hidden layer = 4*(3 weights + 1 bias) = 16, plus output 4+1 = 5, total 21 parameters.
CRITICAL: Linear operations on linear operations remain linear. Hidden layers alone CANNOT learn nonlinearities - need activation functions.
Source: developers.google.com/machine-learning/crash-course/neural-networks/nodes-hidden-layers"""
    },
    {
        "id": "nn_activation_functions",
        "layer": "ml_fundamentals",
        "content": """Neural network activation functions (Google ML Crash Course):
Activation function: nonlinear transform of neuron output before passing to next layer. Enables learning nonlinear relationships.
Common activations: (1) Sigmoid: F(x)=1/(1+e^-x), output 0-1. (2) Tanh: output -1 to 1. (3) ReLU: F(x)=max(0,x).
ReLU preferred: less susceptible to vanishing gradient, easier to compute. Can cause dead ReLU (output stuck at 0).
Node value formula: sigma(w·x + b) where sigma is activation. Stacking nonlinearities on nonlinearities models complex relationships.
Keras provides many activation functions. Start with ReLU.
Source: developers.google.com/machine-learning/crash-course/neural-networks/activation-functions"""
    },
    {
        "id": "nn_backpropagation",
        "layer": "ml_fundamentals",
        "content": """Neural network training: backpropagation (Google ML Crash Course):
Backpropagation: primary training algorithm for neural networks. Enables gradient descent for multi-layer networks. Keras handles it automatically.
Vanishing gradients: lower-layer gradients become very small (product of many small terms). Layers train slowly or not at all. ReLU helps.
Exploding gradients: large weights cause excessively large gradients. Mitigate with batch normalization or lower learning rate.
Dead ReLU units: weighted sum < 0 causes ReLU to output 0 forever, cutting gradient flow. Lower learning rate or use LeakyReLU.
Dropout regularization: randomly drop unit activations during training. 0.0=no dropout, 1.0=drop all (model learns nothing). Higher = stronger regularization.
Source: developers.google.com/machine-learning/crash-course/neural-networks/backpropagation"""
    },
    {
        "id": "nn_multiclass_classification",
        "layer": "ml_fundamentals",
        "content": """Neural network multi-class classification (Google ML Crash Course):
Two approaches: (1) One-vs-all: N separate binary classifiers, one per class. Sigmoid on each output. Probabilities independent, may not sum to 1.
(2) One-vs-one (softmax): probabilities relative to all classes, sum to 1.0. Softmax: p(y=j|x) = exp(wj·x+bj) / sum over k of exp(wk·x+bk).
Softmax layer: hidden layer before output must have same number of nodes as output. Full softmax cheap for few classes, expensive for many.
Candidate sampling: softmax over positive + random sample of negatives. Efficient for large number of classes.
Multi-label (example in multiple classes): use multiple logistic regressions, NOT softmax.
Source: developers.google.com/machine-learning/crash-course/neural-networks/multi-class"""
    },

    # ---- RAG STRUCTURE FOR FURNIT ----
    {
        "id": "rag_dataset_structure",
        "layer": "android_ml_arch",
        "content": """Recommended RAG dataset structure for Furnit code generation:

rag/pytorch/ - torchscript, jit, dispatcher
rag/arm_compute_library/ - gemm, attention, android
rag/android/ - nnapi, scheduler, memory
rag/transformer/ - attention, kernel, optimization

Organize knowledge base this way. Code generation becomes powerful for custom inference engine."""
    },
    {
        "id": "sharp_ultimate_arch",
        "layer": "pytorch_native",
        "content": """Ultimate SHARP inference engine architecture (no ExecuTorch):

Python (export only) -> TorchScript model -> Native C++ engine -> ARM Compute Library kernels -> Android kernel optimized scheduling -> Direct mmap weights

Full FP32. Maximum performance. No framework overhead. Production pathway used by Apple, Meta, ARM."""
    },
    # ---- PLY WRITER OPTIMIZATIONS (Feb 2026) ----
    {
        "id": "ply_buffer_recycling",
        "layer": "performance",
        "content": """PLY writer buffer recycling optimization (all 8 backends).

Problem: Each inference call allocated ByteBuffer.allocateDirect(~127KB) for the PLY batch buffer plus 5 FloatArrays.
DirectByteBuffer cleanup depends on GC finalizers, which are non-deterministic on Android. Under memory pressure
(Part 4 just finished), these lingering buffers cause native memory fragmentation.

Solution: Promote batch buffer and scratch arrays to class-level reusable fields:
  private val plyBatch: ByteBuffer by lazy { ByteBuffer.allocateDirect(BYTES_PER_VERTEX * PLY_BATCH_SIZE).apply { order(ByteOrder.LITTLE_ENDIAN) } }
  private val plyPositions = FloatArray(PLY_BATCH_SIZE * 3)
  private val plyScales = FloatArray(PLY_BATCH_SIZE * 3)
  private val plyRotations = FloatArray(PLY_BATCH_SIZE * 4)
  private val plyColors = FloatArray(PLY_BATCH_SIZE * 3)
  private val plyOpacity = FloatArray(PLY_BATCH_SIZE)

In processGaussianOutput: val batchBuffer = plyBatch; batchBuffer.clear() at start, batchBuffer.clear() after each batch write.
Applied to: SplitOnnxSharp, SplitOnnxFp16Sharp, OnnxInt8Sharp, ExecutorchFp16Sharp, ExecutorchSharp, LiteRTSharp, OnnxSharp, SharpService."""
    },
    {
        "id": "ply_zero_sh_buffer",
        "layer": "performance",
        "content": """PLY writer zero SH block optimization (Direct-to-Direct bulk copy).

Problem: Each Gaussian vertex writes 45 zero SH coefficients (180 bytes). Old approaches:
  - repeat(45) { batchBuffer.putFloat(0f) } → 45 individual JNI calls per vertex × ~1.18M vertices
  - batchBuffer.put(ByteArray(180)) → heap array → JNI → native copy per vertex (JNI crossing)

Solution: Pre-allocate a Direct ByteBuffer for the zero SH block:
  private val zeroSHBuffer: ByteBuffer by lazy { ByteBuffer.allocateDirect(45 * 4).apply { order(ByteOrder.LITTLE_ENDIAN) } }

In the inner loop: zeroSHBuffer.clear(); batchBuffer.put(zeroSHBuffer)
Direct→Direct ByteBuffer.put() uses native memcpy (Unsafe.copyMemory), no JNI heap crossing.
For 1.18M Gaussians: eliminates ~1.18M JNI crossings × 180 bytes = ~212MB of JNI traffic."""
    },
    {
        "id": "ply_batch_io_strategy",
        "layer": "performance",
        "content": """PLY batch I/O strategy: 512 vertices per write syscall.

Each vertex = 62 floats = 248 bytes (BYTES_PER_VERTEX). Writing 1.18M vertices one at a time = 1.18M syscalls.
Batch: accumulate 512 vertices into a 127KB DirectByteBuffer, then one outChannel.write() syscall.
Result: ~2304 syscalls instead of 1.18M. System calls are high-latency "warehouse trips" in the memory hierarchy.

PLY_BATCH_SIZE = 512 chosen to fit L1 cache (~128KB) while amortizing syscall overhead.
Format: binary_little_endian PLY with 62 properties per vertex (xyz, normals, f_dc 3, f_rest 45, opacity, scale 3, rot 4).

Logging to measure: PLY batch loop Xms total (I/O write=Yms, compute=Zms), PLY throughput XMB @ Y.YMB/s.
Look for these in logcat with tags SplitOnnxSharp, ExecutorchFp16Sharp."""
    },
    {
        "id": "ply_lut_optimization",
        "layer": "performance",
        "content": """PLY writer LUT optimizations: LOGIT_LUT and LN_LUT.

LOGIT_LUT (1024 entries): maps opacity [0,1] to logit ln(p/(1-p)). Avoids ln() and division per vertex.
  val LOGIT_LUT = FloatArray(1024) { i -> val p = ...; ln(p / (1f - p)) }
  Usage: LOGIT_LUT[(rawOpacity * lutScale).toInt().coerceIn(0, LOGIT_LUT_SIZE - 1)]

LN_LUT (2048 entries): maps scale values [0.001, 5.0] to ln(x). Avoids ln() per vertex per axis (3 axes).
  fun lnLut(x: Float): Float { return LN_LUT[((x - LN_LUT_MIN) * LN_LUT_SCALE).toInt()] }

For 1.18M Gaussians: eliminates ~4.72M ln() calls (1 opacity + 3 scales per vertex).
Math operations are fast individually but become a bottleneck at 10^6 repetitions."""
    },
    {
        "id": "ply_mmap_tensor_loading",
        "layer": "performance",
        "content": """PLY writer tensor loading via memory-mapped files (SplitOnnxSharp).

Gaussian output tensors (positions, scales, rotations, colors, opacity) are saved to disk between model parts.
Loading for PLY: FileChannel.map(READ_ONLY, offset, size) → FloatBuffer.
This keeps JVM heap lean (~0 bytes for tensor data). OS handles paging from disk cache.

File format: [4 bytes: numDims][numDims * 8 bytes: shape][float data].
Memory map starts at dataOffset = 4 + numDims * 8.

For 1.18M Gaussians: positions = 1.18M * 3 * 4 = ~14MB, scales = ~14MB, rotations = ~19MB, colors = ~14MB, opacity = ~4.7MB.
Total mmap: ~66MB. PLY mmap setup takes ~1-5ms (just page table entries, no data copy)."""
    },

    # ---- CONV+BN OPERATION FUSION (Feb 2026) ----
    {
        "id": "conv_bn_fusion",
        "layer": "quantization",
        "content": """Conv+BN operation fusion in SHARP export scripts.

Problem: SHARP decoder (FPN/UNet) has Conv2d + BatchNorm2d pairs. BN computes y = (x - mean) / sqrt(var + eps) * weight + bias.
This creates intermediate tensors (normalized activations) that consume memory during forward pass.

Solution: fuse_conv_bn(model) recursively walks the model tree, finds Conv2d+BatchNorm2d pairs among direct children
of each module, and fuses them via torch.ao.quantization.fuse_modules(module, pairs, inplace=True).
Fused Conv absorbs BN parameters: W_fused = W_conv * (bn_weight / sqrt(bn_var + eps)), b_fused = (b_conv - bn_mean) * bn_weight / sqrt(bn_var + eps) + bn_bias.

Applied to: export_sharp_executorch_split4.py, export_sharp_executorch_fp16.py, export_sharp_onnx_single.py, export_sharp_litert_split.py.
Called after predictor.eval() and before creating part wrappers.

Benefits: ~15-20% reduction in peak decoder activation memory, smaller exported models, fewer ops for runtime to schedule.
ViT encoder blocks (LayerNorm, not BatchNorm) are unaffected — fusion targets the decoder Conv+BN pairs only."""
    },

    # ---- MULTI-BACKEND ARCHITECTURE (Feb 2026) ----
    {
        "id": "multi_backend_architecture",
        "layer": "sharp_model",
        "content": """SHARP multi-backend architecture on Android (Feb 2026).

8 selectable backends in Settings, each a separate implementation class:
| Backend | Class | Model files | Status |
|---------|-------|-------------|--------|
| ONNX (default) | SplitOnnxSharp | sharp_part{1-4}.onnx + .data | Working, ~5 min |
| ONNX FP16 | SplitOnnxFp16Sharp | sharp_part{1-4}_fp16.onnx | Experimental (CPU FP16 kernel gaps) |
| ONNX INT8 | OnnxInt8Sharp | sharp_single_int8.onnx (715MB) | Single model, crashes at Part 4 activation memory |
| ExecuTorch FP32 | ExecutorchSharp | sharp_split_part{1-4}.pte | Working with XNNPACK |
| ExecuTorch FP16 | ExecutorchFp16Sharp | sharp_split_part{1-3}_fp16.pte + chunked Part 4 | Working, ~3.4 min |
| LiteRT | LiteRTSharp | sharp_part{1-4}_fp16.tflite + chunked 4a/4b | Part 4 crashes (decoder OOM) |
| NativePt | NativePtSharp | sharp_scripted_part{1-4}.ptl | TorchScript Lite |
| NCNN | NcnnSharp | component .ncnn.bin/.param files | ~23 min, serial patches |

SharpService.kt orchestrates: reads backend from SharedPreferences, initializes the selected class, dispatches inference.
BackendConfig.kt: feature flags (ENABLE_ONNX_FP16, ENABLE_EXECUTORCH_FP16, etc.).
SettingsActivity.kt: radio buttons for each backend in Developer section."""
    },
    {
        "id": "executorch_fp16_backend",
        "layer": "executorch",
        "content": """ExecuTorch FP16 backend implementation (ExecutorchFp16Sharp.kt).

Mixed-precision split: Parts 1-3 exported as FP16 (~290MB each), Part 4 as FP32 (~755MB).
Part 4 must be FP32 because F.interpolate (bilinear) in the decoder upcasts FP16→FP32 on CPU,
causing RuntimeError: 'Input type (float) and bias type (c10::Half) should be the same'.

FP16 tensor handling:
  halfFloatToShort(f: Float): Short — converts FP32 to IEEE 754 FP16 representation
  halfShortToFloat(s: Short): Float — converts FP16 back to FP32
  createHalfTensor(data: FloatArray, shape: LongArray): Tensor — creates FP16 tensor from FP32 data
  getOutputAsFloatArray(tensor: Tensor): FloatArray — reads FP16 or FP32 output

Chunked Part 4 for memory: part4a_chunk_512_fp16.pte (FP16), part4a_chunk_65_fp16.pte (FP16), part4b_fp16.pte (FP32 decoder).
Sequentially loaded and destroyed to reduce peak memory from ~755MB to max(~290MB, ~178MB).

Measured performance: Total 205,876ms (~3.4 min). P1+P2=97,975ms, P3=2,791ms, P4(chunked)=93,774ms.
System memory after Part 4b: 0MB available (tight but no crash thanks to chunking)."""
    },
    {
        "id": "chunked_part4_strategy",
        "layer": "performance",
        "content": """Chunked Part 4 strategy for memory-constrained decoder inference.

Problem: Part 4 (decoder + Gaussians) is the largest part (~755-828MB model + ~4GB peak activations).
Causes LMK (Low Memory Killer) on Android across ALL backends (ONNX, LiteRT, ExecuTorch).

Solution: Split Part 4 into 3 sub-chunks loaded and destroyed sequentially:
  Part 4a chunk 512: ViT blocks 12-23 on first 512 tokens → ~577-613MB
  Part 4a chunk 65: ViT blocks 12-23 on remaining 65 tokens → same size
  Part 4b: Decoder + Gaussian output from concatenated normalized tokens → ~178-186MB

Export: Each chunk is a separate .pte/.onnx/.tflite file.
Runtime: Load chunk → forward → save output to disk → destroy module → System.gc() → load next chunk.
Peak memory reduced from ~4GB to max(single_chunk_activations), typically ~1-2GB.

Implemented in: SplitOnnxSharp (ONNX FP32), ExecutorchFp16Sharp (ExecuTorch FP16), LiteRTSharp (LiteRT).
Detection: isChunkedPart4Available() checks if part4a/part4b files exist. Falls back to single Part 4 if not.
Also chunked Part 1 in SplitOnnxSharp: Part 1a (blocks 0-11) + Part 1b (blocks 12-18) for same reason."""
    },
    {
        "id": "memory_wall_android_ml",
        "layer": "performance",
        "content": """Memory wall problem for large vision models on Android.

The 'memory wall' = memory bandwidth and capacity limit inference speed more than compute.
SHARP model: 702M params (2.6GB FP32), decoder activations peak ~4GB, on a device with ~4-6GB available RAM.

Strategies implemented in Furnit to address memory wall:
1. FileChannel.map(READ_ONLY): OS-level data movement, keeps JVM heap lean. ONNX Runtime accesses weights via mmap.
2. Split models: 4 parts loaded/unloaded sequentially. Peak = max(single_part) not sum(all_parts).
3. Chunked decoder: Part 4 split into 3 sub-chunks to cap peak activation memory.
4. Reusable DirectByteBuffers: plyBatch, reusableSaveChunk — avoids GC-dependent buffer cleanup.
5. Conservative thread counts: Part 4 uses 2 threads (not all cores) to reduce per-thread GEMM memory.
6. Arena allocator OFF for Part 4: prevents pre-allocation of large memory pools.
7. System.gc() + Thread.sleep(150) before Part 4: nudges finalizer to release dead DirectByteBuffers.
8. FP16 models: halve weight size and memory bandwidth (Parts 1-3 in ExecuTorch FP16).
9. INT8 quantization: quarter weight size for ONNX INT8 backend.
10. Conv+BN fusion: eliminates BN intermediate tensors, ~15-20% decoder memory reduction."""
    },
    {
        "id": "onnx_session_options_tuning",
        "layer": "onnx_runtime",
        "content": """ONNX Runtime session options tuning for SHARP split model (Feb 2026).

Per-part tuning in SplitOnnxSharp.buildSessionOptions(partNumber):
| Part | OptLevel | Threads | Arena | MemPattern | Notes |
|------|----------|---------|-------|------------|-------|
| 1-3 | ALL_OPT | all cores | ON | true | Max throughput for encoder |
| 4 | ALL_OPT | 2 | OFF | true | Cap GEMM memory in decoder |

Key config entries (via addConfigEntry):
  session.use_mmap = 1          → weights stay on disk, paged by OS
  session.enable_mem_reuse = 1  → reuse activation buffers across ops
  session.intra_op.allow_spinning = 1 → reduce thread wake-up latency

ONNX FP16 (SplitOnnxFp16Sharp): EXTENDED_OPT instead of ALL_OPT to avoid com.microsoft.Gelu fusion
that has no FP16 CPU kernel. node_block_list in convert_onnx_fp16.py keeps LayerNorm in FP32.

ONNX INT8 (OnnxInt8Sharp single model): NO_OPT, 2 threads, no arena, disable prepacking — conservative
because single 715MB INT8 model has massive FP32 activations (INT8 weights, FP32 compute)."""
    },
    {
        "id": "ply_writer_logging",
        "layer": "android_lessons",
        "content": """PLY writer diagnostic logging (added Feb 2026).

SplitOnnxSharp logcat output during PLY write:
  PLY writer: reusable buffers (plyBatch=124KB, zeroSH=180B direct)  ← confirms buffer reuse
  PLY mmap: Xms (5 tensor files memory-mapped)                       ← mmap setup cost
  PLY batch loop: Xms total (I/O write=Yms, compute=Zms)            ← compute vs I/O breakdown
  PLY throughput: 279MB @ XX.XMB/s, batches=2304                     ← throughput measurement
  PLY memory after write: JVM: XXX/512MB, System: XXXXmb available   ← memory impact
  Breakdown: P1=... P2=... P3=... P4=... PLY=Xms                    ← overall timing

ExecutorchFp16Sharp has same PLY logging (tags: ExecutorchFp16Sharp).

Key metrics to compare before/after optimization:
  - compute time (batch loop minus I/O) → measures buffer recycling + SH block improvement
  - throughput MB/s → overall write efficiency
  - memory after write → confirms no lingering DirectByteBuffer pressure

Filter: adb logcat -s SplitOnnxSharp:D | grep PLY"""
    },
    {
        "id": "measured_backend_timings_feb2026",
        "layer": "performance",
        "content": """Measured SHARP backend timings (Feb 2026, Samsung device):

ONNX FP32 Split (SplitOnnxSharp, chunked Part 1 + Part 4):
  Part 1a: ~25s, Part 1b: ~15s, Part 2: ~9s, Part 3: ~1s, Part 4a: ~50s, Part 4b: ~35s
  Total: ~2.5-5 min depending on device state

ONNX INT8 Split (OnnxInt8Sharp, 4 parts):
  Part 1: 30,339ms, Part 2: 8,965ms, Part 3: 1,390ms, Part 4: crashed (activation OOM)

ONNX INT8 Single Model: Session created in 2190ms, crashed during session.run() (peak activation memory)

ExecuTorch FP16 (ExecutorchFp16Sharp, chunked Part 4):
  Part 1+2: 97,975ms (~1.6 min), Part 3: 2,791ms, Part 4 (chunked): 93,774ms (~1.6 min)
  Total: 205,876ms (~3.4 min). Gaussians: 1,179,648. Room: 5.1m × 4.9m × 1.1m.

LiteRT FP16: Parts 1-3 work, Part 4 crashes (decoder activation OOM even with chunked 4a/4b).
ONNX FP16: Fails at Part 1 session creation (com.microsoft.Gelu FP16 kernel not found on CPU EP).

Best reliable backend: ONNX FP32 Split (always completes). Best experimental: ExecuTorch FP16 (~3.4 min)."""
    },

    # ---- BEEWARE ANDROID ----
    {
        "id": "beeware_overview",
        "layer": "beeware_android",
        "content": """BeeWare on Android: Python UI apps on device via Briefcase + Toga + Chaquopy.

Briefcase builds the Android app; Toga provides cross-platform widgets (Button, Label, ImageView, etc.); Chaquopy embeds Python and runs the Toga app inside the Android Activity. The Furnit BeeWare app (com.furnit.beeware.furnit_beeware) mirrors the native Furnit flow: Home, Create Room (single photo), AI Room, Manual Setup, Sharp Room viewer, but ML inference (Sharp) does not run in Python on device—Chaquopy has no torch/executorch Android wheels. Use hand-off to native Furnit app (com.furnit.android) for AI Room, or implement hybrid (native SharpInferenceActivity + startActivityForResult)."""
    },
    {
        "id": "beeware_briefcase_android",
        "layer": "beeware_android",
        "content": """Briefcase Android configuration (pyproject.toml).

[tool.briefcase.app.<app>.android]
requires = ["toga-android"]   # pip packages installed by Chaquopy at build time
permission."android.permission.CAMERA" = true
permission."android.permission.READ_MEDIA_IMAGES" = true
build_gradle_extra_content = "..."  # appended to app/build.gradle
android_manifest_application_extra_content = "..."  # inject <provider>, etc.

Do not add torch or executorch to requires on Android—Chaquopy has no matching wheels; build fails with "No matching distribution found for torch>=2.0.0". BeeWare Android app stays with toga-android only for a working build."""
    },
    {
        "id": "beeware_toga_android_api",
        "layer": "beeware_android",
        "content": """Toga Android (toga-android) API for Furnit BeeWare.

Use app._impl.start_activity(intent) for activity results; intent_result is deprecated. Result from start_activity is the Intent (e.g. from gallery picker); use result.getData() for content URI. app._impl.native is the Android Activity (MainActivity). For startActivity(intent) (e.g. launching native Furnit app), call from the main/UI thread—if the handler runs in an async coroutine, consider runOnUiThread(Runnable) or ensure the event loop runs on the main thread. toga.Image: use src= for image data (data= is deprecated). Dialogs: await app.main_window.dialog(toga.InfoDialog(title, message)) (Toga 0.4+ async)."""
    },
    {
        "id": "beeware_chaquopy_limits",
        "layer": "beeware_android",
        "content": """Chaquopy on Android: Python packages and limits.

Chaquopy installs pip requirements from requirements.txt (generated from Briefcase android requires). Only packages with Android-compatible wheels (e.g. on chaquo.com/pypi-13.1 or PyPI with android tags) can be installed. torch and executorch do not have Android wheels in Chaquopy's index—adding them to requires causes "No matching distribution found for torch>=2.0.0". So Sharp ML cannot run inside the BeeWare Python process on device. Options: (1) hand off to native Furnit app for AI Room, (2) hybrid: BeeWare starts native SharpInferenceActivity with image URI/path and gets PLY path via onActivityResult."""
    },
    {
        "id": "beeware_android_rag_usage",
        "layer": "beeware_android",
        "content": """When to use beeware_android RAG layer.

Query this layer when working on: BeeWare app (beeware/), Briefcase Android build, Toga Android widgets, Chaquopy pip/requirements, photo picker (gallery intent, start_activity), AI Room flow (ExecuTorch unavailable, launch native app), MainActivity/onActivityResult, runOnUiThread, FileProvider for content URIs, or any Python-on-Android behavior in the Furnit project. Combine with executorch or sharp_model when discussing running Sharp in native app vs BeeWare."""
    },
    {
        "id": "paafekt_standard_landscape_camera",
        "layer": "paafekt_standards",
        "content": """Paafekt standard: landscape room opening position (3D viewer).

When a room from a landscape photo is opened, the camera must use the standard 'good position' so the room is framed correctly. Android (SharpRoomActivity WebGL): use L_CAM_X=0, L_CAM_Y=0.00207, L_CAM_Z=-0.130, L_TGT_Z=-0.444. Camera position = (center.x + L_CAM_X*W, center.y + L_CAM_Y*H, center.z + L_CAM_Z*D). Target = (center.x, center.y, center.z + L_TGT_Z*D). Box3 landscape: roomWidth=size.x, roomHeight=size.y, roomDepth=size.z; depth along Z. iOS should match the same viewing angle. See docs/PAAFEKT_STANDARDS.md."""
    },
]


def main():
    print("Indexing Furnit ML knowledge into ChromaDB...")

    collection = get_collection()

    # Clear existing
    try:
        existing = collection.count()
        if existing > 0:
            collection.delete(ids=[c for c in collection.get()["ids"]])
            print(f"Cleared {existing} existing chunks")
    except Exception:
        pass

    ids = []
    documents = []
    metadatas = []

    for chunk in CHUNKS:
        ids.append(chunk["id"])
        documents.append(chunk["content"].strip())
        metadatas.append({"layer": chunk["layer"]})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"\nIndexed {len(CHUNKS)} chunks across layers:")
    layer_counts = {}
    for chunk in CHUNKS:
        layer = chunk["layer"]
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    for layer, count in sorted(layer_counts.items()):
        print(f"  {layer}: {count} chunks")

    # Export for Git
    export = {"chunks": []}
    for chunk in CHUNKS:
        export["chunks"].append({
            "id": chunk["id"],
            "layer": chunk["layer"],
            "content": chunk["content"].strip()
        })
    Path(EXPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(EXPORT_PATH, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nExported to {EXPORT_PATH}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Index Furnit ML RAG")
    ap.add_argument("--index-logs", action="store_true", help="Also index Android logs into android_logs collection")
    ap.add_argument("--from-file", type=str, metavar="PATH", help="Index logs from this file (use with --index-logs)")
    ap.add_argument("--from-adb", action="store_true", help="Index logs from adb logcat -d (use with --index-logs)")
    args = ap.parse_args()

    main()

    if args.index_logs:
        if args.from_adb:
            index_android_logs(from_adb=True)
        elif args.from_file:
            index_android_logs(log_file_path=args.from_file)
        else:
            print("Use --from-file PATH or --from-adb with --index-logs")
