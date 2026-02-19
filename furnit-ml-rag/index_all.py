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
    main()
