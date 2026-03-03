# Ask Ultralytics: ExecuTorch export problems (SHARP / Part4b)

Copy the block below into Ultralytics (Ask AI, forum, or support) to get help on the issues we faced.

---

## Paste this into Ultralytics

```
We're exporting a ViT-based 3D Gaussian splatting model (SHARP) to ExecuTorch for Android. Pipeline: Part1–3 INT8 (XNNPACK), Part4a chunks INT8, Part4b decoder. We hit three issues and would like your guidance.

**1. Part4b FP16 export – mixed precision error**

We want Part4b (decoder + Gaussian heads) in FP16 for quality and to reduce the ~114 s CPU bottleneck. When we export Part4b with use_fp16=True (wrapper.half() and sample_inputs cast to half), torch.export.export fails with:

  RuntimeError: Input type (float) and bias type (c10::Half) should be the same

So somewhere we have float input feeding a half (FP16) module. We already cast the wrapper to .half() and all floating sample_inputs to .half() before export. The model has Conv2d, upsample, decoder, and Gaussian composer. How do you recommend exporting a decoder subgraph to FP16 with ExecuTorch when the rest of the pipeline is INT8/FP32? Is there a standard pattern (e.g. only certain layers in FP16, or a specific export order) to avoid this mixed float/half error?

**2. Vulkan partition not in pip ExecuTorch**

We run: pip install executorch (1.1.0). When we try to use the Vulkan backend for Part4b:

  from executorch.backends.vulkan.partition.vulkan_partitioner import VulkanPartitioner

we get:

  No module named 'executorch.backends.vulkan.partition'

So the Vulkan partitioner isn’t included in the PyPI package. What is the recommended way to get Vulkan-backed .pte export for Android? Do we need to build ExecuTorch from source with a Vulkan build flag, or is there a separate pip extra / package for Vulkan export?

**3. Greedy memory planning – TensorSpec memory offset**

When exporting INT8 parts (and sometimes Part4b) with ExecuTorch’s greedy memory planning (MemoryPlanningPass, alloc_graph_input=False, alloc_graph_output=False), we get:

  Greedy memory planning not available: TensorSpec(...) should have specified memory offset, using default planning

So we fall back to default planning. Is this expected with torch.export + ExecuTorch in recent versions? Is there a compile config or export step we should set so that TensorSpecs get memory offsets and greedy planning can run?

**Context:** PyTorch 2.10, ExecuTorch 1.1.0, Python 3.13. We follow Ultralytics deployment practices (operation fusion, AOT memory planning, mixed precision for decoder heads). Target is Android with ExecuTorch INT8 + optional Part4b FP16 or Vulkan.
```

---

## Short version (if character limit)

```
ExecuTorch export for Android SHARP model:

1) Part4b FP16: export fails with "Input type (float) and bias type (c10::Half) should be the same" after wrapper.half() and casting sample_inputs to half. How to export decoder to FP16 without mixed float/half?

2) Vulkan: pip install executorch has no executorch.backends.vulkan.partition. How to get Vulkan .pte export for Android—build from source or another package?

3) Greedy memory planning fails with "TensorSpec should have specified memory offset". How to enable greedy AOT planning with torch.export + ExecuTorch?
```
