# ExecuTorch memory management

ExecuTorch manages memory by **planning tensor locations in fixed-size memory arenas ahead of time (AOT)**. This works with Vulkan’s low-level, explicit memory API and reduces peak RAM and fragmentation.

## 1. Memory planning pass (compile time)

Before an ExecuTorch program is emitted, a **memory planning pass** runs to determine the size and lifespan of all intermediate tensors. Each tensor is assigned an **ID** and an **offset** within a buffer so that **non-overlapping tensors can share the same allocation**.

### Algorithms

| Algorithm | Behavior | Use |
|----------|----------|-----|
| **Naive** | Concatenates all tensors linearly; no reuse. Upper bound for total memory. | Avoid for large models. |
| **Greedy** | Best-fit reuse: reuses memory based on lifetime; reduces fragmentation and **peak usage**. | **Use for SHARP** (split Part 4 and full Vulkan export). |

### How we use it

- **Export (Python):** When building the `.pte`, we call `to_executorch(ExecutorchBackendConfig(memory_planning_pass=MemoryPlanningPass(memory_planning_algo=greedy, alloc_graph_input=False, alloc_graph_output=False)))`.
  - `alloc_graph_input=False`, `alloc_graph_output=False`: graph input and output buffers are **caller-managed** (app provides them); only **intermediate** tensors are planned. This avoids “Misallocate graph input” and keeps the plan smaller.
  - The **greedy** algorithm is applied to those intermediates so they share memory where lifetimes don’t overlap.

- **Where it’s applied in Furnit (single full Vulkan model):**
  - **Full Vulkan export:** `export_sharp_executorch_full_vulkan.py` uses VulkanPartitioner + greedy MemoryPlanningPass (see script docstring and `to_executorch` call). This applies to the “ExecuTorch Vulkan (single model)” Settings option only.

## 2. Vulkan AOT memory planning

The **Vulkan backend** has its own AOT memory planning, which runs when we **do not** skip it:

- **Export:** `VulkanPartitioner({"skip_memory_planning": False, "force_fp16": True})`.
  - `skip_memory_planning: False` → Vulkan preprocess runs memory planning (buffer reuse, shared allocations where possible).
  - `force_fp16: True` → FP32→FP16 inside the backend; lower GPU memory and bandwidth.

So for Vulkan we get:

1. **Vulkan-side planning** (GPU buffers, descriptor sets) from the partitioner options.
2. **ExecuTorch-side planning** (portable/arena intermediates) from `MemoryPlanningPass(greedy, ...)` in `ExecutorchBackendConfig`.

Both are **compile-time**; the resulting `.pte` carries the plan.

## 3. Runtime (Android)

- **No extra runtime API for “which plan”:** The memory plan is **baked into the `.pte`**. The app does not choose naive vs greedy at load time.
- **Loading:** We use `Module.load(path, Module.LOAD_MODE_MMAP)` so the OS pages in only used weights; cold pages can be evicted under memory pressure (reduces RSS vs loading the whole file into RAM).
- **Releasing:** Call `module.destroy()` (and optionally `release()` on the wrapper) when done so native/Vulkan resources and arenas are freed.

So “memory management” on device is: **build the .pte with greedy + Vulkan AOT planning**, then **load with mmap** and **destroy when done**.

## 4. Custom memory plans (reference)

ExecuTorch allows **custom memory plans** (e.g. different hierarchies like SRAM vs DRAM, or custom allocation strategies). We do not use custom plans in Furnit; we rely on:

- **Greedy** for the portable/arena part.
- **Vulkan’s built-in planning** (skip_memory_planning=False) for the delegated Vulkan part.

## 5. Summary table

| What | Where | Effect |
|------|--------|--------|
| Vulkan AOT planning | Export: `VulkanPartitioner(skip_memory_planning=False)` | GPU buffer reuse, less Vulkan memory. |
| Greedy planning | Export: `MemoryPlanningPass(memory_planning_algo=greedy, alloc_graph_input=False, alloc_graph_output=False)` | Intermediate tensor reuse, lower peak RAM. |
| force_fp16 | Export: `VulkanPartitioner(..., force_fp16=True)` | Lower GPU memory and bandwidth. |
| LOAD_MODE_MMAP | Runtime: `Module.load(path, LOAD_MODE_MMAP)` | Weights paged on demand; lower RSS. |
| destroy() | Runtime: after inference | Frees native/Vulkan and arena memory. |

See also: `EXECUTORCH_VULKAN_OPS.md` (AOT memory planning, Vulkan options), and the export scripts’ docstrings (`export_sharp_executorch_full_vulkan.py`, `export_sharp_executorch_split4.py`).
