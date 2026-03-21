# ExecuTorch Vulkan + `sharp_split_part4b_tile_full.pte` crash

## Symptom

Native **SIGABRT** when the full Vulkan pipeline loads/runs Part4b tiled-full, e.g. after:

`runPart4bTiledFullPipeline: using .../sharp_split_part4b_tile_full.pte`

## Abort text (from `libexecutorch.so`)

```text
vkcompute::vkapi::Error: ... sizes_ubo at .../Tensor.cpp:1050: (sizes_.size() <= 4) is false!
```

## Where it dies

- **Stack:** `vTensor::sizes_ubo()` → `vkcompute::add_concat_node` → `BackendDelegate::Init` / `Method::init` / `Module::execute`
- **When:** Vulkan graph initialization for the **tile_full** Part4b delegate, not a Furnit JNI bug.

## What that means

The Vulkan backend builds a uniform buffer of tensor sizes with the invariant **`sizes_.size() <= 4`**. Something in this **concat** (or related) path is feeding a tensor whose **size vector has more than four entries** (e.g. rank / symbolic shape handling), so the runtime **asserts and aborts**.

**Fix surface:**

1. **Export / graph:** Change the Vulkan LoweredModule so concat (and views around it) never produces operands that violate that assumption, or flatten/reshape before concat as Vulkan expects.
2. **ExecuTorch Vulkan:** Extend `sizes_ubo` / concat lowering to support the shapes this model needs (upstream `pytorch/executorch`).

The app does **not** implement a CPU fallback for Part4b when `useVulkanBackend` is true: `path4bSingle` is only `sharp_split_part4b_vulkan.pte`. Hybrid “Part1+2 on CPU” still runs Part3/4/4b on Vulkan when that flavor is selected.

## Reference log tags

`sharp_executorch_full`, `ExecutorchInt8Sharp`, tombstone `libc` / `DEBUG` abort message as above.
