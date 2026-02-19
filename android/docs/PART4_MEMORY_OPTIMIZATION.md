# Part 4 Memory Optimization (LMK / OOM Mitigation)

## Root cause

Part 4 decoder creates **huge intermediate activations** during the forward pass. Peak memory can reach **~6 GB** (3.3 GB RSS + 2.9 GB swap), causing Android LMK (low memory killer) to terminate the app.

This is **activation memory**, not model weight memory. Splitting the model does not reduce it.

## Solutions (excluding Vulkan)

### Solution 2: Aggressive memory cleanup (implemented)

`ExecutorchSharp.kt` now does before Part 4 forward:

- `System.gc()` + `System.runFinalization()` + 150ms sleep
- Memory check: if avail < 1 GB → abort with user message
- Warning log if avail < 2 GB

### Solution 3: Native allocator

ExecuTorch already uses native memory for tensor data. The Android `Module` API wraps JVM `float[]` in `Tensor.fromBlob()`. Moving to `ByteBuffer.allocateDirect()` would require ExecuTorch API support for native buffers. Not feasible without modifying the ExecuTorch runtime.

### Solution 4: Increase swap (root required)

If the device is rooted, increase zram swap to 8 GB to absorb the activation spike:

```bash
adb push android/scripts/increase_swap_root.sh /data/local/tmp/
adb shell su -c 'sh /data/local/tmp/increase_swap_root.sh'
```

Verify: `adb shell cat /proc/swaps`

Paths may vary by device (Pixel uses `/dev/block/zram0`).

### Solution 5: Stream Part 4 activations (future)

Split the decoder into `part4a.pte`, `part4b.pte`, `part4c.pte` and run sequentially. Each sub-part would have smaller peak activations. Requires re-exporting the model and pipeline changes.

## Best long-term fix: Vulkan Part 4

Export Part 4 with Vulkan backend. GPU memory is separate from system RAM, so activation memory moves off the host. This is the recommended path for production.
