# Alignment with ExecuTorch Vulkan example

This doc compares Furnit’s SHARP ExecuTorch Vulkan export and runtime with the [official ExecuTorch Vulkan example](https://github.com/pytorch/executorch/tree/main/examples/vulkan).

## Export pattern (we follow this)

The example uses:

1. **Export:** `torch.export.export(model, example_inputs, strict=...)`
2. **Compile options:** `compile_options = {}` with `force_fp16`, `skip_memory_planning`, `small_texture_limits`, `require_dynamic_shapes`
3. **Lower:** `to_edge_transform_and_lower(program, partitioner=[VulkanPartitioner(compile_options)], generate_etrecord=...)`
4. **ExecuTorch:** `edge_program.to_executorch()`
5. **Save:** `save_pte_program(exec_prog, output_filename, output_dir)` (or equivalent write of `exec_prog.buffer`)

Furnit’s **export_sharp_executorch_split4.py** does the same flow:

- `torch.export.export(wrapper, sample_inputs, strict=...)`
- `VulkanPartitioner(opts)` with `opts` containing `force_fp16`, `skip_memory_planning` (and `--vulkan-aar-compat` → `force_fp16=False` for AAR shader compatibility)
- `to_edge_transform_and_lower(exported, partitioner=[VulkanPartitioner(opts)], ...)`
- `edge.to_executorch()` (Vulkan path skips greedy memory planning; VulkanPartitioner does AOT planning)
- Write `.pte` via `open(path, "wb").write(et_program.buffer)` (equivalent to saving the program buffer)

So we **follow the same export and lower pattern** as the example.

## Runtime (we follow this)

- **Load:** `Module(path, LoadMode::Mmap)` (extension Module API)
- **Execute:** `module.forward(inputs)`  
Same as in the ExecuTorch Vulkan [profiling tutorial](https://docs.pytorch.org/executorch/1.0/backends/vulkan/tutorials/etvk-profiling-tutorial.html) and example usage.

## Optional example features (implemented)

| Feature | Example | Furnit |
|--------|---------|--------|
| **-d / dynamic shapes** | `require_dynamic_shapes` + dynamic dims | SHARP uses fixed 1536/1280; not used |
| **--small_texture_limits** | `small_texture_limits: True` for (2048,2048,2048) texture limits (e.g. desktop GPUs) | **Implemented.** Pass `--small-texture-limits`; added to Vulkan compile opts in export. |
| **-r / generate_etrecord** | `generate_etrecord=args.etrecord` | **Implemented.** `-r DIR` / `--etrecord DIR`: writes `<DIR>/<part_stem>.etrecord` per part for Inspector. |
| **-b / bundled .bpte** | BundledProgram with test cases, `.bpte` | **Implemented.** `-b` / `--bundled`: writes a `.bpte` per part with one test case (flattened I/O). |
| **--test** | `test_utils.run_and_check_output()` to validate lowered model (needs Vulkan SDK + build from source) | **Implemented.** `-t` / `--test`: runs Vulkan correctness test after each export (relaxed tol for FP16). |

Usage:

```bash
# Export with ETRecord + optional small texture limits (e.g. for desktop Vulkan)
python android/export_sharp_executorch_split4.py --weights ... --sharp-src ... \
  --output-dir android/sharp_vulkan_only --vulkan-aar-compat \
  -r ./etrecord --small-texture-limits

# Also write bundled .bpte and run Vulkan test per part (needs Vulkan SDK for -t)
python android/export_sharp_executorch_split4.py ... -r ./etrecord -b -t
```

Or use `export_sharp_vulkan_only.sh` with env vars: `ETRECORD_DIR=./etrecord`, `SMALL_TEXTURE_LIMITS=1`, `BUNDLED=1`, `RUN_TEST=1` (see script).

## Summary

- **Export and runtime:** We follow the same approach as the [ExecuTorch Vulkan example](https://github.com/pytorch/executorch/tree/main/examples/vulkan): `export` → `to_edge_transform_and_lower` with `VulkanPartitioner(compile_options)` → `to_executorch()` → save `.pte`; load/run with extension Module.
- **Extra in Furnit:** `--vulkan-aar-compat` (force_fp16=False) for compatibility with the executorch-android-vulkan 1.1.0 AAR shaders.
- **Optional flags:** `--small-texture-limits`, `-r/--etrecord DIR`, `-b/--bundled`, `-t/--test` are implemented and wired through all `export_pte` calls.
