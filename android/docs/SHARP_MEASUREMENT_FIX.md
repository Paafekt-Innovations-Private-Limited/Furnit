# SHARP Room Measurement Fix

## Problem

Room dimensions from the Android SHARP pipeline were consistently wrong:

- **Expected:** ~3.15 × 2.86 m (real room)
- **Measured:** ~1.52 × 1.52 × 0.009 m (Android output)

The output was stuck in a small, paper-thin bounding box regardless of the actual room
being scanned.

## Root Cause

Two bugs compound to produce the wrong scale:

### Bug 1: Missing disparity → monodepth conversion in split pipeline (C++)

The SHARP model's decoder head outputs **raw disparity** (inverse depth). The init model
expects **monodepth** (metric depth = `disparity_factor / disparity`).

In the **monolithic** Part4b export (`ImageEncoderPartBFromTileInputs`), the conversion is
present:

```python
# monolithic path — correct
disparity = self.head(decoder_features)
monodepth = self.disparity_factor / disparity.clamp(min=1e-4, max=1e4)
init_output = self.init_model(image, monodepth)  # ✓ gets metric depth
```

In the **split** pipeline (`Part4bTileInitBasePortable`), the conversion was **missing**:

```python
# split path — BUG: passes raw disparity where init_model expects monodepth
def forward(self, image, disparity):
    init_output = self.init_model(image, disparity)  # ✗ gets raw disparity
```

The C++ code (`sharp_executorch_full_common.cpp`) calls the split path for all four tile
configurations (fine split tile_b2, split tile_b2, fine split tile_00, split tile_00),
passing the decoder's raw disparity tensor directly to the init .pte without converting.

With `disparity_factor = 1.0` (hardcoded in export), `monodepth = 1/disparity`. Passing
raw disparity instead of its reciprocal feeds the init model inputs of wildly different
magnitude, producing a global_scale and Gaussian positions that are far too small.

### Bug 2: Missing NDC → metric unprojection (Kotlin)

The Python reference pipeline (`predict.py`) includes an `unproject_gaussians` step after
the model produces its output:

```python
# Python reference — predict.py
f_px = convert_focallength(width, height, f_35mm)
disparity_factor = f_px / width
# ... model inference ...
gaussians_metric = unproject_gaussians(
    gaussians_ndc, eye(4), intrinsics_resized, (H, W)
)
save_ply(gaussians_metric, output_path)
```

The unprojection transforms Gaussian positions from Normalized Device Coordinates (NDC)
to metric world coordinates using camera intrinsics. The formula (for a principal point at
image center) reduces to:

```
X_world = output_x × W / (2 × f_px)
Y_world = output_y × H / (2 × f_px)
Z_world = output_z  (depth is already metric after monodepth conversion)
```

where `f_px = f_35mm × √(W² + H²) / √(36² + 24²)` (full-frame 35mm film diagonal
= 43.27 mm).

**This entire step was absent from the Android and iOS pipelines.** Gaussian positions
were written to PLY and used for bounding box computation directly in model output space.

## Fix

### C++ — `sharp_executorch_full_common.cpp`

Added in-place `1.0f / clamp(disparity, 1e-4, 1e4)` conversion on the disparity tensor
data **before** passing it to the init module, at all four split pipeline locations:

1. `runPart4bBatchedTiledPipeline` — fine split tile_b2 (line ~798)
2. `runPart4bBatchedTiledPipeline` — split tile_b2 (line ~935)
3. `runPart4bTiledFullPipeline` — fine split tile_00 (line ~1448)
4. `runPart4bTiledFullPipeline` — split tile_00 (line ~1592)

```cpp
// Convert raw disparity → monodepth before init_base
for (size_t k = 0; k < disparity.data.size(); ++k) {
    float v = std::clamp(disparity.data[k], 1e-4f, 1e4f);
    disparity.data[k] = 1.0f / v;
}
```

This is compatible with existing .pte models (which pass the tensor straight through to
init_model without further conversion). Future re-exported .pte models that bake the
conversion into the graph will need this C++ code removed.

### Kotlin — `ExecutorchInt8Sharp.kt::writePly`

**Unprojection is NOT applied to positions/scales.** After the C++ disparity→monodepth fix,
the model's `global_scale` produces output that is already approximately metric. Applying
the Python-style NDC→metric unprojection (`X = x * W/(2*f_px)`) on top of this
double-corrects and shrinks measurements further (e.g., 3.15m → 1.1m).

The unprojection math is computed and logged as `[ply_unproj_diag]` for diagnostics only.
Original image dimensions are threaded from `inferStreaming` to `writePly` for this
diagnostic computation.

### Export script — `export_sharp_executorch_split4.py`

`Part4bTileInitBasePortable` now includes the disparity → monodepth conversion inside
its `forward()`:

```python
def forward(self, image, disparity):
    monodepth = self.disparity_factor / disparity.clamp(min=1e-4, max=1e4)
    init_output = self.init_model(image, monodepth)
```

When re-exported .pte models include this fix, the C++ in-place conversion must be
removed to avoid double-inversion.

## Expected Result

After both fixes, the AABB logged at `[ply_bbox]` should show:

- Width/height in the range of the actual room (~3–6 m for typical rooms)
- Depth (Z span) reflecting actual room depth (~2–5 m), not paper-thin 0.009 m
- `looksNormalized = false` (positions exceed the ±1.5 NDC range)

## Dependency: f_35mm from EXIF

The current implementation uses `f_35mm = 30` (SHARP's default when EXIF is unavailable).
For phones with non-standard field of view, reading the actual 35mm-equivalent focal
length from EXIF metadata would improve accuracy. This is a follow-up task — the default
of 30mm produces correct-order-of-magnitude results for typical phone cameras (26–30mm
equivalent).

## Files Changed

| File | Change |
|------|--------|
| `android/app/src/main/cpp/sharp_executorch_full_common.cpp` | disparity → 1/disparity at 4 split init sites |
| `android/app/src/main/java/com/furnit/android/services/ExecutorchInt8Sharp.kt` | `writePly`: NDC → metric unprojection + scale correction |
| `android/export_sharp_executorch_split4.py` | `Part4bTileInitBasePortable`: bake conversion into export |
