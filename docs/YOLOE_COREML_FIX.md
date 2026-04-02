# YOLOE CoreML Model Detection Fix

## Current iOS shipping model

The **Furnit** iOS app loads **YOLOE 26L PF** only: `yoloe-26l-seg-pf_seg_o2m` (preferred) or `yoloe-26l-seg-pf`, via `YOLOEModelService` (see `Furnit/Services/OnDevice/YOLOEModelService.swift`). Export path: `scripts/export_yoloe26_onemany_user.py` / `export_yoloe26_fixed.py` and `scripts/README_YOLOE_COREML.md`.

The remainder of this document is **historical context** from when a broken 26L Core ML export was worked around by preferring an 11L package.

---

## Problem Summary

The iOS app was detecting wrong object classes when using the `yoloe-26l-seg-pf` CoreML model. Instead of detecting "chair" (class 821) or "office chair" (class 2834), the model was outputting random incorrect classes like:
- "plug hat" (class 3128)
- "jam" (class 2260)
- "denim jacket" (class 1378)

**Python inference worked correctly** - running `model.predict("image.heic")` with the same `.pt` weights file correctly detected "chair" at 0.91 confidence.

## Root Cause

### The Issue: Broken Model Head Structure

The `yoloe-26l-seg-pf.pt` model uses `YOLOESegment26` head class which has:
```python
cv3 = None
cv4 = None
```

During CoreML export, Ultralytics calls `model.fuse()` which attempts to fuse batch normalization layers into convolution layers for faster inference. The `fuse()` method tries to iterate over `cv3` and `cv4`:

```python
def fuse(self, verbose=True):
    # ...
    for m in self.model.modules():
        if hasattr(m, 'bn_head'):
            for bn_head in m.bn_head:  # Fails here when bn_head is None
                # ...
```

When `cv3=None` or `cv4=None`, this causes `'NoneType' object is not iterable` error.

### The Workaround That Broke Everything

To work around the fuse() error, the export script (`yolo26.py`) disabled fuse():

```python
# Patch fuse methods to be no-ops BEFORE loading
for name in dir(head):
    cls = getattr(head, name)
    if isinstance(cls, type) and hasattr(cls, 'fuse'):
        cls.fuse = lambda self, *args, **kwargs: None
```

**This allowed the model to export, but the exported CoreML model was corrupted** - the unfused batch normalization layers weren't properly converted, causing the classification head to output garbage class IDs.

## Solution (historical)

### Use yoloe-11l instead of a broken yoloe-26l Core ML export

The `yoloe-11l-seg-pf.pt` model uses `YOLOESegment` head class (not `YOLOESegment26`) which has proper `cv3` and `cv4` heads:

```python
# yoloe-11l head structure
cv3 = ModuleList([...])  # Proper classification heads
cv4 = ModuleList([...])  # Proper coefficient heads
```

This model exports correctly without needing to disable fuse().

### Export Command

```python
from ultralytics import YOLO

model = YOLO("yoloe-11l-seg-pf.pt")
model.export(
    format="coreml",
    imgsz=1280,
    batch=1,
    nms=False,
    half=False,
    simplify=True
)
```

### Model Output Comparison

| Model | Detection Tensor | Shape | Format |
|-------|-----------------|-------|--------|
| yoloe-26l (broken) | var_2346 | [1, 300, 38] | NEW (post-NMS) |
| yoloe-11l (working) | var_2374 | [1, 4621, 33600] | OLD (pre-NMS) |

The 11l model uses the OLD format where:
- Dimension 1 (4621) = 4 (bbox) + 4585 (classes) + 32 (mask coeffs)
- Dimension 2 (33600) = anchor positions (3 scales × 80×80 + 40×40 + 20×20)

## Code Changes Made

### 1. ModelViewerView.swift (line 569-572)

Changed model loading order to prefer 11l:

```swift
let candidateNames = [
    ("yoloe-11l-seg-pf", "mlmodelc"),  // Preferred - has proper cv3/cv4 heads
    ("yoloe-26l-seg-pf", "mlmodelc"),  // fallback
]
```

### 2. SmartyPantsView.swift (line 532-549)

Added support for 11l model output tensor names:

```swift
if let det = output.featureValue(for: "var_2374")?.multiArrayValue,
   let proto = output.featureValue(for: "var_2412")?.multiArrayValue {
    // yoloe-11l model outputs (new export with proper cv3/cv4 heads)
    detArray = det
    protoArray = proto
}
```

### 3. Xcode Project (project.pbxproj)

Added `yoloe-11l-seg-pf.mlpackage` to the project with proper build file references.

## Refactoring: Removed Workaround Code

With the 11l model working correctly, we removed the following workaround code that was added for the broken 26l model:

### Removed: STAGE 8b (Two-Stage Verification)

Previously, the pipeline ran inference twice:
1. First pass on full image → select primary detection
2. **STAGE 8b**: Crop to primary bbox → run inference again → verify class

This was ~160 lines of code that:
- Cropped the pixel buffer to the primary detection bbox
- Ran model inference a second time on the crop
- Looked for a detection filling >50% of the crop
- Updated the primary if coverage >60%

**Why it was needed**: The 26l model was outputting wrong classes, so two-stage verification helped correct misclassifications.

**Why it's removed**: The 11l model correctly classifies objects on the first pass, making two-stage verification unnecessary overhead.

### Removed: `cropPixelBuffer()` Function

~70 lines of code for cropping a CVPixelBuffer to a detection bbox. Only used by STAGE 8b.

### Performance Impact

Removing STAGE 8b eliminates:
- One full model inference (~50-100ms)
- Pixel buffer crop operation (~5ms)
- Detection parsing on crop (~2ms)

**Total savings: ~60-110ms per frame**

## Verification

### Python Verification
```python
from ultralytics import YOLO

# 11l correctly detects chair
model = YOLO("yoloe-11l-seg-pf.pt")
results = model.predict("room.heic")
# Output: chair (class=821) conf=0.92, chair (class=821) conf=0.87
```

### CoreML Verification
```python
import coremltools as ct

model = ct.models.MLModel('yoloe-11l-seg-pf.mlpackage')
print('Outputs:')
for out in model.get_spec().description.output:
    print(f'  {out.name}: {list(out.type.multiArrayType.shape)}')

# Output:
#   var_2374: [1, 4621, 33600]
#   var_2412: [1, 32, 320, 320]
```

## Key Lessons

1. **Don't disable fuse() for CoreML export** - It corrupts the classification head
2. **Model architecture matters** - YOLOESegment26 has broken cv3/cv4, YOLOESegment works
3. **Verify CoreML outputs match Python** - If Python works but CoreML doesn't, the export is corrupted
4. **Check tensor output names** - Different model versions have different output tensor names

## Files

| File | Purpose |
|------|---------|
| `yoloe-11l-seg-pf.mlpackage` | Working CoreML model (preferred) |
| `yoloe-26l-seg-pf.mlpackage` | Broken CoreML model (fallback only) |
| `yoloe-11l-seg-pf.pt` | Source PyTorch weights |
| `yoloe-26l-seg-pf.pt` | Source PyTorch weights (exports broken) |

## LVIS Classes Reference

The model uses LVIS (Large Vocabulary Instance Segmentation) with 4585 classes. Key furniture classes:
- 821: chair
- 2834: office chair
- 595: armchair
- 1089: couch/sofa
- 3893: swivel chair

Classes file: `Furnit/Views/SmartyPants/classes.json`
