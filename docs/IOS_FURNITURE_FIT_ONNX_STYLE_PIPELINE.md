# iOS Furniture Fit — “ONNX-style” pipeline

This document describes the **ONNX-style** segmentation path on iOS: how a frame moves from the camera through model inference, detection post-processing, prototype mask fusion, and compositing. It is aimed at engineers debugging parity with Android or tuning quality/performance.

## Naming: “ONNX-style”

- **ONNX-style** means: **stretch** resize to the model square (like Android’s `createScaledBitmap` / ONNX preprocessing), then **Android-aligned** mask logic (`FurnitureFitOnnxStylePipeline`: bbox-limited prototype fusion, max sigmoid, supporting-table heuristic, etc.).
- On iOS this path runs **Core ML** (`MLModel`) only. There is **no** ONNX Runtime dependency on iOS.

Relevant sources:

| Piece | Location |
|--------|-----------|
| Branching + Core ML ONNX-style entry | `Furnit/Views/FurnitureFit/FurnitureFitView.swift` — `processFrameInner`, `processFrameOnnxStyleCoreML`, `processFrameOnnxStyleCommon` |
| Android-parity helpers | `Furnit/Services/OnDevice/FurnitureFitOnnxStylePipeline.swift` |
| Detection tensor decode | `Furnit/Services/OnDevice/YoloEDetectionParser.swift` |
| User toggle | `Furnit/Models/QualitySettings.swift` — `furnitureFitOnnxStyleEnabled()` / Settings UI |

---

## Feature flag

- **Key:** `furniture_fit_use_onnx_runtime` (legacy name; meaning is “use ONNX-style pipeline”, not “force ORT”).
- **Default:** `true` — ONNX-style stretch + postprocess is the default when a model is available.
- **When `false`:** Furniture Fit uses the **letterbox** Core ML path (different preprocess and mask stages — see the long `processFrameInner` letterbox branch in `FurnitureFitView.swift`).

---

## High-level flow

```
captureOutput (BGRA CVPixelBuffer, camera rotation already applied)
    → processFrame (autoreleasepool)
        → processFrameInner
            → if furnitureFitOnnxStyleEnabled() && mlModel != nil → processFrameOnnxStyleCoreML
            → else: letterbox Core ML pipeline (separate doc-worthy path)
```

All heavy work runs on the **detection queue** (serial); UI updates are dispatched to **main**.

---

## Path A — ONNX-style + Core ML (primary)

### Stage 0 — Input buffer

- **Source:** `AVCaptureVideoDataOutput` BGRA buffer (`processBuffer`).
- **Dimensions:** e.g. **720×1280** (portrait room, 90° rotation) or **1280×720** (landscape). Code uses **buffer width/height** for “landscape”, not `UIDevice` orientation (avoids flat-table mistakes).
- **`isLandscape`:** `bufW > bufH` — used later for optional **display rotation** of the final `UIImage`, not for model input when ONNX-style (always square stretch).

### Stage 1 — Preprocess (stretch)

- **Function:** `resizeStretchToSquare(processBuffer, size: modelInputSize)`.
- **Behavior:** `vImageScale_ARGB8888` from full frame to **modelInputSize × modelInputSize** (aspect **not** preserved — **stretch**, same idea as Android ONNX).
- **`modelInputSize`:** From Core ML `image` constraint `pixelsWide` (typical **640** for current YOLOE seg export); fallback **1280** if unconstrained.
- **Performance:** Reuses a **cached** `CVPixelBuffer` (`cachedStretchBuffer`) sized to `cachedStretchSize` to avoid per-frame allocation.

### Stage 2 — Core ML inference

- **Input name:** `"image"`.
- **If** the model declares **image** type: `MLFeatureValue(pixelBuffer: stretched)`.
- **Else:** `pixelBufferToMLMultiArray(stretched)` for tensor input.
- **`model.prediction(from:)`** — compute units follow Settings (`QualitySettings` / `yoloeCoreMLAllowGPUKey`: default **CPU-only** for stability; optional GPU experimental).
- **Outputs:** `YoloEDetectionParser.extractDetectionAndProto(from:)` resolves known output name pairs, e.g. **`var_2286` (det)** + **`var_2369` (proto)** (export-dependent; see `knownDetectionProtoPairs` in `YoloEDetectionParser.swift`).

### Stage 3 — Shared postprocess (`processFrameOnnxStyleCommon`)

From here, post-processing uses the same `MLMultiArray` det/proto tensors regardless of how inference was run.

---

## `processFrameOnnxStyleCommon` — step by step

### 1) Parse prototype tensor

- **`parsePrototypes(protoArray)`** (`FurnitureFitView.swift`):
  - Accepts **Float16** or **Float32** `MLMultiArray`.
  - Normalizes shape (drops leading batch `1` if present).
  - Expects **32** prototype channels; infers **H×W** (e.g. **160×160** for a 640 input model).
  - Produces **`planes: [Float]`** length **32 × protoH × protoW**, layout suitable for mask dot-products.
  - Uses **Accelerate** / **vImage** for bulk FP16→FP32 where strides are contiguous; reuses `protoRawFloats` / `protoPlanes` buffers.

### 2) Parse detections

- **`YoloEDetectionParser.parseDetections(detArray, confidenceThreshold, classBlacklist: clsToIgnore)`**:
  - **“New” end-to-end layout:** `[1, numDets, featuresPerDet]` with `featuresPerDet < 100`.
  - Per row: **x1, y1, x2, y2, confidence, classIdx**, then **32 mask coefficients** (indices 6…37).
  - Converts box to center **(x, y)** and **w, h** in **model pixel space** (same square as stretch).
  - Filters: finite values, confidence ≥ **`confidenceThreshold`** (view property; SwiftUI wrapper default **0.15**, container default **0.1** — check call site), sanity on box size, **blacklist** by class id.
- **`YoloEDetectionParser.releaseF16Scratch()`** after parse drops a reusable FP16→FP32 scratch buffer to **lower peak memory**.

### 3) NMS + blacklist pass

- If more than one detection: sort by confidence, cap **100**, then **`FurnitureFitNMS.apply(..., iouThreshold: 0.5)`**.
- **`FurnitureFitFilter.excludingClasses(..., blacklist: clsToIgnore)`** again (defense in depth with `blacklist.json`).

**Note:** `FurnitureFitOnnxStylePipeline` defines **`confidenceThreshold: 0.25`** and **`iouThresholdNms: 0.45`** for Android naming parity; the **live** ONNX-style Core path uses the view’s **`confidenceThreshold`** and **IoU 0.5** as commented in code.

### 4) Primary selection (iOS “STAGE 4” scoring)

- **`selectPrimaryIndexCoreFlow(candidates, modelSide:)`** — **not** `FurnitureFitOnnxStylePipeline.pickPrimaryIndex` (that Android-style scorer exists in the enum but is **unused** in the current ONNX-style common path).
- Gates: **min confidence 0.15**, **min normalized area 0.02** (relative to model square).
- Score: **conf^1.5 × areaNorm^0.8 × centerScore^0.5** where `centerScore` favors boxes near the frame center.

### 5) Mask detection list (Android `collectMaskDetections`)

- **`FurnitureFitOnnxStylePipeline.collectMaskDetections(primaryIndex, detections)`**:
  - Starts from **primary**.
  - **Monitor + desk heuristic:** if primary class is in **`monitorLikeClassIds`**, may attach a **supporting table** from **`supportingTableClassIds`** using overlap / gap / size rules (`pickSupportingTableForMonitorScene`).
  - Adds extra detections that **intersect** the primary, are not **encompassing** the primary, pass **size / IoU duplicate** rules, and meet **minimum confidence 0.1**.
  - Returns list with **primary first**, then extras, optional supporting table.

### 6) Expanded primary for mask logits only

- **`onnxStyleExpandedPrimaryForMaskBuild(primary, onnxSide:)`**:
  - Widen/tall **primary** box by **`bboxExpandMargin` (0.08 = 8% per side)** so prototype logits are evaluated **under chair legs / wheels**, clamped so the box stays inside the model square.
- **Fusion list:** `[expandedPrimary] + maskDetections.filter { IoU with original primary < 0.999 && not blacklisted }`  
  - Drops the **unexpanded** primary from fusion by near-duplicate IoU so you don’t double-count the same box.

### 7) Bbox-limited prototype mask (Android loop)

- **`FurnitureFitOnnxStylePipeline.buildBboxLimitedSigmoidMask`**:
  - For each detection in the fusion list, for each **prototype pixel inside that detection’s bbox** (mapped from model space to proto grid):
    - **sum_k coeff[k] × protoPlane[k, y, x]**
    - **sigmoid**
    - **max** across detections at each pixel
  - Threshold **> 0.5** → **0 / 255** planar UInt8 mask, size **protoW × protoH**.

### 8) Optional morphological close

- If **`furnitureFitUseMorphologicalCloseMask`** (currently **`true`**): **`morphologicalBinaryClose3x3Planar8`** on the small mask to reduce speckles / holes.

### 9) Clip proto mask to **tight** primary (model space)

- **Purpose:** Limit spill from secondary fusion to the **tight** detector bbox (not the expanded mask bbox).
- Converts primary **model-space** rect to proto coordinates and **zeros** mask outside **`clipProtoPlanarMaskOutsideRect`**.

### 10) UI geometry (buffer space)

- Scales primary box from model square to **`bufW × bufH`** for:
  - **`primaryBboxInView`** (normalized → mask image view),
  - **`updateAssistedOverlayScale`** / debounced AR path,
  - **`updateAutoScaleFromRoom`** (currently neutral **1×** in generic fit).

### 11) Compositing band vs mask clip band

- **Tight primary** in buffer pixels defines the **detector** box.
- **Composite band** = tight box **expanded** by **`bboxExpandMargin`** on each side (same **0.08**). Only this rectangle is filled; outside stays **fully transparent** so the live camera / scene shows through — fixes “floor eaten” when legs extend below the bbox.

### 12) CPU composite (`compositeCpuBilinearProtoMaskCutout`)

Despite the name, the implementation comment states **vImage `Planar8` scale** proto → full frame (high-quality resampling), **not** per-pixel bilinear in the huge band (avoids mushy edges and full-opacity smear).

Steps:

1. Scale **protoW×protoH** mask to **origW×origH** into **`upscaledPlanarMaskScratch`** (`vImageScale_Planar8`).
2. Allocate **premultiplied RGBA** `CGContext` **origW × origH**.
3. Clear to transparent.
4. For pixels in **[x0,x1)×[y0,y1)** (composite band):
   - Read BGRA from `processBuffer` (respecting `bytesPerRow`).
   - **Smoothstep** on mask value (constants **`edgeLo = 0.36`**, width tied to **0.62 − edgeLo**) for stabler alpha at boundaries.
   - RGB scaled by **~0.92 × alpha** (slight darken), alpha from remapped mask.
5. Non-band pixels remain **alpha 0**.

**Letterbox path** can use a vDSP-accelerated band composite; ONNX-style uses the smoothstep path above inside the same function.

### 13) Debug overlay (optional)

- If **`debugMode`:** draws detection rectangles on the composed image (`drawOnnxStyleDebugDetectionBboxesOnComposedImage`), with **Y flipped** for CGContext vs model coords.

### 14) Display rotation

- If **buffer is landscape** and **`lockedOrientation`** is **not** landscape, and **not** AR camera path: **`rotateCGImage90(clockwise: true)`** before assigning **`maskImageView.image`** so portrait-locked rooms match UI.

### 15) Progress + processing lock

- **`setProgress`** during preprocess / inference / composite.
- **`finishFirstSegmentationIfNeeded`** when mask has any positive pixel.
- **`resetProcessingFlag()`** at end so the next camera frame can run (drops **`isProcessing`**, may resume AR session if that hybrid path is active).

---

## Blacklist

- Loaded once via **`loadBlacklistOnce()`** → **`blacklist.json`** (bundled with Furniture Fit assets).
- Class IDs populate **`clsToIgnore`**; applied in parser and filters.

---

## Android parity notes

- **`FurnitureFitOnnxStylePipeline`** is explicitly documented as mirroring **`FurnitureFitManager`** ONNX behavior (NMS ordering concepts, supporting table, bbox-limited sigmoid mask).
- **Differences to be aware of:**
  - **Primary pick:** iOS ONNX-style common path uses **`selectPrimaryIndexCoreFlow`**, not **`pickPrimaryIndex`** in `FurnitureFitOnnxStylePipeline.swift` (unused).
  - **NMS IoU / pre-NMS cap:** iOS uses **0.5** and **100** in this path; enum has different constants for documentation/Android reference.
  - **Inference:** Android ONNX vs iOS Core ML may differ slightly in numerics; mask logic is shared in Swift after tensors are in `MLMultiArray` form.

---

## Performance and memory

- **Stretch buffer** reused; **prototype** and **composite** scratch arrays grown to fit, not shrunk every frame (see `upscaledPlanarMaskScratch`, `protoPlanes`, etc.).
- **Core ML** inference dominates latency (hundreds of ms on CPU for large exports); ONNX-style **Stage 3–5** (parse/NMS/mask) and composite can add **hundreds of ms** more on busy frames (logs show **~0.9–1.0 s** total possible).
- **`autoreleasepool`** around `processFrame` reduces transient Obj-C pressure per frame.
- **`releaseF16Scratch`** after detection parse reduces peak after heavy FP16 tensors.

---

## Quick reference — constants (typical)

| Item | Value / location |
|------|-------------------|
| `bboxExpandMargin` | `0.08` — `FurnitureFitView.swift` |
| NMS IoU (ONNX-style common path) | `0.5` |
| Pre-NMS top-K | `100` |
| Primary min conf / area (STAGE 4) | `0.15` / `0.02` normalized |
| Morphological close | `furnitureFitUseMorphologicalCloseMask = true` |
| Model input side (typical) | `640` — from Core ML image constraint / `YoloEImageInference.modelInputSize` |
| Proto channels | `32` |

---

## Related reading

- `docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md` — SHARP room vs splat AABB, pinhole vs proportion, fitment ratios, overlay scale (`[FurnitureFitSize]` / `[FurnitureFitOverlay]`).
- `docs/YOLOE_COREML_FIX.md` — Core ML / export issues.
- `docs/ON_DEMAND_RESOURCES.md` — YOLOE ODR (`YOLOEModelService`).
- `scripts/README_YOLOE_COREML.md` — export / conversion scripts.

If you add a new YOLOE export, update **`YoloEDetectionParser.knownDetectionProtoPairs`** and verify **`parsePrototypes`** shape handling for the new proto tensor layout.
