# FurnitureFit Segmentation Pipeline

Real-time furniture segmentation using **RTMDet-Ins-m** (Core ML raw-head export, typically **640×640** input) with instance segmentation masks.

## Architecture Overview

```
Camera/Still Frame → Core ML image input → RTMDet raw heads → Confidence-first NMS →
Mask-head planes → Mask-affinity grouping → Pixel-level cutout union → Overlay gesture/display
```

Current implementation notes:

- `RTMDetImageInference` owns raw-head decoding (`cls/bbox/kernel` at 80/40/20 plus `mask_feat`), confidence-first class-aware NMS, per-instance mask building, mask-affinity grouping, and cached mask rebuilds for live selection.
- `SettingsFurnitureFitImageScanView` uses the same uncapped RTMDet still-image path as the live flow: `maxDetectionCount: nil`, no fixed detection cap, fused instance masks, and pixel-level RGBA union.
- `FurnitureFitContainerView` displays the cutout in `maskImageView`; pinch/pan transforms are applied through `userPinchScale`, `userPanOffset`, and `FurnitureFitOverlayScaling.resolvedTransform`.
- In USDZ / GLB / saved PLY room viewers, the room layer also owns pinch zoom. When a segmented mask is visible, the FurnitureFit overlay must claim two-finger touches so the user scales the segmented cluster rather than the room camera.

## Room viewer brain flow

Room viewers (`ModelViewerView`, `GLBRoomView`, `MeshRoomView`, `SplatRoomView`) embed
`FurnitureFitView` as an inline overlay. Modes are driven by `FurnitureFitSegmentationMode`:

| User action | Mode | Camera preview | Result |
|---|---|---|---|
| Tap **brain** | `segmentPrimary` | Hidden (analysis only) | Auto-segment highest-confidence primary over 3D room |
| Tap **text.viewfinder** | `identifyOnly` + full-video | Live AVCapture preview | Cluster union boxes; tap to pin items |
| Tap **Segment** pill | `segmentSelected` + full-video | Hidden again | Transparent cutout(s) over 3D room for multi-item fitment |
| Tap **Stop** / exit brain | — | — | Return to room browsing |

Key implementation details:

- Brain opens with `segmentPrimary` by default (`toggleFurnitureFit` in room viewer files).
- Full-video is toggled by `showFullVideoWithIdentifications` via the **text.viewfinder** button.
- `shouldShowLiveCameraPreview` is true only in `identifyOnly` (including full-video identify).
- During full-video `segmentSelected`, inference stays live but `previewLayer` hides so mask alpha reveals the room underneath (`isFullVideoSelectedSegmentation`).
- Tap selection is gated to full-video identify (`shouldAllowBoundingBoxTapSelection`).

See `Furnit/diagrams/rtmdet-swift-flow.svg` and `Furnit/docs/README.md` for the smoke test checklist.

## Problems & Solutions

### 1. Memory Crash After Extended Use

**Problem:** App crashed after running segmentation for several minutes. Memory grew unbounded over time.

**Root Causes:**
- `CVPixelBuffer` created every frame in `resizeToSquare()` (~6.5MB/frame)
- `MLMultiArray` created every frame in `pixelBufferToMLMultiArray()` (~19MB/frame)
- Large SGEMM matrices `B` and `C` allocated per frame (`O(planeSize × N)`)
- No `autoreleasepool` around frame processing

**Solutions:**

1. **Reusable CVPixelBuffer:**
```swift
private var cachedSquareBuffer: CVPixelBuffer?
private var cachedSquareSize: Int = 0

// In resizeToSquare():
if cachedSquareSize != size || cachedSquareBuffer == nil {
    // Only create new buffer if size changed
    CVPixelBufferCreate(...)
    cachedSquareBuffer = newBuffer
    cachedSquareSize = size
}
```

2. **Reusable MLMultiArray:**
```swift
private var cachedMLArray: MLMultiArray?
private var cachedMLArraySize: Int = 0

// In pixelBufferToMLMultiArray():
if cachedMLArraySize != width || cachedMLArray == nil {
    cachedMLArray = try? MLMultiArray(shape: [...])
    cachedMLArraySize = width
}
```

3. **Per-candidate SGEMV instead of batched SGEMM:**

Before (memory: `O(planeSize × N)`):
```swift
var B = [Float](repeating: 0, count: 32 * N)      // ~1KB
var C = [Float](repeating: 0, count: planeSize * N)  // ~10MB for N=100
blas_sgemm_rowmajor_transA(...)  // Compute all at once
```

After (memory: `O(planeSize)`):
```swift
private var scratchPrimaryLogits: [Float] = []    // ~100KB
private var scratchCandidateLogits: [Float] = []  // ~100KB (reused per candidate)

for candidate in prunedCandidates {
    blas_sgemv_rowmajor_trans(...)  // Compute one candidate at a time
    // Check overlap immediately, no need to store all results
}
```

4. **Autoreleasepool wrapper:**
```swift
private func processFrame(_ pixelBuffer: CVPixelBuffer) {
    autoreleasepool {
        processFrameInner(pixelBuffer)
    }
}
```

**Result:** Memory usage dropped from ~25MB/frame allocation to near-zero per-frame allocation.

---

### 2. Slow Camera Movement Between Furniture Items

**Problem:** When switching focus between furniture items, the camera response was slow/laggy.

**Root Cause:** Frame handling had unnecessary overhead:
- `sessionGeneration` variable tracking
- `isStarted` guard checks
- Stale frame detection logic

**Solution:** Simplified frame handling:
```swift
func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, ...) {
    let now = Date()
    frameLock.lock()
    let shouldProcess = now.timeIntervalSince(lastProcessTime) >= processInterval && !isProcessing
    if shouldProcess {
        isProcessing = true
        lastProcessTime = now
    }
    frameLock.unlock()

    guard shouldProcess else { return }
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        resetProcessingFlag()
        return
    }
    detectionQueue.async { [weak self] in self?.processFrame(pixelBuffer) }
}
```

---

### 3. Distant Objects Incorrectly Included in Segmentation

**Problem:** Objects far from the primary furniture (e.g., a chair across the room) were being included in the segmentation mask.

**Root Cause:** Mask overlap was computed over the entire prototype space. Objects with no spatial relationship could have "overlapping" masks due to prototype coefficient similarities.

**Solution:** Added bbox intersection check before mask overlap:
```swift
// Skip if candidate bbox doesn't intersect with primary bbox at all
let intersects = !(dx2 < origPX1 || dx1 > origPX2 || dy2 < origPY1 || dy1 > origPY2)
if !intersects {
    if debugMode { logDebug("   ⏭️ [\(i)]: skipped - bbox doesn't intersect primary") }
    continue
}
```

**Result:** Only candidates with spatially overlapping bounding boxes are considered for mask overlap computation.

---

### 4. Too Many Redundant Detections / Wrapper Boxes

**Problem:** RTMDet produced many overlapping detections for the same object, slowing down processing.

**Solution:** Class-aware NMS runs after parsing detections. It is **confidence-first**: highest-confidence boxes are kept first; area is only a deterministic tie-breaker.
```swift
func applyNMS(boxes: [CGRect], scores: [Float], iouThreshold: Float) -> [Int] {
    var indices = scores.enumerated().sorted(by: { $0.element > $1.element }).map { $0.offset }
    var keep = [Int]()

    while !indices.isEmpty {
        let current = indices.removeFirst()
        keep.append(current)

        indices.removeAll { next in
            let intersection = boxes[current].intersection(boxes[next])
            let iou = intersection.area / (boxes[current].area + boxes[next].area - intersection.area)
            return iou > CGFloat(iouThreshold)
        }
    }
    return keep
}

// Current raw-head path:
let selected = classAwareNMS(rawCandidates, iouThreshold: 0.5, limit: maxDetectionCount)
```

For exploratory image scan and live RTMDet, `maxDetectionCount` can be `nil`, so NMS suppresses duplicates without imposing a hard count cap.

---

### 5. Object-Piece Fusion / Mask Affinity

**Problem:** One physical object can appear as multiple RTMDet pieces (chair back, seat, base). Bounding-box overlap alone is not enough to fuse these parts.

**Solution:** Build raw mask planes first, then connect detections whose binary masks have enough pixel-level affinity. The transitive group is used when building per-instance cutouts and cached selected masks.

```swift
let maskAffinityGraph = makeMaskAffinityGraph(rawMaskPlanes)
let groupIndices = maskAffinityGraph.transitiveGroup(seedIndices: [index])
```

This is class-agnostic and applies to every detected object, not chair-specific rules.

---

### 6. Settings Image Scan Drift

**Problem:** Settings image scan had its own detection cap, bbox-overlap clustering, and Core Graphics mask drawing. That made the diagnostic path differ from the live RTMDet path.

**Solution:** Settings image scan now calls `RTMDetImageInference.runInstanceSegmentation` with uncapped detections, consumes fused `instanceMaskImages`, and performs pixel-level RGBA union instead of alpha-blended drawing.

---

### 7. Segmented Cluster Pinch / Main Flow Gesture Ownership

**Problem:** In room viewers, the USDZ / GLB / saved PLY room layer also handles pinch zoom. If the FurnitureFit overlay rejects a touch or returns `nil` from `hitTest`, pinch goes to the room camera instead of resizing the segmented cutout.

**Solution:** When a segmented mask is visible, `FurnitureFitContainerView` claims two-finger touches for overlay pinch. Single-finger pan remains stricter and requires touching actual mask pixels. A padded cluster-bounds hit target makes disconnected object parts easier to pinch.

---

### 8. Incorrect Metal Shader Mask Threshold

**Problem:** Mask threshold in Metal shader was `< 0.5f`, which is arbitrary for binary masks.

**Root Cause:** Mask values are binary (0 or 255), which when read as R8Unorm become 0.0 or 1.0.

**Solution:**
```metal
// Before:
if (m < 0.5f) { ... }

// After:
if (m <= 0.0f) { ... }
```

---

## Pipeline Stages

### STAGE 1: Input / Preprocess
- Current RTMDet Core ML export accepts an image input when available; BGR mean/std normalization is pushed into the model graph.
- Legacy multi-array fallback still supports BGRA → NCHW normalization in Swift.

### STAGE 2: Inference
- Run RTMDet model (`RTMDetImageInference.modelInputSize` / Core ML image constraint)
- Output: detection tensor + prototype masks

### STAGE 3: Parse Outputs
- Extract bounding boxes, confidence scores, class IDs, mask coefficients
- Parse 32-channel prototype masks (160×160)

### STAGE 3b: NMS
- Apply confidence-first class-aware NMS (IoU threshold: 0.5)
- `maxDetectionCount` is optional; `nil` means no artificial cap after suppression

### STAGE 4: Select Primary
- Score = conf^1.5 × area_norm^1.2 × center_term
- Select highest-scoring detection as primary furniture

### STAGE 5a: Prune Candidates
- Filter by confidence (> 0.1)
- Skip if bbox encompasses primary (background detection)
- Skip if bbox doesn't intersect primary

### STAGE 5b: Mask Affinity
- Build raw mask planes from each RTMDet dynamic kernel + `mask_feat`
- Convert planes to bitsets at threshold
- Connect detections by overlap affinity, then use transitive groups for fused object masks

### STAGE 6: Build Mask
- Compute union bbox of kept detections
- Build full-resolution mask via upscaling
- Clip to union bbox

### STAGE 6b: Composite
- CPU path builds full-resolution RGBA cutouts directly from the source buffer
- Combined still-image masks use pixel-level union, not Core Graphics alpha blending
- Active pixels are forced opaque to avoid foggy translucent cutouts

### STAGE 7: Finalize
- Draw debug overlays (bboxes, labels) if debug mode
- Rotate for portrait display if needed
- Present result

---

## Memory Management

### Reused Buffers (Instance Properties)
```swift
// Prototype parsing
private var protoRawFloats: [Float] = []
private var protoPlanes: [Float] = []

// BLAS scratch buffers
private var scratchPrimaryLogits: [Float] = []
private var scratchCandidateLogits: [Float] = []

// CVPixelBuffer & MLMultiArray
private var cachedSquareBuffer: CVPixelBuffer?
private var cachedMLArray: MLMultiArray?

// Metal buffers
private var cachedFusedPlanesBuf: MTLBuffer?
private var cachedFusedCoeffBuf: MTLBuffer?
```

### Memory Logging
```swift
private func logMemory(_ tag: String) {
    var info = mach_task_basic_info()
    // ... get task info ...
    let usedMB = Double(info.resident_size) / 1024.0 / 1024.0
    logDebug("🧠 [\(tag)] Memory: \(String(format: "%.1f", usedMB)) MB")
}
```

Logs at: FRAME START, AFTER INFERENCE, AFTER STAGE 5b, AFTER BUILD MASK, FRAME END

---

## Debug Mode

Enable via `QualitySettings.debugMode`. Shows:
- Timing for each stage
- Memory usage at key points
- Detection details (class, confidence, bbox)
- Filter decisions (kept/rejected with reason)
- Bounding box overlays (red=primary, cyan=kept, green=union)

---

### 9. Full Video Mode — Cluster Bounding Boxes

**Problem:** Full video mode displayed individual detection bounding boxes. Users tapped one piece of a multi-piece object and only that piece was selected.

**Solution:** The mask affinity graph (always built when raw mask planes exist) groups detections into clusters. Full video mode now:
- Displays a single union bounding box per cluster with combined labels.
- Tapping any cluster bbox selects all its members for segmentation.
- Selection highlights apply to entire clusters.

```swift
let clusters = buildClustersFromAffinityGraph(graph, rankedCandidates: candidates)
// Each cluster's union bbox is displayed; tap selects all members
```

---

### 10. Independent Per-Furniture Movement (Multi-Select)

**Problem:** When multiple furniture items were selected and segmented, they could not be moved independently — all items were stacked or moved together.

**Solution:** Regime A (freeze on selection):
- Each selected item gets an independent overlay with a stable `UUID`.
- Items are frozen at selection — not updated from live detections.
- Hit-testing uses each item's **transformed bounding box** (not original position) with alpha fallback.
- Z-order tie-breaking: topmost (last-added) item wins.
- `.began`-miss recovery in `handlePan`: re-resolve if `activeGestureOverlayItemIndex` is nil at `.changed`.
- `overlayPresentationMode: .measuredPlacement` keeps items at detected positions (no auto-centering).
- `resolvedTransform` preserves user pan/pinch even when `debugFreezeOverlayScale` is true.

---

### 11. Debug Bounding Box Drawing

**Problem:** After the RTMDet migration, debug bounding box visualization was lost.

**Solution:** Single `drawDebugDetectionBboxes` helper called from both live and cached segmentation paths. 4-color scheme:
- **Red** — primary detection
- **Orange** — affinity group member (pulled in by mask overlap)
- **Yellow** — explicit pin (user-selected)
- **Cyan** — unselected candidate

Boxes and a burned-in legend are drawn in image space on the CGImage. UIView bbox overlay is suppressed in debug to avoid double-draw.

---

### 12. Thermal & Cadence Management

**Problem:** Live RTMDet ran back-to-back (~100% duty cycle), causing high thermal load. Inference continued during placement (results discarded) and when the app was backgrounded.

**Solution:**

1. **5fps cadence**: `rtmdetLiveTargetInterval = 200ms` creates idle gaps between inference runs.
2. **Placement pause**: `inferencePausedForPlacement` skips inference entirely when independent overlay items are active. Camera preview stays alive.
3. **Background pause**: Observers on `willResignActiveNotification` / `didBecomeActiveNotification` stop/resume the capture session.
4. **Thermal backoff**: Observes `thermalStateDidChangeNotification`:
   - `.nominal` / `.fair` → 200ms (~5fps)
   - `.serious` → 400ms (~2.5fps)
   - `.critical` → pause inference entirely, keep last boxes
5. **Camera ownership**: 150ms settle delay between AR pause and AVCapture start reduces `-17281` contention.

The cadence interval (`rtmdetLiveTargetInterval`) is a single tunable constant adjusted at runtime by thermal backoff.

---

## Key Files

- `FurnitureFitView.swift` - Main pipeline implementation
- `FurnitureFitOverlayScaling.swift` - Overlay transform computation (pan, pinch, assisted scale)
- `CompositeKernels.metal` - GPU compositing shaders
- `MetalMaskLogic.swift` - Metal buffer management
- `classes.json` (per language in `Furnit/xx.lproj/`) — class ID → display name for bbox labels; `Bundle` loads the file for the active locale
- `blacklist.json` - Classes to ignore (rooms, walls, etc.)

## Related docs (room size, pinhole, overlay)

For **Depth Anything USDZ rooms vs saved PLY AABB**, **pinhole vs proportion** furniture sizing, **fitment ratios**, and how **`autoScaleFromRoom`** combines with **AR** and **pinch**, see:

- **`docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md`** (repo root `docs/`).
