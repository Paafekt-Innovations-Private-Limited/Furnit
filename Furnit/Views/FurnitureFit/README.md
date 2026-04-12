# FurnitureFit Segmentation Pipeline

Real-time furniture segmentation using YOLO-E **26L PF** (Core ML / optional ONNX; typically **640×640** input) with instance segmentation masks.

## Architecture Overview

```
Camera Frame → Preprocess → YOLO-E Inference → Parse Outputs → NMS →
Select Primary → Filter Candidates → Build Mask → Composite → Display
```

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

### 4. Too Many Redundant Detections

**Problem:** YOLO-E produced many overlapping detections for the same object, slowing down processing.

**Solution:** Added Non-Maximum Suppression (NMS) after parsing detections:
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

// Usage:
let keptIdx = applyNMS(boxes: boxes, scores: scores, iouThreshold: 0.5)
let candidates: [UnionDet] = keptIdx.map { allDets[$0] }
```

---

### 5. Incorrect Metal Shader Mask Threshold

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

### STAGE 1: Preprocess
- Resize input to square (model constraint, usually **640×640** for 26L PF)
- Letterbox padding with gray (128)
- Convert to MLMultiArray

### STAGE 2: Inference
- Run YOLO-E 26L PF model (`YoloEImageInference.modelInputSize` / Core ML image constraint)
- Output: detection tensor + prototype masks

### STAGE 3: Parse Outputs
- Extract bounding boxes, confidence scores, class IDs, mask coefficients
- Parse 32-channel prototype masks (160×160)

### STAGE 3b: NMS
- Apply Non-Maximum Suppression (IoU threshold: 0.5)
- Reduces redundant overlapping detections

### STAGE 4: Select Primary
- Score = conf^1.5 × area_norm^1.2 × center_term
- Select highest-scoring detection as primary furniture

### STAGE 5a: Prune Candidates
- Filter by confidence (> 0.1)
- Skip if bbox encompasses primary (background detection)
- Skip if bbox doesn't intersect primary

### STAGE 5b: Filter by Mask Overlap (SGEMV)
- Compute primary logits: `scratchPrimaryLogits = planes^T × primary.coeffs`
- For each candidate:
  - Compute candidate logits: `scratchCandidateLogits = planes^T × candidate.coeffs`
  - Count overlap pixels (both masks positive)
  - Reject if no overlap
  - Reject if too large (> 1.5× primary size)
  - Reject if duplicate (bbox IoU > 0.7 with kept detections)

### STAGE 6: Build Mask
- Compute union bbox of kept detections
- Build full-resolution mask via upscaling
- Clip to union bbox

### STAGE 6b: Composite
- **GPU path (Metal):** Fused kernel computes max logits and composites in one pass
- **CPU fallback:** Manual pixel-by-pixel compositing
- Alpha channel: 1.0 where mask positive, 0.0 elsewhere (premultiplied)

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

## Key Files

- `FurnitureFitView.swift` - Main pipeline implementation
- `CompositeKernels.metal` - GPU compositing shaders
- `MetalMaskLogic.swift` - Metal buffer management
- `classes.json` (per language in `Furnit/xx.lproj/`) — class ID → display name for bbox labels; `Bundle` loads the file for the active locale
- `blacklist.json` - Classes to ignore (rooms, walls, etc.)

## Related docs (room size, pinhole, overlay)

For **depth raycast room vs splat AABB**, **pinhole vs proportion** furniture sizing, **fitment ratios**, and how **`autoScaleFromRoom`** combines with **AR** and **pinch**, see:

- **`docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md`** (repo root `docs/`).
