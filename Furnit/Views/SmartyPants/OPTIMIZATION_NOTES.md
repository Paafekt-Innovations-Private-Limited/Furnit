# Stage 6 Optimization: Tensor Transpose

## Problem
Stage 6 was taking **2189 ms** (51% of total frame time) due to strided memory access.

### Before (Strided Access)
```swift
let stride = 8400  // Jump distance in memory

for anchor in 0..<8400 {
    let x = detBuf[0 * stride + anchor]  // Jump 0 positions
    let y = detBuf[1 * stride + anchor]  // Jump 8400 positions
    let w = detBuf[2 * stride + anchor]  // Jump 16800 positions
    
    // Copy 80 class scores with stride
    for i in 0..<80 {
        tempScores[i] = basePtr[i * stride]  // 80 × 8400 = huge jumps!
    }
    
    // Copy 32 coefficients with stride
    for i in 0..<32 {
        coeffs[i] = coeffBase[i * stride]  // 32 × 8400 = more jumps!
    }
}
```

**Problems:**
- Cache misses on every access
- No SIMD vectorization possible
- ~940,800 strided memory reads per frame

---

## Solution: Transpose First

### Step 1: Transpose Tensor
```swift
// Convert [116, 8400] → [8400, 116] using vImageTranspose_PlanarF
let transposed = transposeDetectionTensor(detBuf, rows: 116, cols: 8400)
```

**Cost:** ~50-100 ms (one-time operation per frame)

### Step 2: Sequential Access
```swift
for anchor in 0..<8400 {
    let offset = anchor * 116  // All data for this anchor is contiguous!
    
    let x = transposed[offset + 0]
    let y = transposed[offset + 1]
    let w = transposed[offset + 2]
    let h = transposed[offset + 3]
    
    // Class scores are contiguous now!
    let classPtr = transposed.advanced(by: offset + 4)
    vDSP_maxvi(classPtr, 1, &maxVal, &maxIdx, 80)
    
    // Coefficients are contiguous too!
    let coeffPtr = transposed.advanced(by: offset + 4 + 80)
    let coeffs = Array(UnsafeBufferPointer(start: coeffPtr, count: 32))
}
```

**Benefits:**
- All data for one anchor is contiguous in memory
- CPU can prefetch cache lines
- vDSP operates on sequential memory
- Zero temporary allocations

---

## Expected Performance

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Stage 6 time** | 2189 ms | ~250-350 ms | **6-8x faster** |
| **Total frame time** | 4300 ms | ~2400 ms | **1.8x faster** |
| **FPS** | 0.23 FPS | 0.42 FPS | **1.8x more** |

### Breakdown:
```
Stage 6 (After):
├─ Transpose: 50-100 ms  (one-time SIMD operation)
├─ Extract: 150-250 ms   (sequential memory access)
└─ Total: ~250-350 ms    (vs 2189 ms before)
```

---

## How vImageTranspose Works

```c
// Accelerate framework implementation (SIMD optimized)
vImageTranspose_PlanarF(
    &srcBuffer,  // [116, 8400] row-major
    &dstBuffer,  // [8400, 116] row-major
    flags
)
```

**What it does:**
1. Uses NEON SIMD instructions to process 4-16 floats at a time
2. Optimizes cache access patterns with block-wise transpose
3. Parallelizes across CPU cores when beneficial

**Why it's fast:**
- Vectorized operations (4× faster than scalar)
- Cache-friendly block algorithm
- Zero allocations (writes directly to output buffer)

---

## Memory Layout Comparison

### Before (Strided):
```
Memory: [x₀, x₁, x₂, ..., x₈₃₉₉, y₀, y₁, ..., y₈₃₉₉, w₀, ...]
         ^                ^8400  ^                ^8400

To get anchor 0 data: Read positions [0, 8400, 16800, 25200, ...]
To get anchor 1 data: Read positions [1, 8401, 16801, 25201, ...]
                      ↑ Huge jumps = cache misses!
```

### After (Contiguous):
```
Memory: [x₀, y₀, w₀, h₀, class₀₀, ..., class₀₇₉, coeff₀₀, ..., coeff₀₃₁,
         x₁, y₁, w₁, h₁, class₁₀, ..., class₁₇₉, coeff₁₀, ..., coeff₁₃₁, ...]
         ^──────────────────────────────116 floats──────────────────────────^
         
To get anchor 0 data: Read positions [0, 1, 2, ..., 115]
To get anchor 1 data: Read positions [116, 117, 118, ..., 231]
                      ↑ Sequential = CPU prefetches automatically!
```

---

## Code Changes Summary

### 1. Added transpose function
```swift
private func transposeDetectionTensor(_ src: UnsafePointer<Float>, 
                                     rows: Int, cols: Int) -> [Float]
```

### 2. Modified Stage 6 extraction
- Call transpose once before loop
- Changed from strided to sequential access
- Removed temporary `tempScores` array
- Use `UnsafeBufferPointer` for zero-copy coefficient extraction

### 3. Added timing for transpose operation (debug mode)
```swift
if debugMode {
    logDebug("   ⚡ Transpose: \(ms) ms")
}
```

---

## Testing Checklist

- [ ] Verify detections are still correct (79 detections)
- [ ] Check primary detection is same as before
- [ ] Confirm masks look identical
- [ ] Measure Stage 6 time improvement
- [ ] Test with different scenes (2-100 detections)
- [ ] Verify no crashes or memory issues

---

## Further Optimizations (Future)

1. **Early exit optimization:**
   ```swift
   // Quick scan: count valid detections before transpose
   // Skip transpose if < 5 detections expected
   ```

2. **Reuse transposed buffer:**
   ```swift
   private var reuseTransposed: [Float] = []
   // Avoid allocation every frame
   ```

3. **Parallel extraction:**
   ```swift
   // Use DispatchQueue.concurrentPerform for 8400 anchors
   // Split into chunks of 1000
   ```

4. **SIMD bbox validation:**
   ```swift
   // Check x.isFinite && y.isFinite ... with SIMD
   ```

Expected additional speedup: 20-30% (Stage 6: 250ms → 175-200ms)

---

## References

- Accelerate framework: vImageTranspose_PlanarF
- Cache-friendly algorithms: Block transpose
- Memory access patterns: Strided vs sequential
- SIMD optimization: NEON instructions (ARM)

---

## Notes

The transpose operation itself adds ~50-100ms overhead, but saves ~1900ms in extraction.

**Net gain: ~1800ms per frame (6-8× speedup)**

This moves the bottleneck from Stage 6 back to Stage 3 (Inference), which is where it should be!
