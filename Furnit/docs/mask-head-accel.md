# Problem: un-accelerated mask-head matmul in RTMDet instance segmentation (iOS / Swift)

> **Historical record.** This describes the retired Core ML prototype and its 80×80
> Swift mask head. Production iOS now runs the shared 160×160-mask FP16 TFLite graph
> through LiteRT Metal; see `Furnit/Models/RTMDet/README.md` and
> `Furnit/Views/FurnitureFit/README.md`. Do not use this document as current runtime
> guidance.

## Context
iOS app doing **RTMDet-Ins** instance segmentation on-device via **Core ML**. The model is a
"raw heads" export: the backbone runs on the **Apple Neural Engine** (loaded preferring
`.cpuAndNeuralEngine`), but the **dynamic-conv mask head is computed by us in Swift on the CPU with
plain scalar loops — no Accelerate / BLAS / BNNS / SIMD / Metal**. We want to accelerate just that
mask head while producing numerically-equivalent masks. It runs per frame, on a serial queue, for
each detected instance (≤ 6).

File: `Furnit/Services/OnDevice/RTMDetImageInference.swift`.

## Model raw outputs (3 FPN levels, sides 80/40/20)
- `cls_{80,40,20}`    : `[1, 80, side, side]` — 80 class logits
- `bbox_{80,40,20}`   : `[1, 4, side, side]`
- `kernel_{80,40,20}` : `[1, 169, side, side]` — per-location dynamic-conv weights
- `mask_feat`         : `[1, 8, 80, 80]` — shared mask features (always 80×80). May be **fp16 or
  fp32**, and **strides are not assumed contiguous** (read through a stride-aware accessor).

## The math we want to accelerate
For each selected detection we produce an **80×80 mask**. The detection carries a **169-float
`kernel`** (sampled from `kernel_*` at its grid cell) plus a prior `(priorX, priorY, stride)`. Those
169 floats encode a tiny **per-instance MLP evaluated at all 6,400 mask pixels**:

- **input (10):** `[relX, relY, feat0…feat7]`
  - `gridX = (x + 0.5) * 8`, `gridY = (y + 0.5) * 8`
  - `relX = (priorX - gridX) / (stride * 8)`, `relY = (priorY - gridY) / (stride * 8)`
  - `featc = mask_feat[0, c, y, x]`
- **Layer 1:** `W1[8×10]`, `b1[8]` → ReLU → `h1[8]`
- **Layer 2:** `W2[8×8]`, `b2[8]` → ReLU → `h2[8]`
- **Layer 3:** `W3[1×8]`, `b3[1]` → logit → sigmoid
- **`kernel` packing (169):** `W1 = [0..80)`, `W2 = [80..144)`, `W3 = [144..152)`,
  `b1 = [152..160)`, `b2 = [160..168)`, `b3 = [168]`

**Output contract:** an `[Float]` of length `80*80`, row-major `out[y*80 + x]`, sigmoid values in
`[0,1]` (or `nil`). Consumed downstream by `buildCombinedRawMaskImage`, which samples/thresholds the
plane.

**Key structural fact:** `mask_feat` (the `8×6400` feature matrix) is **shared across all
instances** — only the 2 coordinate rows and the `kernel`/biases differ per instance.

## Current scalar implementation (the hot loop to replace)
```swift
private static func buildRawMaskPlane(candidate: RawCandidate, maskFeat: MLMultiArray) -> [Float]? {
    guard candidate.kernel.count == 169 else { return nil }
    let featAt = nchwReader(for: maskFeat)
    let maskSide = 80
    var out = [Float](repeating: 0, count: maskSide * maskSide)

    let w1 = 0
    let w2 = w1 + 80
    let w3 = w2 + 64
    let b1 = w3 + 8
    let b2 = b1 + 8
    let b3 = b2 + 8

    // Scratch buffers reused across pixels.
    var input = [Float](repeating: 0, count: 10)
    var hidden1 = [Float](repeating: 0, count: 8)
    var hidden2 = [Float](repeating: 0, count: 8)

    for y in 0..<maskSide {
        for x in 0..<maskSide {
            let gridX = (Float(x) + 0.5) * 8
            let gridY = (Float(y) + 0.5) * 8
            input[0] = (candidate.priorX - gridX) / max(1, candidate.stride * 8)
            input[1] = (candidate.priorY - gridY) / max(1, candidate.stride * 8)
            for c in 0..<8 {
                input[2 + c] = featAt(0, c, y, x)
            }

            for o in 0..<8 {
                var sum = candidate.kernel[b1 + o]
                for i in 0..<10 {
                    sum += candidate.kernel[w1 + o * 10 + i] * input[i]
                }
                hidden1[o] = max(0, sum)
            }

            for o in 0..<8 {
                var sum = candidate.kernel[b2 + o]
                for i in 0..<8 {
                    sum += candidate.kernel[w2 + o * 8 + i] * hidden1[i]
                }
                hidden2[o] = max(0, sum)
            }

            var logit = candidate.kernel[b3]
            for i in 0..<8 {
                logit += candidate.kernel[w3 + i] * hidden2[i]
            }
            out[y * maskSide + x] = sigmoid(logit)
        }
    }
    return out
}
```
~**1M scalar multiply-adds per detection per frame**, single-threaded.

## Supporting code
```swift
private struct RawCandidate {
    let box: BoxRecord
    let kernel: [Float]   // 169 floats sampled at the detection's grid cell
    let priorX: Float
    let priorY: Float
    let stride: Float
}

// kernel sampled per detection in decodeRawCandidates:
var kernel = [Float](repeating: 0, count: 169)
for i in 0..<169 { kernel[i] = kernelAt(0, i, y, x) }      // kernelAt = nchwReader(for: kernel_level)

// stride-aware reader (handles fp16/fp32 + non-contiguous strides):
private static func nchwReader(for array: MLMultiArray) -> (Int, Int, Int, Int) -> Float {
    let valueAtOffset = floatReader(for: array)              // fp16/fp32 -> Float
    let strides = array.strides.map(\.intValue)              // [n,c,h,w]
    let s0 = strides[0], s1 = strides[1], s2 = strides[2], s3 = strides[3]
    return { n, c, y, x in valueAtOffset(n*s0 + c*s1 + y*s2 + x*s3) }
}

// caller: builds a mask plane per selected (post-NMS) detection, ≤ maxDetectionCount (6):
let rawMaskPlanes = selected.map { buildRawMaskPlane(candidate: $0, maskFeat: maskFeat) }
```

## Constraints
- Output masks must be numerically equivalent (small fp tolerance is fine).
- iOS, Swift; **Accelerate / BNNS / MPS available**. `model.prediction` (backbone) already runs on
  the ANE.
- Runs on a serial `detectionQueue`, per frame, for each instance.
- `mask_feat` may be fp16 and strided → needs one stride-aware copy to a contiguous `[Float]`
  (or fp16) buffer.

## Questions for the reviewer
1. **Best vectorization:** express as **3 GEMMs over 6,400 columns** — build `F[8×6400]` once
   (shared), then per instance `X[10×6400] = [coordRows(2); F]`,
   `H1 = ReLU(W1·X + b1)`, `H2 = ReLU(W2·H1 + b2)`, `L = W3·H2 + b3`, sigmoid. Use **`cblas_sgemm`**
   vs **BNNS fully-connected** vs **MPS/Metal**? Which is fastest/simplest for an 8-wide,
   6,400-column problem with ≤ 6 instances?
2. **Memory layout:** keep `F` shared across instances; batch all instances into one big GEMM
   (stack kernels) or loop per instance?
3. **Fuse into the model?** Is it worth fusing the whole mask head back **into the Core ML model**
   (so it runs on the ANE) instead — and is the per-pixel coord feature (`relX/relY`) expressible in
   the exported graph?
4. **Precision:** fp16 vs fp32 for the GEMMs given ANE/CPU tradeoffs and the small inner dim (8/10).

## Next step
Once guidance is received, validate against this code (shapes, the 169-weight packing, the
stride/fp16 handling, output equivalence), then implement + compile-verify.

---

## Reviewer guidance (received)

**Strategic verdict: profile before optimizing this loop — it is almost certainly NOT the
bottleneck.** ~1M MACs/instance × 6 ≈ 6M MACs/frame; naive scalar Swift does that in single-digit ms
(<2ms). Observed frame times are 500–900ms, so the time is elsewhere — most likely (a) the Core ML
backbone inference itself, (b) the **full-resolution RGBA cutout loop** in `buildCombinedRawMaskImage`
(≈720×1280 per instance), or (c) the stride-aware closure called per-element. **Instrument those
three first.** If you vectorize the mask head and the frame is still ~700ms, you optimized the wrong
thing.

The GEMM rewrite is still cheap and correct to do, with these answers:

- **Q1 — Use `cblas_sgemm` (Accelerate BLAS).** Skip BNNS (descriptor boilerplate for no gain at this
  size), skip MPS/Metal (GPU encode+sync overhead dwarfs a 1M-MAC workload — adds latency), skip ANE.
  sgemm is NEON-vectorized and ideal for "tiny K (10/8/8), wide N (6400)" — runs in microseconds.
- **Q2 — Share `F`, split GEMM1, loop over instances (no batching).** Build contiguous `F[8×6400]`
  **once per frame** (stride-aware copy). Per instance, don't rebuild `X[10×6400]` (copies F 6×);
  split layer 1 into two accumulating GEMMs so F is reused in place:
  ```
  H1  = W1_feat[8×8]  · F[8×6400]        (beta=0)
  H1 += W1_coord[8×2] · coord[2×6400]    (per-instance coord rows, beta=1)
  H1 += b1 (broadcast) → ReLU
  H2  = ReLU(W2[8×8] · H1 + b2)
  L   = W3[1×8] · H2 + b3 → sigmoid
  ```
  ≤6 instances → plain `for` loop of these small GEMMs. Don't build a block-diagonal/stacked batched
  GEMM (weights + coords differ per instance → no gain, more complexity). Fold bias by pre-filling C
  with the broadcast bias and `beta=1`. ReLU via `vDSP_vthres` (clamp to 0). Sigmoid via `vvexpf`
  over the 6400 logits.
- **Q3 — Do NOT fuse the mask head back into Core ML.** The original blocker stands: per-instance
  kernel selection is a **post-NMS dynamic gather**, which Core ML/ANE mangled (the `(0,0)` mask
  collapse). `relX/relY` is graph-expressible, but the dynamic gather isn't worth re-fighting, and
  there's no perf reason since CPU sgemm here is sub-millisecond.
- **Q4 — fp32 (`sgemm`), not fp16.** Inner dims 8/10 → precision moot; fp16 GEMM gives no CPU speedup
  at this size and complicates "numerically equivalent". Convert `mask_feat` fp16→fp32 **once** during
  the stride-aware contiguous copy (~205KB, trivial).

**Net:** one shared `F` copy + a 6-iteration loop of three `cblas_sgemm` calls with vDSP ReLU/sigmoid
turns the mask head from ~ms to ~µs — **but profile first**; the 700ms most likely lives in the RGBA
cutout loop or the model inference.

## Action taken: per-stage timing instrumentation (do this first)
Added `debug`-gated per-stage timing to `runInstanceSegmentation` / `runRawHeadPostprocess`, surfaced
as a `stageMillis:` line in the RTMDet debug frame footer, covering: resize → input-tensor build →
Core ML prediction → decode+NMS → mask planes (`buildRawMaskPlane`) → combined mask → instance masks
(`buildCombinedRawMaskImage`). Run with debug logging on and read the breakdown before implementing
any GEMM.
