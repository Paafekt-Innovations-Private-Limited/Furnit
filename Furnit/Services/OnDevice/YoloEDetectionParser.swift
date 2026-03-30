import Foundation
import CoreML
import Accelerate

// BLAS via C wrapper (same as FurnitureFitView.swift)
fileprivate typealias BLASInt = Int32

@inline(__always)
fileprivate func yolo_blas_scopy(_ n: BLASInt, _ x: UnsafePointer<Float>, _ incx: BLASInt, _ y: UnsafeMutablePointer<Float>, _ incy: BLASInt) {
    BlasScopy(n, x, incx, y, incy)
}

/// Shared YOLO-E detection tensor decoding for still-image calibration and optional reuse.
enum YoloEDetectionParser {

    private static let knownDetectionProtoPairs: [(det: String, proto: String)] = [
        ("var_2286", "var_2369"),
        ("var_2346", "var_2429"),
        ("var_2374", "var_2412"),
        ("detections", "protos"),
    ]

    // ── Reusable scratch buffer (avoids allocation per frame) ──
    // Thread-safety note: parseDetections is always called from a single
    // serial queue (detectionQueue), so unsynchronized mutation is fine.
    private static var f16ScratchBuffer: [Float] = []

    /// Release the Float16 → Float32 scratch buffer to reduce peak memory usage.
    /// Called after detections are parsed and no further reuse is needed.
    static func releaseF16Scratch() {
        f16ScratchBuffer = []
    }

    /// Bulk Float16 → Float32 using Accelerate (vImageConvert_Planar16FtoPlanarF).
    @inline(__always)
    private static func bulkConvertF16ToF32(
        src: UnsafePointer<UInt16>,
        dst: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var srcBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: src),
            height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<UInt16>.size
        )
        var dstBuf = vImage_Buffer(
            data: dst,
            height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<Float>.size
        )
        vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
    }

    /// Check whether an MLMultiArray's backing store is fully contiguous
    /// (i.e. the strides are the standard C-contiguous strides for its shape).
    @inline(__always)
    private static func isContiguous(_ array: MLMultiArray) -> Bool {
        let dims = array.shape.map { $0.intValue }
        let strides = array.strides.map { $0.intValue }
        guard dims.count == strides.count, !dims.isEmpty else { return false }
        var expected = 1
        for i in stride(from: dims.count - 1, through: 0, by: -1) {
            if strides[i] != expected { return false }
            expected *= dims[i]
        }
        return true
    }

    /// Core parsing logic shared between Float32 and converted Float16 buffers.
    private static func parseDetectionsFromBuffer(
        detBuf: UnsafePointer<Float>,
        detArray: MLMultiArray,
        totalCount: Int,
        confidenceThreshold: Float,
        classBlacklist: Set<Int>
    ) -> [FurnitureFitDetection] {
        let dim1 = detArray.shape[1].intValue
        let dim2 = detArray.shape[2].intValue
        let isNewFormat = dim2 < 100

        var allDets: [FurnitureFitDetection] = []
        allDets.reserveCapacity(512)

        if isNewFormat {
            // ── End-to-end format: [1, numDetections, featuresPerDet] ──
            let numDetections = dim1
            let featuresPerDet = dim2

            // After bulk conversion, data is C-contiguous: row i starts at i * featuresPerDet
            let rowStride = featuresPerDet

            for detIdx in 0..<numDetections {
                let base = detIdx * rowStride

                let x1 = detBuf[base + 0]
                let y1 = detBuf[base + 1]
                let x2 = detBuf[base + 2]
                let y2 = detBuf[base + 3]
                let confidence = detBuf[base + 4]
                let classIdxFloat = detBuf[base + 5]

                guard x1.isFinite, y1.isFinite, x2.isFinite, y2.isFinite,
                      confidence.isFinite, classIdxFloat.isFinite else { continue }

                let w = x2 - x1
                let h = y2 - y1
                let x = (x1 + x2) * 0.5
                let y = (y1 + y2) * 0.5

                guard confidence >= confidenceThreshold, confidence <= 1.0 else { continue }
                guard w > 0, h > 0, w < 2000, h < 2000 else { continue }

                let classIdx = Int(classIdxFloat)
                guard classIdx >= 0, !classBlacklist.contains(classIdx) else { continue }

                // Coefficients: indices 6..<38
                let coeffStart = base + 6
                let coeffEnd = coeffStart + 32
                guard coeffEnd <= totalCount else { continue }

                // Validate all 32 coeffs are finite in one pass
                var coeffs = [Float](repeating: 0, count: 32)
                var validCoeffs = true
                coeffs.withUnsafeMutableBufferPointer { cPtr in
                    guard let cBase = cPtr.baseAddress else {
                        validCoeffs = false
                        return
                    }
                    // Bulk copy from contiguous source
                    memcpy(cBase, detBuf.advanced(by: coeffStart), 32 * MemoryLayout<Float>.size)
                    // Finite check — early exit on NaN/Inf
                    for k in 0..<32 {
                        if !cBase[k].isFinite {
                            validCoeffs = false
                            return
                        }
                    }
                }
                guard validCoeffs else { continue }

                allDets.append(
                    FurnitureFitDetection(
                        x: x,
                        y: y,
                        w: w,
                        h: h,
                        confidence: confidence,
                        classIdx: classIdx,
                        coeffs: coeffs
                    )
                )
            }
        } else {
            // ── One-to-many format: [1, numFeatures, numAnchors] ──
            let numFeatures = dim1
            let numAnchors = dim2
            let numClasses = numFeatures - 4 - 32
            guard numFeatures >= 36, numAnchors > 0, numClasses > 0 else { return [] }

            let stride = numAnchors
            let coeffOffset = 4 + numClasses

            // Pre-compute max class score + argmax for ALL anchors at once.
            // Class rows [4 ..< 4+numClasses] are laid out as contiguous rows of `numAnchors` floats,
            // so iterating over classes gives us cache-friendly, sequential access. We also allow
            // early exit for anchors that already have a very high score.
            var maxScores = [Float](repeating: -Float.greatestFiniteMagnitude, count: numAnchors)
            var maxClassIndices = [Int](repeating: 0, count: numAnchors)
            var anchorDone = [Bool](repeating: false, count: numAnchors)
            var remainingAnchors = numAnchors
            let earlyStopThreshold: Float = 0.9

            for cls in 0..<numClasses {
                if remainingAnchors == 0 {
                    break
                }
                let rowPtr = detBuf.advanced(by: (4 + cls) * stride)
                maxScores.withUnsafeMutableBufferPointer { scores in
                    maxClassIndices.withUnsafeMutableBufferPointer { indices in
                        guard let sBase = scores.baseAddress,
                              let iBase = indices.baseAddress else { return }
                        var a = 0
                        // Process 4 anchors at a time to help the compiler generate SIMD.
                        let limit = numAnchors - (numAnchors % 4)
                        while a < limit {
                            if !anchorDone[a] {
                                let v0 = rowPtr[a]
                                if v0 > sBase[a] {
                                    sBase[a] = v0
                                    iBase[a] = cls
                                    if v0 >= earlyStopThreshold && !anchorDone[a] {
                                        anchorDone[a] = true
                                        remainingAnchors -= 1
                                    }
                                }
                            }
                            if !anchorDone[a + 1] {
                                let v1 = rowPtr[a + 1]
                                if v1 > sBase[a + 1] {
                                    sBase[a + 1] = v1
                                    iBase[a + 1] = cls
                                    if v1 >= earlyStopThreshold && !anchorDone[a + 1] {
                                        anchorDone[a + 1] = true
                                        remainingAnchors -= 1
                                    }
                                }
                            }
                            if !anchorDone[a + 2] {
                                let v2 = rowPtr[a + 2]
                                if v2 > sBase[a + 2] {
                                    sBase[a + 2] = v2
                                    iBase[a + 2] = cls
                                    if v2 >= earlyStopThreshold && !anchorDone[a + 2] {
                                        anchorDone[a + 2] = true
                                        remainingAnchors -= 1
                                    }
                                }
                            }
                            if !anchorDone[a + 3] {
                                let v3 = rowPtr[a + 3]
                                if v3 > sBase[a + 3] {
                                    sBase[a + 3] = v3
                                    iBase[a + 3] = cls
                                    if v3 >= earlyStopThreshold && !anchorDone[a + 3] {
                                        anchorDone[a + 3] = true
                                        remainingAnchors -= 1
                                    }
                                }
                            }
                            a += 4
                        }
                        while a < numAnchors {
                            if !anchorDone[a] {
                                let v = rowPtr[a]
                                if v > sBase[a] {
                                    sBase[a] = v
                                    iBase[a] = cls
                                    if v >= earlyStopThreshold && !anchorDone[a] {
                                        anchorDone[a] = true
                                        remainingAnchors -= 1
                                    }
                                }
                            }
                            a += 1
                        }
                    }
                }
            }

            // Now emit detections using the pre-computed max scores / indices.
            for anchor in 0..<numAnchors {
                let maxVal = maxScores[anchor]
                guard maxVal.isFinite, maxVal >= confidenceThreshold else { continue }

                let classIdx = maxClassIndices[anchor]
                guard classIdx >= 0, !classBlacklist.contains(classIdx) else { continue }

                let x = detBuf[0 * stride + anchor]
                let y = detBuf[1 * stride + anchor]
                let w = detBuf[2 * stride + anchor]
                let h = detBuf[3 * stride + anchor]

                guard x.isFinite, y.isFinite, w.isFinite, h.isFinite, w > 0, h > 0 else { continue }

                // Gather 32 mask coefficients (strided → contiguous) via BLAS
                var coeffs = [Float](repeating: 0, count: 32)
                let coeffBase = detBuf.advanced(by: coeffOffset * stride + anchor)
                yolo_blas_scopy(32, coeffBase, BLASInt(stride), &coeffs, 1)

                allDets.append(
                    FurnitureFitDetection(
                        x: x,
                        y: y,
                        w: w,
                        h: h,
                        confidence: maxVal,
                        classIdx: classIdx,
                        coeffs: coeffs
                    )
                )
            }
        }

        return allDets
    }

    static func parseDetections(
        detArray: MLMultiArray,
        confidenceThreshold: Float,
        classBlacklist: Set<Int>
    ) -> [FurnitureFitDetection] {
        let totalCount = detArray.count

        switch detArray.dataType {
        case .float32:
            let detBuf = detArray.dataPointer.bindMemory(to: Float.self, capacity: totalCount)
            return parseDetectionsFromBuffer(
                detBuf: detBuf,
                detArray: detArray,
                totalCount: totalCount,
                confidenceThreshold: confidenceThreshold,
                classBlacklist: classBlacklist
            )

        case .float16:
            // Grow scratch buffer if needed (never shrinks — avoids repeated alloc)
            if f16ScratchBuffer.count < totalCount {
                f16ScratchBuffer = [Float](repeating: 0, count: totalCount)
            }
            let src16 = detArray.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)

            if isContiguous(detArray) {
                // ✅ Fast path: single bulk Accelerate conversion (~0.15ms for 1.2M elements on A14)
                f16ScratchBuffer.withUnsafeMutableBufferPointer { dst in
                    guard let dstBase = dst.baseAddress else { return }
                    bulkConvertF16ToF32(src: src16, dst: dstBase, count: totalCount)
                }
            } else {
                // Non-contiguous: convert per-row (inner dim contiguous, outer dims strided)
                // For [1, D1, D2] with strides [s0, s1, s2] where s2 == 1
                let dims = detArray.shape.map { $0.intValue }
                let strides = detArray.strides.map { $0.intValue }
                let innerStride = strides.last ?? 1

                if innerStride == 1, dims.count == 3 {
                    let d1 = dims[1], d2 = dims[2]
                    let s0 = strides[0]
                    let s1 = strides[1]

                    f16ScratchBuffer.withUnsafeMutableBufferPointer { dst in
                        guard let dstBase = dst.baseAddress else { return }
                        for i in 0..<dims[0] {
                            for j in 0..<d1 {
                                let srcOffset = i * s0 + j * s1
                                let dstOffset = i * d1 * d2 + j * d2
                                bulkConvertF16ToF32(
                                    src: src16.advanced(by: srcOffset),
                                    dst: dstBase.advanced(by: dstOffset),
                                    count: d2
                                )
                            }
                        }
                    }
                } else {
                    // Fully non-contiguous fallback (extremely rare for CoreML outputs)
                    let src16fp = detArray.dataPointer.bindMemory(to: Float16.self, capacity: totalCount)
                    f16ScratchBuffer.withUnsafeMutableBufferPointer { dst in
                        guard let base = dst.baseAddress else { return }
                        for i in 0..<totalCount {
                            base[i] = Float(src16fp[i])
                        }
                    }
                }
            }

            var result: [FurnitureFitDetection] = []
            f16ScratchBuffer.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                result = parseDetectionsFromBuffer(
                    detBuf: base,
                    detArray: detArray,
                    totalCount: totalCount,
                    confidenceThreshold: confidenceThreshold,
                    classBlacklist: classBlacklist
                )
            }
            return result

        default:
            return []
        }
    }

    static func extractDetectionAndProto(
        from output: MLFeatureProvider
    ) -> (det: MLMultiArray, proto: MLMultiArray, detName: String, protoName: String)? {
        for pair in knownDetectionProtoPairs {
            if let det = output.featureValue(for: pair.det)?.multiArrayValue,
               let proto = output.featureValue(for: pair.proto)?.multiArrayValue {
                return (det, proto, pair.det, pair.proto)
            }
        }

        var detectionCandidate: (name: String, array: MLMultiArray)?
        var protoCandidate: (name: String, array: MLMultiArray)?

        for featureName in output.featureNames.sorted() {
            guard let array = output.featureValue(for: featureName)?.multiArrayValue else { continue }
            let shape = array.shape.map { $0.intValue }
            if shape.count == 4, shape[1] == 32, protoCandidate == nil {
                protoCandidate = (featureName, array)
            } else if shape.count == 3, detectionCandidate == nil {
                detectionCandidate = (featureName, array)
            }
        }

        if let det = detectionCandidate, let proto = protoCandidate {
            return (det.array, proto.array, det.name, proto.name)
        }

        return nil
    }
}
