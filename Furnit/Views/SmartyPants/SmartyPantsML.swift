// SmartyPantsML.swift
// ML model helpers: detection extraction, pixel buffer conversion, prototype buffer

import CoreML
import Accelerate

extension SmartyPantsContainerView {
    
    // MARK: - Pixel Buffer to MLMultiArray
    func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        let t0 = Date()
        let modelSizeNum = NSNumber(value: kModelInputSize)
        guard let array = try? MLMultiArray(shape: [1, 3, modelSizeNum, modelSizeNum], dataType: .float32) else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = kModelInputSize
        let height = kModelInputSize
        let pixelCount = width * height

        // Prepare destination plane pointers in MLMultiArray (Float32)
        let floatSize = MemoryLayout<Float32>.size
        let planeStrideBytes = pixelCount * floatSize
        let rPtr = array.dataPointer.advanced(by: 0 * planeStrideBytes).assumingMemoryBound(to: Float32.self)
        let gPtr = array.dataPointer.advanced(by: 1 * planeStrideBytes).assumingMemoryBound(to: Float32.self)
        let bPtr = array.dataPointer.advanced(by: 2 * planeStrideBytes).assumingMemoryBound(to: Float32.self)

        // vImage source buffer (BGRA8888)
        var srcBuf = vImage_Buffer(data: baseAddress,
                                   height: vImagePixelCount(height),
                                   width: vImagePixelCount(width),
                                   rowBytes: bytesPerRow)

        // Convert BGRA8888 -> ARGB8888, then split into planar 8-bit A,R,G,B
        var argbData = [UInt8](repeating: 0, count: pixelCount * 4)
        var argbBuf = vImage_Buffer(data: &argbData, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width * 4)

        // Permute BGRA (B,G,R,A) to ARGB (A,R,G,B)
        var permute: [UInt8] = [3, 2, 1, 0]
        vImagePermuteChannels_ARGB8888(&srcBuf, &argbBuf, &permute, vImage_Flags(kvImageNoFlags))

        // Temporary planar 8-bit buffers for B, G, R, A
        var b8 = [UInt8](repeating: 0, count: pixelCount)
        var g8 = [UInt8](repeating: 0, count: pixelCount)
        var r8 = [UInt8](repeating: 0, count: pixelCount)
        var a8 = [UInt8](repeating: 0, count: pixelCount)

        b8.withUnsafeMutableBytes { bBytes in
            g8.withUnsafeMutableBytes { gBytes in
                r8.withUnsafeMutableBytes { rBytes in
                    a8.withUnsafeMutableBytes { aBytes in
                        var aBuf = vImage_Buffer(data: aBytes.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                        var rBuf = vImage_Buffer(data: rBytes.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                        var gBuf = vImage_Buffer(data: gBytes.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                        var bBuf = vImage_Buffer(data: bBytes.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)

                        vImageConvert_ARGB8888toPlanar8(&argbBuf, &aBuf, &rBuf, &gBuf, &bBuf, vImage_Flags(kvImageNoFlags))
                    }
                }
            }
        }

        let convertStart = Date()
        // Convert Planar8 -> PlanarF directly into MLMultiArray planes and normalize to [0,1]
        var scaleF: Float = 1.0 / 255.0
        b8.withUnsafeBufferPointer { bU8 in
            vDSP_vfltu8(bU8.baseAddress!, 1, bPtr, 1, vDSP_Length(pixelCount))
            vDSP_vsmul(bPtr, 1, &scaleF, bPtr, 1, vDSP_Length(pixelCount))
        }
        g8.withUnsafeBufferPointer { gU8 in
            vDSP_vfltu8(gU8.baseAddress!, 1, gPtr, 1, vDSP_Length(pixelCount))
            vDSP_vsmul(gPtr, 1, &scaleF, gPtr, 1, vDSP_Length(pixelCount))
        }
        r8.withUnsafeBufferPointer { rU8 in
            vDSP_vfltu8(rU8.baseAddress!, 1, rPtr, 1, vDSP_Length(pixelCount))
            vDSP_vsmul(rPtr, 1, &scaleF, rPtr, 1, vDSP_Length(pixelCount))
        }

        if debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            let conv = Date().timeIntervalSince(convertStart) * 1000.0
            print(String(format: "⏱ pixelBufferToMLMultiArray: total %.2f ms (convert %.2f ms)", dt, conv))
        }

        return array
    }

    // MARK: - Prototype Buffer (subscript access for correct strides)
    func makePrototypeBuffer(from array: MLMultiArray, C: Int, Hp: Int, Wp: Int) -> [Float] {
        let t0 = Date()
        let spatial = Hp * Wp
        let count = C * spatial
        var out = [Float](repeating: 0, count: count)

        // Fast path if data is Float32 and contiguous in [1, C, H, W] layout
        let strides = array.strides.map { $0.intValue }
        let isContiguous = array.dataType == .float32 &&
                           strides.count >= 4 &&
                           strides[3] == 1 &&            // W stride
                           strides[2] == Wp &&           // H stride
                           strides[1] == Hp * Wp         // C stride
        if isContiguous {
            let baseF = array.dataPointer.assumingMemoryBound(to: Float.self)
            for c in 0..<C {
                let src = baseF.advanced(by: c * spatial)
                out.withUnsafeMutableBufferPointer { dst in
                    let d = dst.baseAddress!.advanced(by: c * spatial)
                    memcpy(d, src, spatial * MemoryLayout<Float>.size)
                }
            }
            if debugMode {
                let dt = Date().timeIntervalSince(t0) * 1000.0
                print(String(format: "⏱ makePrototypeBuffer (contiguous memcpy): %.2f ms", dt))
            }
            return out
        }

        // General path: honor MLMultiArray strides. If not Float32, convert entire buffer once.
        let totalCount = array.count
        var tempFloatBuf: UnsafeMutablePointer<Float>? = nil
        defer { tempFloatBuf?.deallocate() }

        let readPtrF: UnsafePointer<Float>
        if array.dataType == .float32 {
            let base = array.dataPointer.assumingMemoryBound(to: Float.self)
            readPtrF = UnsafePointer<Float>(base)
        } else if array.dataType == .float16 {
            // Convert entire buffer to Float once
            let tmp = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
            tempFloatBuf = tmp
            let src = array.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(totalCount), rowBytes: totalCount * MemoryLayout<UInt16>.size)
            var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(tmp), height: 1, width: vImagePixelCount(totalCount), rowBytes: totalCount * MemoryLayout<Float>.size)
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
            readPtrF = UnsafePointer<Float>(tmp)
        } else {
            // Fallback: use subscript (slow but correct) — should rarely happen
            for c in 0..<C {
                for y in 0..<Hp {
                    for x in 0..<Wp {
                        let srcIdx = [0, c, y, x] as [NSNumber]
                        let dstIdx = c * spatial + y * Wp + x
                        out[dstIdx] = array[srcIdx].floatValue
                    }
                }
            }
            if debugMode {
                let dt = Date().timeIntervalSince(t0) * 1000.0
                print(String(format: "⏱ makePrototypeBuffer (fallback subscript): %.2f ms", dt))
            }
            return out
        }

        // Compute offsets using strides in elements
        let s0 = strides.count > 0 ? strides[0] : 0
        let s1 = strides.count > 1 ? strides[1] : 0
        let s2 = strides.count > 2 ? strides[2] : 0
        let s3 = strides.count > 3 ? strides[3] : 0
        precondition(s3 > 0 && s2 > 0 && s1 > 0, "Invalid MLMultiArray strides")

        for c in 0..<C {
            for y in 0..<Hp {
                let dstRowBase = c * spatial + y * Wp
                let srcBaseOffset = 0 * s0 + c * s1 + y * s2
                for x in 0..<Wp {
                    let srcOffset = srcBaseOffset + x * s3
                    out[dstRowBase + x] = readPtrF[srcOffset]
                }
            }
        }

        if debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ makePrototypeBuffer (stride read): %.2f ms", dt))
        }

        return out
    }

    // MARK: - Extract Detections from MLMultiArray
    func extractDetections(from detections: MLMultiArray) -> [DetectionSmarty] {
        let t0 = Date()
        var all: [DetectionSmarty] = []

        guard detections.shape.count == 3 else {
            if debugMode { print("⚠️ extractDetections: Unexpected tensor rank: \(detections.shape)") }
            return []
        }

        let numFeatures = detections.shape[1].intValue
        let numAnchors  = detections.shape[2].intValue
        let numClasses  = numFeatures - 4 - 32

        guard numFeatures >= 36, numAnchors > 0, numClasses > 0 else {
            if debugMode {
                print("⚠️ extractDetections: Invalid tensor dims — features:\(numFeatures) anchors:\(numAnchors) classes:\(numClasses)")
            }
            return []
        }

        let expectedCount = numFeatures * numAnchors
        let totalCount = detections.count
        guard totalCount >= expectedCount else {
            if debugMode { print("⚠️ extractDetections: count mismatch expected=\(expectedCount) got=\(totalCount)") }
            return []
        }

        // Copy/convert to Float buffer
        let detBuf = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
        defer { detBuf.deallocate() }

        let copyStart = Date()
        switch detections.dataType {
        case .float32:
            let src = detections.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(detBuf, src, totalCount * MemoryLayout<Float>.size)
        case .float16:
            let src = detections.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src),
                                       height: 1, width: vImagePixelCount(totalCount),
                                       rowBytes: totalCount * MemoryLayout<UInt16>.size)
            var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(detBuf),
                                       height: 1, width: vImagePixelCount(totalCount),
                                       rowBytes: totalCount * MemoryLayout<Float>.size)
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
        default:
            for i in 0..<totalCount { detBuf[i] = detections[i].floatValue }
        }
        
        if debugMode {
            let copyEnd = Date()
            print(String(format: "⏱ extractDetections copy/convert: %.2f ms", copyEnd.timeIntervalSince(copyStart) * 1000.0))
        }

        let stride = numAnchors
        let coeffOffset = 4 + numClasses

        if debugMode {
            print("📝 Tensor shape: [1, \(numFeatures), \(numAnchors)]")
            print("   → \(numClasses) classes, \(numAnchors) predictions")
            print("   → Mode: CLASS-AGNOSTIC")
            print("   → Using vDSP-optimized max and gather for decode")
        }

        let decodeStart = Date()
        var decodedCount = 0
        for anchor in 0..<numAnchors {
            guard anchor < stride,
                  1 * stride + anchor < totalCount,
                  2 * stride + anchor < totalCount,
                  3 * stride + anchor < totalCount else { continue }

            let x = detBuf[0 * stride + anchor]
            let y = detBuf[1 * stride + anchor]
            let w = detBuf[2 * stride + anchor]
            let h = detBuf[3 * stride + anchor]
            guard x.isFinite, y.isFinite, w.isFinite, h.isFinite, w > 0, h > 0 else { continue }

            // Class-agnostic: take max over class scores (vDSP-optimized)
            let baseConfIdx = 4 * stride + anchor
            var maxVal: Float = 0
            // vDSP_maxv supports strided access; compute max across `numClasses` values starting at baseConfIdx with stride `stride`.
            vDSP_maxv(detBuf.advanced(by: baseConfIdx), vDSP_Stride(stride), &maxVal, vDSP_Length(numClasses))
            // Preserve previous behavior where negative class scores were effectively ignored by starting from 0
            let bestConf = max(0, maxVal)
            guard bestConf > confidenceThreshold else { continue }

            // Mask coefficients (32) via vDSP gather
            var coeffs = [Float](repeating: 0, count: 32)
            let coeffStartIdx = coeffOffset * stride + anchor
            // Ensure bounds for the last gathered index
            let lastCoeffIdx = coeffStartIdx + (32 - 1) * stride
            guard lastCoeffIdx < totalCount else { continue }
            var indices = [vDSP_Length](repeating: 0, count: 32)
            for k in 0..<32 { indices[k] = vDSP_Length(coeffStartIdx + k * stride) }
            coeffs.withUnsafeMutableBufferPointer { dst in
                indices.withUnsafeBufferPointer { idxs in
                    vDSP_vgathr(detBuf, idxs.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(32))
                }
            }

            decodedCount += 1
            all.append(DetectionSmarty(
                x: x, y: y, width: w, height: h,
                confidence: bestConf, classIdx: -1,
                className: "object", maskCoeffs: coeffs
            ))
        }
        
        if debugMode {
            let decodeEnd = Date()
            print(String(format: "⏱ extractDetections decode loop: %.2f ms, decoded: %d", decodeEnd.timeIntervalSince(decodeStart) * 1000.0, decodedCount))
            print("\n📊 DETECTION SUMMARY: \(all.count) total")
            let grouped = Dictionary(grouping: all) { $0.className }
            for (className, dets) in grouped.sorted(by: { $0.value.count > $1.value.count }).prefix(20) {
                let confidences = dets.map { Int($0.confidence * 100) }
                print("  - \(className): \(dets.count)x, conf: \(confidences)%")
            }
            print(String(format: "⏱ extractDetections total: %.2f ms", Date().timeIntervalSince(t0) * 1000.0))
        }

        return all
    }

    // MARK: - NMS
    func applyNMS(_ detections: [DetectionSmarty], iouThreshold: Float) -> [DetectionSmarty] {
        guard !detections.isEmpty else { return [] }
        guard iouThreshold >= 0 && iouThreshold <= 1 else { return detections }
        
        let validDetections = detections.filter { det in
            det.width > 0 && det.height > 0 &&
            det.width.isFinite && det.height.isFinite &&
            det.x.isFinite && det.y.isFinite &&
            det.confidence >= 0 && det.confidence <= 1
        }
        
        guard !validDetections.isEmpty else { return [] }
        
        let sorted = validDetections.sorted { $0.confidence > $1.confidence }
        var kept: [DetectionSmarty] = []
        kept.reserveCapacity(sorted.count)

        for det in sorted {
            var dominated = false
            for k in kept {
                let iou = bboxIoU(det, k)
                if iou.isFinite && iou > iouThreshold {
                    dominated = true
                    break
                }
            }
            if !dominated { kept.append(det) }
        }
        return kept
    }

    func bboxIoU(_ a: DetectionSmarty, _ b: DetectionSmarty) -> Float {
        guard a.width > 0 && a.height > 0 && b.width > 0 && b.height > 0 else { return 0 }
        guard a.width.isFinite && a.height.isFinite && b.width.isFinite && b.height.isFinite else { return 0 }
        guard a.x.isFinite && a.y.isFinite && b.x.isFinite && b.y.isFinite else { return 0 }
        
        let aLeft = a.x - a.width * 0.5, aRight = a.x + a.width * 0.5
        let aTop = a.y - a.height * 0.5, aBottom = a.y + a.height * 0.5
        let bLeft = b.x - b.width * 0.5, bRight = b.x + b.width * 0.5
        let bTop = b.y - b.height * 0.5, bBottom = b.y + b.height * 0.5

        let ix1 = max(aLeft, bLeft), ix2 = min(aRight, bRight)
        let iy1 = max(aTop, bTop), iy2 = min(aBottom, bBottom)
        
        let iw = max(0, ix2 - ix1), ih = max(0, iy2 - iy1)
        let inter = iw * ih
        
        let areaA = a.width * a.height
        let areaB = b.width * b.height
        let union = areaA + areaB - inter
        
        guard union > 0 && union.isFinite && inter.isFinite else { return 0 }
        let iou = inter / union
        return iou.isFinite ? iou : 0
    }
}

