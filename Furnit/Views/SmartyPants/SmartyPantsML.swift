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
        let src = baseAddress.assumingMemoryBound(to: UInt8.self)

        let floatSize = MemoryLayout<Float32>.size
        let planeStrideBytes = pixelCount * floatSize
        let rPtr = array.dataPointer.advanced(by: 0 * planeStrideBytes).assumingMemoryBound(to: Float32.self)
        let gPtr = array.dataPointer.advanced(by: 1 * planeStrideBytes).assumingMemoryBound(to: Float32.self)
        let bPtr = array.dataPointer.advanced(by: 2 * planeStrideBytes).assumingMemoryBound(to: Float32.self)

        var indicesR = [vDSP_Length](repeating: 0, count: width)
        var indicesG = [vDSP_Length](repeating: 0, count: width)
        var indicesB = [vDSP_Length](repeating: 0, count: width)
        for i in 0..<width {
            indicesR[i] = vDSP_Length(2 + i * 4)
            indicesG[i] = vDSP_Length(1 + i * 4)
            indicesB[i] = vDSP_Length(0 + i * 4)
        }

        var rowUInt8 = [UInt8](repeating: 0, count: width * 4)
        var rowFloat = [Float](repeating: 0, count: width * 4)
        var scaleF: Float = 1.0 / 255.0

        for y in 0..<height {
            let rowStart = src.advanced(by: y * bytesPerRow)
            rowUInt8.withUnsafeMutableBytes { dstBytes in
                memcpy(dstBytes.baseAddress!, rowStart, width * 4)
            }

            rowUInt8.withUnsafeBufferPointer { u8Ptr in
                rowFloat.withUnsafeMutableBufferPointer { fPtr in
                    vDSP_vfltu8(u8Ptr.baseAddress!, 1, fPtr.baseAddress!, 1, vDSP_Length(width * 4))
                    vDSP_vsmul(fPtr.baseAddress!, 1, &scaleF, fPtr.baseAddress!, 1, vDSP_Length(width * 4))
                }
            }

            rowFloat.withUnsafeBufferPointer { rf in
                let baseF = rf.baseAddress!
                vDSP_vgathr(baseF, indicesR, 1, rPtr.advanced(by: y * width), 1, vDSP_Length(width))
                vDSP_vgathr(baseF, indicesG, 1, gPtr.advanced(by: y * width), 1, vDSP_Length(width))
                vDSP_vgathr(baseF, indicesB, 1, bPtr.advanced(by: y * width), 1, vDSP_Length(width))
            }
        }

        if debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ pixelBufferToMLMultiArray: %.2f ms", dt))
        }

        return array
    }

    // MARK: - Prototype Buffer (subscript access for correct strides)
    func makePrototypeBuffer(from array: MLMultiArray, C: Int, Hp: Int, Wp: Int) -> [Float] {
        let t0 = Date()
        let spatial = Hp * Wp
        let count = C * spatial
        var out = [Float](repeating: 0, count: count)
        
        // Use subscript access to handle MLMultiArray strides correctly
        // Layout: out[c * spatial + (y * Wp + x)] = array[0, c, y, x]
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
            print(String(format: "⏱ makePrototypeBuffer (subscript): %.2f ms", dt))
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

            all.append(DetectionSmarty(
                x: x, y: y, width: w, height: h,
                confidence: bestConf, classIdx: -1,
                className: "object", maskCoeffs: coeffs
            ))
        }
        
        if debugMode {
            let decodeEnd = Date()
            print(String(format: "⏱ extractDetections decode loop: %.2f ms", decodeEnd.timeIntervalSince(decodeStart) * 1000.0))
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

