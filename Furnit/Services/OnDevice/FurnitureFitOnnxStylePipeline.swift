import Accelerate
import Foundation

/// Android `FurnitureFitManager` ONNX path parity: NMS → primary scoring → supporting-table heuristic → bbox-limited proto logit mask.
enum FurnitureFitOnnxStylePipeline {
    // MARK: Mask thresholds (two different stages)
    //
    // 1. `maskLogitThreshold` — proto resolution only. Used when building the
    //    UInt8 proto binary from `buildBboxLimitedLogitMask*` (diagnostics, maskOn
    //    counts, debug ASCII). Does **not** define the final cutout alpha.
    //
    // 2. Composite gate after bilinear upsample to camera pixels: opaque iff
    //    `upsampled_logit > nativeCompositeUpsampleLogitThreshold()`.
    //    When ``retinaMasksCompositeEnabled`` is `true`, this matches the spirit of Ultralytics
    //    `predict(..., retina_masks=True)`: evaluate the mask at full image resolution (we already
    //    bilinear-upsample the fused logit field from proto to buffer) and use a **logit** gate
    //    like `masks > 0` (sigmoid midpoint), not an extra stricter constant.
    //    With `false`, use ``nativeMaskUpsampleLogitThreshold`` (legacy tighter gate, e.g. 0.25).
    //
    // NOTE: Proto-resolution binary **area** and upsampled binary **area** differ
    // in general: bilinear blends four proto neighbors, producing sub-cell gradients
    // near zero crossings; small positive interpolated values inflate the mask vs
    // per-cell `> 0` on the 160×160 grid.

    /// Proto-only binary mask gate for `buildBboxLimitedLogitMask*` output.
    /// Lowered from 0.0: the _seg_o2m Core ML export (un-fused BN) produces
    /// slightly negative logits (-0.5 to -2) at thin structures (handles, slats)
    /// that the model clearly distinguishes from background (-8 to -14).
    static let maskLogitThreshold: Float = -1.5

    /// When `true`, composite uses ``retinaMaskUpsampleLogitThreshold`` (default `0.0`) so thin
    /// parts are not punched out by a stricter gate. Set `false` to use ``nativeMaskUpsampleLogitThreshold``.
    static let retinaMasksCompositeEnabled: Bool = true

    /// Logit gate after full-res bilinear sample when ``retinaMasksCompositeEnabled`` is `true`.
    /// Lowered from 0.0 to match ``maskLogitThreshold``: captures thin structures
    /// whose logits are slightly negative after bilinear upsample.
    static let retinaMaskUpsampleLogitThreshold: Float = 0.0

    /// Legacy composite gate when ``retinaMasksCompositeEnabled`` is `false`.
    /// `process_mask_native` / GPU bilinear path: opaque iff `upsampled_logit > threshold`.
    /// Lower (~0.2–0.35) restores thin regions that vanish at 0.5; higher reduces bilinear bleed.
    static let nativeMaskUpsampleLogitThreshold: Float = 0.0

    /// Effective post-bilinear logit threshold for GPU + CPU `process_mask_native` compositing.
    static func nativeCompositeUpsampleLogitThreshold() -> Float {
        let base = retinaMasksCompositeEnabled
            ? retinaMaskUpsampleLogitThreshold
            : nativeMaskUpsampleLogitThreshold
        return base + nativeMaskBoundarySharpenExtra
    }

    /// Added to the chosen base threshold when binarizing (GPU and CPU native paths).
    /// Increase slightly if physical see-through gaps (e.g. between legs) look too noisy; keeps
    /// true holes more transparent by requiring a higher logit.
    static let nativeMaskBoundarySharpenExtra: Float = 0.0

    /// Expands the composite ``crop_mask`` band in pixels so bbox edges do not clip foreground.
    static let nativeCompositeBandMarginPx: Int = 1

    // MARK: - Vertical flip (Ultralytics `Instances.flipud` parity)

    /// Row-major `protoW`×`protoH` grid: swaps rows `py` ↔ `protoH - 1 - py`.
    /// Use after mask synthesis when the model input was vertically flipped relative to the camera buffer.
    static func flipProtoFloatGridVertically(_ buffer: inout [Float], protoW: Int, protoH: Int) {
        guard protoW > 0, protoH > 1, buffer.count >= protoW * protoH else { return }
        for py in 0..<(protoH / 2) {
            let rowA = py * protoW
            let rowB = (protoH - 1 - py) * protoW
            for px in 0..<protoW {
                buffer.swapAt(rowA + px, rowB + px)
            }
        }
    }

    /// Same layout as ``flipProtoFloatGridVertically`` for UInt8 proto binaries.
    static func flipProtoUInt8GridVertically(_ buffer: inout [UInt8], protoW: Int, protoH: Int) {
        guard protoW > 0, protoH > 1, buffer.count >= protoW * protoH else { return }
        for py in 0..<(protoH / 2) {
            let rowA = py * protoW
            let rowB = (protoH - 1 - py) * protoW
            for px in 0..<protoW {
                buffer.swapAt(rowA + px, rowB + px)
            }
        }
    }

    /// Bilinear sample of proto `maskLogits` at full-image pixel `(imageX, imageY)`.
    /// Matches PyTorch `F.interpolate(..., mode="bilinear", align_corners=False)`.
    static func bilinearUpsampledLogit(
        maskLogits: [Float],
        protoW: Int,
        protoH: Int,
        origW: Int,
        origH: Int,
        imageX: Int,
        imageY: Int
    ) -> Float {
        guard protoW > 0,
              protoH > 0,
              origW > 0,
              origH > 0,
              maskLogits.count >= protoW * protoH else { return 0 }

        let protoScaleX = Float(protoW) / Float(origW)
        let protoScaleY = Float(protoH) / Float(origH)
        let maxPx = protoW - 1
        let maxPy = protoH - 1

        let fy = (Float(imageY) + 0.5) * protoScaleY - 0.5
        let py0 = max(0, min(maxPy, Int(floor(fy))))
        let py1 = max(0, min(maxPy, py0 + 1))
        let ty = fy - Float(py0)

        let fx = (Float(imageX) + 0.5) * protoScaleX - 0.5
        let px0 = max(0, min(maxPx, Int(floor(fx))))
        let px1 = max(0, min(maxPx, px0 + 1))
        let tx = fx - Float(px0)

        let v00 = maskLogits[py0 * protoW + px0]
        let v10 = maskLogits[py0 * protoW + px1]
        let v01 = maskLogits[py1 * protoW + px0]
        let v11 = maskLogits[py1 * protoW + px1]
        return
            v00 * (1 - tx) * (1 - ty) +
            v10 * tx * (1 - ty) +
            v01 * (1 - tx) * ty +
            v11 * tx * ty
    }

    /// Android `collectMaskDetections` (returns list for mask fusion, primary first).
    /// Merge any other detection whose bbox center lies inside the primary bbox.
    static func collectMaskDetections(
        primaryIndex: Int,
        detections: [FurnitureFitDetection]
    ) -> [FurnitureFitDetection] {
        guard primaryIndex >= 0, primaryIndex < detections.count else { return [] }
        let primaryDetection = detections[primaryIndex]

        let primaryLeft = primaryDetection.x - primaryDetection.w * 0.5
        let primaryTop = primaryDetection.y - primaryDetection.h * 0.5
        let primaryRight = primaryDetection.x + primaryDetection.w * 0.5
        let primaryBottom = primaryDetection.y + primaryDetection.h * 0.5

        var maskDetections: [FurnitureFitDetection] = [primaryDetection]
        for (idx, detection) in detections.enumerated() {
            guard idx != primaryIndex else { continue }
            guard detection.coeffs.count >= 32 else { continue }
            let centerInsidePrimary =
                detection.x >= primaryLeft &&
                detection.x <= primaryRight &&
                detection.y >= primaryTop &&
                detection.y <= primaryBottom
            guard centerInsidePrimary else { continue }
            maskDetections.append(detection)
        }
        return maskDetections
    }

    /// Same as `buildBboxLimitedLogitMask` but also returns the raw float
    /// logits before thresholding, for debug visualization.
    static func buildBboxLimitedLogitMaskWithLogits(
        planes: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Float,
        detections: [FurnitureFitDetection]
    ) -> (binary: [UInt8], logits: [Float]) {
        let hwProto = protoW * protoH
        guard planes.count >= 32 * hwProto else {
            return ([UInt8](repeating: 0, count: hwProto),
                    [Float](repeating: 0, count: hwProto))
        }

        var maskProto = [Float](repeating: 0, count: hwProto)
        var protoPixelTouched = [UInt8](repeating: 0, count: hwProto)
        // Ultralytics-style process_mask: scale boxes from model space
        // (e.g. 640×640) into proto grid (e.g. 160×160) using simple ratios.
        // widthRatio = protoW / modelW, heightRatio = protoH / modelH.
        let widthRatio = Float(protoW) / modelSide
        let heightRatio = Float(protoH) / modelSide
        // Edge bias disabled.
        let edgeBias: Float = 0.0
        let maxX = Float(protoW - 1)
        let maxY = Float(protoH - 1)

        // Optimization: dot product only within each detection’s bbox on the proto grid.
        // Ultralytics does a full 160×160 dot then `crop_mask` after upsample; pixels
        // outside the bbox are zeroed there—so this is equivalent for compositing.

        for detection in detections {
            guard detection.coeffs.count >= 32 else { continue }

            // Map bbox corners from model input size (modelSide×modelSide)
            // into proto grid coordinates (protoW×protoH); clamp defensively so
            // float noise / OOB boxes cannot produce invalid ranges before floor/ceil.
            let x1Proto = max(0, min(maxX, (detection.x - detection.w * 0.5) * widthRatio))
            let y1Proto = max(0, min(maxY, (detection.y - detection.h * 0.5) * heightRatio))
            let x2Proto = max(0, min(maxX, (detection.x + detection.w * 0.5) * widthRatio))
            let y2Proto = max(0, min(maxY, (detection.y + detection.h * 0.5) * heightRatio))

            let bboxLeft = Int(floor(x1Proto - edgeBias)).clamped(to: 0...(protoW - 1))
            let bboxTop = Int(floor(y1Proto - edgeBias)).clamped(to: 0...(protoH - 1))
            let bboxRight = Int(ceil(x2Proto + edgeBias)).clamped(to: 0...(protoW - 1))
            let bboxBottom = Int(ceil(y2Proto + edgeBias)).clamped(to: 0...(protoH - 1))

            for py in bboxTop...bboxBottom {
                let rowBase = py * protoW
                for px in bboxLeft...bboxRight {
                    let protoPixelIndex = rowBase + px
                    var sum: Float = 0
                    var coeffIndex = 0
                    while coeffIndex < 32 {
                        let protoIdx = coeffIndex * hwProto + protoPixelIndex
                        sum += detection.coeffs[coeffIndex] * planes[protoIdx]
                        coeffIndex += 1
                    }
                    if protoPixelTouched[protoPixelIndex] == 0 {
                        maskProto[protoPixelIndex] = sum
                        protoPixelTouched[protoPixelIndex] = 1
                    } else if sum > maskProto[protoPixelIndex] {
                        maskProto[protoPixelIndex] = sum
                    }
                }
            }
        }

        let threshold = maskLogitThreshold
        let binary = maskProto.map { $0 > threshold ? UInt8(255) : UInt8(0) }
        return (binary, maskProto)
    }

    // MARK: - Full-field logit mask (process_mask_native)

    /// Ultralytics `process_mask_native`: coeffs @ protos for ALL 160×160 pixels.
    /// Unlike `buildBboxLimitedLogitMask*`, this does **not** crop to the detection
    /// bbox in proto space — every pixel gets a real logit value, so bilinear
    /// upsampling at bbox edges uses actual negative values instead of artificial
    /// zeros.  For a single detection, this is 160×160×32 ≈ 820 K multiply-adds.
    static func buildFullFieldLogitMask(
        planes: [Float],
        protoW: Int,
        protoH: Int,
        detections: [FurnitureFitDetection]
    ) -> (binary: [UInt8], logits: [Float]) {
        let hwProto = protoW * protoH
        guard planes.count >= 32 * hwProto else {
            return ([UInt8](repeating: 0, count: hwProto),
                    [Float](repeating: 0, count: hwProto))
        }

        let validDetections = detections.filter { $0.coeffs.count >= 32 }
        guard !validDetections.isEmpty else {
            return ([UInt8](repeating: 0, count: hwProto),
                    [Float](repeating: 0, count: hwProto))
        }

        var prototypeMatrixPixelMajor = [Float](repeating: 0, count: hwProto * 32)
        var zero: Float = 0
        prototypeMatrixPixelMajor.withUnsafeMutableBufferPointer { destinationPointer in
            planes.withUnsafeBufferPointer { sourcePointer in
                guard let destinationBase = destinationPointer.baseAddress,
                      let sourceBase = sourcePointer.baseAddress else { return }
                for channelIndex in 0..<32 {
                    let sourceStart = sourceBase.advanced(by: channelIndex * hwProto)
                    let destinationStart = destinationBase.advanced(by: channelIndex)
                    vDSP_vsadd(sourceStart, 1, &zero, destinationStart, 32, vDSP_Length(hwProto))
                }
            }
        }

        var maximumLogits = [Float](repeating: -Float.greatestFiniteMagnitude, count: hwProto)
        let batchSize = 64
        let matrixHeight = vDSP_Length(hwProto)
        let matrixDepth = vDSP_Length(32)
        var batchStart = 0

        while batchStart < validDetections.count {
            let batchEnd = min(validDetections.count, batchStart + batchSize)
            let batchCount = batchEnd - batchStart
            var coefficientMatrix = [Float](repeating: 0, count: 32 * batchCount)

            for detectionOffset in 0..<batchCount {
                let coeffs = validDetections[batchStart + detectionOffset].coeffs
                for channelIndex in 0..<32 {
                    coefficientMatrix[channelIndex * batchCount + detectionOffset] = coeffs[channelIndex]
                }
            }

            var logitsBatch = [Float](repeating: 0, count: hwProto * batchCount)
            prototypeMatrixPixelMajor.withUnsafeBufferPointer { prototypePointer in
                coefficientMatrix.withUnsafeBufferPointer { coefficientPointer in
                    logitsBatch.withUnsafeMutableBufferPointer { logitsPointer in
                        guard let prototypeBase = prototypePointer.baseAddress,
                              let coefficientBase = coefficientPointer.baseAddress,
                              let logitsBase = logitsPointer.baseAddress else { return }
                        vDSP_mmul(
                            prototypeBase,
                            1,
                            coefficientBase,
                            1,
                            logitsBase,
                            1,
                            matrixHeight,
                            vDSP_Length(batchCount),
                            matrixDepth
                        )
                    }
                }
            }

            logitsBatch.withUnsafeBufferPointer { logitsPointer in
                maximumLogits.withUnsafeMutableBufferPointer { maximumPointer in
                    guard let logitsBase = logitsPointer.baseAddress,
                          let maximumBase = maximumPointer.baseAddress else { return }
                    for protoPixelIndex in 0..<hwProto {
                        var rowMaximum: Float = 0
                        vDSP_maxv(
                            logitsBase.advanced(by: protoPixelIndex * batchCount),
                            1,
                            &rowMaximum,
                            vDSP_Length(batchCount)
                        )
                        if rowMaximum > maximumBase[protoPixelIndex] {
                            maximumBase[protoPixelIndex] = rowMaximum
                        }
                    }
                }
            }

            batchStart = batchEnd
        }

        let threshold = maskLogitThreshold
        let binary = maximumLogits.map { $0 > threshold ? UInt8(255) : UInt8(0) }
        return (binary, maximumLogits)
    }

    // MARK: - ASCII mask visualization for debug logs

    /// Samples the proto-resolution mask down to a text grid and returns
    /// multi-line strings you can print to the console.
    ///
    /// - `logits`: raw float logits (protoW × protoH, row-major)
    /// - `binary`: thresholded UInt8 mask (same layout)
    /// - `protoW`, `protoH`: proto dimensions
    /// - `gridCols`, `gridRows`: target ASCII grid size (default 64×32)
    ///
    /// Returns two strings: (logitArt, binaryArt).
    /// Logit art uses ` ░▒▓█` to show intensity; binary art uses `·` / `█`.
    static func asciiMaskVisualization(
        logits: [Float],
        binary: [UInt8],
        protoW: Int,
        protoH: Int,
        gridCols: Int = 64,
        gridRows: Int = 32
    ) -> (logitArt: String, binaryArt: String) {
        guard protoW > 0, protoH > 0,
              logits.count >= protoW * protoH,
              binary.count >= protoW * protoH else {
            return ("(empty)", "(empty)")
        }

        let cols = min(gridCols, protoW)
        let rows = min(gridRows, protoH)

        var logitMin: Float = .greatestFiniteMagnitude
        var logitMax: Float = -.greatestFiniteMagnitude
        for v in logits {
            if v < logitMin { logitMin = v }
            if v > logitMax { logitMax = v }
        }
        let logitRange = logitMax - logitMin
        let logitChars: [Character] = [" ", "░", "▒", "▓", "█"]

        var logitLines = [String]()
        var binaryLines = [String]()

        for row in 0..<rows {
            let srcY = row * protoH / rows
            var logitRow = ""
            var binaryRow = ""
            for col in 0..<cols {
                let srcX = col * protoW / cols
                let idx = srcY * protoW + srcX

                let logitVal = logits[idx]
                if logitRange > 1e-6 {
                    let norm = (logitVal - logitMin) / logitRange
                    let ci = min(logitChars.count - 1, Int(norm * Float(logitChars.count)))
                    logitRow.append(logitChars[ci])
                } else {
                    logitRow.append(logitVal > 0 ? "█" : " ")
                }

                binaryRow.append(binary[idx] > 0 ? "█" : "·")
            }
            logitLines.append(logitRow)
            binaryLines.append(binaryRow)
        }

        let logitStats = String(format: "min=%.3f max=%.3f range=%.3f",
                                logitMin, logitMax, logitRange)
        let onCount = binary.prefix(protoW * protoH).filter { $0 > 0 }.count
        let totalPx = protoW * protoH
        let binaryStats = String(format: "on=%d/%d (%.1f%%)",
                                 onCount, totalPx,
                                 Float(onCount) / Float(max(1, totalPx)) * 100)

        let logitArt = "LOGIT HEATMAP (\(protoW)x\(protoH) → \(cols)x\(rows)) \(logitStats)\n"
            + logitLines.joined(separator: "\n")
        let binaryArt = "BINARY MASK (\(protoW)x\(protoH) → \(cols)x\(rows)) \(binaryStats)\n"
            + binaryLines.joined(separator: "\n")

        return (logitArt, binaryArt)
    }
}

private extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

private extension BinaryInteger {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
