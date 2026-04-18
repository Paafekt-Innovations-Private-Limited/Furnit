import Accelerate
import Foundation

/// Android `FurnitureFitManager` ONNX path parity: NMS → primary scoring → supporting-table heuristic → bbox-limited proto logit mask.
enum FurnitureFitOnnxStylePipeline {

    /// Default detection confidence threshold for ONNX-style helpers.
    /// The live FurnitureFit camera path now uses 0.10 via `FurnitureFitUIView.scoreThreshold`.
    static let confidenceThreshold: Float = 0.10
    static let iouThresholdNms: Float = 0.45
    static let maxDetectionsBeforeNms = 100

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
    //    Legacy `maskUpsampleLogitBias` remains for older docs; the live path also uses optional
    //    morphological close on the composite band.
    //
    // NOTE: Proto-resolution binary **area** and upsampled binary **area** differ
    // in general: bilinear blends four proto neighbors, producing sub-cell gradients
    // near zero crossings; small positive interpolated values inflate the mask vs
    // per-cell `> 0` on the 160×160 grid. `maskUpsampleLogitBias` offsets that bleed.

    /// Proto-only binary mask gate for `buildBboxLimitedLogitMask*` output.
    /// Lowered from 0.0: the _seg_o2m Core ML export (un-fused BN) produces
    /// slightly negative logits (-0.5 to -2) at thin structures (handles, slats)
    /// that the model clearly distinguishes from background (-8 to -14).
    static let maskLogitThreshold: Float = -1.5

    /// Post-bilinear gate for compositing; not the same role as `maskLogitThreshold`.
    /// Tune ~0.06–0.12: lower if thin parts vanish, higher if upscaled mask stays puffier than proto.
    static let maskUpsampleLogitBias: Float = 0.09

    /// When `true`, composite uses ``retinaMaskUpsampleLogitThreshold`` (default `0.0`) so thin
    /// parts are not punched out by a stricter gate. Set `false` to use ``nativeMaskUpsampleLogitThreshold``.
    static let retinaMasksCompositeEnabled: Bool = true

    /// Logit gate after full-res bilinear sample when ``retinaMasksCompositeEnabled`` is `true`.
    /// Lowered from 0.0 to match ``maskLogitThreshold``: captures thin structures
    /// whose logits are slightly negative after bilinear upsample.
    static let retinaMaskUpsampleLogitThreshold: Float = -1.5

    /// Legacy composite gate when ``retinaMasksCompositeEnabled`` is `false`.
    /// `process_mask_native` / GPU bilinear path: opaque iff `upsampled_logit > threshold`.
    /// Lower (~0.2–0.35) restores thin regions that vanish at 0.5; higher reduces bilinear bleed.
    static let nativeMaskUpsampleLogitThreshold: Float = 0.25

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

    /// SAM-style stability score: IoU between masks at (threshold - offset)
    /// and (threshold + offset). Since the high-threshold mask is always a
    /// subset of the low-threshold mask, this reduces to:
    ///
    ///     stability = area(high) / area(low)
    ///
    /// Returns 0 if the low-threshold mask is empty.
    static func calculateMaskStabilityScore(
        logits: [Float],
        threshold: Float,
        offset: Float
    ) -> Float {
        guard !logits.isEmpty, offset > 0 else { return 0 }

        let highThresh = threshold + offset
        let lowThresh = threshold - offset
        var highCount = 0
        var lowCount = 0

        for v in logits {
            if v > highThresh { highCount += 1 }
            if v > lowThresh { lowCount += 1 }
        }

        guard lowCount > 0 else { return 0 }
        return Float(highCount) / Float(lowCount)
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

#if DEBUG
    /// Share of composite-band pixels with bilinear `logit > 0` that fall in `(0, bias]`
    /// (removed when thresholding at `bias`). Values often ~0.05–0.15; sustained **> ~0.2**
    /// suggests a wide shallow-positive margin for that frame.
    static func maskUpsampleBiasSensitivityFraction(
        maskLogits: [Float],
        protoW: Int,
        protoH: Int,
        origW: Int,
        origH: Int,
        xStart: Int,
        xEnd: Int,
        yStart: Int,
        yEnd: Int,
        bias: Float
    ) -> Float {
        guard bias > 0, xStart < xEnd, yStart < yEnd else { return 0 }
        var positiveAtZero: Int = 0
        var inShallowMargin: Int = 0
        for y in yStart..<yEnd {
            for x in xStart..<xEnd {
                let logit = bilinearUpsampledLogit(
                    maskLogits: maskLogits,
                    protoW: protoW,
                    protoH: protoH,
                    origW: origW,
                    origH: origH,
                    imageX: x,
                    imageY: y
                )
                if logit > 0 {
                    positiveAtZero += 1
                    if logit <= bias { inShallowMargin += 1 }
                }
            }
        }
        guard positiveAtZero > 0 else { return 0 }
        return Float(inShallowMargin) / Float(positiveAtZero)
    }
#endif

    private static let includeSupportingTableForMonitorScene = true
    private static let monitorLikeClassIds: Set<Int> = [1063, 2675, 4105]
    private static let supportingTableClassIds: Set<Int> = [1061, 1301, 1325, 1503, 1885, 2324, 2836, 4564]

    private static func pickSupportingTableForMonitorScene(
        primaryDetection: FurnitureFitDetection,
        detections: [FurnitureFitDetection],
        primaryIndex: Int
    ) -> FurnitureFitDetection? {
        if !includeSupportingTableForMonitorScene { return nil }
        if !monitorLikeClassIds.contains(primaryDetection.classIdx) { return nil }

        let primaryLeft = primaryDetection.x - primaryDetection.w * 0.5
        let primaryRight = primaryDetection.x + primaryDetection.w * 0.5
        let primaryBottom = primaryDetection.y + primaryDetection.h * 0.5
        let primaryArea = max(1e-3, primaryDetection.w * primaryDetection.h)

        var best: FurnitureFitDetection?
        var bestScore: Float = -1

        for (idx, detection) in detections.enumerated() {
            if idx == primaryIndex { continue }
            if !supportingTableClassIds.contains(detection.classIdx) { continue }

            let candidateLeft = detection.x - detection.w * 0.5
            let candidateRight = detection.x + detection.w * 0.5
            let candidateTop = detection.y - detection.h * 0.5
            let overlapWidth = max(0, min(primaryRight, candidateRight) - max(primaryLeft, candidateLeft))
            let horizontalOverlapRatio = overlapWidth / max(1e-3, min(primaryDetection.w, detection.w))
            if horizontalOverlapRatio < 0.35 { continue }

            if detection.y <= primaryDetection.y { continue }

            let verticalGap = candidateTop - primaryBottom
            if verticalGap < -primaryDetection.h * 0.20 || verticalGap > primaryDetection.h * 0.60 { continue }

            let widthRatio = detection.w / max(1e-3, primaryDetection.w)
            if widthRatio < 0.75 || widthRatio > 5.0 { continue }

            let areaRatio = (detection.w * detection.h) / primaryArea
            if areaRatio < 0.50 || areaRatio > 12.0 { continue }

            let closenessTerm = 1 - min(1, abs(verticalGap) / max(primaryDetection.h * 0.60, 1e-3))
            let score = detection.confidence * horizontalOverlapRatio * max(0.1, closenessTerm)

            if score > bestScore {
                bestScore = score
                best = detection
            }
        }
        return best
    }

    /// Approximates the candidate mask center from proto pixels inside the candidate bbox.
    private static func candidateMaskCenterInModelSpace(
        detection: FurnitureFitDetection,
        planes: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Float
    ) -> (x: Float, y: Float)? {
        let hwProto = protoW * protoH
        guard protoW > 0,
              protoH > 0,
              modelSide > 0,
              detection.coeffs.count >= 32,
              planes.count >= 32 * hwProto else { return nil }

        let widthRatio = Float(protoW) / modelSide
        let heightRatio = Float(protoH) / modelSide
        let bboxLeft = max(0, min(protoW - 1, Int(floor((detection.x - detection.w * 0.5) * widthRatio))))
        let bboxTop = max(0, min(protoH - 1, Int(floor((detection.y - detection.h * 0.5) * heightRatio))))
        let bboxRight = max(0, min(protoW - 1, Int(ceil((detection.x + detection.w * 0.5) * widthRatio))))
        let bboxBottom = max(0, min(protoH - 1, Int(ceil((detection.y + detection.h * 0.5) * heightRatio))))
        guard bboxRight >= bboxLeft, bboxBottom >= bboxTop else { return nil }

        var sumProtoX: Float = 0
        var sumProtoY: Float = 0
        var onPixelCount: Int = 0
        for protoY in bboxTop...bboxBottom {
            let rowBase = protoY * protoW
            for protoX in bboxLeft...bboxRight {
                let protoPixelIndex = rowBase + protoX
                var logitSum: Float = 0
                var coeffIndex = 0
                while coeffIndex < 32 {
                    let planeIndex = coeffIndex * hwProto + protoPixelIndex
                    logitSum += detection.coeffs[coeffIndex] * planes[planeIndex]
                    coeffIndex += 1
                }
                guard logitSum > maskLogitThreshold else { continue }
                sumProtoX += Float(protoX) + 0.5
                sumProtoY += Float(protoY) + 0.5
                onPixelCount += 1
            }
        }

        guard onPixelCount > 0 else { return nil }
        let centerProtoX = sumProtoX / Float(onPixelCount)
        let centerProtoY = sumProtoY / Float(onPixelCount)
        return (
            x: centerProtoX / widthRatio,
            y: centerProtoY / heightRatio
        )
    }

    /// Android `collectMaskDetections` (returns list for mask fusion, primary first).
    static func collectMaskDetections(
        primaryIndex: Int,
        detections: [FurnitureFitDetection],
        planes: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Float
    ) -> [FurnitureFitDetection] {
        guard primaryIndex >= 0, primaryIndex < detections.count else { return [] }
        let primaryDetection = detections[primaryIndex]

        let supportingTableDetection = pickSupportingTableForMonitorScene(
            primaryDetection: primaryDetection,
            detections: detections,
            primaryIndex: primaryIndex
        )

        let encompassTolerance: Float = 2
        let minimumCandidateConfidence: Float = 0.10
        let bboxDuplicateThreshold: Float = 0.7
        var bboxKept: [FurnitureFitDetection] = []

        for (idx, detection) in detections.enumerated() {
            if idx == primaryIndex || detection.confidence < minimumCandidateConfidence { continue }
            if detection.coeffs.count < 32 { continue }

            let candidateLeft = detection.x - detection.w * 0.5
            let candidateTop = detection.y - detection.h * 0.5
            let candidateRight = detection.x + detection.w * 0.5
            let candidateBottom = detection.y + detection.h * 0.5
            let primaryLeft = primaryDetection.x - primaryDetection.w * 0.5
            let primaryTop = primaryDetection.y - primaryDetection.h * 0.5
            let primaryRight = primaryDetection.x + primaryDetection.w * 0.5
            let primaryBottom = primaryDetection.y + primaryDetection.h * 0.5

            let encompassesPrimary =
                candidateLeft <= primaryLeft + encompassTolerance &&
                candidateTop <= primaryTop + encompassTolerance &&
                candidateRight >= primaryRight - encompassTolerance &&
                candidateBottom >= primaryBottom - encompassTolerance
            if encompassesPrimary { continue }

            guard let candidateMaskCenter = candidateMaskCenterInModelSpace(
                detection: detection,
                planes: planes,
                protoW: protoW,
                protoH: protoH,
                modelSide: modelSide
            ) else { continue }

            let shouldFuse =
                candidateMaskCenter.x >= primaryLeft &&
                candidateMaskCenter.x <= primaryRight &&
                candidateMaskCenter.y >= primaryTop &&
                candidateMaskCenter.y <= primaryBottom
            if !shouldFuse { continue }

            let tooLarge =
                detection.w > primaryDetection.w * 1.5 &&
                detection.h > primaryDetection.h * 1.5
            if tooLarge { continue }

            var shouldSkip = false
            var replaceIndex: Int?
            for (k, keptDetection) in bboxKept.enumerated() {
                let iou = FurnitureFitIoU.calculate(detection, keptDetection)
                if iou > bboxDuplicateThreshold {
                    if detection.confidence > keptDetection.confidence {
                        replaceIndex = k
                    } else {
                        shouldSkip = true
                    }
                    break
                }
            }
            if shouldSkip { continue }
            if let r = replaceIndex {
                bboxKept[r] = detection
            } else {
                bboxKept.append(detection)
            }
        }

        var maskDetections: [FurnitureFitDetection] = [primaryDetection]
        maskDetections.append(contentsOf: bboxKept)
        if let st = supportingTableDetection, !maskDetections.contains(where: { $0.classIdx == st.classIdx && $0.x == st.x && $0.y == st.y }) {
            maskDetections.append(st)
        }
        return maskDetections
    }

    /// Raw YOLOE prototype mask using per-pixel logits inside each detection's
    /// original bbox only. No sigmoid, no morphology, no heuristic expansion.
    static func buildBboxLimitedLogitMask(
        planes: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Float,
        detections: [FurnitureFitDetection]
    ) -> [UInt8] {
        return buildBboxLimitedLogitMaskWithLogits(
            planes: planes, protoW: protoW, protoH: protoH,
            modelSide: modelSide, detections: detections
        ).binary
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
