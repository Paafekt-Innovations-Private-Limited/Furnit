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
    // static let retinaMasksCompositeEnabled: Bool = true
    static let retinaMasksCompositeEnabled: Bool = false

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

    /// Computes what fraction of a child detection's mask pixels overlap the
    /// primary detection's mask pixels at proto resolution.
    static func maskOverlapFraction(
        childCoeffs: [Float],
        primaryCoeffs: [Float],
        protos: [Float],
        protoHeight: Int,
        protoWidth: Int
    ) -> Float {
        let spatialSize = protoHeight * protoWidth
        let numProtos = childCoeffs.count

        guard numProtos > 0, protos.count == numProtos * spatialSize else { return 0 }

        var childLogits = [Float](repeating: 0, count: spatialSize)

        cblas_sgemv(
            CblasRowMajor,
            CblasTrans,
            Int32(numProtos),
            Int32(spatialSize),
            1.0,
            protos,
            Int32(spatialSize),
            childCoeffs,
            1,
            0.0,
            &childLogits,
            1
        )

        var primaryLogits = [Float](repeating: 0, count: spatialSize)

        cblas_sgemv(
            CblasRowMajor,
            CblasTrans,
            Int32(numProtos),
            Int32(spatialSize),
            1.0,
            protos,
            Int32(spatialSize),
            primaryCoeffs,
            1,
            0.0,
            &primaryLogits,
            1
        )

        var childBinary = [Float](repeating: 0, count: spatialSize)
        var primaryBinary = [Float](repeating: 0, count: spatialSize)
        var zero: Float = 0.0
        var one: Float = 1.0

        vDSP_vthres(childLogits, 1, &zero, &childBinary, 1, vDSP_Length(spatialSize))
        vDSP_vclip(childBinary, 1, &zero, &one, &childBinary, 1, vDSP_Length(spatialSize))

        vDSP_vthres(primaryLogits, 1, &zero, &primaryBinary, 1, vDSP_Length(spatialSize))
        vDSP_vclip(primaryBinary, 1, &zero, &one, &primaryBinary, 1, vDSP_Length(spatialSize))

        var intersection = [Float](repeating: 0, count: spatialSize)
        vDSP_vmul(childBinary, 1, primaryBinary, 1, &intersection, 1, vDSP_Length(spatialSize))

        var childPixelCount: Float = 0
        var overlapPixelCount: Float = 0
        vDSP_sve(childBinary, 1, &childPixelCount, vDSP_Length(spatialSize))
        vDSP_sve(intersection, 1, &overlapPixelCount, vDSP_Length(spatialSize))

        guard childPixelCount > 0 else { return 0 }
        return overlapPixelCount / childPixelCount
    }

    /// Pre-compute the primary binary mask once and reuse it for child overlap checks.
    static func buildBinaryMask(
        coeffs: [Float],
        protos: [Float],
        spatialSize: Int
    ) -> [Float] {
        let numProtos = coeffs.count
        var logits = [Float](repeating: 0, count: spatialSize)

        cblas_sgemv(
            CblasRowMajor, CblasTrans,
            Int32(numProtos), Int32(spatialSize),
            1.0, protos, Int32(spatialSize),
            coeffs, 1,
            0.0, &logits, 1
        )

        var binary = [Float](repeating: 0, count: spatialSize)
        var zero: Float = 0.0
        var one: Float = 1.0
        vDSP_vthres(logits, 1, &zero, &binary, 1, vDSP_Length(spatialSize))
        vDSP_vclip(binary, 1, &zero, &one, &binary, 1, vDSP_Length(spatialSize))

        return binary
    }

    static func buildBboxBinaryMask(
        detection: FurnitureFitDetection,
        protoWidth: Int,
        protoHeight: Int,
        modelSide: Int,
        spatialSize: Int
    ) -> [Float] {
        var mask = [Float](repeating: 0, count: spatialSize)
        guard let bbox = protoBounds(
            for: detection,
            protoWidth: protoWidth,
            protoHeight: protoHeight,
            modelSide: modelSide
        ) else { return mask }

        for row in bbox.top...bbox.bottom {
            let start = row * protoWidth + bbox.left
            let length = bbox.right - bbox.left + 1
            for columnOffset in 0..<length {
                mask[start + columnOffset] = 1.0
            }
        }
        return mask
    }

    static func buildCroppedBinaryMask(
        detection: FurnitureFitDetection,
        protos: [Float],
        protoWidth: Int,
        protoHeight: Int,
        modelSide: Int
    ) -> [Float] {
        let spatialSize = protoWidth * protoHeight
        let coefficients = Array(detection.coeffs.prefix(32))
        let numProtos = coefficients.count
        guard numProtos > 0, protos.count == numProtos * spatialSize else {
            return [Float](repeating: 0, count: spatialSize)
        }

        var logits = [Float](repeating: 0, count: spatialSize)
        cblas_sgemv(
            CblasRowMajor,
            CblasTrans,
            Int32(numProtos),
            Int32(spatialSize),
            1.0,
            protos,
            Int32(spatialSize),
            coefficients,
            1,
            0.0,
            &logits,
            1
        )

        let bboxMask = buildBboxBinaryMask(
            detection: detection,
            protoWidth: protoWidth,
            protoHeight: protoHeight,
            modelSide: modelSide,
            spatialSize: spatialSize
        )

#if DEBUG
        var positiveLogitsBeforeCrop: Float = 0
        var positiveLogitsAfterCrop: Float = 0
        for protoPixelIndex in 0..<spatialSize {
            if logits[protoPixelIndex] > 0 {
                positiveLogitsBeforeCrop += 1
            }
        }
#endif

        vDSP_vmul(logits, 1, bboxMask, 1, &logits, 1, vDSP_Length(spatialSize))

#if DEBUG
        for protoPixelIndex in 0..<spatialSize {
            if logits[protoPixelIndex] > 0 {
                positiveLogitsAfterCrop += 1
            }
        }
        if let bbox = protoBounds(
            for: detection,
            protoWidth: protoWidth,
            protoHeight: protoHeight,
            modelSide: modelSide
        ) {
            print(
                "🔲 BBOX CROP class=\(detection.classIdx) " +
                "bbox=(\(bbox.left),\(bbox.top))-(\(bbox.right),\(bbox.bottom)) " +
                "proto=\(protoWidth)x\(protoHeight) " +
                "pixels before=\(Int(positiveLogitsBeforeCrop)) " +
                "after=\(Int(positiveLogitsAfterCrop))"
            )
        }
#endif

        var binary = [Float](repeating: 0, count: spatialSize)
        var zero: Float = 0.0
        var one: Float = 1.0
        vDSP_vthres(logits, 1, &zero, &binary, 1, vDSP_Length(spatialSize))
        vDSP_vclip(binary, 1, &zero, &one, &binary, 1, vDSP_Length(spatialSize))
        return binary
    }

    static func protoBounds(
        for detection: FurnitureFitDetection,
        protoWidth: Int,
        protoHeight: Int,
        modelSide: Int
    ) -> (left: Int, top: Int, right: Int, bottom: Int)? {
        guard protoWidth > 0, protoHeight > 0, modelSide > 0 else { return nil }

        let widthRatio = Float(protoWidth) / Float(modelSide)
        let heightRatio = Float(protoHeight) / Float(modelSide)
        let edgeBias: Float = 0.0
        let maxX = Float(protoWidth - 1)
        let maxY = Float(protoHeight - 1)

        let x1Proto = max(0, min(maxX, (detection.x - detection.w * 0.5) * widthRatio))
        let y1Proto = max(0, min(maxY, (detection.y - detection.h * 0.5) * heightRatio))
        let x2Proto = max(0, min(maxX, (detection.x + detection.w * 0.5) * widthRatio))
        let y2Proto = max(0, min(maxY, (detection.y + detection.h * 0.5) * heightRatio))

        let bboxLeft = Int(floor(x1Proto - edgeBias)).clamped(to: 0...(protoWidth - 1))
        let bboxTop = Int(floor(y1Proto - edgeBias)).clamped(to: 0...(protoHeight - 1))
        let bboxRight = Int(ceil(x2Proto + edgeBias)).clamped(to: 0...(protoWidth - 1))
        let bboxBottom = Int(ceil(y2Proto + edgeBias)).clamped(to: 0...(protoHeight - 1))
        guard bboxLeft <= bboxRight, bboxTop <= bboxBottom else { return nil }
        return (bboxLeft, bboxTop, bboxRight, bboxBottom)
    }

    /// Fast overlap check using a pre-computed primary mask.
    static func childOverlapsFraction(
        childDetection: FurnitureFitDetection,
        primaryBinary: [Float],
        protos: [Float],
        protoWidth: Int,
        protoHeight: Int,
        modelSide: Int
    ) -> Float {
        let spatialSize = protoWidth * protoHeight
        let childCoefficients = Array(childDetection.coeffs.prefix(32))
        let numProtos = childCoefficients.count
        var childLogits = [Float](repeating: 0, count: spatialSize)

        cblas_sgemv(
            CblasRowMajor, CblasTrans,
            Int32(numProtos), Int32(spatialSize),
            1.0, protos, Int32(spatialSize),
            childCoefficients, 1,
            0.0, &childLogits, 1
        )

        guard let bbox = protoBounds(
            for: childDetection,
            protoWidth: protoWidth,
            protoHeight: protoHeight,
            modelSide: modelSide
        ) else { return 0 }

        var bboxMask = [Float](repeating: 0, count: spatialSize)
        for row in bbox.top...bbox.bottom {
            let rowStart = row * protoWidth + bbox.left
            let rowLength = bbox.right - bbox.left + 1
            for columnOffset in 0..<rowLength {
                bboxMask[rowStart + columnOffset] = 1.0
            }
        }
        vDSP_vmul(childLogits, 1, bboxMask, 1, &childLogits, 1, vDSP_Length(spatialSize))

        var childBinary = [Float](repeating: 0, count: spatialSize)
        let childThreshold: Float = 0.0
        var negativeChildThreshold = -childThreshold
        var zero: Float = 0.0
        var one: Float = 1.0
        // Use the standard logit > 0 test after cropping the child mask to its own
        // proto-space bbox so overlap is measured only within the detection region.
        vDSP_vsadd(childLogits, 1, &negativeChildThreshold, &childBinary, 1, vDSP_Length(spatialSize))
        vDSP_vthres(childBinary, 1, &zero, &childBinary, 1, vDSP_Length(spatialSize))
        vDSP_vclip(childBinary, 1, &zero, &one, &childBinary, 1, vDSP_Length(spatialSize))

        var intersection = [Float](repeating: 0, count: spatialSize)
        vDSP_vmul(childBinary, 1, primaryBinary, 1, &intersection, 1, vDSP_Length(spatialSize))

        var childCount: Float = 0
        var overlapCount: Float = 0
        vDSP_sve(childBinary, 1, &childCount, vDSP_Length(spatialSize))
        vDSP_sve(intersection, 1, &overlapCount, vDSP_Length(spatialSize))

        return childCount > 0 ? overlapCount / childCount : 0
    }

    /// Returns the primary first, followed by detections whose proto masks overlap it.
    static func collectMaskDetections(
        primaryIndex: Int,
        detections: [FurnitureFitDetection],
        protos: [Float],
        protoHeight: Int,
        protoWidth: Int,
        modelSide: Int,
        minOverlapFraction: Float = 0.15
    ) -> [FurnitureFitDetection] {
        guard primaryIndex >= 0, primaryIndex < detections.count else { return [] }
        let primary = detections[primaryIndex]
        guard primary.coeffs.count >= 32 else { return [primary] }

        let spatialSize = protoHeight * protoWidth
        let primaryBinary = buildBboxBinaryMask(
            detection: primary,
            protoWidth: protoWidth,
            protoHeight: protoHeight,
            modelSide: modelSide,
            spatialSize: spatialSize
        )

        var maskDetections: [FurnitureFitDetection] = [primary]
        for (idx, detection) in detections.enumerated() {
            guard idx != primaryIndex else { continue }
            guard detection.coeffs.count >= 32 else { continue }

            let overlap = childOverlapsFraction(
                childDetection: detection,
                primaryBinary: primaryBinary,
                protos: protos,
                protoWidth: protoWidth,
                protoHeight: protoHeight,
                modelSide: modelSide
            )

            if overlap >= minOverlapFraction {
                maskDetections.append(detection)
            }
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

    /// Builds a fused binary union by cropping and binarizing each detection independently,
    /// then OR-ing the per-detection masks together.
    static func buildFullFieldLogitMask(
        planes: [Float],
        protoW: Int,
        protoH: Int,
        detections: [FurnitureFitDetection],
        modelSide: Int
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

        var compositeBinary = [Float](repeating: 0, count: hwProto)
        var unionScratch = [Float](repeating: 0, count: hwProto)

        for detection in validDetections {
            let detectionBinary = buildCroppedBinaryMask(
                detection: detection,
                protos: planes,
                protoWidth: protoW,
                protoHeight: protoH,
                modelSide: modelSide
            )
            vDSP_vmax(
                compositeBinary,
                1,
                detectionBinary,
                1,
                &unionScratch,
                1,
                vDSP_Length(hwProto)
            )
            swap(&compositeBinary, &unionScratch)
        }

        let binary = compositeBinary.map { $0 > 0 ? UInt8(255) : UInt8(0) }
        return (binary, compositeBinary)
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
