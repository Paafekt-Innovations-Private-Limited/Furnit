import Accelerate
import CoreML
import CoreVideo
import UIKit

/// Still-image YOLO-E inference using the **same** letterbox + CoreML path as `FurnitureFitView` (camera).
/// - Parameter `classBlacklist`: Furniture Fit loads `blacklist.json` into this; **wall measurement must pass `[]`** so wall/door classes are never stripped.
enum YoloEImageInference {

    struct LetterboxMapping {
        let modelSide: Int
        let gain: Float
        let padX: Int
        let padY: Int
        let sourceWidth: Int
        let sourceHeight: Int
    }

    /// Matches `FurnitureFitView` (`imageConstraint.pixelsWide`, fallback 1280) plus enumerated/range constraints.
    static func modelInputSize(for model: MLModel) -> Int {
        let imageInputDesc = model.modelDescription.inputDescriptionsByName["image"]
        if let imageConstraint = imageInputDesc?.imageConstraint {
            let w = imageConstraint.pixelsWide
            let h = imageConstraint.pixelsHigh
            if w > 0 && h > 0 {
                return Int(w)
            }
            let sc = imageConstraint.sizeConstraint
            if sc.type == .enumerated {
                let sizes = sc.enumeratedImageSizes
                if let best = sizes.max(by: { $0.pixelsWide * $0.pixelsHigh < $1.pixelsWide * $1.pixelsHigh }) {
                    return Int(best.pixelsWide)
                }
            } else if sc.type == .range {
                let r = sc.pixelsWideRange
                let target = 1280
                let lo = Int(r.lowerBound)
                let hi = Int(r.upperBound)
                if lo > 0 && hi >= lo {
                    return min(max(target, lo), hi)
                }
            }
        }
        return 1280
    }

    /// Same pipeline as `FurnitureFitView.processFrame`: letterbox → `prediction` → `YoloEDetectionParser`.
    /// - Note: `classBlacklist` is for Furniture Fit only (`blacklist.json`). Wall measurement **must** pass `[]`.
    static func runDetections(
        image: UIImage,
        model: MLModel,
        classBlacklist: Set<Int>,
        confidenceThreshold: Float = 0.05
    ) throws -> (detections: [FurnitureFitDetection], mapping: LetterboxMapping) {
        guard let pb = uiImageToBGRAPixelBuffer(image) else {
            throw NSError(
                domain: "YoloEImageInference",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "UIImage → CVPixelBuffer failed"],
            )
        }
        let srcW = CVPixelBufferGetWidth(pb)
        let srcH = CVPixelBufferGetHeight(pb)
        let modelSide = modelInputSize(for: model)
        guard let sq = resizeToSquareLetterbox(src: pb, size: modelSide) else {
            throw NSError(
                domain: "YoloEImageInference",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Letterbox resize failed"],
            )
        }

        let inputDesc = model.modelDescription.inputDescriptionsByName["image"]
        let expectsImage = inputDesc?.type == .image
        let inputProvider: MLFeatureProvider
        if expectsImage {
            let imageValue = MLFeatureValue(pixelBuffer: sq.buffer)
            guard let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": imageValue]) else {
                throw NSError(
                    domain: "YoloEImageInference",
                    code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "MLDictionaryFeatureProvider failed"],
                )
            }
            inputProvider = provider
        } else {
            throw NSError(
                domain: "YoloEImageInference",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Model input is not MLFeatureType.image (same as Furniture Fit)"],
            )
        }

        let output = try model.prediction(from: inputProvider)
        guard let pair = YoloEDetectionParser.extractDetectionAndProto(from: output) else {
            throw NSError(
                domain: "YoloEImageInference",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Missing det/proto outputs"],
            )
        }
        let dets = YoloEDetectionParser.parseDetections(
            detArray: pair.det,
            confidenceThreshold: confidenceThreshold,
            classBlacklist: classBlacklist
        )
        YoloEDetectionParser.releaseF16Scratch()
        let map = LetterboxMapping(
            modelSide: modelSide,
            gain: sq.gain,
            padX: sq.padX,
            padY: sq.padY,
            sourceWidth: srcW,
            sourceHeight: srcH
        )
        return (dets, map)
    }

    /// Maps box from letterboxed model space to source pixel coords (same math as `FurnitureFitView` bbox mapping).
    static func mapDetectionToSourceImage(det: FurnitureFitDetection, mapping: LetterboxMapping) -> FurnitureFitDetection {
        let padXf = Float(mapping.padX)
        let padYf = Float(mapping.padY)
        let g = max(mapping.gain, 1e-6)
        let x1 = (det.x - det.w * 0.5 - padXf) / g
        let y1 = (det.y - det.h * 0.5 - padYf) / g
        let x2 = (det.x + det.w * 0.5 - padXf) / g
        let y2 = (det.y + det.h * 0.5 - padYf) / g
        let w = max(1, x2 - x1)
        let h = max(1, y2 - y1)
        let cx = x1 + w * 0.5
        let cy = y1 + h * 0.5
        return FurnitureFitDetection(x: cx, y: cy, w: w, h: h, confidence: det.confidence, classIdx: det.classIdx, coeffs: det.coeffs)
    }

    // MARK: - UIImage → CVPixelBuffer (upright)

    private static func uiImageToBGRAPixelBuffer(_ image: UIImage) -> CVPixelBuffer? {
        let format = UIGraphicsImageRendererFormat()
        format.scale = image.scale
        format.opaque = false
        let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
        let drawn = renderer.image { _ in image.draw(in: CGRect(origin: .zero, size: image.size)) }
        guard let cgImage = drawn.cgImage else { return nil }
        let w = cgImage.width
        let h = cgImage.height
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: w,
            height: h,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    // MARK: - Letterbox (same as FurnitureFitView.resizeToSquare, without instance buffer cache)

    private static func resizeToSquareLetterbox(
        src: CVPixelBuffer,
        size: Int
    ) -> (buffer: CVPixelBuffer, gain: Float, padX: Int, padY: Int, newW: Int, newH: Int)? {
        CVPixelBufferLockBaseAddress(src, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly) }

        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)

        let gain = min(Float(size) / Float(srcW), Float(size) / Float(srcH))
        let newW = Int(Float(srcW) * gain)
        let newH = Int(Float(srcH) * gain)
        let padX = (size - newW) / 2
        let padY = (size - newH) / 2

        var newBuffer: CVPixelBuffer?
        guard CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, nil, &newBuffer) == kCVReturnSuccess,
              let dst = newBuffer else { return nil }

        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }

        guard let srcBase = CVPixelBufferGetBaseAddress(src),
              let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }

        let rowBytes = CVPixelBufferGetBytesPerRow(dst)
        YoloUltralyticsLetterboxFill.fillOpaqueBGRA114(dstBase: dstBase, totalByteCount: rowBytes * size)

        var srcBuffer = vImage_Buffer(data: srcBase, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: CVPixelBufferGetBytesPerRow(src))
        let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)
        let dstRowBytes = CVPixelBufferGetBytesPerRow(dst)
        let offsetPtr = dstPtr.advanced(by: padY * dstRowBytes + padX * 4)
        var dstBuffer = vImage_Buffer(data: offsetPtr, height: vImagePixelCount(newH), width: vImagePixelCount(newW), rowBytes: dstRowBytes)

        guard vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(0)) == kvImageNoError else { return nil }

        return (buffer: dst, gain: gain, padX: padX, padY: padY, newW: newW, newH: newH)
    }
}
