import Accelerate
import CoreML
import CoreVideo
import UIKit

/// Still-image YOLO-E inference using the **same stretch + Core ML path** as ``FurnitureFitView`` (ONNX-style / Android parity).
/// - Parameter `classBlacklist`: Furniture Fit loads `blacklist.json` into this; **wall measurement must pass `[]`** so wall/door classes are never stripped.
enum YoloEImageInference {

    /// Maps detections from the model square back to source pixels (independent scale per axis), matching Furniture Fit ONNX-style preprocessing.
    struct OnnxStyleMapping {
        let modelSide: Int
        let sourceWidth: Int
        let sourceHeight: Int
    }

    /// Square side for stretch (from Core ML `image` constraint). **26L seg PF** exports use **640**; fallback matches that, not legacy 11L 1280.
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
                let target = 640
                let lo = Int(r.lowerBound)
                let hi = Int(r.upperBound)
                if lo > 0 && hi >= lo {
                    return min(max(target, lo), hi)
                }
            }
        }
        return 640
    }

    /// Stretch → `prediction` → `YoloEDetectionParser` (parity with ``FurnitureFitView`` ONNX-style camera path).
    /// - Note: `classBlacklist` is for Furniture Fit only (`blacklist.json`). Wall measurement **must** pass `[]`.
    static func runDetections(
        image: UIImage,
        model: MLModel,
        classBlacklist: Set<Int>,
        confidenceThreshold: Float = 0.05
    ) throws -> (detections: [FurnitureFitDetection], mapping: OnnxStyleMapping) {
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
        guard let stretched = resizeStretchToSquare(src: pb, size: modelSide) else {
            throw NSError(
                domain: "YoloEImageInference",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Stretch resize failed"],
            )
        }

        let inputDesc = model.modelDescription.inputDescriptionsByName["image"]
        let expectsImage = inputDesc?.type == .image
        let inputProvider: MLFeatureProvider
        if expectsImage {
            let imageValue = MLFeatureValue(pixelBuffer: stretched)
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
        let map = OnnxStyleMapping(
            modelSide: modelSide,
            sourceWidth: srcW,
            sourceHeight: srcH
        )
        return (dets, map)
    }

    /// Maps box from stretched model space to source pixel coords (same math as ``FurnitureFitView`` ONNX-style `scaleX` / `scaleY`).
    static func mapDetectionToSourceImage(det: FurnitureFitDetection, mapping: OnnxStyleMapping) -> FurnitureFitDetection {
        let sx = Float(mapping.sourceWidth) / Float(mapping.modelSide)
        let sy = Float(mapping.sourceHeight) / Float(mapping.modelSide)
        return FurnitureFitDetection(
            x: det.x * sx,
            y: det.y * sy,
            w: det.w * sx,
            h: det.h * sy,
            confidence: det.confidence,
            classIdx: det.classIdx,
            coeffs: det.coeffs
        )
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

    // MARK: - Stretch (matches `FurnitureFitView.resizeStretchToSquare`)

    private static func resizeStretchToSquare(src: CVPixelBuffer, size: Int) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(src, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly) }

        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)

        var newBuffer: CVPixelBuffer?
        guard CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, nil, &newBuffer) == kCVReturnSuccess,
              let dst = newBuffer else { return nil }

        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }

        guard let srcBase = CVPixelBufferGetBaseAddress(src),
              let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }

        var srcBuffer = vImage_Buffer(
            data: srcBase,
            height: vImagePixelCount(srcH),
            width: vImagePixelCount(srcW),
            rowBytes: CVPixelBufferGetBytesPerRow(src)
        )
        var dstBuffer = vImage_Buffer(
            data: dstBase,
            height: vImagePixelCount(size),
            width: vImagePixelCount(size),
            rowBytes: CVPixelBufferGetBytesPerRow(dst)
        )
        guard vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(kvImageHighQualityResampling)) == kvImageNoError else {
            return nil
        }
        return dst
    }
}
