import Foundation
import UIKit
import CoreML
import CoreVideo
import Accelerate

// MARK: - Still-image YOLO → calibration boxes (letterbox = FurnitureFit)

enum YoloStillImagePipeline {

    fileprivate static func loadClassBlacklist() -> Set<Int> {
        guard let url = Bundle.main.url(forResource: "blacklist", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            return []
        }
        return Set(dict.keys.compactMap { Int($0) })
    }

    fileprivate static func loadClassNames() -> [Int: String] {
        guard let url = Bundle.main.url(forResource: "classes", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            return [:]
        }
        var result: [Int: String] = [:]
        for (key, value) in dict {
            if let id = Int(key) { result[id] = value }
        }
        return result
    }

    /// Letterbox `pixelBuffer` to square `modelInputSize`, run YOLO-E once, return calibration boxes in **source pixel-buffer coordinates**.
    static func run(
        model: MLModel,
        pixelBuffer: CVPixelBuffer,
        confidenceThreshold: Float,
        classBlacklist: Set<Int>? = nil
    ) throws -> ([YoloCalibrationBox], CGSize) {
        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        let usedSize = CGSize(width: srcW, height: srcH)

        let imageInputDesc = model.modelDescription.inputDescriptionsByName["image"]
        let modelInputSize: Int
        if let imageConstraint = imageInputDesc?.imageConstraint {
            modelInputSize = imageConstraint.pixelsWide
        } else {
            modelInputSize = 1280
        }

        guard let letterboxed = letterboxPixelBuffer(pixelBuffer, size: modelInputSize) else {
            throw NSError(domain: "YoloStillImagePipeline", code: 1, userInfo: [NSLocalizedDescriptionKey: "letterbox failed"])
        }

        let inputProvider: MLFeatureProvider
        let inputDesc = model.modelDescription.inputDescriptionsByName["image"]
        let expectsImage = inputDesc?.type == .image
        if expectsImage {
            guard let imageValue = MLFeatureValue(pixelBuffer: letterboxed.buffer) as MLFeatureValue?,
                  let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": imageValue]) else {
                throw NSError(domain: "YoloStillImagePipeline", code: 2, userInfo: [NSLocalizedDescriptionKey: "image input failed"])
            }
            inputProvider = provider
        } else {
            guard let inputArray = pixelBufferToMLMultiArray(letterboxed.buffer),
                  let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]) else {
                throw NSError(domain: "YoloStillImagePipeline", code: 3, userInfo: [NSLocalizedDescriptionKey: "MLMultiArray input failed"])
            }
            inputProvider = provider
        }

        guard let output = try? model.prediction(from: inputProvider),
              let pair = YoloEDetectionParser.extractDetectionAndProto(from: output) else {
            throw NSError(domain: "YoloStillImagePipeline", code: 4, userInfo: [NSLocalizedDescriptionKey: "inference failed"])
        }

        let blacklist = classBlacklist ?? loadClassBlacklist()
        let dets = YoloEDetectionParser.parseDetections(
            detArray: pair.det,
            confidenceThreshold: confidenceThreshold,
            classBlacklist: blacklist
        )
        let names = loadClassNames()

        let gain = letterboxed.gain
        let padX = CGFloat(letterboxed.padX)
        let padY = CGFloat(letterboxed.padY)

        var boxes: [YoloCalibrationBox] = []
        boxes.reserveCapacity(dets.count)
        for detection in dets {
            let cx = (CGFloat(detection.x) - padX) / CGFloat(gain)
            let cy = (CGFloat(detection.y) - padY) / CGFloat(gain)
            let ww = CGFloat(detection.w) / CGFloat(gain)
            let hh = CGFloat(detection.h) / CGFloat(gain)
            let label = names[detection.classIdx] ?? "unknown"
            boxes.append(YoloCalibrationBox(
                label: label,
                centerX: cx,
                centerY: cy,
                width: ww,
                height: hh,
                confidence: detection.confidence
            ))
        }

        return (boxes, usedSize)
    }

    private struct LetterboxResult {
        let buffer: CVPixelBuffer
        let gain: Float
        let padX: Int
        let padY: Int
    }

    private static func letterboxPixelBuffer(_ src: CVPixelBuffer, size: Int) -> LetterboxResult? {
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

        memset(dstBase, 128, size * size * 4)

        var srcBuffer = vImage_Buffer(data: srcBase, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: CVPixelBufferGetBytesPerRow(src))
        let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)
        let dstRowBytes = CVPixelBufferGetBytesPerRow(dst)
        let offsetPtr = dstPtr.advanced(by: padY * dstRowBytes + padX * 4)
        var dstBuffer = vImage_Buffer(data: offsetPtr, height: vImagePixelCount(newH), width: vImagePixelCount(newW), rowBytes: dstRowBytes)

        guard vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(0)) == kvImageNoError else { return nil }

        return LetterboxResult(buffer: dst, gain: gain, padX: padX, padY: padY)
    }

    private static func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width == height, width > 0 else { return nil }
        guard let array = try? MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32) else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let pixelCount = width * height
        let floatSize = MemoryLayout<Float32>.size
        let planeStrideBytes = pixelCount * floatSize

        let rPtr = array.dataPointer.advanced(by: 0).assumingMemoryBound(to: Float32.self)
        let gPtr = array.dataPointer.advanced(by: planeStrideBytes).assumingMemoryBound(to: Float32.self)
        let bPtr = array.dataPointer.advanced(by: planeStrideBytes * 2).assumingMemoryBound(to: Float32.self)

        let src = baseAddress.assumingMemoryBound(to: UInt8.self)
        var rowU8 = [UInt8](repeating: 0, count: width * 4)
        var rowF = [Float](repeating: 0, count: width * 4)
        var scale: Float = 1.0 / 255.0

        var indicesR = [vDSP_Length](repeating: 0, count: width)
        var indicesG = [vDSP_Length](repeating: 0, count: width)
        var indicesB = [vDSP_Length](repeating: 0, count: width)
        for i in 0..<width {
            indicesR[i] = vDSP_Length(2 + i * 4)
            indicesG[i] = vDSP_Length(1 + i * 4)
            indicesB[i] = vDSP_Length(0 + i * 4)
        }

        for y in 0..<height {
            let rowStart = src.advanced(by: y * bytesPerRow)
            memcpy(&rowU8, rowStart, width * 4)

            rowU8.withUnsafeBufferPointer { u8 in
                rowF.withUnsafeMutableBufferPointer { f in
                    vDSP_vfltu8(u8.baseAddress!, 1, f.baseAddress!, 1, vDSP_Length(width * 4))
                    vDSP_vsmul(f.baseAddress!, 1, &scale, f.baseAddress!, 1, vDSP_Length(width * 4))
                }
            }

            rowF.withUnsafeBufferPointer { rf in
                vDSP_vgathr(rf.baseAddress!, indicesR, 1, rPtr.advanced(by: y * width), 1, vDSP_Length(width))
                vDSP_vgathr(rf.baseAddress!, indicesG, 1, gPtr.advanced(by: y * width), 1, vDSP_Length(width))
                vDSP_vgathr(rf.baseAddress!, indicesB, 1, bPtr.advanced(by: y * width), 1, vDSP_Length(width))
            }
        }
        return array
    }
}

// MARK: - One-shot calibration when opening a saved room

enum RoomYoloRatioCapture {

    static func modelFileExtension(for fileType: ModelFileType) -> String? {
        switch fileType {
        case .ply: return "ply"
        case .meshroom: return "meshroom"
        case .glb: return "glb"
        default: return nil
        }
    }

    /// Runs once per room when ratio metadata is missing; merges into `.meta` and refreshes the model list.
    @MainActor
    static func captureIfMissing(
        savedModel: USDZModel,
        modelManager: USDZModelManager,
        sharpRoomHeightMeters: Float?
    ) async {
        let dbg = AppStateManager.shared.qualitySettings.debugMode
        if dbg {
            logDebug("RoomYoloRatioCapture: captureIfMissing enter fileName=\(savedModel.fileName) fileType=\(savedModel.fileType) isSavedRoom=\(savedModel.isSavedRoom) ratioSettingOn=\(AppStateManager.shared.qualitySettings.enableRatioBasedFurnitureFit) yoloRefImageHeightPx=\(savedModel.yoloRefImageHeightPx.map { String($0) } ?? "nil")")
        }

        guard savedModel.isSavedRoom else {
            if dbg { logDebug("RoomYoloRatioCapture: skip — not a saved room (no .meta calibration path)") }
            return
        }
        guard AppStateManager.shared.qualitySettings.enableRatioBasedFurnitureFit else {
            if dbg { logDebug("RoomYoloRatioCapture: skip — enableRatioBasedFurnitureFit is OFF") }
            return
        }
        guard let roomExt = modelFileExtension(for: savedModel.fileType) else {
            if dbg { logDebug("RoomYoloRatioCapture: skip — unsupported fileType for ratio capture (\(savedModel.fileType))") }
            return
        }

        if let existingRefH = savedModel.yoloRefImageHeightPx {
            if dbg { logDebug("RoomYoloRatioCapture: skip — already calibrated (yoloRefImageHeightPx=\(existingRefH))") }
            return
        }

        let yoloService = YOLOEModelService.shared
        await yoloService.waitForModelReady()
        guard let mlModel = yoloService.model else {
            logDebug("RoomYoloRatioCapture: YOLO model not available")
            return
        }

        let refCandidate = referenceImageURL(fileName: savedModel.fileName, fileType: savedModel.fileType, savedRoomsRoot: modelManager.savedRoomsDirectoryURL)
        guard let refURL = refCandidate,
              FileManager.default.fileExists(atPath: refURL.path),
              let data = try? Data(contentsOf: refURL),
              let uiImage = UIImage(data: data) else {
            if dbg {
                if refCandidate == nil {
                    logDebug("RoomYoloRatioCapture: skip — no referenceImageURL (e.g. missing .meshroom for \(savedModel.fileName))")
                } else if let u = refCandidate, !FileManager.default.fileExists(atPath: u.path) {
                    logDebug("RoomYoloRatioCapture: skip — ref path does not exist: \(u.path)")
                } else {
                    logDebug("RoomYoloRatioCapture: skip — ref file unreadable or not a valid image")
                }
            }
            logDebug("RoomYoloRatioCapture: no reference image for \(savedModel.fileName)")
            return
        }

        let scaled = downscaleImage(uiImage, maxSide: 1280)
        guard let pixelBuffer = makePixelBuffer(from: scaled) else {
            logDebug("RoomYoloRatioCapture: pixel buffer failed")
            return
        }

        do {
            let (boxes, usedSize) = try YoloStillImagePipeline.run(
                model: mlModel,
                pixelBuffer: pixelBuffer,
                confidenceThreshold: 0.12,
                classBlacklist: nil
            )
            if dbg {
                let labelsPreview = boxes.prefix(8).map { "\($0.label):\(String(format: "%.2f", $0.confidence))" }.joined(separator: ", ")
                logDebug("RoomYoloRatioCapture: YOLO still \(boxes.count) boxes on ref \(Int(usedSize.width))×\(Int(usedSize.height)) — \(labelsPreview)\(boxes.count > 8 ? " …" : "")")
            }
            let wallFrac = YoloRatioCalibration.wallHeightFractionOrFullFrame(imageSize: usedSize, boxes: boxes)
            let furnFrac = YoloRatioCalibration.furnitureHeightFractionsByLabel(imageHeight: usedSize.height, boxes: boxes)
            if dbg {
                let keys = furnFrac.keys.sorted().joined(separator: ", ")
                logDebug("RoomYoloRatioCapture: wallFrac=\(String(format: "%.4f", wallFrac)) furnitureMap keys (\(furnFrac.count)): \(keys)")
            }
            let sharpH = sharpRoomHeightMeters ?? savedModel.roomHeight

            try modelManager.mergeYoloCalibrationMetadata(
                fileName: savedModel.fileName,
                modelFileExtension: roomExt,
                wallHeightFrac: wallFrac,
                furnitureFractionsByClass: furnFrac,
                referenceImageHeightPx: Int(usedSize.height.rounded()),
                sharpRoomHeightAtCapture: sharpH
            )
            modelManager.refreshModels()
            logDebug("RoomYoloRatioCapture: wrote YOLO ratios for \(savedModel.fileName) (ref H=\(Int(usedSize.height)))")
        } catch {
            logDebug("RoomYoloRatioCapture: \(error.localizedDescription)")
        }
    }

    private static func referenceImageURL(fileName: String, fileType: ModelFileType, savedRoomsRoot: URL) -> URL? {
        let meshURL = savedRoomsRoot.appendingPathComponent("\(fileName).meshroom")
        switch fileType {
        case .meshroom:
            return meshURL
        case .glb, .ply:
            if FileManager.default.fileExists(atPath: meshURL.path) {
                return meshURL
            }
            return nil
        default:
            return nil
        }
    }

    private static func downscaleImage(_ image: UIImage, maxSide: CGFloat) -> UIImage {
        let w = image.size.width * image.scale
        let h = image.size.height * image.scale
        let longest = max(w, h)
        guard longest > maxSide else { return image }
        let ratio = maxSide / longest
        let newW = floor(w * ratio)
        let newH = floor(h * ratio)
        let newSize = CGSize(width: newW, height: newH)
        let renderer = UIGraphicsImageRenderer(size: newSize)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: newSize))
        }
    }

    private static func makePixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height

        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        var buffer: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &buffer) == kCVReturnSuccess,
              let pixelBuffer = buffer else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixelBuffer
    }
}
