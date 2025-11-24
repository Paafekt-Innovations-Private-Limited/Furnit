import UIKit
import AVFoundation
import CoreML
import Accelerate

// High-performance mask-only overlay view for YOLOE (prototypes: [1,32,160,160])
// - Expects model outputs:
//    prototypes -> MLMultiArray float16 [1,32,160,160]
//    detections  -> MLMultiArray float16 [1,4621,cols]
// - Assumes mask coeffs are the last protoK elements in each detection row (coeffStart = cols - protoK).
// - Renders only instance masks onto a transparent UIImageView (overlaps preserved).
final class SmartyPantsView: UIView {

    // Public: set model before calling processWithYOLO
    var mlModel: MLModel?

    // UI
    private let maskImageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFill
        iv.clipsToBounds = true
        iv.backgroundColor = .clear
        iv.isOpaque = false
        return iv
    }()

    // Inference queue and throttling
    private let detectionQueue = DispatchQueue(label: "com.furnit.smarty.detection", qos: .userInitiated)
    private var lastProcessTime = Date.distantPast
    var processInterval: TimeInterval = 0.07
    private var isProcessing = false

    // Model prototype dims (from your message)
    private let protoK = 32
    private let protoH = 160
    private let protoW = 160

    // Reusable buffer for prototypes converted to Float32
    private var protoFloatBuffer: UnsafeMutablePointer<Float>?
    private var protoFloatCount = 0

    // Score threshold
    var scoreThreshold: Float = 0.25

    // Init
    override init(frame: CGRect) {
        super.init(frame: frame)
        commonInit()
    }
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }
    private func commonInit() {
        backgroundColor = .clear
        addSubview(maskImageView)
        maskImageView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            maskImageView.leadingAnchor.constraint(equalTo: leadingAnchor),
            maskImageView.trailingAnchor.constraint(equalTo: trailingAnchor),
            maskImageView.topAnchor.constraint(equalTo: topAnchor),
            maskImageView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
    }
    deinit {
        protoFloatBuffer?.deallocate()
    }

    // MARK: - Main entry (call with camera frames)
    func processWithYOLO(pixelBuffer: CVPixelBuffer) {
        guard let model = mlModel else { return }
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !isProcessing else { return }
        lastProcessTime = now
        DispatchQueue.main.async { self.isProcessing = true }

        detectionQueue.async { [weak self] in
            guard let self = self else { return }

            // Your project should already provide these helpers. Keep them and return a Float16 MLMultiArray input.
            guard let resized = self.resizePixelBuffer(pixelBuffer, width: 640, height: 640),
                  let inputArray = self.pixelBufferToMLMultiArray(resized),
                  let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]),
                  let output = try? model.prediction(from: inputProvider),
                  let prototypesArr = output.featureValue(for: "p")?.multiArrayValue,
                  let detectionsArr = output.featureValue(for: "var_2421")?.multiArrayValue else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            #if DEBUG
            print("prototypes shape: \(prototypesArr.shape)")
            print("detections shape: \(detectionsArr.shape)")
            #endif

            // Ensure proto buffer allocated
            let protCount = prototypesArr.count
            if self.protoFloatBuffer == nil || self.protoFloatCount != protCount {
                self.protoFloatBuffer?.deallocate()
                self.protoFloatBuffer = UnsafeMutablePointer<Float>.allocate(capacity: protCount)
                self.protoFloatCount = protCount
            }
            guard let protoBuf = self.protoFloatBuffer else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            // Convert prototypes float16 -> float32 into protoBuf
            self.copyFloat16MultiArrayToFloatBuffer(prototypesArr, dest: protoBuf)

            // Detections layout: [1, rows, cols]
            let detShape = detectionsArr.shape.map { $0.intValue }
            guard detShape.count >= 3 else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            let rows = detShape[1]
            let cols = detShape[2]

            // Convert detections to Float buffer
            let detCount = detectionsArr.count
            let detBuf = UnsafeMutablePointer<Float>.allocate(capacity: detCount)
            defer { detBuf.deallocate() }
            self.copyFloat16MultiArrayToFloatBuffer(detectionsArr, dest: detBuf)

            // Coeffs assumed last protoK columns
            let coeffK = self.protoK
            let coeffStart = cols - coeffK
            if coeffStart < 0 {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            // Preallocate mask float buffer for prototype resolution
            let protoPixels = self.protoH * self.protoW
            let maskFloatBuf = UnsafeMutablePointer<Float>.allocate(capacity: protoPixels)
            defer { maskFloatBuf.deallocate() }

            // Canvas size in pixels (match view size)
            let scale = UIScreen.main.scale
            let canvasW = Int(round(self.bounds.width * scale))
            let canvasH = Int(round(self.bounds.height * scale))
            if canvasW == 0 || canvasH == 0 {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            var masksAlpha: [CGImage] = []
            var colors: [UIColor] = []

            // Heuristic score index: usually index 4. If your model differs, change this.
            let scoreIdx = min(4, cols - coeffK - 1)

            // For each detection row, build mask if score passes
            for r in 0..<rows {
                let base = r * cols
                let score = detBuf[base + scoreIdx]
                if score < self.scoreThreshold { continue }

                // Extract coeffs (last K)
                let coeffs = UnsafeMutablePointer<Float>.allocate(capacity: coeffK)
                for k in 0..<coeffK { coeffs[k] = detBuf[base + coeffStart + k] }

                // Build mask from prototypes and coeffs (proto layout from model is [1,C,H,W])
                self.buildMaskFromPrototypesFloatLayout(protoBuf: protoBuf, protoH: self.protoH, protoW: self.protoW, protoK: self.protoK, coeffs: coeffs, outMask: maskFloatBuf)

                if let alphaCG = self.resizeFloatMaskToAlphaImage(maskFloat: maskFloatBuf, srcW: self.protoW, srcH: self.protoH, dstW: canvasW, dstH: canvasH) {
                    masksAlpha.append(alphaCG)
                    colors.append(self.colorForIndex(r))
                }
                coeffs.deallocate()
            }

            let outImage = self.compositeMasksAdditive(masksAlpha: masksAlpha, colors: colors, canvasW: canvasW, canvasH: canvasH)

            DispatchQueue.main.async {
                self.maskImageView.image = outImage
                self.isProcessing = false
            }
        }
    }

    // MARK: - Helpers

    // Convert Float16 MLMultiArray -> Float32 buffer using vImage
    private func copyFloat16MultiArrayToFloatBuffer(_ arr: MLMultiArray, dest: UnsafeMutablePointer<Float>) {
        let count = arr.count
        let src = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
        var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(count), rowBytes: count * MemoryLayout<UInt16>.size)
        var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(dest), height: 1, width: vImagePixelCount(count), rowBytes: count * MemoryLayout<Float>.size)
        let err = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
        if err != kvImageNoError {
            // fallback: manual conversion per element using Accelerate helper
            for i in 0..<count {
                dest[i] = float32FromFloat16Bits(src[i])
            }
        }
    }
    private func float32FromFloat16Bits(_ bits: UInt16) -> Float {
        var b = bits
        var out: Float = 0
        var sbuf = vImage_Buffer(data: &b, height: 1, width: 1, rowBytes: 2)
        var dbuf = vImage_Buffer(data: &out, height: 1, width: 1, rowBytes: 4)
        let _ = vImageConvert_Planar16FtoPlanarF(&sbuf, &dbuf, vImage_Flags(kvImageNoFlags))
        return out
    }

    // Build mask: prototypes memory layout assumed [1, C, H, W] contiguous (channel-major)
    // For pixel (y,x): proto channel value at index = c*H*W + y*W + x
    private func buildMaskFromPrototypesFloatLayout(protoBuf: UnsafeMutablePointer<Float>, protoH: Int, protoW: Int, protoK: Int, coeffs: UnsafeMutablePointer<Float>, outMask: UnsafeMutablePointer<Float>) {
        let HW = protoH * protoW
        for y in 0..<protoH {
            let rowOffset = y * protoW
            for x in 0..<protoW {
                var s: Float = 0
                let pixIdx = rowOffset + x
                for c in 0..<protoK {
                    let pIndex = c * HW + pixIdx
                    s += protoBuf[pIndex] * coeffs[c]
                }
                outMask[pixIdx] = 1.0 / (1.0 + exp(-s))
            }
        }
    }

    // Convert float mask [0..1] to grayscale CGImage using vImage and scale to target size
    private func resizeFloatMaskToAlphaImage(maskFloat: UnsafePointer<Float>, srcW: Int, srcH: Int, dstW: Int, dstH: Int) -> CGImage? {
        let srcCount = srcW * srcH
        let srcU8 = UnsafeMutablePointer<UInt8>.allocate(capacity: srcCount)
        defer { srcU8.deallocate() }
        var srcF = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: maskFloat), height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW * MemoryLayout<Float>.size)
        var dstU8buf = vImage_Buffer(data: srcU8, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW)
        let convErr = vImageConvert_PlanarFtoPlanar8(&srcF, &dstU8buf, 255.0, 0.0, vImage_Flags(kvImageNoFlags))
        if convErr != kvImageNoError { return nil }

        let dstCount = dstW * dstH
        let dstU8 = UnsafeMutablePointer<UInt8>.allocate(capacity: dstCount)
        defer { dstU8.deallocate() }
        var srcBuf = vImage_Buffer(data: srcU8, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW)
        var dstBuf = vImage_Buffer(data: dstU8, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW), rowBytes: dstW)
        let scaleErr = vImageScale_Planar8(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageHighQualityResampling))
        if scaleErr != kvImageNoError { return nil }

        guard let provider = CGDataProvider(data: CFDataCreate(nil, dstU8, dstCount)) else { return nil }
        let cs = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        let cg = CGImage(width: dstW, height: dstH, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: dstW, space: cs, bitmapInfo: bitmapInfo, provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
        return cg
    }

    // Composite masks preserving overlaps onto transparent UIImage
    private func compositeMasksAdditive(masksAlpha: [CGImage], colors: [UIColor], canvasW: Int, canvasH: Int) -> UIImage? {
        guard masksAlpha.count == colors.count else { return nil }
        let scale = UIScreen.main.scale
        let size = CGSize(width: CGFloat(canvasW) / scale, height: CGFloat(canvasH) / scale)
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        guard let ctx = UIGraphicsGetCurrentContext() else { UIGraphicsEndImageContext(); return nil }
        ctx.clear(CGRect(origin: .zero, size: size))
        for i in 0..<masksAlpha.count {
            let alphaImg = masksAlpha[i]
            let color = colors[i]
            ctx.saveGState()
            ctx.clip(to: CGRect(x: 0, y: 0, width: size.width, height: size.height), mask: alphaImg)
            ctx.setFillColor(color.cgColor)
            ctx.setAlpha(1.0)
            ctx.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
            ctx.restoreGState()
        }
        let out = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return out
    }

    // --- Placeholders: replace with your working implementations ---
    func resizePixelBuffer(_ src: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        // Replace with your existing high-quality resize to model input (640x640)
        return src
    }
    func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        // Replace with your existing conversion to MLMultiArray (Float16) expected by the compiled model
        return nil
    }

    private func colorForIndex(_ idx: Int) -> UIColor {
        let palette: [UIColor] = [
            UIColor(red: 0.95, green: 0.3, blue: 0.25, alpha: 1),
            UIColor(red: 0.25, green: 0.6, blue: 0.95, alpha: 1),
            UIColor(red: 0.2, green: 0.8, blue: 0.4, alpha: 1),
            UIColor(red: 0.9, green: 0.6, blue: 0.2, alpha: 1),
            UIColor(red: 0.6, green: 0.3, blue: 0.8, alpha: 1)
        ]
        return palette[idx % palette.count].withAlphaComponent(0.9)
    }
}

extension SmartyPantsView {
    func setOverlayImage(_ img: UIImage?) {
        DispatchQueue.main.async { self.maskImageView.image = img }
    }
    func setModel(_ model: MLModel?) {
        self.mlModel = model
    }
}



