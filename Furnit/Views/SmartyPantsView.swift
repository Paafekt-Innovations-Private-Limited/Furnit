import SwiftUI
import UIKit
import CoreML
import Accelerate
import AVFoundation
import Photos

// SmartyPantsView.swift
// Single-file on-device YOLOE mask decoding + optimized pipeline
// Drop into your project, then instantiate via the provided SwiftUI wrapper.

struct SmartyPantsViewSwiftUI: UIViewRepresentable {
    let mlModel: MLModel?
    var processInterval: TimeInterval = 0.07
    var scoreThreshold: Float = 0.25
    var active: Bool = false
    var debugShowTop1: Bool = true
    var debugSaveImages: Bool = true

    func makeUIView(context: Context) -> SmartyPantsContainerView {
        let v = SmartyPantsContainerView()
        v.processInterval = processInterval
        v.scoreThreshold = scoreThreshold
        v.debugShowTopMask = debugShowTop1
        v.debugSaveImages = debugSaveImages
        v.setModel(mlModel)
        if active { v.startIfNeeded() }
        return v
    }

    func updateUIView(_ uiView: SmartyPantsContainerView, context: Context) {
        uiView.setModel(mlModel)
        uiView.processInterval = processInterval
        uiView.scoreThreshold = scoreThreshold
        uiView.debugShowTopMask = debugShowTop1
        uiView.debugSaveImages = debugSaveImages
        if active { uiView.startIfNeeded() } else { uiView.stop() }
    }

    static func dismantleUIView(_ uiView: SmartyPantsContainerView, coordinator: ()) {
        uiView.stop()
    }
}

final class SmartyPantsContainerView: UIView {
    // Public config
    var processInterval: TimeInterval = 0.07
    var scoreThreshold: Float = 0.25

    // Debug flags
    var debugShowTopMask: Bool = false
    var debugSaveImages: Bool = false

    // UI
    private let maskImageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFill
        iv.backgroundColor = .clear
        iv.isOpaque = false
        iv.clipsToBounds = true
        return iv
    }()

    // ML model
    private var mlModel: MLModel?

    // Queues and throttles
    private let detectionQueue = DispatchQueue(label: "com.furnit.smarty.detection", qos: .userInitiated)
    private var lastProcessTime = Date.distantPast
    private var processing = false

    // Caches and buffers
    private var cachedScoreIdx: Int?
    private var cachedCoeffStart: Int?
    private var detectionsBuf: UnsafeMutablePointer<Float>?
    private var detectionsBufCount: Int = 0

    private var protoFloatBuf: UnsafeMutablePointer<Float>?
    private var protoFloatCount: Int = 0

    private var maskFloatBuf: UnsafeMutablePointer<Float>?
    private var maskFloatBufCount: Int = 0

    private var planar8BufA: UnsafeMutablePointer<UInt8>?
    private var planar8BufB: UnsafeMutablePointer<UInt8>?
    private var planar8BufCount: Int = 0

    // Model protos info
    private let protoK = 32
    private let protoH = 160
    private let protoW = 160

    // Init / layout
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
    override func layoutSubviews() {
        super.layoutSubviews()
        maskImageView.frame = bounds
    }

    deinit {
        protoFloatBuf?.deallocate()
        detectionsBuf?.deallocate()
        maskFloatBuf?.deallocate()
        planar8BufA?.deallocate()
        planar8BufB?.deallocate()
    }

    // Public API
    func setModel(_ model: MLModel?) {
        detectionQueue.sync {
            self.mlModel = model
            print("SmartyPants: model set -> \(model != nil)")
        }
    }
    func startIfNeeded() { /* if you manage capture here start session */ }
    func stop() { /* stop capture if needed */ }

    // MARK: - Main processing entry
    // Call this with a CVPixelBuffer from your camera capture pipeline.
    func processFrame(_ pixelBuffer: CVPixelBuffer) {
        guard let model = mlModel else { return }

        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !processing else { return }
        lastProcessTime = now
        let frameStart = CFAbsoluteTimeGetCurrent()
        DispatchQueue.main.async { self.processing = true }

        detectionQueue.async { [weak self] in
            guard let self = self else { return }
            // Convert buffer -> MLMultiArray input
            guard let inputArray = self.pixelBufferToMLMultiArray(pixelBuffer, width: 640, height: 640) else {
                DispatchQueue.main.async { self.processing = false }
                return
            }
            // Run model
            guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]),
                  let output = try? model.prediction(from: inputProvider) else {
                DispatchQueue.main.async { self.processing = false }
                return
            }
            // Read outputs
            guard let prototypesArr = output.featureValue(for: "p")?.multiArrayValue,
                  let detectionsArr = output.featureValue(for: "var_2421")?.multiArrayValue else {
                DispatchQueue.main.async { self.processing = false }
                return
            }

            // shapes
            let numPredictions = detectionsArr.shape[1].intValue
            let numFeatures = detectionsArr.shape[2].intValue
            let K = self.protoK
            let HW = self.protoH * self.protoW

            print("prototypes shape: \(prototypesArr.shape), detections shape: \(detectionsArr.shape)")

            // Prepare proto buffer (Float32) and reuse
            let protoCount = prototypesArr.count
            if self.protoFloatBuf == nil || self.protoFloatCount != protoCount {
                self.protoFloatBuf?.deallocate()
                self.protoFloatBuf = UnsafeMutablePointer<Float>.allocate(capacity: protoCount)
                self.protoFloatCount = protoCount
            }
            guard let protoBuf = self.protoFloatBuf else {
                DispatchQueue.main.async { self.processing = false }
                return
            }
            // Convert float16 -> float32 (vImage)
            self.copyFloat16MultiArrayToFloatBuffer(prototypesArr, dest: protoBuf)

            // Prepare detections float buffer copy (stride-aware copy)
            let detCount = detectionsArr.count
            if self.detectionsBuf == nil || self.detectionsBufCount != detCount {
                self.detectionsBuf?.deallocate()
                self.detectionsBuf = UnsafeMutablePointer<Float>.allocate(capacity: detCount)
                self.detectionsBufCount = detCount
            }
            guard let detBuf = self.detectionsBuf else {
                DispatchQueue.main.async { self.processing = false }
                return
            }
            self.copyFloat16MultiArrayToFloatBuffer(detectionsArr, dest: detBuf) // safe copy

            // Determine (and cache) score index and coeffStart
            let coeffStartDefault = numFeatures - K
            var coeffStart = self.cachedCoeffStart ?? coeffStartDefault
            var scoreIdx = self.cachedScoreIdx ?? 4
            if self.cachedScoreIdx == nil {
                // scan likely score positions (heuristic)
                var found: Int? = nil
                for f in 4..<min(20, numFeatures - K) {
                    var valid = 0
                    let sampleStart = max(0, numPredictions - 200)
                    for p in sampleStart..<numPredictions {
                        let v = detBuf[p * numFeatures + f]
                        if v.isFinite && v >= 0 && v <= 1 { valid += 1 }
                    }
                    if valid > 50 { found = f; break }
                }
                if let f = found { scoreIdx = f }
                self.cachedScoreIdx = scoreIdx
                self.cachedCoeffStart = coeffStart
            } else {
                // ensure cached coeffStart exists
                if self.cachedCoeffStart == nil { self.cachedCoeffStart = coeffStartDefault }
                coeffStart = self.cachedCoeffStart!
            }

            print("Using cached scoreIdx: \(scoreIdx), coeffStart: \(coeffStart)")

            // Canvas
            let scale = UIScreen.main.scale
            let canvasW = Int(round(self.bounds.width * scale))
            let canvasH = Int(round(self.bounds.height * scale))
            guard canvasW > 0 && canvasH > 0 else {
                DispatchQueue.main.async { self.processing = false }
                return
            }

            // Ensure mask float buffer
            if self.maskFloatBuf == nil || self.maskFloatBufCount != HW {
                self.maskFloatBuf?.deallocate()
                self.maskFloatBuf = UnsafeMutablePointer<Float>.allocate(capacity: HW)
                self.maskFloatBufCount = HW
            }
            guard let maskFloat = self.maskFloatBuf else {
                DispatchQueue.main.async { self.processing = false }
                return
            }

            // Ensure planar8 buffers for resizing
            let dstCount = canvasW * canvasH
            if self.planar8BufCount < max(HW, dstCount) {
                self.planar8BufA?.deallocate()
                self.planar8BufB?.deallocate()
                self.planar8BufA = UnsafeMutablePointer<UInt8>.allocate(capacity: max(HW, dstCount))
                self.planar8BufB = UnsafeMutablePointer<UInt8>.allocate(capacity: max(HW, dstCount))
                self.planar8BufCount = max(HW, dstCount)
            }
            guard let planarA = self.planar8BufA, let planarB = self.planar8BufB else {
                DispatchQueue.main.async { self.processing = false }
                return
            }

            // Collect valid detections with autoscaling & basic sanity checks
            var candidates: [(pred: Int, score: Float, coeffs: [Float])] = []
            for p in 0..<numPredictions {
                let score = detBuf[p * numFeatures + scoreIdx]
                if !score.isFinite || score < self.scoreThreshold { continue }

                // Read raw coeffs
                var raw = [Float](repeating: 0, count: K)
                var maxAbsRaw: Float = 0
                for k in 0..<K {
                    let v = detBuf[p * numFeatures + coeffStart + k]
                    if !v.isFinite { maxAbsRaw = Float.infinity; break }
                    raw[k] = v
                    maxAbsRaw = max(maxAbsRaw, abs(v))
                }
                if !raw[0].isFinite || maxAbsRaw == Float.infinity { continue }

                // Auto-scale heuristics
                var scaleFactor: Float = 1.0
                if maxAbsRaw > 400 { scaleFactor = 255.0 }
                else if maxAbsRaw > 80 { scaleFactor = 64.0 }

                let coeffs = raw.map { $0 / scaleFactor }

                // Reject nearly-uniform coefficient vectors
                var cmin = Float.greatestFiniteMagnitude, cmax = -Float.greatestFiniteMagnitude
                for v in coeffs { cmin = min(cmin, v); cmax = max(cmax, v) }
                if (cmax - cmin) < 1e-4 { continue }

                candidates.append((pred: p, score: score, coeffs: coeffs))
            }

            // Sort candidates by score and limit top-N decode
            candidates.sort { $0.score > $1.score }
            let topN = min(12, candidates.count)
            let toDecode = Array(candidates.prefix(topN))
            print("Found \(candidates.count) candidates, decoding top \(topN)")

            // Prepare prototypes matrix A for BLAS: we need A as (HW x K) row-major.
            // protoBuf currently channel-major: protoBuf[c*HW + i]
            // We will compute s = A * coeffs via cblas_sgemv by providing A as row-major with stride K.
            // Build A_rowMajor buffer once per frame (HW * K)
            let Acount = HW * K
            let Abytes = Acount * MemoryLayout<Float>.size
            let Aptr = UnsafeMutablePointer<Float>.allocate(capacity: Acount)
            // Fill row-major: for row i (pixel), columns k = protoBuf[k*HW + i]
            for i in 0..<HW {
                let baseA = i * K
                for k in 0..<K {
                    Aptr[baseA + k] = protoBuf[k * HW + i]
                }
            }

            var masksAlpha: [CGImage] = []
            var colors: [UIColor] = []

            // Per-detection decode using BLAS + vDSP sigmoid
            for (idx, c) in toDecode.enumerated() {
                let coeffs = c.coeffs
                // allocate coeff vector
                var coeffVec = coeffs // [Float] length K
                // s = A * coeffVec  (A is HW x K row-major)
                // Use cblas_sgemv: y = alpha*A*x + beta*y
                let alpha: Float = 1.0
                var s = [Float](repeating: 0, count: HW)
                // cblas_sgemv expects row-major layout if we use CblasRowMajor
                cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(HW), Int32(K), alpha, Aptr, Int32(K), &coeffVec, 1, 0, &s, 1)
                // apply sigmoid in place: s = 1 / (1 + exp(-s))
                // vDSP doesn't provide exact sigmoid; use vForce's exp then compute
                var negS = s.map { -$0 }
                var expNegS = [Float](repeating: 0, count: HW)
                vvexpf(&expNegS, &negS, [Int32(HW)])
                // s = 1/(1+expNegS)
                var one: Float = 1.0
                for i in 0..<HW {
                    s[i] = 1.0 / (1.0 + expNegS[i])
                }

                // copy s into maskFloat (threshold later)
                for i in 0..<HW { maskFloat[i] = s[i] }

                // compute coverage and threshold
                var validPixels = 0
                var minV: Float = 1, maxV: Float = 0
                for i in 0..<HW {
                    let v = maskFloat[i]
                    minV = min(minV, v); maxV = max(maxV, v)
                    if v > 0.5 { validPixels += 1 } else { maskFloat[i] = 0 }
                }
                let coveragePct = Float(validPixels) / Float(HW) * 100.0

                if self.debugSaveImages {
                    self.saveDebugFloatMask(maskFloat, width: self.protoW, height: self.protoH, name: "mask_proto_\(c.pred)", timestamp: "")
                }

                // Accept masks with moderate coverage
                let minCov = HW * 5 / 100
                let maxCov = HW * 90 / 100
                if validPixels < minCov || validPixels > maxCov {
                    let formattedCoverage = String(format: "%.1f", coveragePct)
                    print("Skip pred \(c.pred) cov \(formattedCoverage)%")
                    continue
                }

                // Resize to canvas with improved vImage pipeline (blur → convert → high-quality scale → post-blur)
                guard let alphaCG = self.resizeFloatMaskToAlphaImageOptimized(maskFloat: maskFloat, srcW: self.protoW, srcH: self.protoH, dstW: canvasW, dstH: canvasH, tmpU8A: planarA, tmpU8B: planarB) else {
                    print("Failed resize")
                    continue
                }

                masksAlpha.append(alphaCG)
                colors.append(self.colorForIndex(idx))
                let formattedCoverage = String(format: "%.1f", coveragePct)
                print("Added mask pred \(c.pred) cov \(formattedCoverage)%")
            }

            Aptr.deallocate()

            let frameTime = CFAbsoluteTimeGetCurrent() - frameStart
            print("Frame total time: \(Int(frameTime * 1000))ms, masks: \(masksAlpha.count)")

            // Composite or top-1
            var outImage: UIImage?
            if self.debugShowTopMask, let top = masksAlpha.first {
                outImage = self.composeSingleMask(top, color: colors.first ?? .red, canvasW: canvasW, canvasH: canvasH)
            } else {
                outImage = self.compositeMasksAdditive(masksAlpha: masksAlpha, colors: colors, canvasW: canvasW, canvasH: canvasH)
            }

            DispatchQueue.main.async {
                self.maskImageView.image = outImage
                self.processing = false
            }
        } // detectionQueue
    }

    // MARK: - Helpers

    // Convert Float16 MLMultiArray -> Float32 buffer using vImage (fast)
    private func copyFloat16MultiArrayToFloatBuffer(_ arr: MLMultiArray, dest: UnsafeMutablePointer<Float>) {
        let count = arr.count
        // If data type is float16 stored as UInt16 bits
        if arr.dataType == .float16 {
            let src = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(count), rowBytes: count * MemoryLayout<UInt16>.size)
            var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(dest), height: 1, width: vImagePixelCount(count), rowBytes: count * MemoryLayout<Float>.size)
            let err = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
            if err != kvImageNoError {
                // fallback elementwise conversion
                for i in 0..<count {
                    dest[i] = float32FromFloat16Bits(src[i])
                }
            }
        } else {
            // float32
            let src = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
            dest.initialize(from: src, count: arr.count)
        }
    }
    private func float32FromFloat16Bits(_ bits: UInt16) -> Float {
        var b = bits
        var out: Float = 0
        var sbuf = vImage_Buffer(data: &b, height: 1, width: 1, rowBytes: 2)
        var dbuf = vImage_Buffer(data: &out, height: 1, width: 1, rowBytes: 4)
        vImageConvert_Planar16FtoPlanarF(&sbuf, &dbuf, vImage_Flags(kvImageNoFlags))
        return out
    }

    // Optimized resize with pre/post blur to avoid striping
    // tmpU8A and tmpU8B are temporary planar8 buffers allocated by caller to avoid allocations
    private func resizeFloatMaskToAlphaImageOptimized(maskFloat: UnsafePointer<Float>, srcW: Int, srcH: Int, dstW: Int, dstH: Int, tmpU8A: UnsafeMutablePointer<UInt8>, tmpU8B: UnsafeMutablePointer<UInt8>) -> CGImage? {
        let srcCount = srcW * srcH

        // temp float buffer for blur
        let tmpFloat = UnsafeMutablePointer<Float>.allocate(capacity: srcCount)
        defer { tmpFloat.deallocate() }

        var srcF = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: maskFloat), height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW * MemoryLayout<Float>.size)
        var tmpF = vImage_Buffer(data: UnsafeMutableRawPointer(tmpFloat), height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW * MemoryLayout<Float>.size)

        // small 3x3 tent blur on float to remove prototype grid
        let kernel: [Float] = [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
        let err = vImageConvolve_PlanarF(&srcF, &tmpF, nil, 0, 0, kernel, 3, 3, 0, vImage_Flags(kvImageEdgeExtend))
        if err != kvImageNoError {
            tmpFloat.initialize(from: maskFloat, count: srcCount)
        }

        // Convert PlanarF -> Planar8 using tmpU8A
        var tmpFForConvert = vImage_Buffer(data: UnsafeMutableRawPointer(tmpFloat), height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW * MemoryLayout<Float>.size)
        var dstU8buf = vImage_Buffer(data: tmpU8A, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW)
        let convErr = vImageConvert_PlanarFtoPlanar8(&tmpFForConvert, &dstU8buf, 255.0, 0.0, vImage_Flags(kvImageNoFlags))
        if convErr != kvImageNoError { return nil }

        // High-quality scale to destination into tmpU8B
        var srcBuf = vImage_Buffer(data: tmpU8A, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW)
        var dstBuf = vImage_Buffer(data: tmpU8B, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW), rowBytes: dstW)
        let scaleErr = vImageScale_Planar8(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageHighQualityResampling))
        if scaleErr != kvImageNoError { return nil }

        // small post box blur to smooth any remaining artifacts
        let postBufPtr = tmpU8B
        let postOutPtr = tmpU8A // reuse other buffer for output
        var postIn = vImage_Buffer(data: postBufPtr, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW), rowBytes: dstW)
        var postOut = vImage_Buffer(data: postOutPtr, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW), rowBytes: dstW)
        let boxErr = vImageBoxConvolve_Planar8(&postIn, &postOut, nil, 0, 0, 3, 3, UInt8(0), vImage_Flags(kvImageEdgeExtend))
        let finalPtr = (boxErr == kvImageNoError) ? postOutPtr : postBufPtr

        guard let provider = CGDataProvider(data: CFDataCreate(nil, finalPtr, dstW * dstH)) else { return nil }
        let cs = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        guard let cg = CGImage(width: dstW, height: dstH, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: dstW, space: cs, bitmapInfo: bitmapInfo, provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { return nil }
        return cg
    }

    // Composite masks additively onto transparent canvas
    private func compositeMasksAdditive(masksAlpha: [CGImage], colors: [UIColor], canvasW: Int, canvasH: Int) -> UIImage? {
        guard masksAlpha.count == colors.count else { return nil }
        let scale = UIScreen.main.scale
        let size = CGSize(width: CGFloat(canvasW)/scale, height: CGFloat(canvasH)/scale)
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        guard let ctx = UIGraphicsGetCurrentContext() else { UIGraphicsEndImageContext(); return nil }
        ctx.clear(CGRect(origin: .zero, size: size))
        for i in 0..<masksAlpha.count {
            let alphaImg = masksAlpha[i]
            let color = colors[i]
            ctx.saveGState()
            // clip to alpha mask and fill with color
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

    // Compose single mask (top-1) as colored image
    private func composeSingleMask(_ alpha: CGImage, color: UIColor, canvasW: Int, canvasH: Int) -> UIImage? {
        let scale = UIScreen.main.scale
        let size = CGSize(width: CGFloat(canvasW)/scale, height: CGFloat(canvasH)/scale)
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        guard let ctx = UIGraphicsGetCurrentContext() else { UIGraphicsEndImageContext(); return nil }
        ctx.clear(CGRect(origin: .zero, size: size))
        ctx.saveGState()
        ctx.clip(to: CGRect(x: 0, y: 0, width: size.width, height: size.height), mask: alpha)
        ctx.setFillColor(color.cgColor)
        ctx.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
        ctx.restoreGState()
        let out = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return out
    }

    // Save float mask as debug PNG (proto resolution)
    private func saveDebugFloatMask(_ maskFloat: UnsafePointer<Float>, width: Int, height: Int, name: String, timestamp: String = "") {
        guard debugSaveImages else { return }
        let count = width * height
        let data = UnsafeMutablePointer<UInt8>.allocate(capacity: count)
        defer { data.deallocate() }
        for i in 0..<count {
            let v = max(0.0, min(1.0, maskFloat[i]))
            data[i] = UInt8(v * 255.0)
        }
        guard let provider = CGDataProvider(data: CFDataCreate(nil, data, count)) else { return }
        let cs = CGColorSpaceCreateDeviceGray()
        guard let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width, space: cs, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { return }
        let ui = UIImage(cgImage: cg, scale: UIScreen.main.scale, orientation: .up)
        saveDebugImage(ui, name: name, timestamp: timestamp)
    }

    // Save UIImage to Photos (debug)
    private func saveDebugImage(_ image: UIImage, name: String, timestamp: String = "") {
        guard debugSaveImages else { return }
        let ts = timestamp.isEmpty ? String(format: "%.0f", Date().timeIntervalSince1970) : timestamp
        let label = "\(name)_\(ts)"
        // overlay label
        let final = addDebugLabel(to: image, label: label)
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            if status == .authorized {
                PHPhotoLibrary.shared().performChanges({
                    PHAssetChangeRequest.creationRequestForAsset(from: final)
                }) { ok, err in
                    if ok { print("Saved debug image: \(label)") } else { print("Save failed: \(err?.localizedDescription ?? "err")") }
                }
            } else { print("No photo permission") }
        }
    }
    private func addDebugLabel(to image: UIImage, label: String) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: image.size)
        return renderer.image { ctx in
            image.draw(at: .zero)
            let r = CGRect(x: 6, y: 6, width: image.size.width - 12, height: 30)
            ctx.cgContext.setFillColor(UIColor.black.withAlphaComponent(0.6).cgColor)
            ctx.cgContext.fill(r)
            let attrs: [NSAttributedString.Key: Any] = [.foregroundColor: UIColor.white, .font: UIFont.systemFont(ofSize: min(18, image.size.width/20))]
            (label as NSString).draw(in: r.insetBy(dx: 6, dy: 4), withAttributes: attrs)
        }
    }

    // MARK: - Utility conversion: CVPixelBuffer -> MLMultiArray (channels-first Float32)
    // Replace with your optimized version if available.
    func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> MLMultiArray? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        // Create Float32 MLMultiArray [1,3,height,width]
        guard let arr = try? MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32) else { return nil }
        // assume kCVPixelFormatType_32BGRA
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        for y in 0..<min(srcH, height) {
            let row = base.advanced(by: y * rowBytes)
            for x in 0..<min(srcW, width) {
                let px = row.advanced(by: x * 4)
                let b = Float(px.load(fromByteOffset: 0, as: UInt8.self)) / 255.0
                let g = Float(px.load(fromByteOffset: 1, as: UInt8.self)) / 255.0
                let r = Float(px.load(fromByteOffset: 2, as: UInt8.self)) / 255.0
                let rIndex = 0*arr.strides[0].intValue + 0*arr.strides[1].intValue + y*arr.strides[2].intValue + x*arr.strides[3].intValue
                let gIndex = 0*arr.strides[0].intValue + 1*arr.strides[1].intValue + y*arr.strides[2].intValue + x*arr.strides[3].intValue
                let bIndex = 0*arr.strides[0].intValue + 2*arr.strides[1].intValue + y*arr.strides[2].intValue + x*arr.strides[3].intValue
                arr[rIndex] = NSNumber(value: r)
                arr[gIndex] = NSNumber(value: g)
                arr[bIndex] = NSNumber(value: b)
            }
        }
        return arr
    }

    // Color palette
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
