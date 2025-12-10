// SmartyPantsProcessing.swift
// Main processing pipeline: processFrame and generateCutoutTwoStage

import CoreML
import CoreImage
import UIKit

extension SmartyPantsContainerView {
    
    // MARK: - Process Frame
    func processFrame(_ pixelBuffer: CVPixelBuffer) {
        let frameStart = Date()
        
        guard let model = mlModel else { return }
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !isProcessing else { return }
        lastProcessTime = now
        isProcessing = true

        if debugMode {
            print("\n🚒 ===== NEW FRAME @ \(now.timeIntervalSince1970) =====")
            print("📌 ========== STAGE 1: FULL FRAME ==========")
        }
        setProgress(0.2, text: "Preprocessing frame…")

        // STAGE 1: Preprocess
        let stage1PreStart = Date()
        guard let resized = resizePixelBufferToSquare(pixelBuffer, size: kModelInputSize) else {
            isProcessing = false
            return
        }
        guard let inputArray = pixelBufferToMLMultiArray(resized) else {
            isProcessing = false
            return
        }
        if debugMode {
            let stage1PreEnd = Date()
            print(String(format: "⏱ Stage1 preprocess (letterbox+toMultiArray): %.2f ms", stage1PreEnd.timeIntervalSince(stage1PreStart) * 1000.0))
        }

        setProgress(0.35, text: "Running detection…")

        // STAGE 1: Inference
        let stage1InfStart = Date()
        guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]) else {
            isProcessing = false
            return
        }
        guard let output = try? model.prediction(from: inputProvider) else {
            isProcessing = false
            return
        }
        if debugMode {
            let stage1InfEnd = Date()
            print(String(format: "⏱ Stage1 model.prediction: %.2f ms", stage1InfEnd.timeIntervalSince(stage1InfStart) * 1000.0))
            let names = output.featureNames.joined(separator: ", ")
            print("📤 Model outputs: \(names)")
        }

        var detectionsArray: MLMultiArray?
        if let arr = output.featureValue(for: "var_1432")?.multiArrayValue {
            detectionsArray = arr
        } else if let arr = output.featureValue(for: "var_2421")?.multiArrayValue {
            detectionsArray = arr
        } else {
            for name in output.featureNames {
                if let arr = output.featureValue(for: name)?.multiArrayValue {
                    let shape = arr.shape.map { $0.intValue }
                    if shape.count == 3 && shape[0] == 1 && shape[1] > 100 {
                        detectionsArray = arr
                        if debugMode { print("   → Using '\(name)' as detections: \(shape)") }
                        break
                    }
                }
            }
        }

        guard let detArray = detectionsArray else {
            isProcessing = false
            return
        }

        guard let prototypesArray = output.featureValue(for: "p")?.multiArrayValue else {
            isProcessing = false
            return
        }

        let decodeStart = Date()
        let stage1DetectionsFull = extractDetections(from: detArray)
        if debugMode {
            let decodeEnd = Date()
            print("📊 Stage 1: \(stage1DetectionsFull.count) detections")
            print(String(format: "⏱ Stage1 detection decode: %.2f ms", decodeEnd.timeIntervalSince(decodeStart) * 1000.0))
        }
        
        // Select primary by confidence × bbox area
        var bestScore: Float = 0
        var primaryBBox: DetectionSmarty? = nil

        for det in stage1DetectionsFull {
            let bboxArea = det.width * det.height
            let score = det.confidence * bboxArea
            if score > bestScore {
                bestScore = score
                primaryBBox = det
            }
        }

        guard let primary = primaryBBox else {
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
            return
        }

        setProgress(0.55, text: "Refining crop…")

        // STAGE 2
        if debugMode { print("\n📌 ========== STAGE 2: CROPPED ==========") }

        var stage2Detections: [DetectionSmarty] = []
        var stage2Prototypes: MLMultiArray? = nil

        let stage2Start = Date()
        if let croppedBuffer = cropPixelBuffer(pixelBuffer, toBBox: primary, padding: 0.0),
           let resizedCrop = resizePixelBufferToSquare(croppedBuffer, size: kModelInputSize),
           let cropInputArray = pixelBufferToMLMultiArray(resizedCrop),
           let cropInputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": cropInputArray]) {
            
            let expectedCount = 1 * 3 * kModelInputSize * kModelInputSize
            guard cropInputArray.count == expectedCount else {
                if debugMode { print("⚠️ Stage2: bad input count:", cropInputArray.count) }
                self.isProcessing = false
                return
            }
            
            let options = MLPredictionOptions()
            options.usesCPUOnly = false
            
            autoreleasepool {
                let stage2InfStart = Date()
                if let cropOutput = try? model.prediction(from: cropInputProvider, options: options) {
                    if debugMode {
                        let stage2InfEnd = Date()
                        print(String(format: "⏱ Stage2 model.prediction: %.2f ms", stage2InfEnd.timeIntervalSince(stage2InfStart) * 1000.0))
                    }
                    
                    var cropDetArray: MLMultiArray?
                    if let arr = cropOutput.featureValue(for: "var_2421")?.multiArrayValue {
                        cropDetArray = arr
                    } else {
                        for name in cropOutput.featureNames {
                            if let arr = cropOutput.featureValue(for: name)?.multiArrayValue {
                                let shape = arr.shape.map { $0.intValue }
                                if shape.count == 3 && shape[0] == 1 && shape[1] > 100 {
                                    cropDetArray = arr
                                    break
                                }
                            }
                        }
                    }
                    
                    if let detArray = cropDetArray,
                       let protoArray = cropOutput.featureValue(for: "p")?.multiArrayValue {
                        let s2DecodeStart = Date()
                        stage2Detections = extractDetections(from: detArray)
                        stage2Prototypes = protoArray
                        if debugMode {
                            let s2DecodeEnd = Date()
                            print("📊 Stage 2: \(stage2Detections.count) detections")
                            print(String(format: "⏱ Stage2 detection decode: %.2f ms", s2DecodeEnd.timeIntervalSince(s2DecodeStart) * 1000.0))
                        }
                    }
                }
            }
        } else {
            if debugMode { print("⚠️ Stage 2: Failed to crop/process") }
        }
        
        if debugMode {
            let stage2End = Date()
            print(String(format: "⏱ Stage2 total (crop+preprocess+infer+decode): %.2f ms",
                         stage2End.timeIntervalSince(stage2Start) * 1000.0))
        }
        
        let rawDetections = extractDetections(from: detArray)
        let uniqueDetections = applyNMS(rawDetections, iouThreshold: 0.6)
        let stage2KeptStage2 = applyNMS(uniqueDetections, iouThreshold: 0.6)
        
        if rawDetections.isEmpty && stage2Detections.isEmpty {
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
            return
        }

        setProgress(0.8, text: "Building mask…")

        let cutoutStart = Date()
        generateCutoutTwoStage(
            stage1Detections: uniqueDetections,
            stage1Prototypes: prototypesArray,
            stage2Detections: stage2KeptStage2,
            stage2Prototypes: stage2Prototypes,
            primaryBBox: primary,
            originalImage: pixelBuffer
        )
        
        if debugMode {
            let cutoutEnd = Date()
            print(String(format: "⏱ generateCutoutTwoStage call: %.2f ms", cutoutEnd.timeIntervalSince(cutoutStart) * 1000.0))
            print(String(format: "🚒 Frame total (processFrame): %.2f ms", cutoutEnd.timeIntervalSince(frameStart) * 1000.0))
        }
    }

    // MARK: - Fast prototype buffer builder (stride-aware, no subscripts)
    /// Flattens MLMultiArray prototypes [1, C, Hp, Wp] into [Float] shaped as C x (Hp*Wp)
    /// Layout matches existing vDSP usage: out[c * (Hp*Wp) + (y*Wp + x)]
    func makePrototypeBufferFast(from prototypes: MLMultiArray, C: Int, Hp: Int, Wp: Int) -> [Float] {
        let spatial = Hp * Wp
        var out = [Float](repeating: 0, count: C * spatial)

        // Expect shape [1, C, Hp, Wp]
        let strides = prototypes.strides.map { $0.intValue }
        guard strides.count == 4 else { return out }

        let sN = strides[0]
        let sC = strides[1]
        let sH = strides[2]
        let sW = strides[3]

        // We assume N == 1, so base offset for N is 0
        switch prototypes.dataType {
        case .float32:
            let base = prototypes.dataPointer.assumingMemoryBound(to: Float.self)
            for c in 0..<C {
                let dstCBase = c * spatial
                let srcCBase = c * sC
                for y in 0..<Hp {
                    let dstRow = dstCBase + y * Wp
                    let srcRow = srcCBase + y * sH
                    var dstIdx = dstRow
                    var srcIdx = srcRow
                    // Unroll inner loop by 1 (straight copy using stride)
                    for _ in 0..<Wp {
                        out[dstIdx] = base[srcIdx]
                        dstIdx += 1
                        srcIdx += sW
                    }
                }
            }
        case .float16:
            // Swift Float16 available; convert per element
            let base = prototypes.dataPointer.assumingMemoryBound(to: Float16.self)
            for c in 0..<C {
                let dstCBase = c * spatial
                let srcCBase = c * sC
                for y in 0..<Hp {
                    let dstRow = dstCBase + y * Wp
                    let srcRow = srcCBase + y * sH
                    var dstIdx = dstRow
                    var srcIdx = srcRow
                    for _ in 0..<Wp {
                        out[dstIdx] = Float(base[srcIdx])
                        dstIdx += 1
                        srcIdx += sW
                    }
                }
            }
        default:
            // Fallback: do nothing (returns zeros). Extend if other types are possible.
            break
        }

        return out
    }

    // MARK: - Generate Cutout Two Stage
    func generateCutoutTwoStage(
        stage1Detections: [DetectionSmarty],
        stage1Prototypes: MLMultiArray,
        stage2Detections: [DetectionSmarty],
        stage2Prototypes: MLMultiArray?,
        primaryBBox: DetectionSmarty,
        originalImage: CVPixelBuffer
    ) {
        let funcStart = Date()
        
        let shape = stage1Prototypes.shape.map { $0.intValue }
        let C = shape[1]
        let Hp = shape[2]
        let Wp = shape[3]
        let spatial = Hp * Wp

        if debugMode {
            print("\n🌈 Generating TWO-STAGE UNION cutout")
            print("   Stage 1: \(stage1Detections.count) detections")
            print("   Stage 2: \(stage2Detections.count) detections (Stage2 coords)")
            print("📐 Prototype shape: C=\(C), H=\(Hp), W=\(Wp)")
        }

        var mappedStage2Detections: [DetectionSmarty] = []

        // Stage 1 prototype buffer
        let protoStage1Start = Date()
        let protoMatrix1 = makePrototypeBufferFast(from: stage1Prototypes, C: C, Hp: Hp, Wp: Wp)
        if debugMode {
            let protoStage1End = Date()
            print(String(format: "⏱ Stage1 prototype buffer build (Accelerate): %.2f ms",
                         protoStage1End.timeIntervalSince(protoStage1Start) * 1000.0))
        }

        var globalMask = [Float](repeating: 0, count: spatial)

        // Map Stage 2 detections to Stage 1 coords
        if let _ = stage2Prototypes, !stage2Detections.isEmpty {
            let padding: Float = 0.1
            let cropX1 = max(0, primaryBBox.x - primaryBBox.width / 2 * (1 + padding))
            let cropY1 = max(0, primaryBBox.y - primaryBBox.height / 2 * (1 + padding))
            let cropX2 = min(kModelInputSizeFloat, primaryBBox.x + primaryBBox.width / 2 * (1 + padding))
            let cropY2 = min(kModelInputSizeFloat, primaryBBox.y + primaryBBox.height / 2 * (1 + padding))
            let cropW = cropX2 - cropX1
            let cropH = cropY2 - cropY1
            let s2ToS1ScaleX = cropW / kModelInputSizeFloat
            let s2ToS1ScaleY = cropH / kModelInputSizeFloat

            for det in stage2Detections {
                let newX = cropX1 + det.x * s2ToS1ScaleX
                let newY = cropY1 + det.y * s2ToS1ScaleY
                let newW = det.width * s2ToS1ScaleX
                let newH = det.height * s2ToS1ScaleY

                let mapped = DetectionSmarty(
                    x: newX, y: newY, width: newW, height: newH,
                    confidence: det.confidence, classIdx: det.classIdx,
                    className: det.className, maskCoeffs: det.maskCoeffs
                )
                mappedStage2Detections.append(mapped)
            }
        }

        let allDetections = stage1Detections + mappedStage2Detections

        // Build globalMask
        let buildStart = Date()
        buildStitchedMask(
            globalMask: &globalMask,
            allDetections: allDetections,
            protoMatrix: protoMatrix1,
            primaryBBox: primaryBBox,
            C: C, Wp: Wp, Hp: Hp
        )

        if debugMode {
            let buildEnd = Date()
            var rawCount = 0
            for i in 0..<spatial { if globalMask[i] > 0 { rawCount += 1 } }
            print(String(format: "⏱ buildGlobalMaskWithOverlapFilter: %.2f ms", buildEnd.timeIntervalSince(buildStart) * 1000.0))
            print("📊 After overlap filter: \(rawCount)/\(spatial) pixels (\(String(format: "%.1f", Float(rawCount)/Float(spatial)*100))%)")
        }
        
        // Update perimeter tracking
        var maskArea = 0
        for i in 0..<spatial { if globalMask[i] > 0 { maskArea += 1 } }
        if maskArea > bestPerimeterArea {
            bestPerimeterMask = globalMask
            bestPerimeterArea = maskArea
        }

        if debugMode {
            let maskCopy = globalMask
            DispatchQueue.main.async {
                self.saveMaskToPhotos(maskCopy, width: Wp, height: Hp, label: "globalMask_raw")
            }
        }

        // Fill inside perimeter
        let ppStart = Date()
        fillInsidePerimeter(&globalMask, width: Wp, height: Hp)
        
        var finalPixelCount = 0
        for i in 0..<spatial { if globalMask[i] > 0 { finalPixelCount += 1 } }

        if debugMode {
            let ppEnd = Date()
            saveMaskToPhotos(globalMask, width: Wp, height: Hp, label: "globalMask_filled")
            print(String(format: "⏱ fillInsidePerimeter: %.2f ms", ppEnd.timeIntervalSince(ppStart) * 1000.0))
            print("📊 FINAL MASK: \(finalPixelCount)/\(spatial) pixels (\(String(format: "%.1f", Float(finalPixelCount)/Float(spatial)*100))%)")
        }

        // RENDER TO IMAGE
        autoreleasepool {
            let renderStart = Date()
            let ciImage = CIImage(cvPixelBuffer: originalImage)
            let width = CVPixelBufferGetWidth(originalImage)
            let height = CVPixelBufferGetHeight(originalImage)

            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
                if debugMode { print("❌ Failed to create CGImage") }
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            guard let ctx = CGContext(
                data: nil, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                if debugMode { print("❌ Failed to create CGContext") }
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

            guard let data = ctx.data else {
                if debugMode { print("❌ CGContext has no data") }
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

            let scaleX = Float(Wp) / Float(width)
            let scaleY = Float(Hp) / Float(height)

            // Apply mask as alpha
            for py in 0..<height {
                let my = min(max(Int(Float(py) * scaleY), 0), Hp - 1)
                let maskRowOffset = my * Wp
                let rowBase = pixels.advanced(by: py * width * 4)

                for px in 0..<width {
                    let mx = min(max(Int(Float(px) * scaleX), 0), Wp - 1)
                    let maskIdx = maskRowOffset + mx
                    let pixelPtr = rowBase.advanced(by: px * 4)
                    
                    if globalMask[maskIdx] > 0 {
                        pixelPtr[3] = 255
                    } else {
                        pixelPtr[0] = 0; pixelPtr[1] = 0; pixelPtr[2] = 0; pixelPtr[3] = 0
                    }
                }
            }
            
            if debugMode {
                drawPerimeterOutline(on: pixels, mask: globalMask,
                                     maskWidth: Wp, maskHeight: Hp,
                                     imageWidth: width, imageHeight: height)
            }

            drawLabelsAndBoxes(ctx: ctx, stage1: stage1Detections, stage2: mappedStage2Detections,
                               imageWidth: width, imageHeight: height, drawBoxes: debugMode)

            if debugMode {
                let renderEnd = Date()
                print(String(format: "⏱ Rendering: %.2f ms", renderEnd.timeIntervalSince(renderStart) * 1000.0))
                print(String(format: "⏱ generateCutoutTwoStage total: %.2f ms", renderEnd.timeIntervalSince(funcStart) * 1000.0))
                print("✅ ==================== FRAME COMPLETE ====================\n")
            }

            if let outCG = ctx.makeImage() {
                DispatchQueue.main.async {
                    self.finishFirstDetectionIfNeeded()
                    self.maskImageView.image = UIImage(cgImage: outCG)
                    self.isProcessing = false
                }
            } else {
                DispatchQueue.main.async { self.isProcessing = false }
            }
        }
    }
}


