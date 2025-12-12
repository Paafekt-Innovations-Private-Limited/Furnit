// SmartyPantsProcessing.swift
// Main processing pipeline: processFrame and generateCutoutTwoStage

import CoreML
import CoreImage
import UIKit
import Accelerate

extension SmartyPantsContainerView {
    
    // MARK: - Process Frame
    func processFrame(_ pixelBuffer: CVPixelBuffer) {
        let frameStart = Date()
        
        guard let model = mlModel else { return }
        let now = Date()
        if debugMode { print("[SP TIMING] ===== New frame @ \(now.timeIntervalSince1970) =====") }
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
        if debugMode {
            let rw = CVPixelBufferGetWidth(resized)
            let rh = CVPixelBufferGetHeight(resized)
            let rbpr = CVPixelBufferGetBytesPerRow(resized)
            print(String(format: "📐 Resized buffer: %dx%d (bytesPerRow=%d)", rw, rh, rbpr))
            print("📐 kModelInputSize: \(kModelInputSize)")
        }
        guard let inputArray = pixelBufferToMLMultiArray(resized) else {
            isProcessing = false
            return
        }
        
        // Ensure model gets Float32 as per Netron graph (Float32 [1,3,960,960])
        var inputArrayF32: MLMultiArray = inputArray
        if inputArray.dataType != .float32 {
            if let converted = upcastToFloat32(inputArray) {
                if debugMode { print("🔁 Upcasted input MLMultiArray from \(inputArray.dataType) to Float32") }
                inputArrayF32 = converted
            } else {
                if debugMode { print("⚠️ Failed to upcast input to Float32; proceeding with \(inputArray.dataType)") }
            }
        }
        
        if debugMode {
            let shape = inputArrayF32.shape.map { $0.intValue }
            let strides = inputArrayF32.strides.map { $0.intValue }
            print("🧮 MLMultiArray shape: \(shape) strides: \(strides) type: \(inputArrayF32.dataType)")
            print("🧮 MLMultiArray count: \(inputArrayF32.count), expected: \(1 * 3 * kModelInputSize * kModelInputSize)")
            if inputArrayF32.count != 1 * 3 * kModelInputSize * kModelInputSize {
                print("⚠️ Input array element count mismatch — aborting frame early")
            }
        }
        if debugMode {
            let stage1PreEnd = Date()
            print(String(format: "⏱ Stage1 preprocess (resizePixelBufferToSquare+toMultiArray): %.2f ms", stage1PreEnd.timeIntervalSince(stage1PreStart) * 1000.0))
        }

        setProgress(0.35, text: "Running detection…")

        // STAGE 1: Inference
        let stage1InfStart = Date()
        let inputProvider: MLDictionaryFeatureProvider
        do {
            if debugMode {
                // Print model input constraints if available
                let inputs = model.modelDescription.inputDescriptionsByName
                if let imgDesc = inputs["image"], let cons = imgDesc.multiArrayConstraint {
                    let expShape = cons.shape.map { $0.intValue }
                    print("📥 Model expects 'image' shape: \(expShape) dataType: \(cons.dataType)")
                } else if let imgDesc = inputs["image"], let imgCons = imgDesc.imageConstraint {
                    print("📥 Model expects 'image' imageConstraint: \(imgCons.pixelsWide)x\(imgCons.pixelsHigh) \(imgCons.pixelFormatType)")
                } else {
                    print("📥 Could not find detailed constraints for 'image' input")
                }
            }
            inputProvider = try MLDictionaryFeatureProvider(dictionary: ["image": inputArrayF32])
        } catch {
            if debugMode { print("❌ MLDictionaryFeatureProvider error: \(error)") }
            isProcessing = false
            return
        }
        let output: MLFeatureProvider
        do {
            // Use CPU-only to avoid ANE "No memory object bound to port" crashes
            let options = MLPredictionOptions()
            options.usesCPUOnly = true
            output = try model.prediction(from: inputProvider, options: options)
        } catch {
            if debugMode { print("❌ model.prediction error: \(error)") }
            isProcessing = false
            return
        }
        if debugMode {
            let stage1InfEnd = Date()
            print(String(format: "⏱ Stage1 model.prediction: %.2f ms", stage1InfEnd.timeIntervalSince(stage1InfStart) * 1000.0))
            let names = output.featureNames.joined(separator: ", ")
            print("📤 Model outputs: \(names)")
            for name in output.featureNames {
                if let arr = output.featureValue(for: name)?.multiArrayValue {
                    let shp = arr.shape.map { $0.intValue }
                    print("   • Output tensor '\(name)' shape: \(shp) type: \(arr.dataType)")
                }
            }
        }

        var detectionsArray: MLMultiArray?
        if let arr = output.featureValue(for: "var_2497")?.multiArrayValue {
            detectionsArray = arr
            if debugMode { print("   → Using 'var_2497' as detections: \((arr.shape.map { $0.intValue }))") }
        } else if let arr = output.featureValue(for: "var_1432")?.multiArrayValue {
            detectionsArray = arr
        } else if let arr = output.featureValue(for: "var_2421")?.multiArrayValue {
            detectionsArray = arr
        } else {
            for name in output.featureNames {
                if let arr = output.featureValue(for: name)?.multiArrayValue {
                    let shape = arr.shape.map { $0.intValue }
                    if shape.count == 3 && shape[0] == 1 {
                        // Accept either [1, features, anchors] or [1, anchors, features]
                        if shape[1] > 100 || shape[2] > 100 {
                            detectionsArray = arr
                            if debugMode { print("   → Using '\(name)' as detections: \(shape)") }
                            break
                        }
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
        if debugMode {
            let pShape = prototypesArray.shape.map { $0.intValue }
            print("🧪 Prototypes 'p' shape: \(pShape) type: \(prototypesArray.dataType)")
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
           let resizedCrop = resizePixelBufferToSquare(croppedBuffer, size: kModelInputSize) {
            
            if debugMode {
                let cw = CVPixelBufferGetWidth(resizedCrop)
                let ch = CVPixelBufferGetHeight(resizedCrop)
                print("📐 Stage2 resized crop: \(cw)x\(ch)")
            }
            
            if let cropInputArray = pixelBufferToMLMultiArray(resizedCrop) {
                var cropInputArrayF32: MLMultiArray = cropInputArray
                if cropInputArray.dataType != .float32 {
                    if let converted = upcastToFloat32(cropInputArray) {
                        if debugMode { print("🔁 Upcasted Stage2 input MLMultiArray from \(cropInputArray.dataType) to Float32") }
                        cropInputArrayF32 = converted
                    } else {
                        if debugMode { print("⚠️ Failed to upcast Stage2 input to Float32; proceeding with \(cropInputArray.dataType)") }
                    }
                }
                
                if let cropInputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": cropInputArrayF32]) {
                    
                    let expectedCount = 1 * 3 * kModelInputSize * kModelInputSize
                    guard cropInputArrayF32.count == expectedCount else {
                        if debugMode { print("⚠️ Stage2: bad input count:", cropInputArrayF32.count) }
                        self.isProcessing = false
                        return
                    }
                    
                    let options = MLPredictionOptions()
                    options.usesCPUOnly = true
                    
                    autoreleasepool {
                        let stage2InfStart = Date()
                        if let cropOutput = try? model.prediction(from: cropInputProvider, options: options) {
                            if debugMode {
                                let stage2InfEnd = Date()
                                print(String(format: "⏱ Stage2 model.prediction: %.2f ms", stage2InfEnd.timeIntervalSince(stage2InfStart) * 1000.0))
                            }
                            
                            var cropDetArray: MLMultiArray?
                            if let arr = cropOutput.featureValue(for: "var_2497")?.multiArrayValue {
                                cropDetArray = arr
                            } else if let arr = cropOutput.featureValue(for: "var_2421")?.multiArrayValue {
                                cropDetArray = arr
                            } else {
                                for name in cropOutput.featureNames {
                                    if let arr = cropOutput.featureValue(for: name)?.multiArrayValue {
                                        let shape = arr.shape.map { $0.intValue }
                                        if shape.count == 3 && shape[0] == 1 {
                                            if shape[1] > 100 || shape[2] > 100 {
                                                cropDetArray = arr
                                                break
                                            }
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
                }
            } else {
                if debugMode { print("⚠️ Stage 2: Failed to create MLMultiArray from resized crop") }
            }
        } else {
            if debugMode { print("⚠️ Stage 2: Failed to crop/process") }
        }
        
        if debugMode {
            let stage2End = Date()
            print(String(format: "⏱ Stage2 total (crop+preprocess+infer+decode): %.2f ms",
                         stage2End.timeIntervalSince(stage2Start) * 1000.0))
        }
        
        // Reuse decoded detections instead of decoding again
        let rawDetections = stage1DetectionsFull
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
        let cutoutEnd = Date()
        print(String(format: "[SP TIMING] generateCutoutTwoStage call: %.2f ms", cutoutEnd.timeIntervalSince(cutoutStart) * 1000.0))
        
        if debugMode {
            let frameEnd = Date()
            print(String(format: "[SP TIMING] Frame total (processFrame): %.2f ms", frameEnd.timeIntervalSince(frameStart) * 1000.0))
        }
    }

    func makePrototypeBufferFast(from prototypes: MLMultiArray, C: Int, Hp: Int, Wp: Int) -> [Float] {
        let spatial = Hp * Wp
        let total = C * spatial
        var out = [Float](repeating: 0, count: total)

        // Expect shape [1, C, Hp, Wp]
        let strides = prototypes.strides.map { $0.intValue }
        guard strides.count == 4 else { return out }

        let sN = strides[0]
        let sC = strides[1]
        let sH = strides[2]
        let sW = strides[3]

        // Fast path: contiguous [1, C, Hp, Wp] layout
        // N: C*Hp*Wp, C: Hp*Wp, H: Wp, W: 1
        if sN == C * spatial && sC == spatial && sH == Wp && sW == 1 {
            switch prototypes.dataType {
            case .float32:
                // Direct bulk copy via BLAS
                let src = prototypes.dataPointer.assumingMemoryBound(to: Float.self)
                out.withUnsafeMutableBufferPointer { dst in
                    if let dstBase = dst.baseAddress {
                        cblas_scopy(Int32(total), src, 1, dstBase, 1)
                    }
                }
                if debugMode {
                    print("📦 [PROTO] BLAS scopy (.float32 contiguous)")
                }
                return out

            case .float16:
                // Single linear pass: Float16 -> Float
                let src = prototypes.dataPointer.assumingMemoryBound(to: Float16.self)
                out.withUnsafeMutableBufferPointer { dst in
                    if let dstBase = dst.baseAddress {
                        var i = 0
                        while i < total {
                            dstBase[i] = Float(src[i])
                            i += 1
                        }
                    }
                }
                if debugMode {
                    print("📦 [PROTO] Fast contiguous convert (.float16)")
                }
                return out

            default:
                if debugMode {
                    print("⚠️ [PROTO] Unsupported MLMultiArray dataType in contiguous path: \(prototypes.dataType)")
                }
                return out
            }
        }

        // Fallback: general stride-aware path
        switch prototypes.dataType {
        case .float32:
            let base = prototypes.dataPointer.assumingMemoryBound(to: Float.self)

            out.withUnsafeMutableBufferPointer { dst in
                guard let dstBase = dst.baseAddress else { return }

                for c in 0..<C {
                    let dstCBase = c * spatial
                    let srcCBase = c * sC

                    for y in 0..<Hp {
                        let dstRow = dstCBase + y * Wp
                        let srcRow = srcCBase + y * sH

                        // Use BLAS to copy one row with stride sW
                        cblas_scopy(
                            Int32(Wp),
                            base + srcRow,
                            Int32(sW),
                            dstBase + dstRow,
                            1
                        )
                    }
                }
            }

            if debugMode {
                print("📦 [PROTO] BLAS scopy (.float32 strided)")
            }

        case .float16:
            let base = prototypes.dataPointer.assumingMemoryBound(to: Float16.self)

            out.withUnsafeMutableBufferPointer { dst in
                guard let dstBase = dst.baseAddress else { return }

                for c in 0..<C {
                    let dstCBase = c * spatial
                    let srcCBase = c * sC

                    for y in 0..<Hp {
                        let dstRow = dstCBase + y * Wp
                        let srcRow = srcCBase + y * sH

                        var dstIdx = dstRow
                        var srcIdx = srcRow
                        for _ in 0..<Wp {
                            dstBase[dstIdx] = Float(base[srcIdx])
                            dstIdx += 1
                            srcIdx += sW
                        }
                    }
                }
            }

            if debugMode {
                print("📦 [PROTO] fallback convert (.float16 strided)")
            }

        default:
            if debugMode {
                print("⚠️ [PROTO] Unsupported MLMultiArray dataType in strided path: \(prototypes.dataType)")
            }
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

        // Build globalMask using Accelerate-backed overlap filter (see SmartyPantsMask.swift)
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
            var sum: Float = 0
            globalMask.withUnsafeBufferPointer { ptr in
                if let base = ptr.baseAddress {
                    vDSP_sve(base, 1, &sum, vDSP_Length(spatial))
                }
            }
            let rawCount = Int(sum.rounded())
            print(String(format: "⏱ buildGlobalMaskWithOverlapFilter: %.2f ms",
                         buildEnd.timeIntervalSince(buildStart) * 1000.0))
            print(String(format: "[SP TIMING] buildGlobalMaskWithOverlapFilter: %.2f ms",
                         buildEnd.timeIntervalSince(buildStart) * 1000.0))
            print("📊 After overlap filter: \(rawCount)/\(spatial) pixels (\(String(format: "%.1f", Float(rawCount)/Float(spatial)*100))%)")
        }
        
        // Update perimeter tracking (area of current mask)
        var areaSum: Float = 0
        globalMask.withUnsafeBufferPointer { ptr in
            if let base = ptr.baseAddress {
                vDSP_sve(base, 1, &areaSum, vDSP_Length(spatial))
            }
        }
        let maskArea = Int(areaSum.rounded())
        
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
        
        var finalSum: Float = 0
        globalMask.withUnsafeBufferPointer { ptr in
            if let base = ptr.baseAddress {
                vDSP_sve(base, 1, &finalSum, vDSP_Length(spatial))
            }
        }
        let finalPixelCount = Int(finalSum.rounded())

        if debugMode {
            let ppEnd = Date()
            print(String(format: "⏱ fillInsidePerimeter: %.2f ms", ppEnd.timeIntervalSince(ppStart) * 1000.0))
            print(String(format: "[SP TIMING] fillInsidePerimeter: %.2f ms", ppEnd.timeIntervalSince(ppStart) * 1000.0))
            saveMaskToPhotos(globalMask, width: Wp, height: Hp, label: "globalMask_filled")
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
    
    // Helper: Upcast MLMultiArray to Float32 for models that require Float32 input
    private func upcastToFloat32(_ array: MLMultiArray) -> MLMultiArray? {
        if array.dataType == .float32 { return array }
        guard let out = try? MLMultiArray(shape: array.shape, dataType: .float32) else { return nil }
        let total = array.count
        if array.dataType == .float16 {
            let src = array.dataPointer.bindMemory(to: UInt16.self, capacity: total)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src),
                                       height: 1, width: vImagePixelCount(total),
                                       rowBytes: total * MemoryLayout<UInt16>.size)
            var dstBuf = vImage_Buffer(data: out.dataPointer,
                                       height: 1, width: vImagePixelCount(total),
                                       rowBytes: total * MemoryLayout<Float>.size)
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
            return out
        }
        // Fallback: generic element-wise copy
        for i in 0..<total { out[i] = array[i] }
        return out
    }
}
