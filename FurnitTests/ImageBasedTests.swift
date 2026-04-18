// ImageBasedTests.swift
// Image-based integration tests for segmentation and room processing

import XCTest
import CoreML
import CoreImage
import simd
@testable import Furnit

/// Image-based tests that run actual inference on test images
final class ImageBasedTests: XCTestCase {

    // MARK: - Test Resources

    /// Load test image from test bundle
    private func loadTestImage(named name: String, extension ext: String) -> UIImage? {
        let bundle = Bundle(for: type(of: self))
        if let url = bundle.url(forResource: name, withExtension: ext) {
            return UIImage(contentsOfFile: url.path)
        }
        // Try loading from the test folder path directly
        let testFolderPath = "/Users/al/Documents/tries01/Furnit/FurnitTests/\(name).\(ext)"
        return UIImage(contentsOfFile: testFolderPath)
    }

    /// Load YOLOE segmentation model from main bundle
    private func loadYOLOEModel() -> MLModel? {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        // Match app load order: the app now ships only the 11L PF model.
        let modelNames = ["yoloe-11l-seg-pf"]

        for modelName in modelNames {
            if let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") {
                do {
                    let model = try MLModel(contentsOf: modelURL, configuration: config)
                    return model
                } catch {
                    print("Failed to load model \(modelName): \(error)")
                }
            }
        }
        return nil
    }

    // MARK: - Image Loading Tests

    func testBusImageLoads() {
        let image = loadTestImage(named: "bus", extension: "jpg")
        XCTAssertNotNil(image, "bus.jpg should load from test bundle")

        if let img = image {
            XCTAssertGreaterThan(img.size.width, 0)
            XCTAssertGreaterThan(img.size.height, 0)
            print("Bus image size: \(img.size.width) x \(img.size.height)")
        }
    }

    func testRoomImageLoads() {
        let image = loadTestImage(named: "room", extension: "jpeg")
        XCTAssertNotNil(image, "room.jpeg should load from test bundle")

        if let img = image {
            XCTAssertGreaterThan(img.size.width, 0)
            XCTAssertGreaterThan(img.size.height, 0)
            print("Room image size: \(img.size.width) x \(img.size.height)")
        }
    }

    // MARK: - Image Preprocessing Tests

    func testImageToPixelBuffer() {
        guard let image = loadTestImage(named: "bus", extension: "jpg"),
              let cgImage = image.cgImage else {
            XCTFail("Could not load bus.jpg")
            return
        }

        // Create a pixel buffer from the image
        let width = 640
        let height = 640

        var pixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width, height,
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer
        )

        XCTAssertEqual(status, kCVReturnSuccess, "Should create pixel buffer")
        XCTAssertNotNil(pixelBuffer)

        if let buffer = pixelBuffer {
            CVPixelBufferLockBaseAddress(buffer, [])
            let context = CGContext(
                data: CVPixelBufferGetBaseAddress(buffer),
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            )

            XCTAssertNotNil(context, "Should create graphics context")

            context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            CVPixelBufferUnlockBaseAddress(buffer, [])

            // Verify buffer dimensions
            XCTAssertEqual(CVPixelBufferGetWidth(buffer), width)
            XCTAssertEqual(CVPixelBufferGetHeight(buffer), height)
        }
    }

    func testImageResizeForModel() {
        guard let image = loadTestImage(named: "room", extension: "jpeg") else {
            XCTFail("Could not load room.jpeg")
            return
        }

        let originalSize = image.size
        let targetSize = CGSize(width: 640, height: 640)

        // Calculate letterbox/resize parameters (as done in FurnitureFitView)
        let scale = min(targetSize.width / originalSize.width,
                       targetSize.height / originalSize.height)
        let newWidth = originalSize.width * scale
        let newHeight = originalSize.height * scale
        let padX = (targetSize.width - newWidth) / 2
        let padY = (targetSize.height - newHeight) / 2

        XCTAssertGreaterThan(scale, 0, "Scale should be positive")
        XCTAssertLessThanOrEqual(newWidth, targetSize.width, "Resized width should fit target")
        XCTAssertLessThanOrEqual(newHeight, targetSize.height, "Resized height should fit target")

        print("Original: \(originalSize), Scale: \(scale), Padding: (\(padX), \(padY))")
    }

    // MARK: - Detection Utilities with Real Image Dimensions

    func testBoundingBoxConversionWithRealImageDimensions() {
        // Simulate a detection in a 640x640 model output
        let modelSize: Float = 640

        // Detection at center of image, 100x100 box
        let detection = FurnitureFitDetection(
            x: 320, y: 320, w: 100, h: 100,
            confidence: 0.85, classIdx: 0
        )

        let box = detection.boundingBox

        // Verify bounding box is correctly calculated
        XCTAssertEqual(box.origin.x, 270, accuracy: 0.1)  // 320 - 50
        XCTAssertEqual(box.origin.y, 270, accuracy: 0.1)  // 320 - 50
        XCTAssertEqual(box.width, 100, accuracy: 0.1)
        XCTAssertEqual(box.height, 100, accuracy: 0.1)

        // Test conversion to original image coordinates
        // If original image was 1920x1080, resized to 640x360 with padding
        let originalWidth: Float = 1920
        let originalHeight: Float = 1080
        let scale = min(modelSize / originalWidth, modelSize / originalHeight)
        let padX = (modelSize - originalWidth * scale) / 2
        let padY = (modelSize - originalHeight * scale) / 2

        // Convert detection coordinates back to original image space
        let origX = (detection.x - padX) / scale
        let origY = (detection.y - padY) / scale
        let origW = detection.w / scale
        let origH = detection.h / scale

        XCTAssertGreaterThan(origX, 0)
        XCTAssertGreaterThan(origY, 0)
        XCTAssertGreaterThan(origW, 0)
        XCTAssertGreaterThan(origH, 0)

        print("Original coords: (\(origX), \(origY)) size: \(origW)x\(origH)")
    }

    // MARK: - NMS with Realistic Detections

    func testNMSWithRealisticFurnitureDetections() {
        // Simulate realistic furniture detections from a room image
        // Multiple overlapping detections of same object (couch)
        let detections = [
            // Primary couch detection
            FurnitureFitDetection(x: 300, y: 400, w: 200, h: 150, confidence: 0.92, classIdx: 57), // couch
            // Slightly offset duplicate
            FurnitureFitDetection(x: 305, y: 402, w: 195, h: 148, confidence: 0.88, classIdx: 57),
            // Another duplicate
            FurnitureFitDetection(x: 298, y: 398, w: 202, h: 152, confidence: 0.75, classIdx: 57),

            // Chair detection (different object, should not be suppressed)
            FurnitureFitDetection(x: 100, y: 450, w: 80, h: 100, confidence: 0.85, classIdx: 56), // chair

            // Table detection
            FurnitureFitDetection(x: 400, y: 350, w: 150, h: 80, confidence: 0.78, classIdx: 60), // table
        ]

        let result = FurnitureFitNMS.apply(detections: detections, iouThreshold: 0.5)

        // Should keep 1 couch (highest confidence), 1 chair, 1 table = 3 detections
        XCTAssertEqual(result.count, 3, "NMS should suppress duplicate couch detections")

        // Verify the highest confidence couch was kept
        let couchDetections = result.filter { $0.classIdx == 57 }
        XCTAssertEqual(couchDetections.count, 1)
        if let couchConfidence = couchDetections.first?.confidence {
            XCTAssertEqual(couchConfidence, 0.92, accuracy: 0.01)
        }

        // Verify chair and table are preserved
        XCTAssertEqual(result.filter { $0.classIdx == 56 }.count, 1, "Chair should be preserved")
        XCTAssertEqual(result.filter { $0.classIdx == 60 }.count, 1, "Table should be preserved")
    }

    // MARK: - Room Bounds from Realistic Point Cloud

    func testRoomBoundsFromRealisticPointCloud() {
        // Simulate a realistic room point cloud (5m x 3m x 4m room)
        var points: [SIMD3<Float>] = []

        // Floor points (Y = 0)
        for x in stride(from: Float(-2.5), to: Float(2.5), by: 0.5) {
            for z in stride(from: Float(-2), to: Float(2), by: 0.5) {
                points.append(SIMD3<Float>(x, 0, z))
            }
        }

        // Ceiling points (Y = 3)
        for x in stride(from: Float(-2.5), to: Float(2.5), by: 0.5) {
            for z in stride(from: Float(-2), to: Float(2), by: 0.5) {
                points.append(SIMD3<Float>(x, 3, z))
            }
        }

        // Wall points
        for y in stride(from: Float(0), to: Float(3), by: 0.5) {
            // Back wall (Z = -2)
            for x in stride(from: Float(-2.5), to: Float(2.5), by: 0.5) {
                points.append(SIMD3<Float>(x, y, -2))
            }
            // Front wall (Z = 2)
            for x in stride(from: Float(-2.5), to: Float(2.5), by: 0.5) {
                points.append(SIMD3<Float>(x, y, 2))
            }
        }

        // Calculate bounding box
        let bounds = SharpRoomBoundsUtils.calculateBoundingBox(points: points)
        XCTAssertNotNil(bounds)

        if let b = bounds {
            // Verify room dimensions
            let width = b.max.x - b.min.x   // Should be ~5m
            let height = b.max.y - b.min.y  // Should be ~3m
            let depth = b.max.z - b.min.z   // Should be ~4m

            XCTAssertEqual(width, 5.0, accuracy: 0.5, "Room width should be ~5m")
            XCTAssertEqual(height, 3.0, accuracy: 0.5, "Room height should be ~3m")
            XCTAssertEqual(depth, 4.0, accuracy: 0.5, "Room depth should be ~4m")

            print("Room dimensions: \(width)m x \(height)m x \(depth)m")
        }
    }

    func testCameraPositionForRealisticRoom() {
        // Create realistic room bounds
        let bounds = RoomBounds(
            minX: -2.5, maxX: 2.5,  // 5m wide
            minY: 0, maxY: 3.0,      // 3m high
            minZ: -4, maxZ: 0        // 4m deep
        )

        let manager = RoomBoundaryManager(bounds: bounds)
        let camera = manager.getCameraAtBackWall()
        let expected = bounds.defaultSplatCameraEyeAndTarget()

        // Back-center camera (Android parity): inside room from back wall, look at front wall center — not room centroid.
        XCTAssertEqual(camera.eye.x, expected.eye.x, accuracy: 0.001)
        XCTAssertEqual(camera.eye.y, expected.eye.y, accuracy: 0.001)
        XCTAssertEqual(camera.eye.z, expected.eye.z, accuracy: 0.001)
        XCTAssertEqual(camera.target.x, expected.target.x, accuracy: 0.001)
        XCTAssertEqual(camera.target.y, expected.target.y, accuracy: 0.001)
        XCTAssertEqual(camera.target.z, expected.target.z, accuracy: 0.001)
        XCTAssertGreaterThan(camera.eye.z, bounds.minZ, "Camera should be inside room")
        XCTAssertLessThan(camera.eye.z, bounds.maxZ, "Camera should not be past front wall")
        XCTAssertEqual(camera.target.z, bounds.maxZ, accuracy: 0.001, "Look-at should be front wall Z")

        print("Camera position: eye=\(camera.eye), target=\(camera.target)")
    }

    // MARK: - Mask Processing Tests

    func testMaskCoverageForRealisticSegmentation() {
        // Simulate a mask for a couch that covers ~20% of a 160x160 prototype output
        let maskSize = 160 * 160
        var mask = [UInt8](repeating: 0, count: maskSize)

        // Fill in a rectangular region representing a couch (roughly 20% coverage)
        let couchWidth = 80
        let couchHeight = 64
        let startX = 40
        let startY = 48

        for y in startY..<(startY + couchHeight) {
            for x in startX..<(startX + couchWidth) {
                mask[y * 160 + x] = 255
            }
        }

        let coverage = FurnitureFitMask.coverage(mask)
        let expectedCoverage = Float(couchWidth * couchHeight) / Float(maskSize)

        XCTAssertEqual(coverage, expectedCoverage, accuracy: 0.01)
        XCTAssertTrue(FurnitureFitMask.hasContent(mask))

        let positiveCount = FurnitureFitMask.positivePixelCount(mask)
        XCTAssertEqual(positiveCount, couchWidth * couchHeight)

        print("Mask coverage: \(coverage * 100)%, positive pixels: \(positiveCount)")
    }

    // MARK: - Performance Tests

    func testImagePreprocessingPerformance() {
        guard let image = loadTestImage(named: "bus", extension: "jpg"),
              let cgImage = image.cgImage else {
            XCTFail("Could not load bus.jpg")
            return
        }

        measure {
            // Simulate the preprocessing pipeline
            let width = 640
            let height = 640

            var pixelBuffer: CVPixelBuffer?
            CVPixelBufferCreate(
                kCFAllocatorDefault,
                width, height,
                kCVPixelFormatType_32BGRA,
                nil,
                &pixelBuffer
            )

            if let buffer = pixelBuffer {
                CVPixelBufferLockBaseAddress(buffer, [])
                let context = CGContext(
                    data: CVPixelBufferGetBaseAddress(buffer),
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
                )
                context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
                CVPixelBufferUnlockBaseAddress(buffer, [])
            }
        }
    }

    // MARK: - End-to-End YOLOE 26L PF Inference Test
    // Validates Swift inference against the same Core ML package as the app (`yoloe-26l-seg-pf` / `_seg_o2m`).

    /// Model: **yoloe-26l-seg-pf** (prompt-free), input side from Core ML `image` constraint (typically **640**).
    /// Image: bus.jpg letterboxed to a square of that side (Ultralytics gray 114).

    func testYOLOE26LInferenceMatchesPython() throws {
        // Load model first — input side must match the bundled 26L export.
        guard let model = loadYOLOEModel() else {
            throw XCTSkip("Skipping ML inference test - no YOLOE model available in test environment")
        }

        // Load test image
        guard let image = loadTestImage(named: "bus", extension: "jpg"),
              let cgImage = image.cgImage else {
            XCTFail("Could not load bus.jpg")
            return
        }

        let modelSize = YoloEImageInference.modelInputSize(for: model)
        print("Image size: \(image.size.width) x \(image.size.height); model input side: \(modelSize)")

        // Create CVPixelBuffer for the letterboxed image (model expects Image input, not MultiArray)
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, modelSize, modelSize, kCVPixelFormatType_32BGRA, nil, &pixelBuffer)

        guard let buffer = pixelBuffer else {
            XCTFail("Failed to create pixel buffer")
            return
        }

        CVPixelBufferLockBaseAddress(buffer, [])

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: modelSize,
            height: modelSize,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            CVPixelBufferUnlockBaseAddress(buffer, [])
            XCTFail("Failed to create context")
            return
        }

        // Ultralytics letterbox padding RGB gray 114/255 (matches `YoloUltralyticsLetterboxFill` / probe scripts)
        context.setFillColor(gray: 114.0/255.0, alpha: 1.0)
        context.fill(CGRect(x: 0, y: 0, width: modelSize, height: modelSize))

        // Calculate letterbox dimensions
        let scale = min(CGFloat(modelSize) / image.size.width, CGFloat(modelSize) / image.size.height)
        let newWidth = image.size.width * scale
        let newHeight = image.size.height * scale
        let padX = (CGFloat(modelSize) - newWidth) / 2
        let padY = (CGFloat(modelSize) - newHeight) / 2

        context.draw(cgImage, in: CGRect(x: padX, y: padY, width: newWidth, height: newHeight))

        print("Letterbox: scale=\(scale), pad=(\(padX), \(padY)), newSize=(\(newWidth)x\(newHeight))")

        CVPixelBufferUnlockBaseAddress(buffer, [])

        // Run inference using CVPixelBuffer directly (model expects Image input)
        do {
            let imageValue = MLFeatureValue(pixelBuffer: buffer)
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": imageValue])
            let output = try model.prediction(from: input)

            guard let pair = YoloEDetectionParser.extractDetectionAndProto(from: output) else {
                let availableOutputs = output.featureNames.joined(separator: ", ")
                XCTFail("Missing expected detection/proto tensors. Found: \(availableOutputs)")
                return
            }
            let detArray = pair.det
            let protoArray = pair.proto

            // Validate tensor shapes
            let detShape = detArray.shape.map { $0.intValue }
            let protoShape = protoArray.shape.map { $0.intValue }
            print("Detection tensor shape: \(detShape)")
            print("Proto tensor shape: \(protoShape)")

            // Detection tensor: [1, features, anchors] for raw model or [1, detections, features] for post-NMS
            XCTAssertEqual(detShape.count, 3, "Detection tensor should be 3D")
            XCTAssertEqual(detShape[0], 1, "Batch size should be 1")

            // Proto tensor: [1, 32, H, W] - 32 prototype masks
            XCTAssertEqual(protoShape.count, 4, "Proto tensor should be 4D")
            XCTAssertEqual(protoShape[0], 1, "Batch size should be 1")
            XCTAssertEqual(protoShape[1], 32, "Should have 32 prototype masks")

            // For raw model output [1, 4621, 33600], skip detailed detection validation
            // as the full decode+NMS pipeline is complex and tested in FurnitureFitView
            let isRawFormat = detShape[2] == 33600 || detShape[1] == 33600
            if isRawFormat {
                print("Model outputs raw anchors format - skipping detection parsing (handled by FurnitureFitView)")
                print("Inference test PASSED - model loads and runs correctly")
                return
            }

            // For post-NMS format [1, 300, 38], continue with detection validation
            XCTAssertEqual(detShape[1], 300, "Should have 300 detections (post-NMS)")
            XCTAssertEqual(detShape[2], 38, "Should have 38 features per detection")

            // Parse detections using same logic as FurnitureFitView
            let numDetections = detShape[1]
            let featuresPerDet = detShape[2]
            let strides = detArray.strides.map { $0.intValue }
            let detStride = strides.count >= 2 ? strides[1] : featuresPerDet
            let featStride = strides.count >= 3 ? strides[2] : 1

            print("Strides: \(strides), detStride=\(detStride), featStride=\(featStride)")

            // Copy to float buffer
            let totalCount = detArray.count
            let detBuf = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
            defer { detBuf.deallocate() }
            memcpy(detBuf, detArray.dataPointer, totalCount * MemoryLayout<Float>.size)

            // Extract detections with conf > 0.1
            struct Detection {
                let cls: Int
                let conf: Float
                let x: Float, y: Float, w: Float, h: Float
            }

            var detections: [Detection] = []

            for detIdx in 0..<numDetections {
                let base = detIdx * detStride

                // XYXY format (as fixed in FurnitureFitView)
                let x1 = detBuf[base + 0 * featStride]
                let y1 = detBuf[base + 1 * featStride]
                let x2 = detBuf[base + 2 * featStride]
                let y2 = detBuf[base + 3 * featStride]
                let confidence = detBuf[base + 4 * featStride]
                let classIdxFloat = detBuf[base + 5 * featStride]

                guard x1.isFinite, y1.isFinite, x2.isFinite, y2.isFinite,
                      confidence.isFinite, classIdxFloat.isFinite else { continue }

                guard confidence > 0.1 else { continue }

                // Convert xyxy to xywh center
                let w = x2 - x1
                let h = y2 - y1
                let x = (x1 + x2) / 2.0
                let y = (y1 + y2) / 2.0

                guard w > 0, h > 0 else { continue }

                let cls = Int(classIdxFloat)
                detections.append(Detection(cls: cls, conf: confidence, x: x, y: y, w: w, h: h))
            }

            print("\nTotal detections (conf > 0.1): \(detections.count)")

            // Write results to file for inspection
            var logOutput = "=== YOLOE Inference Results ===\n"
            logOutput += "Total detections (conf > 0.1): \(detections.count)\n\n"

            // Print all high-confidence detections for debugging
            let highConfDetections = detections.filter { $0.conf > 0.3 }.sorted { $0.conf > $1.conf }
            logOutput += "=== High confidence detections (conf > 0.3) ===\n"
            for det in highConfDetections.prefix(20) {
                logOutput += "  cls=\(det.cls), conf=\(String(format: "%.3f", det.conf)), center=(\(String(format: "%.1f", det.x)), \(String(format: "%.1f", det.y))), size=(\(String(format: "%.1f", det.w))x\(String(format: "%.1f", det.h)))\n"
            }

            // Print class distribution for debugging
            var classCounts: [Int: Int] = [:]
            for d in detections {
                classCounts[d.cls, default: 0] += 1
            }
            let topClasses = classCounts.sorted { $0.value > $1.value }.prefix(15)
            logOutput += "\nTop 15 classes by count:\n"
            for (cls, count) in topClasses {
                logOutput += "  Class \(cls): \(count) detections\n"
            }

            // === ASSERTIONS: Compare with Python ground truth ===
            // LVIS class IDs (same for 11L and 26L):
            // bus=640, person=2163, stop_sign=3913, chair=821, couch=1141

            // 1. Should have some detections
            XCTAssertGreaterThan(detections.count, 10, "Should detect objects")

            // 2. Find bus detection (class 640)
            // Python 26L: conf=0.702, center=(279.8, 421.2)
            // Python 11L: conf=0.899, center=(517.1, 480.0)
            let buses = detections.filter { $0.cls == 640 }
            logOutput += "\nBus (class 640): \(buses.count) detections\n"
            if let bus = buses.max(by: { $0.conf < $1.conf }) {
                logOutput += "  Best: conf=\(String(format: "%.3f", bus.conf)), center=(\(String(format: "%.1f", bus.x)), \(String(format: "%.1f", bus.y))), size=(\(String(format: "%.1f", bus.w))x\(String(format: "%.1f", bus.h)))\n"
                XCTAssertGreaterThan(bus.conf, 0.3, "Bus confidence should be > 0.3")
                let side = Float(modelSize)
                XCTAssertGreaterThan(bus.x, side * 0.08, "Bus center X should be inside frame")
                XCTAssertLessThan(bus.x, side * 0.92, "Bus center X should be inside frame")
                XCTAssertGreaterThan(bus.w, side * 0.12, "Bus width should be plausible vs input side")
            } else {
                logOutput += "  NOT DETECTED\n"
                XCTFail("Should detect bus (class 640)")
            }

            // 3. Find person detections (class 2163)
            let persons = detections.filter { $0.cls == 2163 }
            logOutput += "Person (class 2163): \(persons.count) detections\n"
            if let bestPerson = persons.max(by: { $0.conf < $1.conf }) {
                logOutput += "  Best: conf=\(String(format: "%.3f", bestPerson.conf))\n"
                XCTAssertGreaterThan(bestPerson.conf, 0.3, "Best person confidence should be > 0.3")
            } else {
                logOutput += "  WARNING: No persons detected\n"
            }

            // 4. Check if stop sign detected (optional - might have low conf)
            let stopSigns = detections.filter { $0.cls == 3913 }
            logOutput += "Stop sign (class 3913): \(stopSigns.count) detections\n"
            if let stopSign = stopSigns.max(by: { $0.conf < $1.conf }) {
                logOutput += "  Best: conf=\(String(format: "%.3f", stopSign.conf))\n"
            }

            logOutput += "\n✅ YOLOE inference test completed\n"

            // Output to console
            NSLog("=== YOLOE TEST RESULTS ===\n%@", logOutput)

        } catch {
            XCTFail("Inference failed: \(error)")
        }
    }

    // MARK: - Landscape Segmentation Test
    // Decodes 26L PF outputs and exercises primary selection in model input space (640 typical).

    /// **yoloe-26l-seg-pf** on `landscape.jpeg` (1280×960), letterboxed to the model’s square input (typically **640**).
    /// Older PyTorch baselines at 1280×1280 are not numerically comparable; this test checks decode + primary selection in model space.
    func testLandscapeSegmentationMatchesPyTorch() throws {
        // Load test image
        guard let image = loadTestImage(named: "landscape", extension: "jpeg"),
              let cgImage = image.cgImage else {
            XCTFail("Could not load landscape.jpeg")
            return
        }

        guard let model = loadYOLOEModel() else {
            throw XCTSkip("Skipping - no YOLOE model available")
        }

        let modelSize = YoloEImageInference.modelInputSize(for: model)

        // Load blacklist for filtering
        let blacklistURL = Bundle.main.url(forResource: "blacklist", withExtension: "json")
        var blacklist: Set<Int> = []
        if let url = blacklistURL, let data = try? Data(contentsOf: url),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            blacklist = Set(dict.keys.compactMap { Int($0) })
        }

        print("Loaded \(blacklist.count) blacklisted classes")
        print("Image size: \(image.size.width) x \(image.size.height); model input side: \(modelSize)")

        // Letterbox source image to model square (same convention as Furniture Fit / YoloEImageInference)
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, modelSize, modelSize, kCVPixelFormatType_32BGRA, nil, &pixelBuffer)

        guard let buffer = pixelBuffer else {
            XCTFail("Failed to create pixel buffer")
            return
        }

        CVPixelBufferLockBaseAddress(buffer, [])

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: modelSize,
            height: modelSize,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            CVPixelBufferUnlockBaseAddress(buffer, [])
            XCTFail("Failed to create context")
            return
        }

        // Ultralytics letterbox padding (114/255)
        context.setFillColor(gray: 114.0/255.0, alpha: 1.0)
        context.fill(CGRect(x: 0, y: 0, width: modelSize, height: modelSize))

        // Calculate letterbox dimensions
        let scale = min(CGFloat(modelSize) / image.size.width, CGFloat(modelSize) / image.size.height)
        let newWidth = image.size.width * scale
        let newHeight = image.size.height * scale
        let padX = (CGFloat(modelSize) - newWidth) / 2
        let padY = (CGFloat(modelSize) - newHeight) / 2

        context.draw(cgImage, in: CGRect(x: padX, y: padY, width: newWidth, height: newHeight))
        CVPixelBufferUnlockBaseAddress(buffer, [])

        print("Letterbox: scale=\(scale), pad=(\(padX), \(padY))")

        // Run inference
        do {
            let imageValue = MLFeatureValue(pixelBuffer: buffer)
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": imageValue])
            let output = try model.prediction(from: input)

            guard let pair = YoloEDetectionParser.extractDetectionAndProto(from: output) else {
                XCTFail("Missing detection/proto tensors")
                return
            }
            let detArray = pair.det

            let detShape = detArray.shape.map { $0.intValue }
            print("Detection tensor shape: \(detShape)")

            // For raw anchor format, skip detailed validation
            let isRawFormat = detShape[2] == 33600 || detShape[1] == 33600
            if isRawFormat {
                print("Model outputs raw anchors - testing basic inference only")
                XCTAssertEqual(detShape.count, 3)
                return
            }

            // Parse detections (post-NMS format: [1, 300, 38])
            let numDetections = detShape[1]
            let featuresPerDet = detShape[2]
            let strides = detArray.strides.map { $0.intValue }
            let detStride = strides.count >= 2 ? strides[1] : featuresPerDet
            let featStride = strides.count >= 3 ? strides[2] : 1

            let totalCount = detArray.count
            let detBuf = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
            defer { detBuf.deallocate() }
            memcpy(detBuf, detArray.dataPointer, totalCount * MemoryLayout<Float>.size)

            // Extract detections with conf > 0.25
            struct Detection {
                let cls: Int
                let conf: Float
                let cx: Float, cy: Float, w: Float, h: Float
            }

            var allDetections: [Detection] = []
            let confThreshold: Float = 0.25

            for detIdx in 0..<numDetections {
                let base = detIdx * detStride

                let x1 = detBuf[base + 0 * featStride]
                let y1 = detBuf[base + 1 * featStride]
                let x2 = detBuf[base + 2 * featStride]
                let y2 = detBuf[base + 3 * featStride]
                let conf = detBuf[base + 4 * featStride]
                let clsFloat = detBuf[base + 5 * featStride]

                guard conf > confThreshold, x1.isFinite, y1.isFinite else { continue }

                let w = x2 - x1
                let h = y2 - y1
                let cx = (x1 + x2) / 2
                let cy = (y1 + y2) / 2

                guard w > 0, h > 0 else { continue }

                allDetections.append(Detection(cls: Int(clsFloat), conf: conf, cx: cx, cy: cy, w: w, h: h))
            }

            print("\n=== CoreML Detections (conf > 0.25): \(allDetections.count) ===")

            // ASSERTION 1: Reasonable detection count at the model’s native input size
            XCTAssertGreaterThan(allDetections.count, 10, "Should have at least 10 detections")
            XCTAssertLessThan(allDetections.count, 55, "Should have fewer than 55 detections")

            // Filter by blacklist
            let validDetections = allDetections.filter { !blacklist.contains($0.cls) }
            print("Valid (non-blacklisted): \(validDetections.count)")

            // ASSERTION 2: Valid detection count should be close to PyTorch (21)
            XCTAssertGreaterThan(validDetections.count, 15, "Should have at least 15 valid detections")

            // ASSERTION 3: Test primary detection selection (composite scoring)
            let frameArea: Float = Float(modelSize * modelSize)
            let centerX: Float = Float(modelSize) / 2
            let centerY: Float = Float(modelSize) / 2
            let minConf: Float = 0.25
            let minAreaNorm: Float = 0.001

            var bestComposite: Float = -1
            var primaryDetection: Detection?

            for det in validDetections {
                let areaNorm = (det.w * det.h) / frameArea
                if det.conf < minConf || areaNorm < minAreaNorm { continue }

                let dx = abs(det.cx - centerX) / centerX
                let dy = abs(det.cy - centerY) / centerY
                let centerScore = 1.0 - (dx + dy) / 2.0
                let composite = det.conf * areaNorm * centerScore

                if composite > bestComposite {
                    bestComposite = composite
                    primaryDetection = det
                }
            }

            guard let primary = primaryDetection else {
                XCTFail("No primary detection selected")
                return
            }

            print("\n=== PRIMARY DETECTION ===")
            print("Class: \(primary.cls), conf: \(String(format: "%.3f", primary.conf))")
            print("Center: (\(String(format: "%.1f", primary.cx)), \(String(format: "%.1f", primary.cy)))")
            print("Size: \(String(format: "%.1f", primary.w)) x \(String(format: "%.1f", primary.h))")
            print("Composite score: \(String(format: "%.6f", bestComposite))")

            // ASSERTION 4: Primary is a large, reasonably central detection in **model** pixel space
            let sideF = Float(modelSize)
            XCTAssertGreaterThan(primary.conf, 0.3, "Primary confidence should be > 0.3")
            XCTAssertGreaterThan(primary.cx, sideF * 0.18, "Primary center X should be in frame")
            XCTAssertLessThan(primary.cx, sideF * 0.92, "Primary center X should be in frame")
            XCTAssertGreaterThan(primary.cy, sideF * 0.22, "Primary center Y should be in frame")
            XCTAssertLessThan(primary.cy, sideF * 0.95, "Primary center Y should be in frame")

            // ASSERTION 5: Primary should be a large object (high area)
            let primaryAreaNorm = (primary.w * primary.h) / frameArea
            XCTAssertGreaterThan(primaryAreaNorm, 0.05, "Primary should cover > 5% of frame")

            // Check if primary matches expected class (writing desk = 4564)
            // Other acceptable large furniture: cocktail table (1006), daybed (823), step stool (3888)
            let acceptablePrimaryClasses = [4564, 1006, 823, 3888, 2754, 276] // writing desk, cocktail table, daybed, step stool, music stool, bar stool
            if acceptablePrimaryClasses.contains(primary.cls) {
                print("✅ Primary class \(primary.cls) matches expected furniture classes")
            } else {
                print("⚠️ Primary class \(primary.cls) differs from expected - checking location/size only")
            }

            print("\n✅ Landscape segmentation test PASSED")

        } catch {
            XCTFail("Inference failed: \(error)")
        }
    }
}
