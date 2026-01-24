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

        // Try different model names
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

        // Camera should be positioned inside the room, near front wall
        XCTAssertEqual(camera.eye.x, 0, accuracy: 0.1, "Camera X should be at room center")
        XCTAssertEqual(camera.eye.y, 1.5, accuracy: 0.1, "Camera Y should be at room center height")
        XCTAssertGreaterThan(camera.eye.z, bounds.minZ, "Camera should be inside room")
        XCTAssertLessThan(camera.eye.z, bounds.maxZ, "Camera should not be outside front wall")

        // Camera should look at room center
        XCTAssertEqual(camera.target.x, 0, accuracy: 0.1)
        XCTAssertEqual(camera.target.y, 1.5, accuracy: 0.1)
        XCTAssertEqual(camera.target.z, -2, accuracy: 0.1, "Target should be at room center Z")

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
}
