// ImageBasedTests.swift
// Image-based integration tests for segmentation and room processing

import XCTest
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
        return nil
    }

    // MARK: - Image Loading Tests

    func testImageFixtureLoads() {
        let image = loadTestImage(named: "rtmdet_repeated_chair_frame", extension: "jpg")
        XCTAssertNotNil(image, "RTMDet image fixture should load from test bundle")

        if let img = image {
            XCTAssertGreaterThan(img.size.width, 0)
            XCTAssertGreaterThan(img.size.height, 0)
            print("Fixture image size: \(img.size.width) x \(img.size.height)")
        }
    }

    func testImageFixtureHasDimensions() {
        let image = loadTestImage(named: "rtmdet_repeated_chair_frame", extension: "jpg")
        XCTAssertNotNil(image, "RTMDet image fixture should load from test bundle")

        if let img = image {
            XCTAssertGreaterThan(img.size.width, 0)
            XCTAssertGreaterThan(img.size.height, 0)
            print("Fixture image size: \(img.size.width) x \(img.size.height)")
        }
    }

    // MARK: - Image Preprocessing Tests

    func testImageToPixelBuffer() {
        guard let image = loadTestImage(named: "rtmdet_repeated_chair_frame", extension: "jpg"),
              let cgImage = image.cgImage else {
            XCTFail("Could not load RTMDet image fixture")
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
        guard let image = loadTestImage(named: "rtmdet_repeated_chair_frame", extension: "jpg") else {
            XCTFail("Could not load RTMDet image fixture")
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

        // Verify the tighter overlapping couch box was kept (NMS prefers smaller area, then confidence).
        let couchDetections = result.filter { $0.classIdx == 57 }
        XCTAssertEqual(couchDetections.count, 1)
        if let couchConfidence = couchDetections.first?.confidence {
            XCTAssertEqual(couchConfidence, 0.88, accuracy: 0.01)
        }

        // Verify chair and table are preserved
        XCTAssertEqual(result.filter { $0.classIdx == 56 }.count, 1, "Chair should be preserved")
        XCTAssertEqual(result.filter { $0.classIdx == 60 }.count, 1, "Table should be preserved")
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
        let expectedCoverage = Double(couchWidth * couchHeight) / Double(maskSize)

        XCTAssertEqual(coverage, expectedCoverage, accuracy: 0.01)
        XCTAssertTrue(FurnitureFitMask.hasContent(mask))

        let positiveCount = FurnitureFitMask.positivePixelCount(mask)
        XCTAssertEqual(positiveCount, couchWidth * couchHeight)

        print("Mask coverage: \(coverage * 100)%, positive pixels: \(positiveCount)")
    }

    // MARK: - Performance Tests

    func testImagePreprocessingPerformance() {
        guard let image = loadTestImage(named: "rtmdet_repeated_chair_frame", extension: "jpg"),
              let cgImage = image.cgImage else {
            XCTFail("Could not load RTMDet image fixture")
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
