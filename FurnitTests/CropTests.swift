// CropTests.swift
// Tests for front wall image cropping functionality

import XCTest
import UIKit
@testable import Furnit

/// Tests for the cropImageToFrontWall functionality used in manual room creation
final class CropTests: XCTestCase {

    // MARK: - Test Resources

    /// Load test image from test bundle or direct path
    private func loadTestImage(named name: String, extension ext: String) -> UIImage? {
        let bundle = Bundle(for: type(of: self))
        if let url = bundle.url(forResource: name, withExtension: ext) {
            return UIImage(contentsOfFile: url.path)
        }
        // Try loading from the test folder path directly
        let testFolderPath = "/Users/al/Documents/tries01/Furnit/FurnitTests/\(name).\(ext)"
        return UIImage(contentsOfFile: testFolderPath)
    }

    // MARK: - Crop Function Implementation (copied from SinglePhotoRoomViewer for testing)

    /// Crop image to the selected front wall boundaries
    /// This is the same implementation that should be in SinglePhotoRoomViewer
    private func cropImageToFrontWall(image: UIImage, leftX: CGFloat, rightX: CGFloat, ceilingY: CGFloat, floorY: CGFloat) -> UIImage {
        print("🔲 [cropImageToFrontWall] Starting crop with boundaries: L=\(leftX), R=\(rightX), T=\(ceilingY), B=\(floorY)")
        print("   Input image: \(image.size), orientation: \(image.imageOrientation.rawValue)")

        // First, normalize orientation so CGImage matches what user saw
        let normalizedImage = image.imageOrientation == .up ? image : image.fixedOrientation()

        guard let cgImage = normalizedImage.cgImage else {
            print("⚠️ [cropImageToFrontWall] Failed to get CGImage, returning original")
            return image
        }

        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)
        print("   CGImage size: \(Int(imageWidth))x\(Int(imageHeight))")

        // Convert normalized coordinates (0-1) to pixel coordinates
        let cropX = leftX * imageWidth
        let cropY = ceilingY * imageHeight
        let cropWidth = (rightX - leftX) * imageWidth
        let cropHeight = (floorY - ceilingY) * imageHeight

        // Ensure valid crop rect
        let cropRect = CGRect(
            x: max(0, cropX),
            y: max(0, cropY),
            width: min(cropWidth, imageWidth - cropX),
            height: min(cropHeight, imageHeight - cropY)
        )

        print("   Crop rect: x=\(Int(cropRect.minX)), y=\(Int(cropRect.minY)), w=\(Int(cropRect.width)), h=\(Int(cropRect.height))")

        // Perform the crop
        guard cropRect.width > 0, cropRect.height > 0,
              let croppedCGImage = cgImage.cropping(to: cropRect) else {
            print("⚠️ [cropImageToFrontWall] Invalid crop rect, returning original image")
            return image
        }

        // Return cropped image with .up orientation (already normalized)
        let croppedImage = UIImage(cgImage: croppedCGImage, scale: normalizedImage.scale, orientation: .up)
        print("✅ [cropImageToFrontWall] Cropped image from \(Int(imageWidth))x\(Int(imageHeight)) to \(Int(cropRect.width))x\(Int(cropRect.height))")

        return croppedImage
    }

    // MARK: - Basic Crop Tests

    func testCropRoomImageWithDefaultBoundaries() {
        // Load room image
        guard let image = loadTestImage(named: "room", extension: "jpeg") else {
            XCTFail("Could not load room.jpeg - make sure it's in FurnitTests folder")
            return
        }

        print("\n=== Test: Crop with default boundaries ===")
        print("Original image size: \(image.size)")
        print("Original orientation: \(image.imageOrientation.rawValue)")

        // Default boundaries from RoomBoundaryDetectionView
        let leftX: CGFloat = 0.12
        let rightX: CGFloat = 0.88
        let ceilingY: CGFloat = 0.15
        let floorY: CGFloat = 0.85

        let cropped = cropImageToFrontWall(
            image: image,
            leftX: leftX,
            rightX: rightX,
            ceilingY: ceilingY,
            floorY: floorY
        )

        // Calculate expected dimensions
        let expectedWidth = (rightX - leftX) * image.size.width
        let expectedHeight = (floorY - ceilingY) * image.size.height

        print("\nExpected cropped size: \(expectedWidth) x \(expectedHeight)")
        print("Actual cropped size: \(cropped.size)")

        // Verify crop happened
        XCTAssertLessThan(cropped.size.width, image.size.width, "Cropped width should be less than original")
        XCTAssertLessThan(cropped.size.height, image.size.height, "Cropped height should be less than original")

        // Verify approximate dimensions (allowing for rounding)
        XCTAssertEqual(cropped.size.width, expectedWidth, accuracy: 2.0, "Cropped width should match expected")
        XCTAssertEqual(cropped.size.height, expectedHeight, accuracy: 2.0, "Cropped height should match expected")

        // Verify the cropped image is valid
        XCTAssertNotNil(cropped.cgImage, "Cropped image should have valid CGImage")

        print("✅ Test passed - image was cropped correctly")
    }

    func testCropWithCustomBoundaries() {
        guard let image = loadTestImage(named: "room", extension: "jpeg") else {
            XCTFail("Could not load room.jpeg")
            return
        }

        print("\n=== Test: Crop with custom boundaries ===")

        // Custom boundaries - crop to center 50% of image
        let leftX: CGFloat = 0.25
        let rightX: CGFloat = 0.75
        let ceilingY: CGFloat = 0.25
        let floorY: CGFloat = 0.75

        let cropped = cropImageToFrontWall(
            image: image,
            leftX: leftX,
            rightX: rightX,
            ceilingY: ceilingY,
            floorY: floorY
        )

        // Should be approximately 50% of original in each dimension
        let expectedWidth = 0.5 * image.size.width
        let expectedHeight = 0.5 * image.size.height

        print("Original: \(image.size)")
        print("Expected: \(expectedWidth) x \(expectedHeight)")
        print("Actual: \(cropped.size)")

        XCTAssertEqual(cropped.size.width, expectedWidth, accuracy: 2.0)
        XCTAssertEqual(cropped.size.height, expectedHeight, accuracy: 2.0)

        print("✅ Test passed")
    }

    func testCropWithTightBoundaries() {
        guard let image = loadTestImage(named: "room", extension: "jpeg") else {
            XCTFail("Could not load room.jpeg")
            return
        }

        print("\n=== Test: Crop with tight boundaries (10% margins) ===")

        // Tight crop - only 10% margins
        let leftX: CGFloat = 0.10
        let rightX: CGFloat = 0.90
        let ceilingY: CGFloat = 0.10
        let floorY: CGFloat = 0.90

        let cropped = cropImageToFrontWall(
            image: image,
            leftX: leftX,
            rightX: rightX,
            ceilingY: ceilingY,
            floorY: floorY
        )

        let expectedWidth = 0.8 * image.size.width
        let expectedHeight = 0.8 * image.size.height

        print("Original: \(image.size)")
        print("Expected: \(expectedWidth) x \(expectedHeight)")
        print("Actual: \(cropped.size)")

        XCTAssertEqual(cropped.size.width, expectedWidth, accuracy: 2.0)
        XCTAssertEqual(cropped.size.height, expectedHeight, accuracy: 2.0)

        print("✅ Test passed")
    }

    // MARK: - Orientation Tests

    func testCropWithRotatedImage() {
        guard let originalImage = loadTestImage(named: "room", extension: "jpeg"),
              let cgImage = originalImage.cgImage else {
            XCTFail("Could not load room.jpeg")
            return
        }

        print("\n=== Test: Crop with rotated image (simulating camera orientation) ===")

        // Create a rotated version of the image (simulating .right orientation from camera)
        let rotatedImage = UIImage(cgImage: cgImage, scale: originalImage.scale, orientation: .right)

        print("Original image size: \(originalImage.size), orientation: \(originalImage.imageOrientation.rawValue)")
        print("Rotated image size: \(rotatedImage.size), orientation: \(rotatedImage.imageOrientation.rawValue)")

        // Crop the rotated image
        let cropped = cropImageToFrontWall(
            image: rotatedImage,
            leftX: 0.12,
            rightX: 0.88,
            ceilingY: 0.15,
            floorY: 0.85
        )

        print("Cropped image size: \(cropped.size), orientation: \(cropped.imageOrientation.rawValue)")

        // The cropped image should be valid and have reasonable dimensions
        XCTAssertNotNil(cropped.cgImage)
        XCTAssertGreaterThan(cropped.size.width, 0)
        XCTAssertGreaterThan(cropped.size.height, 0)

        // After fixedOrientation, the orientation should be .up
        XCTAssertEqual(cropped.imageOrientation, .up, "Cropped image should have .up orientation")

        print("✅ Test passed - rotated image was cropped correctly")
    }

    func testCropPreservesImageQuality() {
        guard let image = loadTestImage(named: "room", extension: "jpeg"),
              let originalCGImage = image.cgImage else {
            XCTFail("Could not load room.jpeg")
            return
        }

        print("\n=== Test: Crop preserves image quality ===")

        let cropped = cropImageToFrontWall(
            image: image,
            leftX: 0.2,
            rightX: 0.8,
            ceilingY: 0.2,
            floorY: 0.8
        )

        guard let croppedCGImage = cropped.cgImage else {
            XCTFail("Cropped image has no CGImage")
            return
        }

        // Verify bits per component is preserved
        XCTAssertEqual(croppedCGImage.bitsPerComponent, originalCGImage.bitsPerComponent,
                      "Bits per component should be preserved")

        // Verify the cropped image can be converted to data (is valid)
        let jpegData = cropped.jpegData(compressionQuality: 0.9)
        XCTAssertNotNil(jpegData, "Should be able to convert cropped image to JPEG")
        XCTAssertGreaterThan(jpegData?.count ?? 0, 1000, "JPEG data should have reasonable size")

        print("Original bits per component: \(originalCGImage.bitsPerComponent)")
        print("Cropped bits per component: \(croppedCGImage.bitsPerComponent)")
        print("JPEG data size: \(jpegData?.count ?? 0) bytes")

        print("✅ Test passed")
    }

    // MARK: - Edge Cases

    func testCropWithFullImageBoundaries() {
        guard let image = loadTestImage(named: "room", extension: "jpeg") else {
            XCTFail("Could not load room.jpeg")
            return
        }

        print("\n=== Test: Crop with full image boundaries (0,0 to 1,1) ===")

        // Full image crop (should return same dimensions)
        let cropped = cropImageToFrontWall(
            image: image,
            leftX: 0.0,
            rightX: 1.0,
            ceilingY: 0.0,
            floorY: 1.0
        )

        print("Original: \(image.size)")
        print("Cropped: \(cropped.size)")

        // Should be same size (or very close due to rounding)
        XCTAssertEqual(cropped.size.width, image.size.width, accuracy: 1.0)
        XCTAssertEqual(cropped.size.height, image.size.height, accuracy: 1.0)

        print("✅ Test passed")
    }

    func testCropWithMinimalBoundaries() {
        guard let image = loadTestImage(named: "room", extension: "jpeg") else {
            XCTFail("Could not load room.jpeg")
            return
        }

        print("\n=== Test: Crop with minimal boundaries (small region) ===")

        // Very small crop region
        let cropped = cropImageToFrontWall(
            image: image,
            leftX: 0.45,
            rightX: 0.55,
            ceilingY: 0.45,
            floorY: 0.55
        )

        // Should be approximately 10% of original in each dimension
        let expectedWidth = 0.1 * image.size.width
        let expectedHeight = 0.1 * image.size.height

        print("Original: \(image.size)")
        print("Expected: \(expectedWidth) x \(expectedHeight)")
        print("Actual: \(cropped.size)")

        XCTAssertEqual(cropped.size.width, expectedWidth, accuracy: 2.0)
        XCTAssertEqual(cropped.size.height, expectedHeight, accuracy: 2.0)

        print("✅ Test passed")
    }

    // MARK: - Integration Test with fixedOrientation

    func testFixedOrientationExtension() {
        guard let originalImage = loadTestImage(named: "room", extension: "jpeg"),
              let cgImage = originalImage.cgImage else {
            XCTFail("Could not load room.jpeg")
            return
        }

        print("\n=== Test: UIImage.fixedOrientation() extension ===")

        // Test all orientations
        let orientations: [UIImage.Orientation] = [.up, .down, .left, .right, .upMirrored, .downMirrored, .leftMirrored, .rightMirrored]

        for orientation in orientations {
            let rotatedImage = UIImage(cgImage: cgImage, scale: originalImage.scale, orientation: orientation)
            let fixedImage = rotatedImage.fixedOrientation()

            print("  Orientation \(orientation.rawValue): original=\(rotatedImage.size), fixed=\(fixedImage.size), fixedOrientation=\(fixedImage.imageOrientation.rawValue)")

            // Fixed image should always have .up orientation
            XCTAssertEqual(fixedImage.imageOrientation, .up, "Fixed image should have .up orientation for input orientation \(orientation.rawValue)")

            // Fixed image should have valid CGImage
            XCTAssertNotNil(fixedImage.cgImage, "Fixed image should have valid CGImage")
        }

        print("✅ Test passed - all orientations handled correctly")
    }

    // MARK: - Save Cropped Image for Visual Inspection

    func testCropAndSaveForInspection() {
        guard let image = loadTestImage(named: "room", extension: "jpeg") else {
            XCTFail("Could not load room.jpeg")
            return
        }

        print("\n=== Test: Crop and save for visual inspection ===")

        // Use realistic boundaries that a user might select
        let cropped = cropImageToFrontWall(
            image: image,
            leftX: 0.12,
            rightX: 0.88,
            ceilingY: 0.15,
            floorY: 0.85
        )

        // Save to temp directory for visual inspection
        let tempDir = FileManager.default.temporaryDirectory
        let originalPath = tempDir.appendingPathComponent("crop_test_original.jpg")
        let croppedPath = tempDir.appendingPathComponent("crop_test_cropped.jpg")

        if let originalData = image.jpegData(compressionQuality: 0.9) {
            try? originalData.write(to: originalPath)
            print("Original saved to: \(originalPath.path)")
        }

        if let croppedData = cropped.jpegData(compressionQuality: 0.9) {
            try? croppedData.write(to: croppedPath)
            print("Cropped saved to: \(croppedPath.path)")
        }

        print("\nOriginal size: \(image.size)")
        print("Cropped size: \(cropped.size)")
        print("\nOpen these files to visually verify the crop is correct:")
        print("  open \"\(originalPath.path)\"")
        print("  open \"\(croppedPath.path)\"")

        // Basic verification
        XCTAssertLessThan(cropped.size.width, image.size.width)
        XCTAssertLessThan(cropped.size.height, image.size.height)

        print("\n✅ Test passed - images saved for inspection")
    }
}
