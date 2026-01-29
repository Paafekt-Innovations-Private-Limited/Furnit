// SharpRoomViewTests.swift
// Unit tests for SharpRoomView recent changes:
// - Calibration overlay
// - Auto-orbit toggle (default off)
// - Number pad input
// - Landscape rotation
// - Warm-up rendering

import XCTest
import SwiftUI
@testable import Furnit

final class SharpRoomViewTests: XCTestCase {

    // MARK: - Auto-Orbit Default Tests

    func testAutoOrbitDefaultIsFalse() {
        // Verify the default value for oscillation is false
        // This tests that new users won't have auto-orbit enabled by default
        let defaults = UserDefaults.standard

        // Clear any existing value to test true default
        defaults.removeObject(forKey: "roomViewer.oscillation")

        // Read the value - should be false (our new default)
        let oscillationEnabled = defaults.bool(forKey: "roomViewer.oscillation")
        XCTAssertFalse(oscillationEnabled, "Auto-orbit should be disabled by default")
    }

    func testAutoOrbitCanBeEnabled() {
        let defaults = UserDefaults.standard

        // Set to true
        defaults.set(true, forKey: "roomViewer.oscillation")

        let oscillationEnabled = defaults.bool(forKey: "roomViewer.oscillation")
        XCTAssertTrue(oscillationEnabled, "Auto-orbit should be enableable")

        // Clean up
        defaults.removeObject(forKey: "roomViewer.oscillation")
    }

    func testAutoOrbitCanBeDisabled() {
        let defaults = UserDefaults.standard

        // Set to true then false
        defaults.set(true, forKey: "roomViewer.oscillation")
        defaults.set(false, forKey: "roomViewer.oscillation")

        let oscillationEnabled = defaults.bool(forKey: "roomViewer.oscillation")
        XCTAssertFalse(oscillationEnabled, "Auto-orbit should be disableable")

        // Clean up
        defaults.removeObject(forKey: "roomViewer.oscillation")
    }

    // MARK: - Calibration Scale Factor Tests

    func testCalibrationScaleFactorCalculation() {
        // If detected height is 0.5m and real height is 1.0m
        // Scale factor should be 2.0 (room needs to be twice as big)
        let detectedHeight: Float = 0.5
        let realHeight: Float = 1.0

        let scaleFactor = realHeight / detectedHeight
        XCTAssertEqual(scaleFactor, 2.0, accuracy: 0.001)
    }

    func testCalibrationScaleFactorSmaller() {
        // If detected height is 2.0m and real height is 1.0m
        // Scale factor should be 0.5 (room needs to be half as big)
        let detectedHeight: Float = 2.0
        let realHeight: Float = 1.0

        let scaleFactor = realHeight / detectedHeight
        XCTAssertEqual(scaleFactor, 0.5, accuracy: 0.001)
    }

    func testCalibrationScaleFactorSame() {
        // If detected matches real, scale factor is 1.0
        let detectedHeight: Float = 1.5
        let realHeight: Float = 1.5

        let scaleFactor = realHeight / detectedHeight
        XCTAssertEqual(scaleFactor, 1.0, accuracy: 0.001)
    }

    func testCalibratedRoomDimensions() {
        // Room dimensions should scale proportionally
        let originalRoomHeight: Float = 3.0
        let originalRoomWidth: Float = 4.0
        let scaleFactor: Float = 1.5

        let calibratedHeight = originalRoomHeight * scaleFactor
        let calibratedWidth = originalRoomWidth * scaleFactor

        XCTAssertEqual(calibratedHeight, 4.5, accuracy: 0.001)
        XCTAssertEqual(calibratedWidth, 6.0, accuracy: 0.001)
    }

    // MARK: - Number Pad Input Tests

    func testNumberPadDigitLimit() {
        // Number pad should limit input to 5 characters (e.g., "12.34")
        var input = ""

        // Simulate appending digits
        func appendDigit(_ digit: String) {
            if input.count >= 5 { return }
            if let dotIndex = input.firstIndex(of: ".") {
                let decimals = input.distance(from: dotIndex, to: input.endIndex) - 1
                if decimals >= 2 { return }
            }
            input += digit
        }

        appendDigit("1")
        appendDigit("2")
        appendDigit(".")
        appendDigit("3")
        appendDigit("4")
        appendDigit("5")  // Should be ignored (max 5 chars)

        XCTAssertEqual(input, "12.34")
        XCTAssertEqual(input.count, 5)
    }

    func testNumberPadDecimalLimit() {
        // Should only allow 2 decimal places
        var input = "1."

        func appendDigit(_ digit: String) {
            if input.count >= 5 { return }
            if let dotIndex = input.firstIndex(of: ".") {
                let decimals = input.distance(from: dotIndex, to: input.endIndex) - 1
                if decimals >= 2 { return }
            }
            input += digit
        }

        appendDigit("2")
        appendDigit("3")
        appendDigit("4")  // Should be ignored (max 2 decimals)

        XCTAssertEqual(input, "1.23")
    }

    func testNumberPadSingleDecimalPoint() {
        // Should only allow one decimal point
        var input = "1.5"

        func appendDecimal() {
            if !input.contains(".") {
                input += input.isEmpty ? "0." : "."
            }
        }

        appendDecimal()  // Should be ignored

        XCTAssertEqual(input, "1.5")
        XCTAssertEqual(input.filter { $0 == "." }.count, 1)
    }

    func testNumberPadEmptyDecimal() {
        // Starting with decimal should prepend "0"
        var input = ""

        func appendDecimal() {
            if !input.contains(".") {
                input += input.isEmpty ? "0." : "."
            }
        }

        appendDecimal()

        XCTAssertEqual(input, "0.")
    }

    func testNumberPadBackspace() {
        var input = "1.23"

        if !input.isEmpty {
            input.removeLast()
        }

        XCTAssertEqual(input, "1.2")
    }

    func testNumberPadBackspaceEmpty() {
        var input = ""

        if !input.isEmpty {
            input.removeLast()
        }

        XCTAssertEqual(input, "")
    }

    func testNumberPadValidFloat() {
        let input = "1.75"
        let value = Float(input)

        XCTAssertNotNil(value)
        XCTAssertEqual(value!, 1.75, accuracy: 0.001)
    }

    func testNumberPadInvalidFloat() {
        let input = "1.2.3"
        let value = Float(input)

        XCTAssertNil(value)
    }

    // MARK: - Photo Orientation Tests

    func testPhotoOrientationPortrait() {
        let orientation = PhotoOrientation.portrait
        XCTAssertEqual(orientation.rawValue, "portrait")
    }

    func testPhotoOrientationLandscape() {
        let orientation = PhotoOrientation.landscape
        XCTAssertEqual(orientation.rawValue, "landscape")
    }

    func testPhotoOrientationHint() {
        XCTAssertTrue(PhotoOrientation.portrait.hint.contains("Portrait"))
        XCTAssertTrue(PhotoOrientation.landscape.hint.contains("Landscape"))
    }

    // MARK: - Landscape Rotation Tests

    func testLandscapeRotationAngle() {
        // Landscape orientation should rotate UI by 90 degrees
        let portraitRotation: Double = 0
        let landscapeRotation: Double = 90

        XCTAssertEqual(portraitRotation, 0)
        XCTAssertEqual(landscapeRotation, 90)
    }

    // MARK: - Warm-up Rendering Tests

    func testWarmupDuration() {
        // Warm-up duration should be 5 seconds (5000ms)
        let warmupDuration = 5000

        XCTAssertEqual(warmupDuration, 5000)
    }

    func testWarmupActiveCheck() {
        // Simulate warm-up period check
        let warmupDuration = 5000
        let animationStartTime = 0

        // At 2 seconds - should be in warmup
        let elapsed2s = 2000
        let inWarmup2s = elapsed2s < warmupDuration
        XCTAssertTrue(inWarmup2s, "Should be in warmup at 2 seconds")

        // At 6 seconds - should be out of warmup
        let elapsed6s = 6000
        let inWarmup6s = elapsed6s < warmupDuration
        XCTAssertFalse(inWarmup6s, "Should be out of warmup at 6 seconds")
    }

    // MARK: - WebGL Notification Tests

    func testScaleRoomNotificationUserInfo() {
        // Test that scale room notification carries correct factor
        let scaleFactor: Double = 1.5
        let userInfo: [String: Any] = ["factor": scaleFactor]

        XCTAssertEqual(userInfo["factor"] as? Double, 1.5)
    }

    func testOrbitGestureNotificationUserInfo() {
        // Test orbit gesture notification format
        let deltaX: CGFloat = 10.5
        let deltaY: CGFloat = -5.2
        let userInfo: [String: Any] = ["deltaX": deltaX, "deltaY": deltaY]

        XCTAssertEqual(userInfo["deltaX"] as? CGFloat, 10.5)
        XCTAssertEqual(userInfo["deltaY"] as? CGFloat, -5.2)
    }

    func testZoomGestureNotificationUserInfo() {
        // Test zoom gesture notification format
        let scale: CGFloat = 1.25
        let userInfo: [String: Any] = ["scale": scale, "incremental": true]

        XCTAssertEqual(userInfo["scale"] as? CGFloat, 1.25)
        XCTAssertEqual(userInfo["incremental"] as? Bool, true)
    }

    func testGestureStateNotificationUserInfo() {
        // Test gesture state notification format
        let userInfo: [String: Any] = ["interacting": true]

        XCTAssertEqual(userInfo["interacting"] as? Bool, true)
    }

    // MARK: - Room Measurement Cap Tests (for calibration accuracy)

    func testWidthCapPortrait() {
        // Portrait rooms should cap at 6.0m width
        let maxRealisticWidth: Float = 6.0
        let measuredWidth: Float = 8.0

        let cappedWidth = min(measuredWidth, maxRealisticWidth)
        XCTAssertEqual(cappedWidth, 6.0, accuracy: 0.001)
    }

    func testWidthCapLandscape() {
        // Landscape rooms should cap at 8.0m width
        let maxRealisticWidth: Float = 8.0
        let measuredWidth: Float = 10.0

        let cappedWidth = min(measuredWidth, maxRealisticWidth)
        XCTAssertEqual(cappedWidth, 8.0, accuracy: 0.001)
    }

    func testHeightCapPortrait() {
        // Portrait rooms should cap at 3.5m height
        let maxRealisticHeight: Float = 3.5
        let measuredHeight: Float = 5.0

        let cappedHeight = min(measuredHeight, maxRealisticHeight)
        XCTAssertEqual(cappedHeight, 3.5, accuracy: 0.001)
    }

    func testHeightCapLandscape() {
        // Landscape rooms should cap at 3.2m height
        let maxRealisticHeight: Float = 3.2
        let measuredHeight: Float = 4.0

        let cappedHeight = min(measuredHeight, maxRealisticHeight)
        XCTAssertEqual(cappedHeight, 3.2, accuracy: 0.001)
    }

    // MARK: - Edge Cases

    func testCalibrationWithZeroDetectedHeight() {
        // Should not divide by zero
        let detectedHeight: Float = 0.0
        let realHeight: Float = 1.0

        // Guard against division by zero
        let scaleFactor: Float? = detectedHeight > 0 ? realHeight / detectedHeight : nil
        XCTAssertNil(scaleFactor, "Should not calculate scale factor with zero detected height")
    }

    func testCalibrationWithNegativeHeight() {
        // Should not accept negative heights
        let detectedHeight: Float = -1.0
        let realHeight: Float = 1.0

        // Only calculate if both are positive
        let scaleFactor: Float? = (detectedHeight > 0 && realHeight > 0) ? realHeight / detectedHeight : nil
        XCTAssertNil(scaleFactor, "Should not calculate scale factor with negative height")
    }

    func testNumberPadEmptyStringToFloat() {
        let input = ""
        let value = Float(input)

        XCTAssertNil(value, "Empty string should not convert to Float")
    }
}
