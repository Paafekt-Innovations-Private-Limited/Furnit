import XCTest
import UIKit

final class SavedRoomNavigationUITests: XCTestCase {
    private let roomName = "E2E Saved Room 0715"
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = true
        app = XCUIApplication()
        app.launchArguments = ["-PaafektScreenshotHome", "-PaafektSavedRoomE2E"]
        app.launchEnvironment["PAAFEKT_UI_TEST_ROOM_NAME"] = roomName
        app.launch()
    }

    override func tearDownWithError() throws {
        app?.terminate()
        let cleanupApp = XCUIApplication()
        cleanupApp.launchArguments = ["-PaafektScreenshotHome", "-PaafektSavedRoomE2E"]
        cleanupApp.launchEnvironment["PAAFEKT_UI_TEST_ROOM_NAME"] = roomName
        cleanupApp.launch()
        cleanupApp.terminate()
    }

    func testCreateSaveReopenAndNavigateWithoutRendererHoles() throws {
        let createRoom = app.buttons["create_room_button"]
        XCTAssertTrue(createRoom.waitForExistence(timeout: 20), "Create Room did not appear")
        createRoom.tap()

        let aiRoom = app.buttons["ai_room_option"]
        XCTAssertTrue(aiRoom.waitForExistence(timeout: 20), "Injected room photo did not reach the method picker")
        aiRoom.tap()

        let previewViewport = app.descendants(matching: .any)["preview_room_viewport"]
        XCTAssertTrue(previewViewport.waitForExistence(timeout: 45), "AI preview did not open")
        waitForRendering()
        let preview = screenshot(named: "01-preview")
        assertNoRendererHoles(preview, in: previewViewport.frame, rendererGray: 31, label: "preview")

        let saveRoom = app.buttons["save_room_button"]
        XCTAssertTrue(saveRoom.waitForExistence(timeout: 15), "Preview Save button is unavailable")
        saveRoom.tap()

        let nameInput = app.textFields["room_name_input"]
        XCTAssertTrue(nameInput.waitForExistence(timeout: 10), "Room-name sheet did not open")
        nameInput.tap()
        nameInput.typeText(roomName)
        app.buttons["room_name_save_button"].tap()

        let savedRoom = app.staticTexts[roomName]
        XCTAssertTrue(savedRoom.waitForExistence(timeout: 300), "Saved room did not return to Home")
        savedRoom.tap()

        let savedViewport = app.descendants(matching: .any)["saved_room_viewport"]
        XCTAssertTrue(savedViewport.waitForExistence(timeout: 60), "Saved RealityKit room did not open")
        waitForRendering()

        let baseline = screenshot(named: "02-saved-baseline")
        assertNoRendererHoles(baseline, in: savedViewport.frame, rendererGray: 31, label: "saved baseline")

        // The stable iOS save format is a textured photo plane. Exercise its zoom, pan and
        // D-pad paths while checking that the saved renderer never exposes its background.
        savedViewport.pinch(withScale: 1.65, velocity: 1.0)
        waitForRendering()
        let afterPinch = screenshot(named: "03-after-pinch")
        assertFrameChanged(baseline, afterPinch, in: savedViewport.frame, action: "pinch zoom")
        assertNoRendererHoles(afterPinch, in: savedViewport.frame, rendererGray: 31, label: "after pinch")

        savedViewport.swipeRight(velocity: .slow)
        waitForRendering()
        let afterLook = screenshot(named: "04-after-look")
        assertFrameChanged(afterPinch, afterLook, in: savedViewport.frame, action: "one-finger pan")
        assertNoRendererHoles(afterLook, in: savedViewport.frame, rendererGray: 31, label: "after pan")

        // Step back in the opposite direction so this verifies D-pad handling even if the swipe
        // reached the photo plane's pan boundary.
        let dpadLeft = app.buttons["camera_dpad_left"]
        XCTAssertTrue(dpadLeft.waitForExistence(timeout: 10), "Saved-room D-pad is unavailable")
        dpadLeft.tap()
        waitForRendering()
        let afterDpad = screenshot(named: "05-after-dpad")
        assertFrameChanged(afterLook, afterDpad, in: savedViewport.frame, action: "D-pad")
        assertNoRendererHoles(afterDpad, in: savedViewport.frame, rendererGray: 31, label: "after D-pad")
    }

    private func waitForRendering() {
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 5))
        Thread.sleep(forTimeInterval: 0.8)
    }

    private func screenshot(named name: String) -> UIImage {
        let capture = XCUIScreen.main.screenshot()
        let attachment = XCTAttachment(screenshot: capture)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        guard let image = UIImage(data: capture.pngRepresentation) else {
            XCTFail("Could not decode screenshot \(name)")
            return UIImage()
        }
        return image
    }

    private func assertNoRendererHoles(
        _ image: UIImage,
        in frame: CGRect,
        rendererGray: UInt8,
        label: String
    ) {
        guard let raster = ScreenshotRaster(image: image, appFrame: app.frame) else {
            XCTFail("Could not read \(label) screenshot")
            return
        }
        let crop = raster.insetPixelRect(for: frame)
        var sampled = 0
        var rendererPixels = 0
        for y in stride(from: crop.minY, to: crop.maxY, by: 4) {
            for x in stride(from: crop.minX, to: crop.maxX, by: 4) {
                let pixel = raster.pixel(x: x, y: y)
                sampled += 1
                if abs(Int(pixel.r) - Int(rendererGray)) <= 3,
                   abs(Int(pixel.g) - Int(rendererGray)) <= 3,
                   abs(Int(pixel.b) - Int(rendererGray)) <= 3 {
                    rendererPixels += 1
                }
            }
        }
        let ratio = Double(rendererPixels) / Double(max(sampled, 1))
        XCTAssertLessThan(ratio, 0.025, "\(label) contains \(Int(ratio * 100))% renderer-background holes")
    }

    private func assertFrameChanged(
        _ before: UIImage,
        _ after: UIImage,
        in frame: CGRect,
        action: String
    ) {
        guard let first = ScreenshotRaster(image: before, appFrame: app.frame),
              let second = ScreenshotRaster(image: after, appFrame: app.frame) else {
            XCTFail("Could not compare screenshots after \(action)")
            return
        }
        let crop = first.insetPixelRect(for: frame)
        var totalDelta = 0
        var channels = 0
        for y in stride(from: crop.minY, to: crop.maxY, by: 8) {
            for x in stride(from: crop.minX, to: crop.maxX, by: 8) {
                let lhs = first.pixel(x: x, y: y)
                let rhs = second.pixel(x: x, y: y)
                totalDelta += abs(Int(lhs.r) - Int(rhs.r))
                totalDelta += abs(Int(lhs.g) - Int(rhs.g))
                totalDelta += abs(Int(lhs.b) - Int(rhs.b))
                channels += 3
            }
        }
        let meanDelta = Double(totalDelta) / Double(max(channels, 1))
        XCTAssertGreaterThan(meanDelta, 0.7, "\(action) did not move the rendered camera (mean delta \(meanDelta))")
    }
}

private struct ScreenshotRaster {
    struct Pixel { let r: UInt8; let g: UInt8; let b: UInt8 }

    let width: Int
    let height: Int
    let bytes: [UInt8]
    let scaleX: CGFloat
    let scaleY: CGFloat

    init?(image: UIImage, appFrame: CGRect) {
        guard let cgImage = image.cgImage, appFrame.width > 0, appFrame.height > 0 else { return nil }
        width = cgImage.width
        height = cgImage.height
        scaleX = CGFloat(width) / appFrame.width
        scaleY = CGFloat(height) / appFrame.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        bytes = pixels
    }

    func insetPixelRect(for pointRect: CGRect) -> (minX: Int, minY: Int, maxX: Int, maxY: Int) {
        let horizontalInset = pointRect.width * 0.10
        let verticalInset = pointRect.height * 0.14
        return (
            max(0, Int((pointRect.minX + horizontalInset) * scaleX)),
            max(0, Int((pointRect.minY + verticalInset) * scaleY)),
            min(width, Int((pointRect.maxX - horizontalInset) * scaleX)),
            min(height, Int((pointRect.maxY - verticalInset) * scaleY))
        )
    }

    func pixel(x: Int, y: Int) -> Pixel {
        let offset = (y * width + x) * 4
        return Pixel(r: bytes[offset], g: bytes[offset + 1], b: bytes[offset + 2])
    }
}
