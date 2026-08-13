import UIKit
import XCTest
@testable import Furnit

final class SplatRoomViewTests: XCTestCase {
    private let autoOrbitKey = "roomViewer.oscillation"

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: autoOrbitKey)
        super.tearDown()
    }

    func testAutoOrbitDefaultIsFalse() {
        UserDefaults.standard.removeObject(forKey: autoOrbitKey)

        XCTAssertFalse(UserDefaults.standard.bool(forKey: autoOrbitKey))
    }

    func testAutoOrbitCanBeEnabled() {
        UserDefaults.standard.set(true, forKey: autoOrbitKey)

        XCTAssertTrue(UserDefaults.standard.bool(forKey: autoOrbitKey))
    }

    func testAutoOrbitCanBeDisabled() {
        UserDefaults.standard.set(true, forKey: autoOrbitKey)
        UserDefaults.standard.set(false, forKey: autoOrbitKey)

        XCTAssertFalse(UserDefaults.standard.bool(forKey: autoOrbitKey))
    }

    func testPhotoOrientationRawValues() {
        XCTAssertEqual(PhotoOrientation.portrait.rawValue, "portrait")
        XCTAssertEqual(PhotoOrientation.landscape.rawValue, "landscape")
    }

    func testPhotoOrientationHints() {
        XCTAssertTrue(PhotoOrientation.portrait.hint.contains("Portrait"))
        XCTAssertTrue(PhotoOrientation.landscape.hint.contains("Landscape"))
    }

    func testPhotoOrientationUsesNormalizedDisplayDimensions() {
        let portrait = makeImage(width: 80, height: 120)
        let landscape = makeImage(width: 120, height: 80)
        let square = makeImage(width: 100, height: 100)

        XCTAssertEqual(PhotoOrientation.detect(from: portrait), .portrait)
        XCTAssertEqual(PhotoOrientation.detect(from: landscape), .landscape)
        XCTAssertEqual(PhotoOrientation.detect(from: square), .square)
    }

    func testPhotoOrientationAppliesExifAxisSwap() throws {
        let sensorLandscape = makeImage(width: 120, height: 80)
        let cgImage = try XCTUnwrap(sensorLandscape.cgImage)
        let portraitViaExif = UIImage(cgImage: cgImage, scale: 1, orientation: .right)

        XCTAssertEqual(PhotoOrientation.detect(from: portraitViaExif), .portrait)
    }

    func testDepthAnythingDepthMeshPreservesLandscapePhotoAspect() {
        let dimensions = DepthAnythingRoomReconstructor.depthMeshDisplayDimensions(
            imageWidth: 1600,
            imageHeight: 900,
            measuredHeightMeters: 2.7
        )

        XCTAssertEqual(dimensions.height, 2.7, accuracy: 0.0001)
        XCTAssertEqual(dimensions.width / dimensions.height, 1600.0 / 900.0, accuracy: 0.0001)
    }

    func testDepthAnythingDepthMeshPreservesPortraitAspectAndMinimumWidth() {
        let dimensions = DepthAnythingRoomReconstructor.depthMeshDisplayDimensions(
            imageWidth: 900,
            imageHeight: 1600,
            measuredHeightMeters: 2.4
        )

        XCTAssertGreaterThanOrEqual(dimensions.width, 2.0)
        XCTAssertEqual(dimensions.width / dimensions.height, 900.0 / 1600.0, accuracy: 0.0001)
    }

    private func makeImage(width: CGFloat, height: CGFloat) -> UIImage {
        UIGraphicsImageRenderer(size: CGSize(width: width, height: height)).image { context in
            UIColor.black.setFill()
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        }
    }
}
