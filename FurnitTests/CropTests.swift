import XCTest
import UIKit
@testable import Furnit

final class CropTests: XCTestCase {
    func testFixedOrientationExtension() throws {
        let bundle = Bundle(for: type(of: self))
        let imageURL = try XCTUnwrap(
            bundle.url(forResource: "rtmdet_repeated_chair_frame", withExtension: "jpg")
        )
        let originalImage = try XCTUnwrap(UIImage(contentsOfFile: imageURL.path))
        let cgImage = try XCTUnwrap(originalImage.cgImage)
        let orientations: [UIImage.Orientation] = [
            .up,
            .down,
            .left,
            .right,
            .upMirrored,
            .downMirrored,
            .leftMirrored,
            .rightMirrored,
        ]

        for orientation in orientations {
            let rotatedImage = UIImage(
                cgImage: cgImage,
                scale: originalImage.scale,
                orientation: orientation
            )
            let fixedImage = rotatedImage.fixedOrientation()

            XCTAssertEqual(
                fixedImage.imageOrientation,
                .up,
                "Fixed image should have .up orientation for input orientation \(orientation.rawValue)"
            )
            XCTAssertNotNil(fixedImage.cgImage)
        }
    }
}
