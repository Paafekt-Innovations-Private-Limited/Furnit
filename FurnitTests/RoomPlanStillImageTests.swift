import XCTest
import UIKit

#if canImport(RoomPlan)
import RoomPlan
#endif

final class RoomPlanStillImageTests: XCTestCase {

    func testRoomPlanCannotRunFromSingleStillImage() throws {
        let imagePath = "/Users/al/Downloads/WhatsApp Image 2026-07-06 at 10.21.21.jpeg"
        guard FileManager.default.fileExists(atPath: imagePath),
              let image = UIImage(contentsOfFile: imagePath) else {
            throw XCTSkip("Still-image fixture is not available at \(imagePath)")
        }
        XCTAssertGreaterThan(image.size.width, 0)
        XCTAssertGreaterThan(image.size.height, 0)

        #if canImport(RoomPlan)
        if #available(iOS 16.0, *) {
            // RoomPlan exposes live LiDAR capture (`RoomCaptureSession` / `RoomCaptureView`)
            // and captured-room export types. It does not expose an API that accepts UIImage,
            // CGImage, or CVPixelBuffer and returns a floor plan from a single RGB photo.
            XCTAssertNotNil(RoomCaptureSession.self)
            throw XCTSkip("RoomPlan is live LiDAR capture only; there is no still-image RoomPlan API to invoke for this JPEG.")
        } else {
            throw XCTSkip("RoomPlan requires iOS 16 or newer.")
        }
        #else
        throw XCTSkip("RoomPlan framework is not available in this test environment.")
        #endif
    }
}
