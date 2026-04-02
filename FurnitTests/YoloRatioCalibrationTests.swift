import XCTest
@testable import Furnit

final class YoloRatioCalibrationTests: XCTestCase {

    func testCanonicalLabelMapsSofaToCouch() {
        XCTAssertEqual(YoloRatioCalibration.canonicalLabel(raw: "  Sofa "), "couch")
        XCTAssertEqual(YoloRatioCalibration.canonicalLabel(raw: "dining table"), "dining table")
    }

    func testWallHeightFractionUsesWidestWallLikeBox() {
        let imageSize = CGSize(width: 800, height: 600)
        let boxes: [YoloCalibrationBox] = [
            YoloCalibrationBox(label: "chair", centerX: 400, centerY: 300, width: 80, height: 120, confidence: 0.9),
            YoloCalibrationBox(label: "segment", centerX: 50, centerY: 300, width: 500, height: 200, confidence: 0.5)
        ]
        let frac = YoloRatioCalibration.wallHeightFractionOrFullFrame(imageSize: imageSize, boxes: boxes)
        XCTAssertGreaterThan(frac, 0.2)
        XCTAssertLessThanOrEqual(frac, 1.0)
    }

    func testWallHeightFractionFullFrameWhenNoWall() {
        let imageSize = CGSize(width: 400, height: 400)
        let boxes: [YoloCalibrationBox] = [
            YoloCalibrationBox(label: "chair", centerX: 200, centerY: 200, width: 60, height: 90, confidence: 0.9)
        ]
        XCTAssertEqual(YoloRatioCalibration.wallHeightFractionOrFullFrame(imageSize: imageSize, boxes: boxes), 1.0, accuracy: 0.001)
    }

    func testFurnitureHeightFractionsMedianByLabel() {
        let imageHeight: CGFloat = 1000
        let boxes: [YoloCalibrationBox] = [
            YoloCalibrationBox(label: "chair", centerX: 100, centerY: 500, width: 50, height: 200, confidence: 0.9),
            YoloCalibrationBox(label: "chair", centerX: 200, centerY: 500, width: 50, height: 400, confidence: 0.8)
        ]
        let map = YoloRatioCalibration.furnitureHeightFractionsByLabel(imageHeight: imageHeight, boxes: boxes)
        // 75th percentile of [0.2, 0.4] clamped fractions → upper sample
        XCTAssertEqual(map["chair"], 0.4, accuracy: 0.02)
    }

    func testCropRectForTargetFurnitureFraction() {
        let rect = FurnitureFitRatioGeometry.cropRectForTargetFurnitureFraction(
            frameWidth: 1000,
            frameHeight: 800,
            bboxMinX: 400,
            bboxMinY: 200,
            bboxMaxX: 600,
            bboxMaxY: 500,
            rTarget: 0.35
        )
        XCTAssertNotNil(rect)
        guard let crop = rect else { return }
        XCTAssertGreaterThanOrEqual(crop.width, 32)
        XCTAssertGreaterThanOrEqual(crop.height, 32)
    }
}
