// FurnitureFitTests.swift
// Unit tests for FurnitureFit segmentation utilities

import XCTest
@testable import Furnit

final class FurnitureFitTests: XCTestCase {

    // MARK: - Detection Tests

    func testDetectionInitialization() {
        let detection = FurnitureFitDetection(
            x: 100, y: 100, w: 50, h: 50,
            confidence: 0.9, classIdx: 5, coeffs: [1.0, 2.0, 3.0]
        )

        XCTAssertEqual(detection.x, 100)
        XCTAssertEqual(detection.y, 100)
        XCTAssertEqual(detection.w, 50)
        XCTAssertEqual(detection.h, 50)
        XCTAssertEqual(detection.confidence, 0.9)
        XCTAssertEqual(detection.classIdx, 5)
        XCTAssertEqual(detection.coeffs.count, 3)
    }

    func testDetectionBoundingBox() {
        // Center at (100, 100), size 50x50
        let detection = FurnitureFitDetection(
            x: 100, y: 100, w: 50, h: 50,
            confidence: 0.9, classIdx: 0
        )

        let box = detection.boundingBox
        XCTAssertEqual(box.origin.x, 75, accuracy: 0.001)  // 100 - 25
        XCTAssertEqual(box.origin.y, 75, accuracy: 0.001)  // 100 - 25
        XCTAssertEqual(box.width, 50, accuracy: 0.001)
        XCTAssertEqual(box.height, 50, accuracy: 0.001)
    }

    func testDetectionEquality() {
        let det1 = FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 5)
        let det2 = FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 5)
        let det3 = FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.8, classIdx: 5)

        XCTAssertEqual(det1, det2)
        XCTAssertNotEqual(det1, det3)
    }

    // MARK: - IoU Tests

    func testIoUPerfectOverlap() {
        let det1 = FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 0)
        let det2 = FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.8, classIdx: 0)

        let iou = FurnitureFitIoU.calculate(det1, det2)
        XCTAssertEqual(iou, 1.0, accuracy: 0.001)
    }

    func testIoUNoOverlap() {
        let det1 = FurnitureFitDetection(x: 50, y: 50, w: 50, h: 50, confidence: 0.9, classIdx: 0)
        let det2 = FurnitureFitDetection(x: 200, y: 200, w: 50, h: 50, confidence: 0.8, classIdx: 0)

        let iou = FurnitureFitIoU.calculate(det1, det2)
        XCTAssertEqual(iou, 0.0, accuracy: 0.001)
    }

    func testIoUPartialOverlap() {
        // Two boxes that overlap by 25x50 area
        let det1 = FurnitureFitDetection(x: 50, y: 50, w: 50, h: 50, confidence: 0.9, classIdx: 0)
        let det2 = FurnitureFitDetection(x: 75, y: 50, w: 50, h: 50, confidence: 0.8, classIdx: 0)

        // det1: (25-75, 25-75), det2: (50-100, 25-75)
        // Intersection: (50-75, 25-75) = 25x50 = 1250
        // Union: 2500 + 2500 - 1250 = 3750
        // IoU = 1250/3750 = 0.333...

        let iou = FurnitureFitIoU.calculate(det1, det2)
        XCTAssertEqual(iou, 0.333, accuracy: 0.01)
    }

    func testIoUCGRectPerfectOverlap() {
        let rect1 = CGRect(x: 0, y: 0, width: 100, height: 100)
        let rect2 = CGRect(x: 0, y: 0, width: 100, height: 100)

        let iou = FurnitureFitIoU.calculate(rect1, rect2)
        XCTAssertEqual(iou, 1.0, accuracy: 0.001)
    }

    func testIoUCGRectNoOverlap() {
        let rect1 = CGRect(x: 0, y: 0, width: 50, height: 50)
        let rect2 = CGRect(x: 100, y: 100, width: 50, height: 50)

        let iou = FurnitureFitIoU.calculate(rect1, rect2)
        XCTAssertEqual(iou, 0.0, accuracy: 0.001)
    }

    func testIoUCGRectPartialOverlap() {
        let rect1 = CGRect(x: 0, y: 0, width: 100, height: 100)
        let rect2 = CGRect(x: 50, y: 50, width: 100, height: 100)

        // Intersection: (50,50) to (100,100) = 50x50 = 2500
        // Union: 10000 + 10000 - 2500 = 17500
        // IoU = 2500/17500 = 0.142857...

        let iou = FurnitureFitIoU.calculate(rect1, rect2)
        XCTAssertEqual(iou, 0.1428, accuracy: 0.01)
    }

    // MARK: - NMS Tests

    func testNMSEmptyInput() {
        let result = FurnitureFitNMS.apply(detections: [], iouThreshold: 0.5)
        XCTAssertTrue(result.isEmpty)
    }

    func testNMSSingleDetection() {
        let detection = FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 0)
        let result = FurnitureFitNMS.apply(detections: [detection], iouThreshold: 0.5)

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], detection)
    }

    func testNMSNoOverlap() {
        let detections = [
            FurnitureFitDetection(x: 50, y: 50, w: 50, h: 50, confidence: 0.9, classIdx: 0),
            FurnitureFitDetection(x: 200, y: 200, w: 50, h: 50, confidence: 0.8, classIdx: 0)
        ]

        let result = FurnitureFitNMS.apply(detections: detections, iouThreshold: 0.5)
        XCTAssertEqual(result.count, 2)
    }

    func testNMSHighOverlapSuppression() {
        // Two nearly identical boxes - should suppress one
        let detections = [
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 0),
            FurnitureFitDetection(x: 102, y: 102, w: 50, h: 50, confidence: 0.7, classIdx: 0)
        ]

        let result = FurnitureFitNMS.apply(detections: detections, iouThreshold: 0.5)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].confidence, 0.9)  // Higher confidence kept
    }

    func testNMSKeepsHighestConfidence() {
        let detections = [
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.5, classIdx: 0),
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 0),
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.7, classIdx: 0)
        ]

        let result = FurnitureFitNMS.apply(detections: detections, iouThreshold: 0.5)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].confidence, 0.9)
    }

    func testNMSWithBoxesAndScores() {
        let boxes = [
            CGRect(x: 0, y: 0, width: 100, height: 100),
            CGRect(x: 10, y: 10, width: 100, height: 100),  // High overlap with first
            CGRect(x: 200, y: 200, width: 100, height: 100)  // No overlap
        ]
        let scores: [Float] = [0.9, 0.7, 0.8]

        let keptIndices = FurnitureFitNMS.apply(boxes: boxes, scores: scores, iouThreshold: 0.5)

        XCTAssertEqual(keptIndices.count, 2)
        XCTAssertTrue(keptIndices.contains(0))  // Highest score
        XCTAssertTrue(keptIndices.contains(2))  // No overlap
        XCTAssertFalse(keptIndices.contains(1))  // Suppressed
    }

    func testNMSLowThreshold() {
        // With very low threshold, even small overlaps cause suppression
        let detections = [
            FurnitureFitDetection(x: 100, y: 100, w: 100, h: 100, confidence: 0.9, classIdx: 0),
            FurnitureFitDetection(x: 150, y: 100, w: 100, h: 100, confidence: 0.8, classIdx: 0)
        ]

        let result = FurnitureFitNMS.apply(detections: detections, iouThreshold: 0.1)
        XCTAssertEqual(result.count, 1)
    }

    func testNMSHighThreshold() {
        // With very high threshold, only perfect overlaps cause suppression
        let detections = [
            FurnitureFitDetection(x: 100, y: 100, w: 100, h: 100, confidence: 0.9, classIdx: 0),
            FurnitureFitDetection(x: 150, y: 100, w: 100, h: 100, confidence: 0.8, classIdx: 0)
        ]

        let result = FurnitureFitNMS.apply(detections: detections, iouThreshold: 0.9)
        XCTAssertEqual(result.count, 2)
    }

    // MARK: - Filter Tests

    func testFilterByConfidence() {
        let detections = [
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 0),
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.5, classIdx: 0),
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.3, classIdx: 0)
        ]

        let filtered = FurnitureFitFilter.byConfidence(detections, threshold: 0.6)
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered[0].confidence, 0.9)
    }

    func testFilterByConfidenceEdgeCase() {
        let detections = [
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.5, classIdx: 0)
        ]

        let filtered = FurnitureFitFilter.byConfidence(detections, threshold: 0.5)
        XCTAssertEqual(filtered.count, 1)  // Equal to threshold should pass
    }

    func testFilterExcludingClasses() {
        let detections = [
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 1),
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 2),
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 3)
        ]

        let blacklist: Set<Int> = [1, 3]
        let filtered = FurnitureFitFilter.excludingClasses(detections, blacklist: blacklist)

        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered[0].classIdx, 2)
    }

    func testFilterCombined() {
        let detections = [
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 1),  // Blacklisted
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.3, classIdx: 2),  // Low confidence
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.8, classIdx: 3)   // Pass
        ]

        let filtered = FurnitureFitFilter.apply(
            detections: detections,
            confidenceThreshold: 0.5,
            classBlacklist: [1]
        )

        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered[0].classIdx, 3)
    }

    // MARK: - Mask Tests

    func testMaskHasContent() {
        let emptyMask: [UInt8] = [0, 0, 0, 0]
        let maskWithContent: [UInt8] = [0, 255, 0, 0]

        XCTAssertFalse(FurnitureFitMask.hasContent(emptyMask))
        XCTAssertTrue(FurnitureFitMask.hasContent(maskWithContent))
    }

    func testMaskPositivePixelCount() {
        let mask: [UInt8] = [0, 255, 255, 0, 255, 0]
        XCTAssertEqual(FurnitureFitMask.positivePixelCount(mask), 3)
    }

    func testMaskCoverage() {
        let mask: [UInt8] = [0, 255, 255, 0]  // 2 out of 4 = 50%
        XCTAssertEqual(FurnitureFitMask.coverage(mask), 0.5, accuracy: 0.001)
    }

    func testMaskCoverageEmpty() {
        let emptyMask: [UInt8] = []
        XCTAssertEqual(FurnitureFitMask.coverage(emptyMask), 0.0)
    }

    func testMaskThreshold() {
        let floatMask: [Float] = [-0.5, 0.0, 0.5, 1.0]
        let thresholded = FurnitureFitMask.threshold(floatMask, threshold: 0.0)

        XCTAssertEqual(thresholded, [0, 0, 255, 255])
    }

    func testMaskThresholdCustom() {
        let floatMask: [Float] = [0.1, 0.3, 0.5, 0.7]
        let thresholded = FurnitureFitMask.threshold(floatMask, threshold: 0.4)

        XCTAssertEqual(thresholded, [0, 0, 255, 255])
    }

    // MARK: - Performance Tests

    func testNMSPerformanceWith100Detections() {
        var detections: [FurnitureFitDetection] = []
        for i in 0..<100 {
            detections.append(FurnitureFitDetection(
                x: Float(i * 10),
                y: Float(i * 10),
                w: 50,
                h: 50,
                confidence: Float.random(in: 0.1...1.0),
                classIdx: Int.random(in: 0...10)
            ))
        }

        measure {
            _ = FurnitureFitNMS.apply(detections: detections, iouThreshold: 0.5)
        }
    }

    func testIoUPerformance() {
        let det1 = FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 0)
        let det2 = FurnitureFitDetection(x: 110, y: 110, w: 50, h: 50, confidence: 0.8, classIdx: 0)

        measure {
            for _ in 0..<10000 {
                _ = FurnitureFitIoU.calculate(det1, det2)
            }
        }
    }

    // MARK: - Rotation Decision Tests

    func testShouldRotateForPortrait_LandscapeBufferPortraitRoom() {
        // Landscape buffer with portrait room = should rotate
        let shouldRotate = FurnitureFitRotation.shouldRotateForPortrait(
            isLandscapeBuffer: true,
            isLandscapeRoom: false  // portrait room
        )
        XCTAssertTrue(shouldRotate)
    }

    func testShouldRotateForPortrait_LandscapeBufferLandscapeRoom() {
        // Landscape buffer with landscape room = should NOT rotate (room is naturally landscape)
        let shouldRotate = FurnitureFitRotation.shouldRotateForPortrait(
            isLandscapeBuffer: true,
            isLandscapeRoom: true  // landscape room
        )
        XCTAssertFalse(shouldRotate)
    }

    func testShouldRotateForPortrait_PortraitBuffer() {
        // Portrait buffer = should never rotate regardless of room orientation
        XCTAssertFalse(FurnitureFitRotation.shouldRotateForPortrait(
            isLandscapeBuffer: false,
            isLandscapeRoom: false  // portrait room
        ))
        XCTAssertFalse(FurnitureFitRotation.shouldRotateForPortrait(
            isLandscapeBuffer: false,
            isLandscapeRoom: true  // landscape room
        ))
    }

    func testShouldRotateForPortrait_SquareRoom() {
        // Square room (not landscape) with landscape buffer = should rotate
        let shouldRotate = FurnitureFitRotation.shouldRotateForPortrait(
            isLandscapeBuffer: true,
            isLandscapeRoom: false  // square room treated as not landscape
        )
        XCTAssertTrue(shouldRotate)
    }

    func testIsLandscape() {
        XCTAssertTrue(FurnitureFitRotation.isLandscape(width: 1920, height: 1080))
        XCTAssertTrue(FurnitureFitRotation.isLandscape(width: 1280, height: 720))
        XCTAssertFalse(FurnitureFitRotation.isLandscape(width: 1080, height: 1920))
        XCTAssertFalse(FurnitureFitRotation.isLandscape(width: 720, height: 1280))
        XCTAssertFalse(FurnitureFitRotation.isLandscape(width: 1000, height: 1000))  // Square
    }

    func testRotateClockwise_LandscapeLeft() {
        // UIDeviceOrientation.landscapeLeft.rawValue == 4
        XCTAssertTrue(FurnitureFitRotation.rotateClockwise(deviceOrientationRawValue: 4))
    }

    func testRotateClockwise_LandscapeRight() {
        // UIDeviceOrientation.landscapeRight.rawValue == 3
        XCTAssertFalse(FurnitureFitRotation.rotateClockwise(deviceOrientationRawValue: 3))
    }

    func testRotateClockwise_Portrait() {
        // UIDeviceOrientation.portrait.rawValue == 1
        XCTAssertFalse(FurnitureFitRotation.rotateClockwise(deviceOrientationRawValue: 1))
    }

    // MARK: - BBox Intersection Tests

    func testBBoxIntersects_Overlapping() {
        let box1 = (x1: Float(0), y1: Float(0), x2: Float(100), y2: Float(100))
        let box2 = (x1: Float(50), y1: Float(50), x2: Float(150), y2: Float(150))
        XCTAssertTrue(FurnitureFitBBox.intersects(box1, box2))
    }

    func testBBoxIntersects_NoOverlap() {
        let box1 = (x1: Float(0), y1: Float(0), x2: Float(50), y2: Float(50))
        let box2 = (x1: Float(100), y1: Float(100), x2: Float(150), y2: Float(150))
        XCTAssertFalse(FurnitureFitBBox.intersects(box1, box2))
    }

    func testBBoxIntersects_Touching() {
        // Boxes that share an edge
        let box1 = (x1: Float(0), y1: Float(0), x2: Float(50), y2: Float(50))
        let box2 = (x1: Float(50), y1: Float(0), x2: Float(100), y2: Float(50))
        // Touching at x=50 means x2 == x1, which is NOT < so they do intersect
        XCTAssertTrue(FurnitureFitBBox.intersects(box1, box2))
    }

    func testBBoxIntersects_OneInsideOther() {
        let outer = (x1: Float(0), y1: Float(0), x2: Float(100), y2: Float(100))
        let inner = (x1: Float(25), y1: Float(25), x2: Float(75), y2: Float(75))
        XCTAssertTrue(FurnitureFitBBox.intersects(outer, inner))
        XCTAssertTrue(FurnitureFitBBox.intersects(inner, outer))
    }

    func testBBoxEncompasses_LargerContaining() {
        let candidate = (x1: Float(0), y1: Float(0), x2: Float(200), y2: Float(200))
        let primary = (x1: Float(50), y1: Float(50), x2: Float(100), y2: Float(100))
        XCTAssertTrue(FurnitureFitBBox.encompasses(candidate: candidate, primary: primary))
    }

    func testBBoxEncompasses_SmallerCandidate() {
        let candidate = (x1: Float(50), y1: Float(50), x2: Float(100), y2: Float(100))
        let primary = (x1: Float(0), y1: Float(0), x2: Float(200), y2: Float(200))
        XCTAssertFalse(FurnitureFitBBox.encompasses(candidate: candidate, primary: primary))
    }

    func testBBoxEncompasses_PartialOverlap() {
        let candidate = (x1: Float(0), y1: Float(0), x2: Float(100), y2: Float(100))
        let primary = (x1: Float(50), y1: Float(50), x2: Float(150), y2: Float(150))
        XCTAssertFalse(FurnitureFitBBox.encompasses(candidate: candidate, primary: primary))
    }

    func testBBoxEncompasses_SlightlyLargerButNotContaining() {
        // Candidate is larger but doesn't fully contain primary
        let candidate = (x1: Float(0), y1: Float(0), x2: Float(150), y2: Float(150))
        let primary = (x1: Float(100), y1: Float(100), x2: Float(200), y2: Float(200))
        XCTAssertFalse(FurnitureFitBBox.encompasses(candidate: candidate, primary: primary))
    }

    // MARK: - PhotoOrientation Tests

    func testPhotoOrientationRawValues() {
        XCTAssertEqual(PhotoOrientation.portrait.rawValue, "portrait")
        XCTAssertEqual(PhotoOrientation.landscape.rawValue, "landscape")
        XCTAssertEqual(PhotoOrientation.square.rawValue, "square")
    }

    func testPhotoOrientationCodable() throws {
        let orientation = PhotoOrientation.landscape
        let encoded = try JSONEncoder().encode(orientation)
        let decoded = try JSONDecoder().decode(PhotoOrientation.self, from: encoded)
        XCTAssertEqual(decoded, orientation)
    }

    func testPhotoOrientationFromRawValue() {
        XCTAssertEqual(PhotoOrientation(rawValue: "portrait"), .portrait)
        XCTAssertEqual(PhotoOrientation(rawValue: "landscape"), .landscape)
        XCTAssertEqual(PhotoOrientation(rawValue: "square"), .square)
        XCTAssertNil(PhotoOrientation(rawValue: "invalid"))
    }

    // MARK: - Primary Selection Tests

    func testPrimarySelectionPrefersBalancedAreaScoreInsideShortlist() {
        let candidates = [
            FurnitureFitDetection(x: 80, y: 80, w: 10, h: 10, confidence: 0.92, classIdx: 1),
            FurnitureFitDetection(x: 120, y: 120, w: 20, h: 20, confidence: 0.80, classIdx: 1),
            FurnitureFitDetection(x: 160, y: 160, w: 18, h: 18, confidence: 0.78, classIdx: 1)
        ]
        let config = FurnitureFitPrimarySelectionConfig(
            minimumConfidence: 0.57,
            preferHighestConfidence: false,
            areaShortlistCount: 3,
            persistenceIoUThreshold: 0.45,
            switchRequiredFrames: 3,
            confidenceSwitchGain: 1.08,
            confidenceSwitchMargin: 0.03
        )

        let selectedIndex = FurnitureFitPrimarySelection.selectPrimaryIndex(
            candidates: candidates,
            config: config
        )

        XCTAssertEqual(selectedIndex, 1)
    }

    func testStableAutoPrimaryRequiresChallengerPersistenceBeforeSwitching() {
        let config = FurnitureFitPrimarySelectionConfig(
            minimumConfidence: 0.57,
            preferHighestConfidence: true,
            areaShortlistCount: 3,
            persistenceIoUThreshold: 0.45,
            switchRequiredFrames: 3,
            confidenceSwitchGain: 1.08,
            confidenceSwitchMargin: 0.03
        )
        var selectionState = FurnitureFitAutoPrimarySelectionState()
        let stableCandidate = FurnitureFitDetection(x: 100, y: 100, w: 60, h: 60, confidence: 0.72, classIdx: 1)
        let challengerCandidate = FurnitureFitDetection(x: 220, y: 220, w: 70, h: 70, confidence: 0.80, classIdx: 1)

        let initialIndex = FurnitureFitPrimarySelection.selectStableAutoPrimaryIndex(
            candidates: [stableCandidate],
            config: config,
            state: &selectionState
        )
        XCTAssertEqual(initialIndex, 0)

        let frameCandidates = [stableCandidate, challengerCandidate]
        let firstChallengerIndex = FurnitureFitPrimarySelection.selectStableAutoPrimaryIndex(
            candidates: frameCandidates,
            config: config,
            state: &selectionState
        )
        let secondChallengerIndex = FurnitureFitPrimarySelection.selectStableAutoPrimaryIndex(
            candidates: frameCandidates,
            config: config,
            state: &selectionState
        )
        let thirdChallengerIndex = FurnitureFitPrimarySelection.selectStableAutoPrimaryIndex(
            candidates: frameCandidates,
            config: config,
            state: &selectionState
        )

        XCTAssertEqual(firstChallengerIndex, 0)
        XCTAssertEqual(secondChallengerIndex, 0)
        XCTAssertEqual(thirdChallengerIndex, 1)
    }

    // MARK: - Tap Selection Tests

    func testTapSelectionRequiresPositiveMaskHitWhenSnapshotExists() {
        let tapPoint = CGPoint(x: 50, y: 50)
        let positiveCandidate = FurnitureFitDetection(
            x: 100, y: 100, w: 120, h: 120,
            confidence: 0.8, classIdx: 1, coeffs: [Float](repeating: 1, count: 32)
        )
        let negativeCandidate = FurnitureFitDetection(
            x: 100, y: 100, w: 120, h: 120,
            confidence: 0.95, classIdx: 1, coeffs: [Float](repeating: -1, count: 32)
        )
        let snapshot = FurnitureFitTapMaskSnapshot(
            planes: [Float](repeating: 1, count: 32),
            protoWidth: 1,
            protoHeight: 1,
            modelSide: 200,
            imageWidth: 100,
            imageHeight: 100
        )
        let context = FurnitureFitTapSelectionContext(
            pointInMaskView: tapPoint,
            maskViewBounds: CGRect(x: 0, y: 0, width: 100, height: 100),
            candidateRectsInView: [
                CGRect(x: 0, y: 0, width: 100, height: 100),
                CGRect(x: 0, y: 0, width: 100, height: 100)
            ],
            candidates: [negativeCandidate, positiveCandidate],
            tapMaskSnapshot: snapshot,
            isShowingLiveVideoIdentifications: false
        )

        let selectedIndex = FurnitureFitTapSelection.candidateIndex(context: context)

        XCTAssertEqual(selectedIndex, 1)
    }

    func testTapSelectionFallsBackToSmallerBoxWithoutMaskSnapshot() {
        let context = FurnitureFitTapSelectionContext(
            pointInMaskView: CGPoint(x: 50, y: 50),
            maskViewBounds: CGRect(x: 0, y: 0, width: 100, height: 100),
            candidateRectsInView: [
                CGRect(x: 0, y: 0, width: 80, height: 80),
                CGRect(x: 10, y: 10, width: 40, height: 40)
            ],
            candidates: [
                FurnitureFitDetection(x: 100, y: 100, w: 140, h: 140, confidence: 0.99, classIdx: 1),
                FurnitureFitDetection(x: 100, y: 100, w: 80, h: 80, confidence: 0.6, classIdx: 1)
            ],
            tapMaskSnapshot: FurnitureFitTapMaskSnapshot(
                planes: [],
                protoWidth: 0,
                protoHeight: 0,
                modelSide: 0,
                imageWidth: 0,
                imageHeight: 0
            ),
            isShowingLiveVideoIdentifications: true
        )

        let selectedIndex = FurnitureFitTapSelection.candidateIndex(context: context)

        XCTAssertEqual(selectedIndex, 1)
    }
}
