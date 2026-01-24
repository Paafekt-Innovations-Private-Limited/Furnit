// IntegrationTests.swift
// Integration tests to verify utility functions match original code behavior

import XCTest
import simd
import CoreGraphics
@testable import Furnit

/// Integration tests that verify utility functions produce identical results
/// to the original implementations in the main codebase.
final class IntegrationTests: XCTestCase {

    // MARK: - IoU Integration Tests

    /// Test that FurnitureFitIoU.calculate matches the original iou() function logic
    /// Original: private func iou(_ a: UnionDet, _ b: UnionDet) -> Float in FurnitureFitView.swift
    func testIoUMatchesOriginalImplementation() {
        // Test cases with known expected values based on original algorithm:
        // ax1 = a.x - a.w * 0.5, ax2 = a.x + a.w * 0.5
        // intersection = max(0, min(ax2,bx2) - max(ax1,bx1)) * max(0, min(ay2,by2) - max(ay1,by1))
        // union = a.w*a.h + b.w*b.h - intersection
        // iou = intersection / union

        let testCases: [(a: FurnitureFitDetection, b: FurnitureFitDetection, expectedIoU: Float)] = [
            // Perfect overlap
            (
                FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 0),
                FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.8, classIdx: 0),
                1.0
            ),
            // No overlap - boxes far apart
            (
                FurnitureFitDetection(x: 50, y: 50, w: 50, h: 50, confidence: 0.9, classIdx: 0),
                FurnitureFitDetection(x: 200, y: 200, w: 50, h: 50, confidence: 0.8, classIdx: 0),
                0.0
            ),
            // Partial overlap - calculate manually
            // a: center(100,100), size(50,50) -> (75-125, 75-125)
            // b: center(120,100), size(50,50) -> (95-145, 75-125)
            // intersection: x=[95,125]=30, y=[75,125]=50 -> 30*50=1500
            // union: 2500 + 2500 - 1500 = 3500
            // iou = 1500/3500 = 0.4286
            (
                FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 0),
                FurnitureFitDetection(x: 120, y: 100, w: 50, h: 50, confidence: 0.8, classIdx: 0),
                0.4286
            ),
            // Edge touching - no overlap
            // a: center(50,50), size(50,50) -> (25-75, 25-75)
            // b: center(100,50), size(50,50) -> (75-125, 25-75)
            // intersection: x=[75,75]=0, y=[25,75]=50 -> 0
            (
                FurnitureFitDetection(x: 50, y: 50, w: 50, h: 50, confidence: 0.9, classIdx: 0),
                FurnitureFitDetection(x: 100, y: 50, w: 50, h: 50, confidence: 0.8, classIdx: 0),
                0.0
            ),
            // One box inside another
            // a: center(100,100), size(100,100) -> (50-150, 50-150)
            // b: center(100,100), size(50,50) -> (75-125, 75-125)
            // intersection: 50*50 = 2500
            // union: 10000 + 2500 - 2500 = 10000
            // iou = 2500/10000 = 0.25
            (
                FurnitureFitDetection(x: 100, y: 100, w: 100, h: 100, confidence: 0.9, classIdx: 0),
                FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.8, classIdx: 0),
                0.25
            )
        ]

        for (index, testCase) in testCases.enumerated() {
            let utilityResult = FurnitureFitIoU.calculate(testCase.a, testCase.b)
            XCTAssertEqual(utilityResult, testCase.expectedIoU, accuracy: 0.01,
                "IoU mismatch in test case \(index): expected \(testCase.expectedIoU), got \(utilityResult)")
        }
    }

    /// Test IoU symmetry - iou(a,b) should equal iou(b,a)
    func testIoUSymmetry() {
        let detections = [
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 0),
            FurnitureFitDetection(x: 120, y: 110, w: 60, h: 40, confidence: 0.8, classIdx: 1),
            FurnitureFitDetection(x: 80, y: 90, w: 70, h: 70, confidence: 0.7, classIdx: 2)
        ]

        for i in 0..<detections.count {
            for j in 0..<detections.count {
                let iouAB = FurnitureFitIoU.calculate(detections[i], detections[j])
                let iouBA = FurnitureFitIoU.calculate(detections[j], detections[i])
                XCTAssertEqual(iouAB, iouBA, accuracy: 0.0001,
                    "IoU not symmetric for detections \(i) and \(j)")
            }
        }
    }

    // MARK: - NMS Integration Tests

    /// Test that FurnitureFitNMS.apply matches original applyNMS behavior
    func testNMSMatchesOriginalImplementation() {
        // Test case 1: No suppression needed (no overlap)
        let noOverlapDetections = [
            FurnitureFitDetection(x: 50, y: 50, w: 40, h: 40, confidence: 0.9, classIdx: 0),
            FurnitureFitDetection(x: 150, y: 50, w: 40, h: 40, confidence: 0.8, classIdx: 0),
            FurnitureFitDetection(x: 250, y: 50, w: 40, h: 40, confidence: 0.7, classIdx: 0)
        ]
        let noOverlapResult = FurnitureFitNMS.apply(detections: noOverlapDetections, iouThreshold: 0.5)
        XCTAssertEqual(noOverlapResult.count, 3, "NMS should keep all non-overlapping detections")

        // Test case 2: High overlap - should suppress lower confidence
        let highOverlapDetections = [
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 0),
            FurnitureFitDetection(x: 105, y: 105, w: 50, h: 50, confidence: 0.7, classIdx: 0),
            FurnitureFitDetection(x: 102, y: 98, w: 50, h: 50, confidence: 0.5, classIdx: 0)
        ]
        let highOverlapResult = FurnitureFitNMS.apply(detections: highOverlapDetections, iouThreshold: 0.5)
        XCTAssertEqual(highOverlapResult.count, 1, "NMS should suppress overlapping detections")
        XCTAssertEqual(highOverlapResult[0].confidence, 0.9, "NMS should keep highest confidence")

        // Test case 3: Mixed - some overlap, some not
        let mixedDetections = [
            FurnitureFitDetection(x: 50, y: 50, w: 50, h: 50, confidence: 0.9, classIdx: 0),
            FurnitureFitDetection(x: 55, y: 55, w: 50, h: 50, confidence: 0.85, classIdx: 0),  // Overlaps with first
            FurnitureFitDetection(x: 200, y: 200, w: 50, h: 50, confidence: 0.8, classIdx: 0)  // No overlap
        ]
        let mixedResult = FurnitureFitNMS.apply(detections: mixedDetections, iouThreshold: 0.5)
        XCTAssertEqual(mixedResult.count, 2, "NMS should keep non-overlapping + highest confidence overlapping")
    }

    /// Test NMS with varying thresholds
    func testNMSThresholdBehavior() {
        let detections = [
            FurnitureFitDetection(x: 100, y: 100, w: 100, h: 100, confidence: 0.9, classIdx: 0),
            FurnitureFitDetection(x: 130, y: 100, w: 100, h: 100, confidence: 0.8, classIdx: 0)
        ]

        // Low threshold - even small overlap suppresses
        let lowThresholdResult = FurnitureFitNMS.apply(detections: detections, iouThreshold: 0.1)
        XCTAssertEqual(lowThresholdResult.count, 1, "Low threshold should suppress more")

        // High threshold - only high overlap suppresses
        let highThresholdResult = FurnitureFitNMS.apply(detections: detections, iouThreshold: 0.9)
        XCTAssertEqual(highThresholdResult.count, 2, "High threshold should suppress less")
    }

    /// Test NMS ordering - should process by confidence descending
    func testNMSConfidenceOrdering() {
        let detections = [
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.5, classIdx: 0),
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 0),
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.7, classIdx: 0)
        ]

        let result = FurnitureFitNMS.apply(detections: detections, iouThreshold: 0.5)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].confidence, 0.9, "Should keep detection with highest confidence")
    }

    // MARK: - Sharp Room Integration Tests

    /// Test that SharpRoomBoundsUtils calculations match RoomBounds computed properties
    func testBoundsUtilsMatchRoomBoundsStruct() {
        let bounds = RoomBounds(minX: -2, maxX: 2, minY: 0, maxY: 3, minZ: -5, maxZ: -1)

        // Compare utility calculations with RoomBounds computed properties
        XCTAssertEqual(
            SharpRoomBoundsUtils.calculateCenter(min: bounds.minX, max: bounds.maxX),
            bounds.centerX, accuracy: 0.001,
            "centerX calculation mismatch"
        )
        XCTAssertEqual(
            SharpRoomBoundsUtils.calculateCenter(min: bounds.minY, max: bounds.maxY),
            bounds.centerY, accuracy: 0.001,
            "centerY calculation mismatch"
        )
        XCTAssertEqual(
            SharpRoomBoundsUtils.calculateCenter(min: bounds.minZ, max: bounds.maxZ),
            bounds.centerZ, accuracy: 0.001,
            "centerZ calculation mismatch"
        )
        XCTAssertEqual(
            SharpRoomBoundsUtils.calculateDimension(min: bounds.minX, max: bounds.maxX),
            bounds.width, accuracy: 0.001,
            "width calculation mismatch"
        )
        XCTAssertEqual(
            SharpRoomBoundsUtils.calculateDimension(min: bounds.minY, max: bounds.maxY),
            bounds.height, accuracy: 0.001,
            "height calculation mismatch"
        )
        XCTAssertEqual(
            SharpRoomBoundsUtils.calculateDimension(min: bounds.minZ, max: bounds.maxZ),
            bounds.depth, accuracy: 0.001,
            "depth calculation mismatch"
        )
    }

    /// Test camera position calculation matches RoomBoundaryManager.getCameraAtBackWall
    func testCameraUtilsMatchBoundaryManager() {
        let bounds = RoomBounds(minX: -2, maxX: 2, minY: -1.5, maxY: 1.5, minZ: -5, maxZ: -1)
        let manager = RoomBoundaryManager(bounds: bounds)

        // Get camera position from manager
        let managerCamera = manager.getCameraAtBackWall(fovDegrees: 60)

        // Calculate using utility (with same insideFactor of 0.15)
        let utilityCamera = SharpRoomCameraUtils.calculateCameraPosition(
            frontWallZ: manager.frontWallZ,
            backWallZ: manager.backWallZ,
            centerX: manager.centerX,
            centerY: manager.centerY,
            centerZ: manager.centerZ,
            insideFactor: 0.15
        )

        XCTAssertEqual(utilityCamera.eye.x, managerCamera.eye.x, accuracy: 0.001, "eye.x mismatch")
        XCTAssertEqual(utilityCamera.eye.y, managerCamera.eye.y, accuracy: 0.001, "eye.y mismatch")
        XCTAssertEqual(utilityCamera.eye.z, managerCamera.eye.z, accuracy: 0.001, "eye.z mismatch")
        XCTAssertEqual(utilityCamera.target.x, managerCamera.target.x, accuracy: 0.001, "target.x mismatch")
        XCTAssertEqual(utilityCamera.target.y, managerCamera.target.y, accuracy: 0.001, "target.y mismatch")
        XCTAssertEqual(utilityCamera.target.z, managerCamera.target.z, accuracy: 0.001, "target.z mismatch")
    }

    // MARK: - Plane Detection Integration Tests

    /// Test plane normal calculation matches expected geometric behavior
    func testPlaneNormalGeometry() {
        // XY plane should have normal along Z
        let xyPlaneNormal = SharpRoomPlaneUtils.calculatePlaneNormal(
            p1: SIMD3<Float>(0, 0, 0),
            p2: SIMD3<Float>(1, 0, 0),
            p3: SIMD3<Float>(0, 1, 0)
        )
        XCTAssertNotNil(xyPlaneNormal)
        XCTAssertEqual(abs(xyPlaneNormal!.z), 1.0, accuracy: 0.001, "XY plane normal should be along Z")

        // XZ plane should have normal along Y
        let xzPlaneNormal = SharpRoomPlaneUtils.calculatePlaneNormal(
            p1: SIMD3<Float>(0, 0, 0),
            p2: SIMD3<Float>(1, 0, 0),
            p3: SIMD3<Float>(0, 0, 1)
        )
        XCTAssertNotNil(xzPlaneNormal)
        XCTAssertEqual(abs(xzPlaneNormal!.y), 1.0, accuracy: 0.001, "XZ plane normal should be along Y")

        // YZ plane should have normal along X
        let yzPlaneNormal = SharpRoomPlaneUtils.calculatePlaneNormal(
            p1: SIMD3<Float>(0, 0, 0),
            p2: SIMD3<Float>(0, 1, 0),
            p3: SIMD3<Float>(0, 0, 1)
        )
        XCTAssertNotNil(yzPlaneNormal)
        XCTAssertEqual(abs(yzPlaneNormal!.x), 1.0, accuracy: 0.001, "YZ plane normal should be along X")
    }

    /// Test distance to plane calculation
    func testDistanceToPlaneCalculation() {
        // Horizontal plane at y=0
        let planePoint = SIMD3<Float>(0, 0, 0)
        let planeNormal = SIMD3<Float>(0, 1, 0)

        // Points at various heights
        let points: [(point: SIMD3<Float>, expectedDist: Float)] = [
            (SIMD3<Float>(0, 0, 0), 0.0),      // On plane
            (SIMD3<Float>(5, 0, -3), 0.0),     // On plane, different x,z
            (SIMD3<Float>(0, 3, 0), 3.0),      // 3 units above
            (SIMD3<Float>(0, -2, 0), -2.0),    // 2 units below
            (SIMD3<Float>(10, 5, -7), 5.0)     // Above plane, different x,z
        ]

        for (point, expected) in points {
            let distance = SharpRoomPlaneUtils.distanceToPlane(
                point: point,
                planePoint: planePoint,
                planeNormal: planeNormal
            )
            XCTAssertEqual(distance, expected, accuracy: 0.001,
                "Distance mismatch for point \(point)")
        }
    }

    // MARK: - Filter Integration Tests

    /// Test filter produces same results as manual filtering
    func testFilterMatchesManualFiltering() {
        let detections = [
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.9, classIdx: 1),
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.5, classIdx: 2),
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.7, classIdx: 3),
            FurnitureFitDetection(x: 100, y: 100, w: 50, h: 50, confidence: 0.3, classIdx: 1)
        ]

        let threshold: Float = 0.6
        let blacklist: Set<Int> = [1]

        // Manual filtering
        let manualFiltered = detections.filter { detection in
            detection.confidence >= threshold && !blacklist.contains(detection.classIdx)
        }

        // Utility filtering
        let utilityFiltered = FurnitureFitFilter.apply(
            detections: detections,
            confidenceThreshold: threshold,
            classBlacklist: blacklist
        )

        XCTAssertEqual(manualFiltered.count, utilityFiltered.count,
            "Filter count mismatch: manual=\(manualFiltered.count), utility=\(utilityFiltered.count)")

        for (manual, utility) in zip(manualFiltered, utilityFiltered) {
            XCTAssertEqual(manual.confidence, utility.confidence)
            XCTAssertEqual(manual.classIdx, utility.classIdx)
        }
    }

    // MARK: - Mask Integration Tests

    /// Test mask utilities produce correct results
    func testMaskUtilitiesCorrectness() {
        // Create test mask with known properties
        let mask: [UInt8] = [0, 0, 255, 255, 0, 255, 0, 0, 255, 0]
        // 4 positive pixels out of 10 = 40% coverage

        XCTAssertTrue(FurnitureFitMask.hasContent(mask))
        XCTAssertEqual(FurnitureFitMask.positivePixelCount(mask), 4)
        XCTAssertEqual(FurnitureFitMask.coverage(mask), 0.4, accuracy: 0.001)

        // Empty mask
        let emptyMask: [UInt8] = [0, 0, 0, 0]
        XCTAssertFalse(FurnitureFitMask.hasContent(emptyMask))
        XCTAssertEqual(FurnitureFitMask.positivePixelCount(emptyMask), 0)
        XCTAssertEqual(FurnitureFitMask.coverage(emptyMask), 0.0)

        // Full mask
        let fullMask: [UInt8] = [255, 255, 255, 255]
        XCTAssertTrue(FurnitureFitMask.hasContent(fullMask))
        XCTAssertEqual(FurnitureFitMask.positivePixelCount(fullMask), 4)
        XCTAssertEqual(FurnitureFitMask.coverage(fullMask), 1.0)
    }

    /// Test mask thresholding
    func testMaskThresholding() {
        let floatMask: [Float] = [-1.0, -0.1, 0.0, 0.1, 0.5, 1.0]

        // Default threshold (0.0)
        let defaultThreshold = FurnitureFitMask.threshold(floatMask)
        XCTAssertEqual(defaultThreshold, [0, 0, 0, 255, 255, 255])

        // Custom threshold (0.3)
        let customThreshold = FurnitureFitMask.threshold(floatMask, threshold: 0.3)
        XCTAssertEqual(customThreshold, [0, 0, 0, 0, 255, 255])
    }

    // MARK: - Vector Utils Integration Tests

    /// Test vector angle calculation matches expected geometric values
    func testVectorAngleGeometry() {
        // Parallel vectors - 0 degrees
        XCTAssertEqual(
            SharpRoomVectorUtils.angleBetween(SIMD3<Float>(1, 0, 0), SIMD3<Float>(2, 0, 0)),
            0, accuracy: 0.001
        )

        // Perpendicular vectors - 90 degrees
        XCTAssertEqual(
            SharpRoomVectorUtils.angleBetween(SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0)),
            Float.pi / 2, accuracy: 0.001
        )

        // Opposite vectors - 180 degrees
        XCTAssertEqual(
            SharpRoomVectorUtils.angleBetween(SIMD3<Float>(1, 0, 0), SIMD3<Float>(-1, 0, 0)),
            Float.pi, accuracy: 0.001
        )

        // 45 degree angle
        XCTAssertEqual(
            SharpRoomVectorUtils.angleBetween(SIMD3<Float>(1, 0, 0), SIMD3<Float>(1, 1, 0)),
            Float.pi / 4, accuracy: 0.001
        )
    }

    /// Test distance calculation matches expected values
    func testDistanceCalculation() {
        // 3-4-5 triangle
        XCTAssertEqual(
            SharpRoomVectorUtils.distance(SIMD3<Float>(0, 0, 0), SIMD3<Float>(3, 4, 0)),
            5.0, accuracy: 0.001
        )

        // Same point
        XCTAssertEqual(
            SharpRoomVectorUtils.distance(SIMD3<Float>(5, 5, 5), SIMD3<Float>(5, 5, 5)),
            0.0, accuracy: 0.001
        )

        // 3D diagonal
        XCTAssertEqual(
            SharpRoomVectorUtils.distance(SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 1, 1)),
            sqrt(3.0), accuracy: 0.001
        )
    }

    // MARK: - Stress Tests

    /// Stress test NMS with many detections
    func testNMSStressTest() {
        var detections: [FurnitureFitDetection] = []

        // Create 3 highly overlapping detections at each of 10 locations
        // Each group has 3 nearly-identical boxes with decreasing confidence
        for cluster in 0..<10 {
            let baseX = Float(cluster % 5) * 200  // Well separated clusters
            let baseY = Float(cluster / 5) * 200

            // 3 overlapping detections per cluster (almost identical position)
            for j in 0..<3 {
                detections.append(FurnitureFitDetection(
                    x: baseX + Float(j),  // 1-pixel offset
                    y: baseY + Float(j),
                    w: 50,
                    h: 50,
                    confidence: 0.9 - Float(j) * 0.2,  // 0.9, 0.7, 0.5
                    classIdx: 0
                ))
            }
        }

        // 30 total detections
        XCTAssertEqual(detections.count, 30)

        let result = FurnitureFitNMS.apply(detections: detections, iouThreshold: 0.5)

        // Should suppress 2 out of 3 per cluster, keeping 10 detections
        XCTAssertLessThan(result.count, detections.count, "NMS should reduce detection count")

        // All kept detections should have IoU <= threshold
        for i in 0..<result.count {
            for j in (i+1)..<result.count {
                let iou = FurnitureFitIoU.calculate(result[i], result[j])
                XCTAssertLessThanOrEqual(iou, 0.5,
                    "NMS should not keep detections with IoU > threshold")
            }
        }
    }

    /// Stress test bounding box calculation
    func testBoundingBoxStressTest() {
        var points: [SIMD3<Float>] = []

        // Create 10000 random points
        for _ in 0..<10000 {
            points.append(SIMD3<Float>(
                Float.random(in: -100...100),
                Float.random(in: -100...100),
                Float.random(in: -100...100)
            ))
        }

        let result = SharpRoomBoundsUtils.calculateBoundingBox(points: points)
        XCTAssertNotNil(result)

        // Verify all points are inside bounds
        for point in points {
            XCTAssertTrue(SharpRoomBoundsUtils.isPointInside(
                point: point,
                minBound: result!.min,
                maxBound: result!.max
            ))
        }
    }
}
