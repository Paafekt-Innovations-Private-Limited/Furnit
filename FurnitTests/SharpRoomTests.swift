// SharpRoomTests.swift
// Unit tests for Sharp Room functionality

import XCTest
import simd
@testable import Furnit

final class SharpRoomTests: XCTestCase {

    // MARK: - Room Bounds Utils Tests

    func testCalculateCenter() {
        XCTAssertEqual(SharpRoomBoundsUtils.calculateCenter(min: 0, max: 10), 5.0, accuracy: 0.001)
        XCTAssertEqual(SharpRoomBoundsUtils.calculateCenter(min: -5, max: 5), 0.0, accuracy: 0.001)
        XCTAssertEqual(SharpRoomBoundsUtils.calculateCenter(min: -10, max: -2), -6.0, accuracy: 0.001)
    }

    func testCalculateDimension() {
        XCTAssertEqual(SharpRoomBoundsUtils.calculateDimension(min: 0, max: 10), 10.0, accuracy: 0.001)
        XCTAssertEqual(SharpRoomBoundsUtils.calculateDimension(min: -5, max: 5), 10.0, accuracy: 0.001)
        XCTAssertEqual(SharpRoomBoundsUtils.calculateDimension(min: 2, max: 2), 0.0, accuracy: 0.001)
    }

    func testCalculateVolume() {
        XCTAssertEqual(SharpRoomBoundsUtils.calculateVolume(width: 4, height: 3, depth: 5), 60.0, accuracy: 0.001)
        XCTAssertEqual(SharpRoomBoundsUtils.calculateVolume(width: 2, height: 2, depth: 2), 8.0, accuracy: 0.001)
        XCTAssertEqual(SharpRoomBoundsUtils.calculateVolume(width: 0, height: 5, depth: 5), 0.0, accuracy: 0.001)
    }

    func testIsPointInside() {
        let minBound = SIMD3<Float>(0, 0, 0)
        let maxBound = SIMD3<Float>(10, 10, 10)

        // Inside
        XCTAssertTrue(SharpRoomBoundsUtils.isPointInside(point: SIMD3<Float>(5, 5, 5), minBound: minBound, maxBound: maxBound))

        // On boundary
        XCTAssertTrue(SharpRoomBoundsUtils.isPointInside(point: SIMD3<Float>(0, 0, 0), minBound: minBound, maxBound: maxBound))
        XCTAssertTrue(SharpRoomBoundsUtils.isPointInside(point: SIMD3<Float>(10, 10, 10), minBound: minBound, maxBound: maxBound))

        // Outside
        XCTAssertFalse(SharpRoomBoundsUtils.isPointInside(point: SIMD3<Float>(-1, 5, 5), minBound: minBound, maxBound: maxBound))
        XCTAssertFalse(SharpRoomBoundsUtils.isPointInside(point: SIMD3<Float>(11, 5, 5), minBound: minBound, maxBound: maxBound))
    }

    func testCalculateBoundingBox() {
        let points = [
            SIMD3<Float>(1, 2, 3),
            SIMD3<Float>(-1, -2, -3),
            SIMD3<Float>(5, 0, 1)
        ]

        let result = SharpRoomBoundsUtils.calculateBoundingBox(points: points)
        XCTAssertNotNil(result)

        XCTAssertEqual(result!.min.x, -1, accuracy: 0.001)
        XCTAssertEqual(result!.min.y, -2, accuracy: 0.001)
        XCTAssertEqual(result!.min.z, -3, accuracy: 0.001)
        XCTAssertEqual(result!.max.x, 5, accuracy: 0.001)
        XCTAssertEqual(result!.max.y, 2, accuracy: 0.001)
        XCTAssertEqual(result!.max.z, 3, accuracy: 0.001)
    }

    func testCalculateBoundingBoxEmpty() {
        let result = SharpRoomBoundsUtils.calculateBoundingBox(points: [])
        XCTAssertNil(result)
    }

    func testExpandBounds() {
        let minBound = SIMD3<Float>(0, 0, 0)
        let maxBound = SIMD3<Float>(10, 10, 10)

        let result = SharpRoomBoundsUtils.expandBounds(min: minBound, max: maxBound, margin: 2.0)

        XCTAssertEqual(result.min.x, -2, accuracy: 0.001)
        XCTAssertEqual(result.min.y, -2, accuracy: 0.001)
        XCTAssertEqual(result.min.z, -2, accuracy: 0.001)
        XCTAssertEqual(result.max.x, 12, accuracy: 0.001)
        XCTAssertEqual(result.max.y, 12, accuracy: 0.001)
        XCTAssertEqual(result.max.z, 12, accuracy: 0.001)
    }

    // MARK: - Camera Utils Tests

    func testCalculateCameraPosition() {
        let result = SharpRoomCameraUtils.calculateCameraPosition(
            frontWallZ: -1,
            backWallZ: -5,
            centerX: 0,
            centerY: 1.5,
            centerZ: -3,
            insideFactor: 0.15
        )

        // Eye should be at frontWallZ - depth * 0.15 = -1 - 4 * 0.15 = -1.6
        XCTAssertEqual(result.eye.x, 0, accuracy: 0.001)
        XCTAssertEqual(result.eye.y, 1.5, accuracy: 0.001)
        XCTAssertEqual(result.eye.z, -1.6, accuracy: 0.001)

        // Target should be at center
        XCTAssertEqual(result.target.x, 0, accuracy: 0.001)
        XCTAssertEqual(result.target.y, 1.5, accuracy: 0.001)
        XCTAssertEqual(result.target.z, -3, accuracy: 0.001)
    }

    func testCalculateCameraPositionAtCenter() {
        let result = SharpRoomCameraUtils.calculateCameraPosition(
            frontWallZ: 0,
            backWallZ: -10,
            centerX: 5,
            centerY: 2,
            centerZ: -5,
            insideFactor: 0.5  // At room center
        )

        // Eye should be at center depth
        XCTAssertEqual(result.eye.z, -5, accuracy: 0.001)
    }

    func testCalculateOptimalDistance() {
        let distance = SharpRoomCameraUtils.calculateOptimalDistance(
            roomWidth: 4.0,
            roomHeight: 3.0,
            fovDegrees: 60.0
        )

        // For 60 degree FOV, distance = (maxDim/2) / tan(30°) = 2 / 0.577 ≈ 3.46
        XCTAssertEqual(distance, 3.46, accuracy: 0.1)
    }

    func testCalculateLookDirection() {
        let eye = SIMD3<Float>(0, 0, 5)
        let target = SIMD3<Float>(0, 0, 0)

        let direction = SharpRoomCameraUtils.calculateLookDirection(from: eye, to: target)

        XCTAssertEqual(direction.x, 0, accuracy: 0.001)
        XCTAssertEqual(direction.y, 0, accuracy: 0.001)
        XCTAssertEqual(direction.z, -1, accuracy: 0.001)
    }

    func testCalculateLookDirectionSamePoint() {
        let point = SIMD3<Float>(5, 5, 5)
        let direction = SharpRoomCameraUtils.calculateLookDirection(from: point, to: point)

        // Should return default direction
        XCTAssertEqual(direction.z, -1, accuracy: 0.001)
    }

    // MARK: - Plane Utils Tests

    func testCalculatePlaneNormal() {
        // XY plane (normal should be Z)
        let p1 = SIMD3<Float>(0, 0, 0)
        let p2 = SIMD3<Float>(1, 0, 0)
        let p3 = SIMD3<Float>(0, 1, 0)

        let normal = SharpRoomPlaneUtils.calculatePlaneNormal(p1: p1, p2: p2, p3: p3)
        XCTAssertNotNil(normal)
        XCTAssertEqual(abs(normal!.z), 1.0, accuracy: 0.001)
    }

    func testCalculatePlaneNormalCollinear() {
        // Collinear points - should return nil
        let p1 = SIMD3<Float>(0, 0, 0)
        let p2 = SIMD3<Float>(1, 0, 0)
        let p3 = SIMD3<Float>(2, 0, 0)

        let normal = SharpRoomPlaneUtils.calculatePlaneNormal(p1: p1, p2: p2, p3: p3)
        XCTAssertNil(normal)
    }

    func testDistanceToPlane() {
        let planePoint = SIMD3<Float>(0, 0, 0)
        let planeNormal = SIMD3<Float>(0, 1, 0)  // Horizontal plane

        // Point above plane
        let abovePoint = SIMD3<Float>(0, 5, 0)
        XCTAssertEqual(SharpRoomPlaneUtils.distanceToPlane(point: abovePoint, planePoint: planePoint, planeNormal: planeNormal), 5.0, accuracy: 0.001)

        // Point below plane
        let belowPoint = SIMD3<Float>(0, -3, 0)
        XCTAssertEqual(SharpRoomPlaneUtils.distanceToPlane(point: belowPoint, planePoint: planePoint, planeNormal: planeNormal), -3.0, accuracy: 0.001)

        // Point on plane
        let onPlane = SIMD3<Float>(10, 0, -5)
        XCTAssertEqual(SharpRoomPlaneUtils.distanceToPlane(point: onPlane, planePoint: planePoint, planeNormal: planeNormal), 0.0, accuracy: 0.001)
    }

    func testIsPlaneVertical() {
        // Vertical wall (normal in XZ plane)
        let verticalNormal = SIMD3<Float>(1, 0, 0)
        XCTAssertTrue(SharpRoomPlaneUtils.isPlaneVertical(normal: verticalNormal))

        // Horizontal floor (normal along Y)
        let horizontalNormal = SIMD3<Float>(0, 1, 0)
        XCTAssertFalse(SharpRoomPlaneUtils.isPlaneVertical(normal: horizontalNormal))

        // Slightly tilted but still vertical
        let tiltedVertical = simd_normalize(SIMD3<Float>(1, 0.1, 0))
        XCTAssertTrue(SharpRoomPlaneUtils.isPlaneVertical(normal: tiltedVertical))
    }

    func testIsPlaneHorizontal() {
        // Horizontal floor
        let horizontalNormal = SIMD3<Float>(0, 1, 0)
        XCTAssertTrue(SharpRoomPlaneUtils.isPlaneHorizontal(normal: horizontalNormal))

        // Vertical wall
        let verticalNormal = SIMD3<Float>(1, 0, 0)
        XCTAssertFalse(SharpRoomPlaneUtils.isPlaneHorizontal(normal: verticalNormal))
    }

    func testCountInliers() {
        let planePoint = SIMD3<Float>(0, 0, 0)
        let planeNormal = SIMD3<Float>(0, 1, 0)

        let points = [
            SIMD3<Float>(0, 0.01, 0),   // Very close - inlier
            SIMD3<Float>(1, 0, 1),      // On plane - inlier
            SIMD3<Float>(0, 0.5, 0),    // Far - outlier
            SIMD3<Float>(-1, -0.02, 2)  // Close - inlier
        ]

        let count = SharpRoomPlaneUtils.countInliers(points: points, planePoint: planePoint, planeNormal: planeNormal, threshold: 0.1)
        XCTAssertEqual(count, 3)
    }

    func testGetInliers() {
        let planePoint = SIMD3<Float>(0, 0, 0)
        let planeNormal = SIMD3<Float>(0, 1, 0)

        let points = [
            SIMD3<Float>(0, 0, 0),
            SIMD3<Float>(1, 0.5, 1),  // Outlier
            SIMD3<Float>(2, 0.01, 2)
        ]

        let inliers = SharpRoomPlaneUtils.getInliers(points: points, planePoint: planePoint, planeNormal: planeNormal, threshold: 0.1)
        XCTAssertEqual(inliers.count, 2)
    }

    // MARK: - Measurement Utils Tests

    func testApplyHeightCap() {
        // Below cap - unchanged
        XCTAssertEqual(SharpRoomMeasurementUtils.applyHeightCap(measuredHeight: 3.0), 3.0, accuracy: 0.001)

        // Above cap - scaled down
        XCTAssertEqual(SharpRoomMeasurementUtils.applyHeightCap(measuredHeight: 6.0, maxRealisticHeight: 4.0, scaleFactor: 0.7), 4.2, accuracy: 0.001)
    }

    func testCalculateConfidence() {
        // Below minimum
        XCTAssertEqual(SharpRoomMeasurementUtils.calculateConfidence(pointCount: 50, minExpected: 100, maxExpected: 1000), 0.0)

        // At minimum
        XCTAssertEqual(SharpRoomMeasurementUtils.calculateConfidence(pointCount: 100, minExpected: 100, maxExpected: 1000), 0.0)

        // At maximum
        XCTAssertEqual(SharpRoomMeasurementUtils.calculateConfidence(pointCount: 1000, minExpected: 100, maxExpected: 1000), 1.0)

        // Above maximum
        XCTAssertEqual(SharpRoomMeasurementUtils.calculateConfidence(pointCount: 2000, minExpected: 100, maxExpected: 1000), 1.0)

        // Mid-range
        XCTAssertEqual(SharpRoomMeasurementUtils.calculateConfidence(pointCount: 550, minExpected: 100, maxExpected: 1000), 0.5, accuracy: 0.001)
    }

    func testFormatDimensions() {
        XCTAssertEqual(SharpRoomMeasurementUtils.formatDimensions(width: 4.5, height: 3.2), "4.5 × 3.2")
        XCTAssertEqual(SharpRoomMeasurementUtils.formatDimensions(width: 10.0, height: 8.0), "10.0 × 8.0")
    }

    func testMetersToFeet() {
        XCTAssertEqual(SharpRoomMeasurementUtils.metersToFeet(1.0), 3.28084, accuracy: 0.001)
        XCTAssertEqual(SharpRoomMeasurementUtils.metersToFeet(3.0), 9.84252, accuracy: 0.001)
    }

    func testFeetToMeters() {
        XCTAssertEqual(SharpRoomMeasurementUtils.feetToMeters(3.28084), 1.0, accuracy: 0.001)
        XCTAssertEqual(SharpRoomMeasurementUtils.feetToMeters(10.0), 3.048, accuracy: 0.001)
    }

    func testCalculateFloorArea() {
        XCTAssertEqual(SharpRoomMeasurementUtils.calculateFloorArea(width: 4, depth: 5), 20.0, accuracy: 0.001)
    }

    func testCalculateWallArea() {
        // Room 4x3x5 (width x height x depth)
        // Wall area = 2*(4*3) + 2*(5*3) = 24 + 30 = 54
        XCTAssertEqual(SharpRoomMeasurementUtils.calculateWallArea(width: 4, height: 3, depth: 5), 54.0, accuracy: 0.001)
    }

    // MARK: - Vector Utils Tests

    func testAngleBetweenParallel() {
        let v1 = SIMD3<Float>(1, 0, 0)
        let v2 = SIMD3<Float>(2, 0, 0)

        let angle = SharpRoomVectorUtils.angleBetween(v1, v2)
        XCTAssertEqual(angle, 0, accuracy: 0.001)
    }

    func testAngleBetweenPerpendicular() {
        let v1 = SIMD3<Float>(1, 0, 0)
        let v2 = SIMD3<Float>(0, 1, 0)

        let angle = SharpRoomVectorUtils.angleBetween(v1, v2)
        XCTAssertEqual(angle, .pi / 2, accuracy: 0.001)
    }

    func testAngleBetweenOpposite() {
        let v1 = SIMD3<Float>(1, 0, 0)
        let v2 = SIMD3<Float>(-1, 0, 0)

        let angle = SharpRoomVectorUtils.angleBetween(v1, v2)
        XCTAssertEqual(angle, .pi, accuracy: 0.001)
    }

    func testProjectOntoPlane() {
        let vector = SIMD3<Float>(1, 1, 0)
        let planeNormal = SIMD3<Float>(0, 1, 0)  // XZ plane

        let projected = SharpRoomVectorUtils.projectOntoPlane(vector, planeNormal: planeNormal)

        XCTAssertEqual(projected.x, 1, accuracy: 0.001)
        XCTAssertEqual(projected.y, 0, accuracy: 0.001)
        XCTAssertEqual(projected.z, 0, accuracy: 0.001)
    }

    func testMidpoint() {
        let p1 = SIMD3<Float>(0, 0, 0)
        let p2 = SIMD3<Float>(10, 20, 30)

        let mid = SharpRoomVectorUtils.midpoint(p1, p2)

        XCTAssertEqual(mid.x, 5, accuracy: 0.001)
        XCTAssertEqual(mid.y, 10, accuracy: 0.001)
        XCTAssertEqual(mid.z, 15, accuracy: 0.001)
    }

    func testDistance() {
        let p1 = SIMD3<Float>(0, 0, 0)
        let p2 = SIMD3<Float>(3, 4, 0)

        let dist = SharpRoomVectorUtils.distance(p1, p2)
        XCTAssertEqual(dist, 5.0, accuracy: 0.001)
    }

    // MARK: - RoomBounds Tests

    func testRoomBoundsComputedProperties() {
        let bounds = RoomBounds(minX: -2, maxX: 2, minY: 0, maxY: 3, minZ: -5, maxZ: -1)

        XCTAssertEqual(bounds.width, 4.0, accuracy: 0.001)
        XCTAssertEqual(bounds.height, 3.0, accuracy: 0.001)
        XCTAssertEqual(bounds.depth, 4.0, accuracy: 0.001)
        XCTAssertEqual(bounds.centerX, 0.0, accuracy: 0.001)
        XCTAssertEqual(bounds.centerY, 1.5, accuracy: 0.001)
        XCTAssertEqual(bounds.centerZ, -3.0, accuracy: 0.001)
        XCTAssertEqual(bounds.frontZ, -1.0, accuracy: 0.001)
    }

    // MARK: - RoomBoundaryManager Tests

    func testRoomBoundaryManagerDefaultBounds() {
        let defaultBounds = RoomBoundaryManager.defaultBounds

        XCTAssertEqual(defaultBounds.minX, -2)
        XCTAssertEqual(defaultBounds.maxX, 2)
        XCTAssertEqual(defaultBounds.width, 4)
    }

    func testRoomBoundaryManagerCameraPosition() {
        let bounds = RoomBounds(minX: -2, maxX: 2, minY: -1.5, maxY: 1.5, minZ: -5, maxZ: -1)
        let manager = RoomBoundaryManager(bounds: bounds)

        XCTAssertEqual(manager.width, 4.0, accuracy: 0.001)
        XCTAssertEqual(manager.height, 3.0, accuracy: 0.001)
        XCTAssertEqual(manager.depth, 4.0, accuracy: 0.001)
        XCTAssertEqual(manager.frontWallZ, -1.0, accuracy: 0.001)
        XCTAssertEqual(manager.backWallZ, -5.0, accuracy: 0.001)
    }

    // MARK: - Performance Tests

    func testBoundingBoxPerformanceWith10000Points() {
        var points: [SIMD3<Float>] = []
        for _ in 0..<10000 {
            points.append(SIMD3<Float>(
                Float.random(in: -10...10),
                Float.random(in: -10...10),
                Float.random(in: -10...10)
            ))
        }

        measure {
            _ = SharpRoomBoundsUtils.calculateBoundingBox(points: points)
        }
    }

    func testPlaneInlierCountPerformance() {
        var points: [SIMD3<Float>] = []
        for _ in 0..<5000 {
            points.append(SIMD3<Float>(
                Float.random(in: -10...10),
                Float.random(in: -0.1...0.1),  // Near the XZ plane
                Float.random(in: -10...10)
            ))
        }

        let planePoint = SIMD3<Float>(0, 0, 0)
        let planeNormal = SIMD3<Float>(0, 1, 0)

        measure {
            _ = SharpRoomPlaneUtils.countInliers(points: points, planePoint: planePoint, planeNormal: planeNormal, threshold: 0.2)
        }
    }
}
