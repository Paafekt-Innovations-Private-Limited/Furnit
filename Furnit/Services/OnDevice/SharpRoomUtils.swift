// SharpRoomUtils.swift
// Utility functions for Sharp Room functionality - extracted for testability

import Foundation
import simd

// MARK: - Room Bounds Utilities

/// Utility functions for room bounds calculations
public struct SharpRoomBoundsUtils {

    /// Calculate center point from min/max bounds
    public static func calculateCenter(min: Float, max: Float) -> Float {
        return (min + max) / 2.0
    }

    /// Calculate dimension (width/height/depth) from min/max
    public static func calculateDimension(min: Float, max: Float) -> Float {
        return max - min
    }

    /// Calculate volume from bounds
    public static func calculateVolume(width: Float, height: Float, depth: Float) -> Float {
        return width * height * depth
    }

    /// Check if a point is inside bounds
    public static func isPointInside(
        point: SIMD3<Float>,
        minBound: SIMD3<Float>,
        maxBound: SIMD3<Float>
    ) -> Bool {
        return point.x >= minBound.x && point.x <= maxBound.x &&
               point.y >= minBound.y && point.y <= maxBound.y &&
               point.z >= minBound.z && point.z <= maxBound.z
    }

    /// Calculate bounding box from a set of points
    public static func calculateBoundingBox(points: [SIMD3<Float>]) -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        guard !points.isEmpty else { return nil }

        var minBound = SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
        var maxBound = SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)

        for point in points {
            minBound = min(minBound, point)
            maxBound = max(maxBound, point)
        }

        return (minBound, maxBound)
    }

    /// Expand bounds by a margin
    public static func expandBounds(
        min: SIMD3<Float>,
        max: SIMD3<Float>,
        margin: Float
    ) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        let marginVec = SIMD3<Float>(margin, margin, margin)
        return (min - marginVec, max + marginVec)
    }
}

// MARK: - Camera Position Calculator

/// Utility for calculating camera positions in a room
public struct SharpRoomCameraUtils {

    /// Calculate camera position inside room looking at center
    /// - Parameters:
    ///   - frontWallZ: Z coordinate of front wall (closest to camera)
    ///   - backWallZ: Z coordinate of back wall (farthest from camera)
    ///   - centerX: X coordinate of room center
    ///   - centerY: Y coordinate of room center
    ///   - centerZ: Z coordinate of room center
    ///   - insideFactor: How far inside the room (0.0 = at front wall, 0.5 = center, 1.0 = at back wall)
    /// - Returns: Tuple of eye position and target position
    public static func calculateCameraPosition(
        frontWallZ: Float,
        backWallZ: Float,
        centerX: Float,
        centerY: Float,
        centerZ: Float,
        insideFactor: Float = 0.15
    ) -> (eye: SIMD3<Float>, target: SIMD3<Float>) {
        let depth = frontWallZ - backWallZ
        let eyeZ = frontWallZ - depth * insideFactor

        let eye = SIMD3<Float>(centerX, centerY, eyeZ)
        let target = SIMD3<Float>(centerX, centerY, centerZ)

        return (eye, target)
    }

    /// Calculate optimal camera distance based on room dimensions and FOV
    public static func calculateOptimalDistance(
        roomWidth: Float,
        roomHeight: Float,
        fovDegrees: Float
    ) -> Float {
        let fovRadians = fovDegrees * .pi / 180.0
        let halfFov = fovRadians / 2.0

        // Use the larger dimension to ensure room fits in view
        let maxDimension = Swift.max(roomWidth, roomHeight)

        // Distance needed to see the full dimension
        let distance = (maxDimension / 2.0) / tan(halfFov)

        return distance
    }

    /// Calculate look-at direction vector
    public static func calculateLookDirection(
        from eye: SIMD3<Float>,
        to target: SIMD3<Float>
    ) -> SIMD3<Float> {
        let direction = target - eye
        let length = simd_length(direction)
        return length > 0 ? direction / length : SIMD3<Float>(0, 0, -1)
    }
}

// MARK: - Plane Detection Utilities

/// Utility functions for plane detection and RANSAC
public struct SharpRoomPlaneUtils {

    /// Calculate plane normal from 3 points
    /// - Returns: Normalized plane normal, or nil if points are collinear
    public static func calculatePlaneNormal(
        p1: SIMD3<Float>,
        p2: SIMD3<Float>,
        p3: SIMD3<Float>
    ) -> SIMD3<Float>? {
        let v1 = p2 - p1
        let v2 = p3 - p1
        var normal = cross(v1, v2)

        let normalLength = simd_length(normal)
        guard normalLength > 1e-6 else { return nil }

        return normal / normalLength
    }

    /// Calculate signed distance from point to plane
    /// - Parameters:
    ///   - point: The point to measure distance from
    ///   - planePoint: A point on the plane
    ///   - planeNormal: The plane's normal vector (should be normalized)
    /// - Returns: Signed distance (positive if on normal side, negative if opposite)
    public static func distanceToPlane(
        point: SIMD3<Float>,
        planePoint: SIMD3<Float>,
        planeNormal: SIMD3<Float>
    ) -> Float {
        return dot(planeNormal, point - planePoint)
    }

    /// Check if a plane is approximately vertical
    /// - Parameters:
    ///   - normal: The plane's normal vector
    ///   - threshold: Maximum Y component of normal to be considered vertical (radians from horizontal)
    /// - Returns: True if the plane is approximately vertical
    public static func isPlaneVertical(normal: SIMD3<Float>, threshold: Float = 0.3) -> Bool {
        // A vertical plane has a normal mostly in the XZ plane (small Y component)
        return abs(normal.y) < sin(threshold)
    }

    /// Check if a plane is approximately horizontal
    public static func isPlaneHorizontal(normal: SIMD3<Float>, threshold: Float = 0.3) -> Bool {
        // A horizontal plane has a normal mostly along Y axis
        return abs(normal.y) > cos(threshold)
    }

    /// Count inliers for a plane
    public static func countInliers(
        points: [SIMD3<Float>],
        planePoint: SIMD3<Float>,
        planeNormal: SIMD3<Float>,
        threshold: Float
    ) -> Int {
        return points.filter { point in
            abs(distanceToPlane(point: point, planePoint: planePoint, planeNormal: planeNormal)) < threshold
        }.count
    }

    /// Get inlier points for a plane
    public static func getInliers(
        points: [SIMD3<Float>],
        planePoint: SIMD3<Float>,
        planeNormal: SIMD3<Float>,
        threshold: Float
    ) -> [SIMD3<Float>] {
        return points.filter { point in
            abs(distanceToPlane(point: point, planePoint: planePoint, planeNormal: planeNormal)) < threshold
        }
    }
}

// MARK: - Measurement Utilities

/// Utility functions for room measurements
public struct SharpRoomMeasurementUtils {

    /// Apply realistic height cap to measured room height
    public static func applyHeightCap(
        measuredHeight: Float,
        maxRealisticHeight: Float = 4.0,
        scaleFactor: Float = 0.7
    ) -> Float {
        if measuredHeight > maxRealisticHeight {
            return measuredHeight * scaleFactor
        }
        return measuredHeight
    }

    /// Calculate confidence score based on point count and expected count
    public static func calculateConfidence(
        pointCount: Int,
        minExpected: Int,
        maxExpected: Int
    ) -> Float {
        guard pointCount >= minExpected else { return 0.0 }

        if pointCount >= maxExpected {
            return 1.0
        }

        return Float(pointCount - minExpected) / Float(maxExpected - minExpected)
    }

    /// Format dimensions for display
    public static func formatDimensions(width: Float, height: Float) -> String {
        return String(format: "%.1f × %.1f", width, height)
    }

    /// Convert meters to feet
    public static func metersToFeet(_ meters: Float) -> Float {
        return meters * 3.28084
    }

    /// Convert feet to meters
    public static func feetToMeters(_ feet: Float) -> Float {
        return feet / 3.28084
    }

    /// Calculate room area (floor area)
    public static func calculateFloorArea(width: Float, depth: Float) -> Float {
        return width * depth
    }

    /// Calculate total wall area
    public static func calculateWallArea(width: Float, height: Float, depth: Float) -> Float {
        // 2 walls of width x height + 2 walls of depth x height
        return 2 * (width * height) + 2 * (depth * height)
    }
}

// MARK: - Vector Utilities

/// Utility functions for 3D vector operations
public struct SharpRoomVectorUtils {

    /// Calculate angle between two vectors in radians
    public static func angleBetween(_ v1: SIMD3<Float>, _ v2: SIMD3<Float>) -> Float {
        let len1 = simd_length(v1)
        let len2 = simd_length(v2)
        guard len1 > 0 && len2 > 0 else { return 0 }

        let cosAngle = dot(v1, v2) / (len1 * len2)
        // Clamp to avoid NaN from acos
        let clampedCos = Swift.max(-1.0, Swift.min(1.0, cosAngle))
        return acos(clampedCos)
    }

    /// Project a vector onto a plane
    public static func projectOntoPlane(_ vector: SIMD3<Float>, planeNormal: SIMD3<Float>) -> SIMD3<Float> {
        let normalizedNormal = simd_normalize(planeNormal)
        return vector - dot(vector, normalizedNormal) * normalizedNormal
    }

    /// Calculate the midpoint between two points
    public static func midpoint(_ p1: SIMD3<Float>, _ p2: SIMD3<Float>) -> SIMD3<Float> {
        return (p1 + p2) / 2.0
    }

    /// Calculate distance between two points
    public static func distance(_ p1: SIMD3<Float>, _ p2: SIMD3<Float>) -> Float {
        return simd_length(p2 - p1)
    }
}
