import Foundation
import simd

/// Room measurement results from plane detection
struct RoomMeasurements {
    /// Front wall width in model units
    let frontWallWidth: Float
    /// Front wall height in model units
    let frontWallHeight: Float
    /// Room depth (distance from front to back wall)
    let roomDepth: Float
    /// Total bounding box dimensions
    let boundingBox: SIMD3<Float>
    /// Number of points used for front wall detection
    let frontWallPointCount: Int
    /// Confidence score (0-1) based on plane fit quality
    let confidence: Float

    /// Formatted string for display
    var frontWallDescription: String {
        String(format: "%.1f × %.1f", frontWallWidth, frontWallHeight)
    }
}

/// Detects planes and measures room dimensions from Gaussian splat positions
class RoomMeasurement {

    // MARK: - Configuration

    /// Distance threshold for RANSAC inlier detection
    private static let ransacThreshold: Float = 0.05

    /// Number of RANSAC iterations
    private static let ransacIterations: Int = 100

    /// Minimum points required to consider a valid plane
    private static let minPlanePoints: Int = 100

    /// Angle threshold (radians) to consider a plane as vertical
    private static let verticalThreshold: Float = 0.3  // ~17 degrees from vertical

    // MARK: - Plane Detection

    /// Detect planes and measure room from Gaussian positions
    /// - Parameter positions: Array of (x, y, z) positions
    /// - Returns: Room measurements if front wall detected
    static func measureRoom(positions: [(Float, Float, Float)]) -> RoomMeasurements? {
        guard positions.count >= minPlanePoints else {
            logDebug("RoomMeasurement: Not enough points (\(positions.count))")
            return nil
        }

        // Convert to SIMD for faster math
        let points = positions.map { SIMD3<Float>($0.0, $0.1, $0.2) }

        // Compute overall bounding box
        var minBound = SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
        var maxBound = SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)

        for p in points {
            minBound = min(minBound, p)
            maxBound = max(maxBound, p)
        }

        let boundingBox = maxBound - minBound
        logDebug("RoomMeasurement: Bounding box: \(boundingBox.x) × \(boundingBox.y) × \(boundingBox.z)")

        // Find the front wall - typically the plane at max Z (back of room from camera view)
        // or min Z depending on orientation. We'll check both and pick the larger vertical plane.

        // Strategy: The front wall should be:
        // 1. A vertical plane (normal mostly in Z direction)
        // 2. At one of the Z extremes
        // 3. The largest such plane

        // Sample points near the front (min Z) and back (max Z) of the room
        let zRange = boundingBox.z
        let frontZThreshold = minBound.z + zRange * 0.15  // Front 15% of room
        let backZThreshold = maxBound.z - zRange * 0.15   // Back 15% of room

        let frontPoints = points.filter { $0.z <= frontZThreshold }
        let backPoints = points.filter { $0.z >= backZThreshold }

        logDebug("RoomMeasurement: Front region points: \(frontPoints.count), Back region points: \(backPoints.count)")

        // Detect plane in the region with more points (likely the visible wall)
        let (wallPoints, wallNormal, wallConfidence) = detectLargestVerticalPlane(
            in: frontPoints.count > backPoints.count ? frontPoints : backPoints
        )

        guard wallPoints.count >= minPlanePoints else {
            logDebug("RoomMeasurement: No valid wall plane found")
            // Fall back to bounding box estimation
            return RoomMeasurements(
                frontWallWidth: boundingBox.x,
                frontWallHeight: boundingBox.y,
                roomDepth: boundingBox.z,
                boundingBox: boundingBox,
                frontWallPointCount: 0,
                confidence: 0.3
            )
        }

        // Measure the wall dimensions from inlier points
        let wallDimensions = measurePlaneDimensions(points: wallPoints, normal: wallNormal)

        logDebug("RoomMeasurement: Front wall detected:")
        logDebug("  Width: \(wallDimensions.width)")
        logDebug("  Height: \(wallDimensions.height)")
        logDebug("  Points: \(wallPoints.count)")
        logDebug("  Confidence: \(wallConfidence)")

        return RoomMeasurements(
            frontWallWidth: wallDimensions.width,
            frontWallHeight: wallDimensions.height,
            roomDepth: boundingBox.z,
            boundingBox: boundingBox,
            frontWallPointCount: wallPoints.count,
            confidence: wallConfidence
        )
    }

    // MARK: - RANSAC Plane Detection

    /// Detect the largest vertical plane using RANSAC
    private static func detectLargestVerticalPlane(in points: [SIMD3<Float>]) -> (inliers: [SIMD3<Float>], normal: SIMD3<Float>, confidence: Float) {
        guard points.count >= 3 else {
            return ([], SIMD3<Float>(0, 0, 1), 0)
        }

        var bestInliers: [SIMD3<Float>] = []
        var bestNormal = SIMD3<Float>(0, 0, 1)
        var bestScore: Float = 0

        for _ in 0..<ransacIterations {
            // Randomly sample 3 points
            let indices = (0..<points.count).shuffled().prefix(3)
            let samples = indices.map { points[$0] }

            // Compute plane from 3 points
            let v1 = samples[1] - samples[0]
            let v2 = samples[2] - samples[0]
            var normal = cross(v1, v2)

            let normalLength = length(normal)
            guard normalLength > 1e-6 else { continue }
            normal /= normalLength

            // Check if plane is roughly vertical (normal should be mostly horizontal)
            // A vertical wall has a normal in the XZ plane (small Y component)
            let verticalness = abs(normal.y)
            if verticalness > sin(verticalThreshold) {
                continue  // Not vertical enough, skip
            }

            // Count inliers
            let d = -dot(normal, samples[0])
            var inliers: [SIMD3<Float>] = []

            for p in points {
                let distance = abs(dot(normal, p) + d)
                if distance < ransacThreshold {
                    inliers.append(p)
                }
            }

            // Score: number of inliers weighted by how vertical the plane is
            let score = Float(inliers.count) * (1.0 - verticalness)

            if score > bestScore {
                bestScore = score
                bestInliers = inliers
                bestNormal = normal
            }
        }

        let confidence = min(Float(bestInliers.count) / Float(max(points.count, 1)), 1.0)
        return (bestInliers, bestNormal, confidence)
    }

    // MARK: - Dimension Measurement

    /// Measure width and height of a detected plane
    private static func measurePlaneDimensions(points: [SIMD3<Float>], normal: SIMD3<Float>) -> (width: Float, height: Float) {
        guard points.count >= 2 else {
            return (0, 0)
        }

        // Create local coordinate system on the plane
        // Up vector is world Y (assuming room is upright)
        let worldUp = SIMD3<Float>(0, 1, 0)

        // Horizontal axis on the plane (perpendicular to normal and up)
        var horizontal = cross(worldUp, normal)
        let hLength = length(horizontal)
        if hLength > 1e-6 {
            horizontal /= hLength
        } else {
            // Normal is pointing up/down, use X as horizontal
            horizontal = SIMD3<Float>(1, 0, 0)
        }

        // Vertical axis on the plane
        var vertical = cross(normal, horizontal)
        let vLength = length(vertical)
        if vLength > 1e-6 {
            vertical /= vLength
        } else {
            vertical = worldUp
        }

        // Project all points onto local 2D coordinates
        var minH: Float = .greatestFiniteMagnitude
        var maxH: Float = -.greatestFiniteMagnitude
        var minV: Float = .greatestFiniteMagnitude
        var maxV: Float = -.greatestFiniteMagnitude

        for p in points {
            let h = dot(p, horizontal)
            let v = dot(p, vertical)
            minH = min(minH, h)
            maxH = max(maxH, h)
            minV = min(minV, v)
            maxV = max(maxV, v)
        }

        return (width: maxH - minH, height: maxV - minV)
    }
}
