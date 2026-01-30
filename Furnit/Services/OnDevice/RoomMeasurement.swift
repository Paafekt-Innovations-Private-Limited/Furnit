import Foundation
import simd
import UIKit

// MARK: - Photo Orientation

/// Orientation of the source photo used for 3D reconstruction
enum PhotoOrientation: String, Codable {
    case portrait
    case landscape
    case square

    /// Detect orientation from EXIF metadata
    /// iPhone camera: .up = landscape (sensor native), .left/.right = portrait (rotated)
    static func detect(from image: UIImage) -> PhotoOrientation {
        let exif = image.imageOrientation
        let uiWidth = image.size.width
        let uiHeight = image.size.height
        let cgWidth = image.cgImage?.width ?? 0
        let cgHeight = image.cgImage?.height ?? 0

        logDebug("📐 [Orientation] ========== DETECTION START ==========")
        logDebug("📐 [Orientation] EXIF raw value: \(exif.rawValue)")
        logDebug("📐 [Orientation] EXIF name: \(exif)")
        logDebug("📐 [Orientation] UIImage.size: \(uiWidth) x \(uiHeight)")
        logDebug("📐 [Orientation] CGImage size: \(cgWidth) x \(cgHeight)")

        switch exif {
        case .up, .upMirrored, .down, .downMirrored:
            // No rotation = sensor's native landscape orientation
            logDebug("📐 [Orientation] EXIF .up/.down → landscape")
            logDebug("📐 [Orientation] ========== RESULT: LANDSCAPE ==========")
            return .landscape

        case .left, .leftMirrored, .right, .rightMirrored:
            // 90° rotation = phone held portrait
            logDebug("📐 [Orientation] EXIF .left/.right → portrait")
            logDebug("📐 [Orientation] ========== RESULT: PORTRAIT ==========")
            return .portrait

        @unknown default:
            logDebug("📐 [Orientation] EXIF unknown → portrait (default)")
            logDebug("📐 [Orientation] ========== RESULT: PORTRAIT (default) ==========")
            return .portrait
        }
    }

    /// User-friendly description
    var hint: String {
        switch self {
        case .landscape:
            return "Landscape – better for full wall width"
        case .portrait:
            return "Portrait – great for tall furniture"
        case .square:
            return "Square"
        }
    }
}

/// Actual min/max bounds of the room in 3D space
struct RoomBounds {
    let minX: Float
    let maxX: Float
    let minY: Float
    let maxY: Float
    let minZ: Float
    let maxZ: Float

    var centerX: Float { (minX + maxX) / 2 }
    var centerY: Float { (minY + maxY) / 2 }
    var centerZ: Float { (minZ + maxZ) / 2 }
    var width: Float { maxX - minX }
    var height: Float { maxY - minY }
    var depth: Float { maxZ - minZ }
    /// Front wall Z (closest to camera, largest Z value)
    var frontZ: Float { maxZ }
}

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
    /// Actual min/max bounds in 3D space
    let actualBounds: RoomBounds?
    /// Number of points used for front wall detection
    let frontWallPointCount: Int
    /// Confidence score (0-1) based on plane fit quality
    let confidence: Float
    /// Orientation of the source photo
    let photoOrientation: PhotoOrientation

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
    /// - Parameters:
    ///   - positions: Array of (x, y, z) positions
    ///   - photoOrientation: Orientation of the source photo (affects height heuristics)
    /// - Returns: Room measurements based on front wall detection
    static func measureRoom(positions: [(Float, Float, Float)], photoOrientation: PhotoOrientation = .landscape) -> RoomMeasurements? {
        guard positions.count >= minPlanePoints else {
            logDebug("RoomMeasurement: Not enough points (\(positions.count))")
            return nil
        }

        logDebug("RoomMeasurement: Orientation = \(photoOrientation.rawValue), points = \(positions.count)")

        // Convert to SIMD for faster math
        let points = positions.map { SIMD3<Float>($0.0, $0.1, $0.2) }

        // Compute overall bounding box
        guard let bounds = SharpRoomBoundsUtils.calculateBoundingBox(points: points) else {
            logDebug("RoomMeasurement: Failed to calculate bounding box")
            return nil
        }
        let minBound = bounds.min
        let maxBound = bounds.max
        let boundingBox = maxBound - minBound

        logDebug("RoomMeasurement: Full bounding box: \(String(format: "%.2f", boundingBox.x)) × \(String(format: "%.2f", boundingBox.y)) × \(String(format: "%.2f", boundingBox.z))")

        // FRONT WALL DETECTION using Z histogram
        // Find the dominant Z plane (where most points cluster = likely a wall)
        let zCoords = points.map { $0.z }
        let zMin = zCoords.min() ?? 0
        let zMax = zCoords.max() ?? 1
        let zRange = zMax - zMin

        // Create histogram with 20 bins
        let numBins = 20
        let binSize = zRange / Float(numBins)
        var histogram = [Int](repeating: 0, count: numBins)

        for z in zCoords {
            let binIndex = min(numBins - 1, max(0, Int((z - zMin) / binSize)))
            histogram[binIndex] += 1
        }

        // Find the bin with most points (dominant Z plane)
        let maxBinIndex = histogram.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        let dominantZ = zMin + (Float(maxBinIndex) + 0.5) * binSize

        logDebug("RoomMeasurement: Z histogram - dominant bin \(maxBinIndex), Z = \(String(format: "%.2f", dominantZ))")

        // Filter points near dominant Z plane (within 1 bin width tolerance)
        let zTolerance = binSize * 1.5
        let frontWallPoints = points.filter { abs($0.z - dominantZ) <= zTolerance }

        logDebug("RoomMeasurement: Front wall points: \(frontWallPoints.count) at Z ≈ \(String(format: "%.2f", dominantZ))")

        // Measure X and Y extent of front wall points
        var roomWidth: Float
        var roomHeight: Float
        let roomDepth = boundingBox.z

        if frontWallPoints.count >= 100 {
            let wallXCoords = frontWallPoints.map { $0.x }
            let wallYCoords = frontWallPoints.map { $0.y }

            // Use percentile trimming - tighter for Y (height) to cut fog
            let sortedX = wallXCoords.sorted()
            let sortedY = wallYCoords.sorted()

            // Width: 5th-95th percentile (walls visible edge to edge)
            let lowIdxX = Int(Double(sortedX.count) * 0.05)
            let highIdxX = Int(Double(sortedX.count) * 0.95) - 1

            // Height: 15th-85th percentile (floor/ceiling fog is worse)
            let lowIdxY = Int(Double(sortedY.count) * 0.15)
            let highIdxY = Int(Double(sortedY.count) * 0.85) - 1

            roomWidth = sortedX[min(sortedX.count - 1, highIdxX)] - sortedX[max(0, lowIdxX)]
            roomHeight = sortedY[min(sortedY.count - 1, highIdxY)] - sortedY[max(0, lowIdxY)]

            logDebug("RoomMeasurement: Front wall measured (X:5-95%, Y:15-85%): \(String(format: "%.2f", roomWidth)) × \(String(format: "%.2f", roomHeight))")
        } else {
            // Fallback to full bounding box if not enough front wall points
            logDebug("RoomMeasurement: Not enough front wall points, using bounding box")
            roomWidth = boundingBox.x
            roomHeight = boundingBox.y
        }

        // Apply realistic caps as safety net
        let maxRealisticWidth: Float = photoOrientation == .portrait ? 6.0 : 8.0
        let maxRealisticHeight: Float = photoOrientation == .portrait ? 3.5 : 3.2
        let minRealisticHeight: Float = 2.2

        if roomWidth > maxRealisticWidth {
            logDebug("RoomMeasurement: Width \(String(format: "%.2f", roomWidth)) > max, capping to \(maxRealisticWidth)")
            roomWidth = maxRealisticWidth
        }

        if roomHeight > maxRealisticHeight {
            logDebug("RoomMeasurement: Height \(String(format: "%.2f", roomHeight)) > max, capping to \(maxRealisticHeight)")
            roomHeight = maxRealisticHeight
        } else if roomHeight < minRealisticHeight {
            logDebug("RoomMeasurement: Height \(String(format: "%.2f", roomHeight)) < min, using \(minRealisticHeight)")
            roomHeight = minRealisticHeight
        }

        logDebug("RoomMeasurement: FINAL dimensions: \(String(format: "%.2f", roomWidth)) × \(String(format: "%.2f", roomHeight)) m")

        // Create actual bounds
        let actualBounds = RoomBounds(
            minX: minBound.x, maxX: maxBound.x,
            minY: minBound.y, maxY: maxBound.y,
            minZ: minBound.z, maxZ: maxBound.z
        )

        return RoomMeasurements(
            frontWallWidth: roomWidth,
            frontWallHeight: roomHeight,
            roomDepth: roomDepth,
            boundingBox: boundingBox,
            actualBounds: actualBounds,
            frontWallPointCount: positions.count,
            confidence: 0.8,
            photoOrientation: photoOrientation
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
