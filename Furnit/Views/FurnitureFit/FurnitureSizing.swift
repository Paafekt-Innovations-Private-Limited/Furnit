import Foundation
import CoreGraphics

/// Real-world furniture dimensions in meters
/// Used to scale segmented furniture relative to room dimensions from SHARP
struct FurnitureDimensions {
    let width: Float   // Width in meters
    let height: Float  // Height in meters
    let depth: Float   // Depth in meters (optional, for 3D placement)

    /// Common furniture dimensions based on standard sizes
    static let standardDimensions: [Int: FurnitureDimensions] = [
        // Bed (class 375) - Queen size bed
        375: FurnitureDimensions(width: 1.6, height: 0.6, depth: 2.0),

        // Chair (class 821) - Standard dining/office chair
        821: FurnitureDimensions(width: 0.5, height: 0.9, depth: 0.5),

        // Couch/Sofa (class 1141) - 3-seater sofa
        1141: FurnitureDimensions(width: 2.0, height: 0.85, depth: 0.9),

        // Table (class 1301) - Dining table
        1301: FurnitureDimensions(width: 1.5, height: 0.75, depth: 0.9),
    ]

    /// Get dimensions for a furniture class, with fallback
    static func forClass(_ classIdx: Int) -> FurnitureDimensions? {
        return standardDimensions[classIdx]
    }
}

/// Calculates scaling factors for furniture relative to room dimensions
class FurnitureSizingCalculator {

    /// Room dimensions from SHARP output (in meters)
    let roomWidth: Float
    let roomHeight: Float

    /// Screen/view dimensions (in pixels)
    let viewWidth: CGFloat
    let viewHeight: CGFloat

    /// Meters per pixel for the current view
    var metersPerPixelX: Float {
        return roomWidth / Float(viewWidth)
    }

    var metersPerPixelY: Float {
        return roomHeight / Float(viewHeight)
    }

    init(roomWidth: Float, roomHeight: Float, viewWidth: CGFloat, viewHeight: CGFloat) {
        self.roomWidth = roomWidth
        self.roomHeight = roomHeight
        self.viewWidth = viewWidth
        self.viewHeight = viewHeight
    }

    /// Calculate the expected pixel dimensions for furniture of a given class
    /// - Parameters:
    ///   - classIdx: The YOLO class index
    ///   - detectedWidthPixels: The detected width in pixels from segmentation
    ///   - detectedHeightPixels: The detected height in pixels from segmentation
    /// - Returns: Tuple of (scaledWidth, scaledHeight) in pixels, or nil if class not recognized
    func calculateScaledDimensions(
        classIdx: Int,
        detectedWidthPixels: CGFloat,
        detectedHeightPixels: CGFloat
    ) -> (width: CGFloat, height: CGFloat)? {

        guard let furnitureDims = FurnitureDimensions.forClass(classIdx) else {
            logDebug("📏 [FurnitureSizing] Unknown class \(classIdx), no scaling applied")
            return nil
        }

        // Calculate expected pixel dimensions based on real-world size
        let expectedWidthPixels = CGFloat(furnitureDims.width / metersPerPixelX)
        let expectedHeightPixels = CGFloat(furnitureDims.height / metersPerPixelY)

        logDebug("📏 [FurnitureSizing] Class \(classIdx):")
        logDebug("   Real size: \(furnitureDims.width)m × \(furnitureDims.height)m")
        logDebug("   Room: \(roomWidth)m × \(roomHeight)m")
        logDebug("   View: \(viewWidth)px × \(viewHeight)px")
        logDebug("   Meters/pixel: \(metersPerPixelX) × \(metersPerPixelY)")
        logDebug("   Detected: \(Int(detectedWidthPixels))px × \(Int(detectedHeightPixels))px")
        logDebug("   Expected: \(Int(expectedWidthPixels))px × \(Int(expectedHeightPixels))px")

        return (width: expectedWidthPixels, height: expectedHeightPixels)
    }

    /// Calculate scale factor to resize detected furniture to real-world proportions
    /// - Parameters:
    ///   - classIdx: The YOLO class index
    ///   - detectedWidthPixels: The detected width in pixels
    ///   - detectedHeightPixels: The detected height in pixels
    /// - Returns: Scale factor to apply, or 1.0 if class not recognized
    func calculateScaleFactor(
        classIdx: Int,
        detectedWidthPixels: CGFloat,
        detectedHeightPixels: CGFloat
    ) -> CGFloat {

        guard let expectedDims = calculateScaledDimensions(
            classIdx: classIdx,
            detectedWidthPixels: detectedWidthPixels,
            detectedHeightPixels: detectedHeightPixels
        ) else {
            return 1.0
        }

        // Calculate scale factor based on the larger dimension to maintain aspect ratio
        let scaleX = expectedDims.width / detectedWidthPixels
        let scaleY = expectedDims.height / detectedHeightPixels

        // Use the average to balance both dimensions
        let scaleFactor = (scaleX + scaleY) / 2.0

        // Clamp to reasonable range (0.25x to 4x)
        let clampedScale = max(0.25, min(4.0, scaleFactor))

        logDebug("📏 [FurnitureSizing] Scale factor: \(scaleFactor) → clamped: \(clampedScale)")

        return clampedScale
    }

    /// Estimate real-world dimensions of detected furniture
    /// - Parameters:
    ///   - detectedWidthPixels: Detected width in pixels
    ///   - detectedHeightPixels: Detected height in pixels
    /// - Returns: Estimated real-world dimensions in meters
    func estimateRealWorldSize(
        detectedWidthPixels: CGFloat,
        detectedHeightPixels: CGFloat
    ) -> (widthMeters: Float, heightMeters: Float) {

        let widthMeters = Float(detectedWidthPixels) * metersPerPixelX
        let heightMeters = Float(detectedHeightPixels) * metersPerPixelY

        return (widthMeters: widthMeters, heightMeters: heightMeters)
    }
}

// MARK: - Extension for FurnitureFitDetection

extension FurnitureFitDetection {
    /// Get real-world dimensions for this detection
    func realWorldDimensions(calculator: FurnitureSizingCalculator) -> FurnitureDimensions? {
        return FurnitureDimensions.forClass(classIdx)
    }

    /// Calculate scale factor for this detection relative to room
    func scaleFactor(calculator: FurnitureSizingCalculator) -> CGFloat {
        return calculator.calculateScaleFactor(
            classIdx: classIdx,
            detectedWidthPixels: CGFloat(w),
            detectedHeightPixels: CGFloat(h)
        )
    }
}
