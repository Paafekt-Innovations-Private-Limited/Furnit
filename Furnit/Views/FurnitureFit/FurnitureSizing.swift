import Foundation
import CoreGraphics

/// Real-world furniture dimensions in meters.
/// NOTE: We no longer keep any global \"standard\" sizes per class; FurnitureFit
/// should not rely on hard-coded catalog dimensions.
struct FurnitureDimensions {
    let width: Float   // Width in meters
    let height: Float  // Height in meters
    let depth: Float   // Depth in meters (optional, for 3D placement)
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
    /// Get real-world dimensions for this detection based on meters-per-pixel only.
    func realWorldDimensions(calculator: FurnitureSizingCalculator) -> FurnitureDimensions? {
        let est = calculator.estimateRealWorldSize(
            detectedWidthPixels: CGFloat(w),
            detectedHeightPixels: CGFloat(h)
        )
        return FurnitureDimensions(width: est.widthMeters, height: est.heightMeters, depth: 0)
    }
}
