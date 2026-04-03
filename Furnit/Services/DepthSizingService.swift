import CoreGraphics
import Foundation

public struct DepthSizingService {
    public struct CameraIntrinsics: Sendable, Equatable {
        public let focalLengthMM: Float
        public let sensorWidthMM: Float
        public let imageWidthPx: Float
        public let imageHeightPx: Float

        public init(
            focalLengthMM: Float,
            sensorWidthMM: Float,
            imageWidthPx: Float,
            imageHeightPx: Float
        ) {
            self.focalLengthMM = focalLengthMM
            self.sensorWidthMM = sensorWidthMM
            self.imageWidthPx = imageWidthPx
            self.imageHeightPx = imageHeightPx
        }

        public var horizontalFieldOfViewRadians: Float {
            guard focalLengthMM > 0, sensorWidthMM > 0 else { return 0 }
            return 2 * atan(sensorWidthMM / (2 * focalLengthMM))
        }

        public func focalLengthPixels(viewportWidth: CGFloat) -> Float {
            guard sensorWidthMM > 0, imageWidthPx > 0, viewportWidth > 0 else { return 0 }
            let effectiveImageWidth = imageWidthPx > 0 ? imageWidthPx : Float(viewportWidth)
            return (focalLengthMM / sensorWidthMM) * effectiveImageWidth
        }
    }

    private let intrinsics: CameraIntrinsics
    private let sceneToMeters: Float

    public init(intrinsics: CameraIntrinsics, sceneToMeters: Float) {
        self.intrinsics = intrinsics
        self.sceneToMeters = max(sceneToMeters, 0.0001)
    }

    public func estimateMetricWidth(
        leftScreenX: CGFloat,
        rightScreenX: CGFloat,
        depth: Float,
        viewportWidth: CGFloat
    ) -> Float {
        let pixelSpan = Float(abs(rightScreenX - leftScreenX))
        let focalPx = intrinsics.focalLengthPixels(viewportWidth: viewportWidth)
        guard focalPx > 0, depth > 0 else { return 0 }
        let sceneWidth = pixelSpan * (depth / focalPx)
        return sceneWidth * sceneToMeters
    }

    public func estimateMetricHeight(
        topScreenY: CGFloat,
        bottomScreenY: CGFloat,
        depth: Float,
        viewportWidth: CGFloat
    ) -> Float {
        let pixelSpan = Float(abs(bottomScreenY - topScreenY))
        let focalPx = intrinsics.focalLengthPixels(viewportWidth: viewportWidth)
        guard focalPx > 0, depth > 0 else { return 0 }
        let sceneHeight = pixelSpan * (depth / focalPx)
        return sceneHeight * sceneToMeters
    }
}
