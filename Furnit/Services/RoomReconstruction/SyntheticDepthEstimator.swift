import CoreImage
import UIKit

// MARK: - Synthetic Depth Estimator
/// Legacy non-ML depth placeholder used by ``SinglePhotoRoomReconstructor``.
/// This is not MiDaS and does not attempt to load a depth model.
final class SyntheticDepthEstimator {
    init() {
        logDebug("🧠 [SyntheticDepthEstimator] Initializing")
    }

    func estimateDepth(from image: UIImage) async -> CIImage? {
        logDebug("🔬 [SyntheticDepthEstimator] Estimating synthetic depth from image")
        guard let cgImage = image.cgImage else {
            logDebug("❌ [SyntheticDepthEstimator] Failed to get CGImage")
            return nil
        }

        return generateSyntheticDepthMap(from: CIImage(cgImage: cgImage))
    }

    private func generateSyntheticDepthMap(from image: CIImage) -> CIImage {
        logDebug("🎨 [SyntheticDepthEstimator] Generating synthetic depth map")

        guard let grayscale = CIFilter(name: "CIPhotoEffectMono")?.apply(image: image),
              let edges = CIFilter(name: "CIEdges")?.apply(image: grayscale, intensity: 2.0) else {
            logDebug("⚠️ [SyntheticDepthEstimator] Filter failed, returning original")
            return image
        }

        logDebug("✅ [SyntheticDepthEstimator] Synthetic depth map created")
        return edges
    }
}

// MARK: - CIFilter Extensions
extension CIFilter {
    func apply(image: CIImage) -> CIImage? {
        setValue(image, forKey: kCIInputImageKey)
        return outputImage
    }

    func apply(image: CIImage, intensity: Double) -> CIImage? {
        setValue(image, forKey: kCIInputImageKey)
        setValue(intensity, forKey: kCIInputIntensityKey)
        return outputImage
    }
}
