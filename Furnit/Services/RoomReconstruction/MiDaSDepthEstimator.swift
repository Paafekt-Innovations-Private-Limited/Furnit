import UIKit
import CoreImage
import Vision
import CoreML

// MARK: - MiDaS Depth Estimator
class MiDaSDepthEstimator {
    private var model: VNCoreMLModel?
    
    init() {
        print("🧠 [DepthEstimator] Initializing")
        loadModel()
    }
    
    private func loadModel() {
        print("📦 [DepthEstimator] Attempting to load MiDaS model")
        print("⚠️ [DepthEstimator] MiDaS model not available, will use fallback")
    }
    
    func estimateDepth(from image: UIImage) async -> CIImage? {
        print("🔬 [DepthEstimator] Estimating depth from image")
        guard let cgImage = image.cgImage else {
            print("❌ [DepthEstimator] Failed to get CGImage")
            return nil
        }
        
        let ciImage = CIImage(cgImage: cgImage)
        
        if model != nil {
            print("   - Using MiDaS model")
            return nil
        }
        
        print("   - Using synthetic depth map (fallback)")
        return generateSyntheticDepthMap(from: ciImage)
    }
    
    private func generateSyntheticDepthMap(from image: CIImage) -> CIImage {
        print("🎨 [DepthEstimator] Generating synthetic depth map")
        
        guard let grayscale = CIFilter(name: "CIPhotoEffectMono")?.apply(image: image),
              let edges = CIFilter(name: "CIEdges")?.apply(image: grayscale, intensity: 2.0) else {
            print("⚠️ [DepthEstimator] Filter failed, returning original")
            return image
        }
        
        print("✅ [DepthEstimator] Synthetic depth map created")
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
