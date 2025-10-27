import UIKit
import CoreImage

// MARK: - Texture Processor
class TextureProcessor {
    func inpaintMissingRegions(_ image: UIImage, mask: UIImage) -> UIImage {
        print("🎨 [TextureProcessor] Inpainting missing regions")
        
        guard let cgImage = image.cgImage else {
            print("⚠️ [TextureProcessor] No CGImage available")
            return image
        }
        guard let maskCG = mask.cgImage else {
            print("⚠️ [TextureProcessor] Mask lacks CGImage; returning original image")
            return image
        }
        
        let ciImage = CIImage(cgImage: cgImage)
        let blurred = ciImage.applyingGaussianBlur(sigma: 10.0)
        
        guard let filter = CIFilter(name: "CIBlendWithMask") else {
            print("⚠️ [TextureProcessor] Missing CIBlendWithMask filter")
            return image
        }
        filter.setValue(blurred, forKey: "inputImage")
        filter.setValue(ciImage, forKey: "inputBackgroundImage")
        filter.setValue(CIImage(cgImage: maskCG), forKey: "inputMaskImage")
        
        let context = CIContext()
        guard let output = filter.outputImage,
              let cgOutput = context.createCGImage(output, from: output.extent) else {
            print("⚠️ [TextureProcessor] Inpainting failed")
            return image
        }
        
        print("✅ [TextureProcessor] Inpainting complete")
        return UIImage(cgImage: cgOutput)
    }
}
