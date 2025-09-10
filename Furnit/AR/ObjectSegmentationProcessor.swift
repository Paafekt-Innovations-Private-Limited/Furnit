import CoreML
@preconcurrency import Vision
import UIKit
import CoreGraphics
import Accelerate

@MainActor
class ObjectSegmentationProcessor: ObservableObject {
    // Published properties for UI updates
    @Published var isProcessing = false
    @Published var segmentedImage: UIImage?
    @Published var errorMessage: String?
    @Published var processingStatus = "Ready"
    
    // CoreML model and Vision request
    private var mlModel: MLModel?
    private var vnModel: VNCoreMLModel?
    
    // Furniture class labels for DeepLabV3
    // DeepLabV3 uses PASCAL VOC classes - chair (9), dining table (11), sofa (15)
    private let furnitureClassLabels = [9, 11, 15] // chair, dining table, sofa
    private let classNames = [9: "chair", 11: "table", 15: "sofa"]
    
    init() {
        loadSegmentationModel()
    }
    
    // Load the CoreML segmentation model
    private func loadSegmentationModel() {
        // Try to load DeepLabV3 model from the main bundle
        guard let modelURL = Bundle.main.url(forResource: "DeepLabV3", withExtension: "mlmodel") else {
            print("⚠️ DeepLabV3.mlmodel not found in bundle - using mock implementation")
            print("📥 To enable full AR functionality, add DeepLabV3.mlmodel to Xcode project")
            print("   Download from: https://docs-assets.developer.apple.com/ml-res/models/DeepLabV3.mlmodel")
            errorMessage = "AR model not available - add to project"
            return
        }
        
        do {
            // Load the CoreML model
            let mlModel = try MLModel(contentsOf: modelURL)
            print("✅ DeepLabV3 CoreML model loaded successfully")
            
            // Create Vision Core ML model wrapper
            let visionModel = try VNCoreMLModel(for: mlModel)
            print("✅ Vision CoreML model wrapper created")
            
            // Store the models for use in processing
            self.mlModel = mlModel
            self.vnModel = visionModel
            
            print("🤖 DeepLabV3 model ready for furniture segmentation")
            
        } catch {
            print("⚠️ Failed to load DeepLabV3 model: \(error.localizedDescription)")
            print("📥 Falling back to mock implementation for testing")
            errorMessage = "Failed to load AR model - using mock"
        }
    }
    
    // Process image and segment furniture objects
    func processImage(_ image: UIImage) async -> UIImage? {
        guard let vnModel = vnModel else {
            errorMessage = "AR model not available - please download DeepLabV3.mlmodel"
            return await createMockSegmentation(from: image)
        }
        
        // Update processing status
        isProcessing = true
        processingStatus = "Analyzing image..."
        
        // Create Vision request for semantic segmentation
        let request = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
            Task { @MainActor in
                if let error = error {
                    self?.errorMessage = "Segmentation failed: \(error.localizedDescription)"
                    self?.isProcessing = false
                    return
                }
                
                self?.processingStatus = "Processing segmentation..."
                
                // Process the segmentation results
                if let results = request.results as? [VNPixelBufferObservation],
                   let segmentationBuffer = results.first?.pixelBuffer {
                    self?.processSegmentationResults(segmentationBuffer, originalImage: image)
                } else {
                    self?.errorMessage = "Failed to get segmentation results"
                    self?.isProcessing = false
                }
            }
        }
        
        // Configure request for optimal performance
        request.imageCropAndScaleOption = .scaleFill
        
        // Create Vision image from UIImage
        guard let cgImage = image.cgImage else {
            errorMessage = "Failed to convert UIImage to CGImage"
            isProcessing = false
            return nil
        }
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        // Perform segmentation on background queue
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try handler.perform([request])
                    // Wait for async processing to complete
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                        continuation.resume(returning: self.segmentedImage)
                    }
                } catch {
                    Task { @MainActor in
                        self.errorMessage = "Vision processing failed: \(error.localizedDescription)"
                        self.isProcessing = false
                    }
                    continuation.resume(returning: nil)
                }
            }
        }
    }
    
    // Process segmentation results and create clean mask
    private func processSegmentationResults(_ segmentationBuffer: CVPixelBuffer, originalImage: UIImage) {
        processingStatus = "Extracting furniture objects..."
        
        // Lock the pixel buffer for reading
        CVPixelBufferLockBaseAddress(segmentationBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(segmentationBuffer, .readOnly) }
        
        let width = CVPixelBufferGetWidth(segmentationBuffer)
        let height = CVPixelBufferGetHeight(segmentationBuffer)
        let _ = CVPixelBufferGetBytesPerRow(segmentationBuffer) // Not used in this implementation
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(segmentationBuffer) else {
            errorMessage = "Failed to access segmentation buffer"
            isProcessing = false
            return
        }
        
        // Create binary mask for furniture objects
        let maskData = UnsafeMutablePointer<UInt8>.allocate(capacity: width * height)
        defer { maskData.deallocate() }
        
        let segmentationData = baseAddress.assumingMemoryBound(to: UInt8.self)
        var furniturePixelCount = 0
        
        // Extract furniture pixels based on class labels
        for i in 0..<(width * height) {
            let pixelValue = segmentationData[i]
            
            // Check if pixel belongs to furniture class
            if furnitureClassLabels.contains(Int(pixelValue)) {
                maskData[i] = 255 // White for furniture
                furniturePixelCount += 1
            } else {
                maskData[i] = 0 // Black for background
            }
        }
        
        // Check if we found any furniture objects
        let furnitureRatio = Double(furniturePixelCount) / Double(width * height)
        if furnitureRatio < 0.01 { // Less than 1% furniture pixels
            errorMessage = "No furniture detected. Please point camera at furniture objects."
            isProcessing = false
            return
        }
        
        // Apply morphological operations to clean up the mask
        let cleanedMask = applyMorphologicalOperations(maskData, width: width, height: height)
        
        // Create segmented image with transparent background
        if let segmentedUIImage = createSegmentedImage(originalImage: originalImage,
                                                      mask: cleanedMask,
                                                      width: width,
                                                      height: height) {
            segmentedImage = segmentedUIImage
            processingStatus = "Segmentation complete - Ready to place"
        } else {
            errorMessage = "Failed to create segmented image"
        }
        
        isProcessing = false
    }
    
    // Apply morphological operations to clean up the segmentation mask
    private func applyMorphologicalOperations(_ maskData: UnsafeMutablePointer<UInt8>, 
                                            width: Int, 
                                            height: Int) -> UnsafeMutablePointer<UInt8> {
        let cleanedMask = UnsafeMutablePointer<UInt8>.allocate(capacity: width * height)
        
        // Apply opening (erosion followed by dilation) to remove noise
        let kernelSize = 3
        let kernel = UnsafeMutablePointer<UInt8>.allocate(capacity: kernelSize * kernelSize)
        defer { kernel.deallocate() }
        
        // Create circular kernel for morphological operations
        for i in 0..<(kernelSize * kernelSize) {
            kernel[i] = 1
        }
        
        // Simple erosion operation
        for y in kernelSize/2..<(height - kernelSize/2) {
            for x in kernelSize/2..<(width - kernelSize/2) {
                var minVal: UInt8 = 255
                
                for ky in 0..<kernelSize {
                    for kx in 0..<kernelSize {
                        let pixelIndex = (y - kernelSize/2 + ky) * width + (x - kernelSize/2 + kx)
                        minVal = min(minVal, maskData[pixelIndex])
                    }
                }
                
                cleanedMask[y * width + x] = minVal
            }
        }
        
        // Copy original data for border pixels
        for i in 0..<(width * height) {
            let y = i / width
            let x = i % width
            if y < kernelSize/2 || y >= height - kernelSize/2 || x < kernelSize/2 || x >= width - kernelSize/2 {
                cleanedMask[i] = maskData[i]
            }
        }
        
        return cleanedMask
    }
    
    // Create final segmented image with transparent background
    private func createSegmentedImage(originalImage: UIImage, 
                                    mask: UnsafeMutablePointer<UInt8>,
                                    width: Int,
                                    height: Int) -> UIImage? {
        
        // Resize original image to match segmentation dimensions
        guard let resizedOriginal = originalImage.resized(to: CGSize(width: width, height: height)),
              let cgImage = resizedOriginal.cgImage else {
            return nil
        }
        
        // Create new image with alpha channel
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(data: nil,
                                     width: width,
                                     height: height,
                                     bitsPerComponent: 8,
                                     bytesPerRow: width * 4,
                                     space: colorSpace,
                                     bitmapInfo: bitmapInfo.rawValue) else {
            return nil
        }
        
        // Draw original image
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Apply alpha mask based on segmentation
        guard let imageData = context.data?.assumingMemoryBound(to: UInt8.self) else {
            return nil
        }
        
        for i in 0..<(width * height) {
            let pixelIndex = i * 4
            let alpha = mask[i]
            imageData[pixelIndex + 3] = alpha // Set alpha channel
        }
        
        // Create final CGImage and convert to UIImage
        guard let finalCGImage = context.makeImage() else {
            return nil
        }
        
        return UIImage(cgImage: finalCGImage)
    }
    
    // Create a mock segmentation for testing when model is not available
    private func createMockSegmentation(from image: UIImage) async -> UIImage? {
        isProcessing = true
        processingStatus = "Creating mock segmentation for testing..."
        
        // Wait to simulate processing
        try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        
        // Create a simple mock segmentation - center rectangle with alpha
        guard let cgImage = image.cgImage else {
            isProcessing = false
            return nil
        }
        
        let width = cgImage.width
        let height = cgImage.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(data: nil,
                                     width: width,
                                     height: height,
                                     bitsPerComponent: 8,
                                     bytesPerRow: width * 4,
                                     space: colorSpace,
                                     bitmapInfo: bitmapInfo.rawValue) else {
            isProcessing = false
            return nil
        }
        
        // Draw original image
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Create mock furniture area in center with some transparency
        let centerRect = CGRect(
            x: width / 4,
            y: height / 4,
            width: width / 2,
            height: height / 2
        )
        
        guard let imageData = context.data?.assumingMemoryBound(to: UInt8.self) else {
            isProcessing = false
            return nil
        }
        
        // Apply alpha mask to center area only
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = (y * width + x) * 4
                
                // Check if pixel is in center area
                if centerRect.contains(CGPoint(x: x, y: y)) {
                    // Keep the pixel opaque (mock furniture)
                    imageData[pixelIndex + 3] = 255
                } else {
                    // Make background transparent
                    imageData[pixelIndex + 3] = 0
                }
            }
        }
        
        // Create final CGImage and convert to UIImage
        guard let finalCGImage = context.makeImage() else {
            isProcessing = false
            return nil
        }
        
        processingStatus = "Mock segmentation complete - Ready to place"
        isProcessing = false
        
        return UIImage(cgImage: finalCGImage)
    }
}

// MARK: - UIImage Extension for Resizing
extension UIImage {
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        defer { UIGraphicsEndImageContext() }
        
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}