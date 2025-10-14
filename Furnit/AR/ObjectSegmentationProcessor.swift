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
    // DeepLabV3 uses PASCAL VOC classes - expanded to include more furniture items
    private let furnitureClassLabels = [
        9,  // chair - basic seating furniture
        11, // dining table - tables for dining
        18, // sofa - couches and sofas  
        16, // potted plant - decorative furniture/accessories
        20  // tv monitor - entertainment furniture
    ]
    
    // PASCAL VOC 2012 class names for debugging
    private let pascalVOCClasses = [
        0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 
        5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 
        10: "cow", 11: "dining table", 12: "dog", 13: "horse", 14: "motorbike", 
        15: "person", 16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv monitor"
    ]
    
    init() {
        loadSegmentationModel()
    }
    
    // Load the CoreML segmentation model
    private func loadSegmentationModel() {
        print("🔍 MODELS IN BUNDLE: \(Bundle.main.paths(forResourcesOfType: "mlmodel", inDirectory: nil))")
        print("🔍 COMPILED MODELS: \(Bundle.main.paths(forResourcesOfType: "mlmodelc", inDirectory: nil))")
            
        // Debug: Check what's actually in the bundle
        print("🔍 Looking for models in bundle...")
        if let resourcePath = Bundle.main.resourcePath {
            do {
                let files = try FileManager.default.contentsOfDirectory(atPath: resourcePath)
                let mlmodels = files.filter { $0.contains(".mlmodel") }
                print("📦 Found .mlmodel files: \(mlmodels)")
                
                // Also check for compiled models
                let compiled = files.filter { $0.contains(".mlmodelc") }
                print("📦 Found .mlmodelc files: \(compiled)")
            } catch {
                print("❌ Error listing bundle: \(error)")
            }
        }
        
        // Also try direct path
        if let path = Bundle.main.path(forResource: "DeepLabV3", ofType: "mlmodelc") {
            print("✅ Found DeepLabV3 at: \(path)")
        } else {
            print("❌ DeepLabV3.mlmodelc not found in bundle")
        }
        
        // Your existing model loading code...
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
                    print("⚠️ Vision request error: \(error.localizedDescription)")
                    return
                }
                
                self?.processingStatus = "Processing segmentation..."
                
                // Debug: Log the actual result types we're getting
                print("🔍 Vision request completed. Result count: \(request.results?.count ?? 0)")
                if let results = request.results {
                    for (index, result) in results.enumerated() {
                        print("   Result \(index): \(type(of: result))")
                    }
                }
                
                // DeepLabV3 returns VNCoreMLFeatureValueObservation with MLMultiArray
                if let results = request.results as? [VNCoreMLFeatureValueObservation],
                   let featureValue = results.first?.featureValue,
                   let multiArray = featureValue.multiArrayValue {
                    print("✅ Got MLMultiArray segmentation results: \(multiArray.shape)")
                    self?.processSegmentationMultiArray(multiArray, originalImage: image)
                } else if let results = request.results as? [VNPixelBufferObservation],
                          let segmentationBuffer = results.first?.pixelBuffer {
                    // Fallback to pixel buffer processing (some models use this format)
                    print("✅ Got pixel buffer segmentation results")
                    self?.processSegmentationResults(segmentationBuffer, originalImage: image)
                } else {
                    // Enhanced error logging
                    let resultTypes = request.results?.map { type(of: $0) } ?? []
                    self?.errorMessage = "Unexpected Vision result format. Got: \(resultTypes)"
                    print("⚠️ Failed to get segmentation results. Result types: \(resultTypes)")
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
        
        // Debug: Count all detected classes
        var classPixelCounts: [Int: Int] = [:]
        
        // Extract furniture pixels based on class labels
        for i in 0..<(width * height) {
            let pixelValue = segmentationData[i]
            let classIndex = Int(pixelValue)
            
            // Count pixels for each class
            classPixelCounts[classIndex, default: 0] += 1
            
            // Check if pixel belongs to furniture class
            if furnitureClassLabels.contains(classIndex) {
                maskData[i] = 255 // White for furniture
                furniturePixelCount += 1
            } else {
                maskData[i] = 0 // Black for background
            }
        }
        
        // Debug: Print detected classes and their pixel counts
        print("🔍 Segmentation Analysis:")
        print("   Image size: \(width)x\(height) = \(width * height) pixels")
        print("   Detected classes with >0.1% coverage:")
        let totalPixels = width * height
        let sortedClasses = classPixelCounts.sorted { $0.value > $1.value }
        
        for (classIndex, pixelCount) in sortedClasses {
            let percentage = Double(pixelCount) / Double(totalPixels) * 100
            if percentage > 0.1 {
                let className = getClassName(for: classIndex)
                print("     Class \(classIndex) (\(className)): \(pixelCount) pixels (\(String(format: "%.1f", percentage))%)")
            }
        }
        
        // Debug furniture detection specifically
        let furniturePixels = furnitureClassLabels.reduce(0) { sum, classLabel in
            sum + (classPixelCounts[classLabel] ?? 0)
        }
        let furniturePercentage = Double(furniturePixels) / Double(totalPixels) * 100
        print("   Furniture pixels (classes 9,11,16,18,20): \(furniturePixels) (\(String(format: "%.3f", furniturePercentage))%)")
        
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
    
    // Process MLMultiArray segmentation results from DeepLabV3
    private func processSegmentationMultiArray(_ multiArray: MLMultiArray, originalImage: UIImage) {
        processingStatus = "Extracting furniture from MLMultiArray..."
        
        // DeepLabV3 outputs 513x513 array of class indices
        let shape = multiArray.shape
        guard shape.count >= 2 else {
            errorMessage = "Invalid MLMultiArray shape: \(shape)"
            isProcessing = false
            return
        }
        
        let width = shape[1].intValue  // Should be 513
        let height = shape[0].intValue // Should be 513
        let totalPixels = width * height
        
        print("🔍 Processing MLMultiArray segmentation:")
        print("   Array shape: \(shape)")
        print("   Dimensions: \(width)x\(height) = \(totalPixels) pixels")
        print("   Data type: \(multiArray.dataType.rawValue)")
        
        // Convert MLMultiArray to pointer for efficient access
        guard let dataPointer = try? UnsafeBufferPointer<Int32>(multiArray) else {
            errorMessage = "Failed to access MLMultiArray data"
            isProcessing = false
            return
        }
        
        // Create binary mask for furniture objects
        let maskData = UnsafeMutablePointer<UInt8>.allocate(capacity: totalPixels)
        defer { maskData.deallocate() }
        
        var furniturePixelCount = 0
        var classPixelCounts: [Int: Int] = [:]
        
        // Process each pixel in the segmentation array
        for i in 0..<totalPixels {
            let classIndex = Int(dataPointer[i])
            
            // Count pixels for each detected class
            classPixelCounts[classIndex, default: 0] += 1
            
            // Check if pixel belongs to furniture class
            if furnitureClassLabels.contains(classIndex) {
                maskData[i] = 255 // White for furniture
                furniturePixelCount += 1
            } else {
                maskData[i] = 0 // Black for background
            }
        }
        
        // Debug: Print detected classes and their pixel counts
        print("🔍 MLMultiArray Segmentation Analysis:")
        print("   Image size: \(width)x\(height) = \(totalPixels) pixels")
        print("   Detected classes with >0.1% coverage:")
        let sortedClasses = classPixelCounts.sorted { $0.value > $1.value }
        
        for (classIndex, pixelCount) in sortedClasses {
            let percentage = Double(pixelCount) / Double(totalPixels) * 100
            if percentage > 0.1 {
                let className = getClassName(for: classIndex)
                print("     Class \(classIndex) (\(className)): \(pixelCount) pixels (\(String(format: "%.1f", percentage))%)")
            }
        }
        
        // Debug furniture detection specifically
        let furniturePixels = furnitureClassLabels.reduce(0) { sum, classLabel in
            sum + (classPixelCounts[classLabel] ?? 0)
        }
        let furniturePercentage = Double(furniturePixels) / Double(totalPixels) * 100
        print("   Furniture pixels (classes 9,11,16,18,20): \(furniturePixels) (\(String(format: "%.3f", furniturePercentage))%)")
        
        // Check if we found any furniture objects
        let furnitureRatio = Double(furniturePixelCount) / Double(totalPixels)
        if furnitureRatio < 0.01 { // Less than 1% furniture pixels
            errorMessage = "No furniture detected in image. Try pointing at chairs, tables, or sofas."
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
    
    // Helper function to get class name for debugging
    private func getClassName(for classIndex: Int) -> String {
        return pascalVOCClasses[classIndex] ?? "unknown(\(classIndex))"
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
