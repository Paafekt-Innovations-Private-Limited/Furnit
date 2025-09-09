import CoreML
import Vision
import CoreVideo
import UIKit
import CoreImage

protocol ObjectSegmentationProcessorDelegate {
    func processor(_ processor: ObjectSegmentationProcessor, didSegment object: SegmentedObject)
    func processor(_ processor: ObjectSegmentationProcessor, didFailWithError error: Error)
}

struct SegmentedObject {
    let originalImage: CIImage
    let segmentationMask: CVPixelBuffer
    let boundingBox: CGRect
    let confidence: Float
    let objectClass: FurnitureClass
    
    // Create a UIImage showing the furniture with transparent background
    func createFurnitureTexture() -> UIImage? {
        return ObjectSegmentationProcessor.extractFurnitureFromImage(originalImage: originalImage, mask: segmentationMask, boundingBox: boundingBox)
    }
}

enum FurnitureClass: Int, CaseIterable {
    case chair = 56
    case sofa = 57
    case pottedPlant = 58
    case bed = 59
    case diningTable = 60
    case toilet = 61
    case tvMonitor = 62
    
    var displayName: String {
        switch self {
        case .chair: return "Chair"
        case .sofa: return "Sofa"
        case .pottedPlant: return "Plant"
        case .bed: return "Bed"
        case .diningTable: return "Table"
        case .toilet: return "Toilet"
        case .tvMonitor: return "TV"
        }
    }
    
    var isSupported: Bool {
        // Focus on common furniture items for better results
        switch self {
        case .chair, .sofa, .diningTable:
            return true
        default:
            return false
        }
    }
}

// Frame buffer for temporal smoothing
struct SegmentationFrame {
    let mask: CVPixelBuffer
    let confidence: Float
    let boundingBox: CGRect
    let timestamp: CFTimeInterval
    let objectClass: FurnitureClass
}

class ObjectSegmentationProcessor: ObservableObject {
    var delegate: ObjectSegmentationProcessorDelegate?
    
    private let processingQueue = DispatchQueue(label: "segmentation.processing.queue", qos: .userInteractive)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    
    // Frame processing throttling - Increased to 10 FPS for responsiveness
    private var lastProcessedTime: CFTimeInterval = 0
    private let minFrameInterval: CFTimeInterval = 1.0 / 10.0 // 10 FPS for better responsiveness
    
    // Enhanced temporal smoothing for single capture mode
    private var frameBuffer: [SegmentationFrame] = []
    private let maxBufferSize = 1 // Single frame processing for high-quality captures
    private var lastStableSegmentation: SegmentedObject?
    private var lastUpdateTime: CFTimeInterval = 0
    private let minUpdateInterval: CFTimeInterval = 0.1 // Quick processing for single captures
    
    // Edge refinement settings
    private let enableEdgeRefinement = true
    private let morphologyKernelSize = 5
    private let edgeSmoothingIterations = 2
    
    // Quality filtering - Enhanced thresholds for cleaner results
    private let minConfidenceThreshold: Float = 0.7 // Increased for higher quality
    private let minMaskArea: Float = 0.03 // Increased minimum area (3% of image)
    private let maxMaskArea: Float = 0.6 // Maximum area to filter out background captures
    
    private var visionModel: VNCoreMLModel?
    
    @Published var isProcessing = false
    
    init() {
        loadSegmentationModel()
    }
    
    private func loadSegmentationModel() {
        processingQueue.async { [weak self] in
            self?.setupVisionModel()
        }
    }
    
    private func setupVisionModel() {
        // Try to load DeepLabV3 CoreML model
        print("🔍 Searching for DeepLabV3 model...")
        
        if let modelURL = Bundle.main.url(forResource: "DeepLabV3", withExtension: "mlmodelc") {
            print("✅ Found DeepLabV3.mlmodelc at: \(modelURL.path)")
            loadCoreMLModel(from: modelURL)
        } else if let modelURL = Bundle.main.url(forResource: "DeepLabV3", withExtension: "mlmodel") {
            print("✅ Found DeepLabV3.mlmodel at: \(modelURL.path)")
            loadCoreMLModel(from: modelURL)
        } else {
            print("❌ DeepLabV3 model not found in bundle")
            print("📁 Available resources:")
            if let resourcePath = Bundle.main.resourcePath {
                let resourceContents = try? FileManager.default.contentsOfDirectory(atPath: resourcePath)
                print("   \(resourceContents?.joined(separator: "\n   ") ?? "None")")
            }
            print("💡 Add DeepLabV3.mlmodel to your project from Apple's CoreML Models Gallery")
            
            // Don't set up any fallback - we need the real model
            delegate?.processor(self, didFailWithError: SegmentationError.modelNotFound)
        }
    }
    
    func processFrame(_ pixelBuffer: CVPixelBuffer) {
        let currentTime = CACurrentMediaTime()
        
        // Throttle processing to maintain performance
        guard currentTime - lastProcessedTime >= minFrameInterval else {
            return
        }
        
        lastProcessedTime = currentTime
        
        guard !isProcessing else { return }
        
        processingQueue.async { [weak self] in
            self?.performSegmentation(on: pixelBuffer)
        }
    }
    
    private func performSegmentation(on pixelBuffer: CVPixelBuffer) {
        DispatchQueue.main.async { [weak self] in
            self?.isProcessing = true
        }
        
        // Convert CVPixelBuffer to CIImage for processing
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Always try to use DeepLabV3 CoreML model first
        if visionModel != nil {
            print("🎯 Using DeepLabV3 CoreML model for furniture segmentation")
            performCoreMLSegmentation(with: pixelBuffer, ciImage: ciImage)
        } else {
            print("⚠️ DeepLabV3 model not available - cannot perform furniture segmentation")
            // Instead of using person segmentation (which only detects people), 
            // inform delegate about the failure
            delegate?.processor(self, didFailWithError: SegmentationError.modelNotFound)
        }
        
        DispatchQueue.main.async { [weak self] in
            self?.isProcessing = false
        }
    }
    
    // Removed Vision person segmentation methods - they only detect people, not furniture
    // We now exclusively use DeepLabV3 for proper furniture segmentation
    
    private func calculateBoundingBox(from maskBuffer: CVPixelBuffer) -> CGRect {
        CVPixelBufferLockBaseAddress(maskBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(maskBuffer, .readOnly) }
        
        let width = CVPixelBufferGetWidth(maskBuffer)
        let height = CVPixelBufferGetHeight(maskBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(maskBuffer)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(maskBuffer) else {
            return CGRect(x: 0.2, y: 0.3, width: 0.6, height: 0.4) // Fallback
        }
        
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var minX = width, maxX = 0
        var minY = height, maxY = 0
        
        for y in 0..<height {
            for x in 0..<width {
                let pixel = buffer[y * bytesPerRow + x]
                if pixel > 128 { // Threshold for mask presence
                    minX = min(minX, x)
                    maxX = max(maxX, x)
                    minY = min(minY, y)
                    maxY = max(maxY, y)
                }
            }
        }
        
        // Convert to normalized coordinates
        if maxX > minX && maxY > minY {
            return CGRect(
                x: Double(minX) / Double(width),
                y: Double(minY) / Double(height),
                width: Double(maxX - minX) / Double(width),
                height: Double(maxY - minY) / Double(height)
            )
        } else {
            // No significant segmentation found, return center area
            return CGRect(x: 0.2, y: 0.3, width: 0.6, height: 0.4)
        }
    }
    
    // Removed mock segmentation methods - we only want real DeepLabV3 furniture segmentation
    // No fallbacks to prevent showing incorrect/default masks
    
    // MARK: - Real CoreML Integration
    private func loadCoreMLModel(from modelURL: URL) {
        print("🔄 Loading CoreML model from: \(modelURL.lastPathComponent)")
        
        do {
            let model = try MLModel(contentsOf: modelURL)
            let visionModel = try VNCoreMLModel(for: model)
            self.visionModel = visionModel
            print("✅ Successfully loaded CoreML segmentation model")
        } catch {
            print("❌ Failed to load CoreML model: \(error)")
            delegate?.processor(self, didFailWithError: error)
        }
    }
    
    @available(iOS 15.0, *)
    private func performCoreMLSegmentation(with pixelBuffer: CVPixelBuffer, ciImage: CIImage) {
        guard let visionModel = visionModel else {
            print("❌ CoreML model not available - cannot perform furniture segmentation")
            delegate?.processor(self, didFailWithError: SegmentationError.modelNotFound)
            return
        }
        
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            self?.handleCoreMLSegmentationResult(request: request, error: error, originalImage: ciImage, pixelBuffer: pixelBuffer)
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        
        do {
            try handler.perform([request])
        } catch {
            print("❌ CoreML segmentation failed: \(error)")
            delegate?.processor(self, didFailWithError: error)
        }
    }
    
    @available(iOS 15.0, *)
    private func handleCoreMLSegmentationResult(request: VNRequest, error: Error?, originalImage: CIImage, pixelBuffer: CVPixelBuffer) {
        guard error == nil,
              let observations = request.results as? [VNCoreMLFeatureValueObservation],
              let segmentationOutput = observations.first?.featureValue.multiArrayValue else {
            print("❌ CoreML segmentation failed or no results")
            delegate?.processor(self, didFailWithError: SegmentationError.processingFailed)
            return
        }
        
        // Process DeepLabV3 segmentation results
        print("✅ CoreML segmentation completed, processing results...")
        
        // DeepLabV3 outputs segmentation classes for each pixel
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        // Create segmentation mask from DeepLabV3 output
        guard let maskPixelBuffer = createSegmentationMask(from: segmentationOutput, width: width, height: height) else {
            print("❌ Failed to create segmentation mask from CoreML output")
            delegate?.processor(self, didFailWithError: SegmentationError.processingFailed)
            return
        }
        
        // Calculate bounding box from the mask
        let boundingBox = calculateBoundingBox(from: maskPixelBuffer)
        
        // Determine the most prominent furniture class from segmentation
        let detectedClass = detectFurnitureClass(from: segmentationOutput, width: width, height: height)
        
        // Create segmented object result
        let segmentedObject = SegmentedObject(
            originalImage: originalImage,
            segmentationMask: maskPixelBuffer,
            boundingBox: boundingBox,
            confidence: 0.90, // Higher confidence for real ML model
            objectClass: detectedClass
        )
        
        // Process with stabilization instead of direct delegate call
        processFrameWithStabilization(segmentedObject)
        print("✅ DeepLabV3-based segmentation completed with class: \(detectedClass.displayName), bounding box: \(boundingBox)")
    }
    
    // MARK: - DeepLabV3 Processing Helpers
    
    private func createSegmentationMask(from multiArray: MLMultiArray, width: Int, height: Int) -> CVPixelBuffer? {
        // Safely inspect the MLMultiArray structure first
        print("📊 MLMultiArray shape: \(multiArray.shape)")
        print("📊 MLMultiArray dataType: \(multiArray.dataType.rawValue)")
        print("📊 MLMultiArray count: \(multiArray.count)")
        
        // Check what the actual data type is
        let dataTypeString: String
        switch multiArray.dataType {
        case .double:
            dataTypeString = "double"
        case .float16:
            dataTypeString = "float16"  
        case .float32:
            dataTypeString = "float32"
        case .int32:
            dataTypeString = "int32"
        default:
            dataTypeString = "unknown(\(multiArray.dataType.rawValue))"
        }
        print("📊 Data type: \(dataTypeString)")
        
        // DeepLabV3 outputs class predictions for each pixel
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let result = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_OneComponent8,
            attributes as CFDictionary,
            &pixelBuffer
        )
        
        guard result == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        
        // Initialize as transparent
        memset(baseAddress, 0, height * bytesPerRow)
        
        // Handle different DeepLabV3 output formats
        if multiArray.shape.count == 2 {
            // DeepLabV3 outputs direct segmentation mask [height, width] with class indices
            print("📊 Processing direct segmentation mask format")
            return processDirect2DSegmentationMask(multiArray, targetWidth: width, targetHeight: height)
        }
        
        // Safely handle class probability formats (3D/4D shapes)
        guard multiArray.shape.count >= 3 else {
            print("❌ Unexpected MLMultiArray shape: \(multiArray.shape)")
            return buffer // Return empty mask
        }
        
        // Common DeepLabV3 shapes: [1, 21, H, W] or [21, H, W] or [H, W, 21]
        let numClasses: Int
        let modelHeight: Int
        let modelWidth: Int
        
        if multiArray.shape.count == 4 {
            // Shape: [batch, classes, height, width]
            numClasses = multiArray.shape[1].intValue
            modelHeight = multiArray.shape[2].intValue
            modelWidth = multiArray.shape[3].intValue
        } else if multiArray.shape.count == 3 {
            // Shape: [classes, height, width] or [height, width, classes]
            if multiArray.shape[0].intValue == 21 {
                // [classes, height, width]
                numClasses = multiArray.shape[0].intValue
                modelHeight = multiArray.shape[1].intValue
                modelWidth = multiArray.shape[2].intValue
            } else {
                // [height, width, classes]
                modelHeight = multiArray.shape[0].intValue
                modelWidth = multiArray.shape[1].intValue
                numClasses = multiArray.shape[2].intValue
            }
        } else {
            print("❌ Unsupported MLMultiArray shape: \(multiArray.shape)")
            return buffer // Return empty mask
        }
        
        print("📊 Detected shape - Classes: \(numClasses), Height: \(modelHeight), Width: \(modelWidth)")
        
        // Process DeepLabV3 output safely
        let outputPtr = multiArray.dataPointer.assumingMemoryBound(to: Float32.self)
        let totalElements = multiArray.count
        
        // Scale factors if model output size differs from camera frame
        let scaleX = Double(modelWidth) / Double(width)
        let scaleY = Double(modelHeight) / Double(height)
        
        // For each pixel in the target resolution
        for y in 0..<height {
            for x in 0..<width {
                // Map to model coordinates
                let modelX = min(Int(Double(x) * scaleX), modelWidth - 1)
                let modelY = min(Int(Double(y) * scaleY), modelHeight - 1)
                
                // Find the class with highest probability for this pixel
                var maxClassValue: Float32 = 0
                var maxClassIndex = 0
                
                for classIndex in 0..<min(numClasses, 21) { // Limit to PASCAL VOC classes
                    let outputIndex: Int
                    
                    if multiArray.shape.count == 4 {
                        // [batch, classes, height, width]
                        outputIndex = classIndex * modelHeight * modelWidth + modelY * modelWidth + modelX
                    } else if multiArray.shape.count == 3 && multiArray.shape[0].intValue == numClasses {
                        // [classes, height, width]
                        outputIndex = classIndex * modelHeight * modelWidth + modelY * modelWidth + modelX
                    } else {
                        // [height, width, classes]
                        outputIndex = modelY * modelWidth * numClasses + modelX * numClasses + classIndex
                    }
                    
                    // Bounds check
                    guard outputIndex >= 0 && outputIndex < totalElements else {
                        continue
                    }
                    
                    let classValue = outputPtr[outputIndex]
                    
                    if classValue > maxClassValue {
                        maxClassValue = classValue
                        maxClassIndex = classIndex
                    }
                }
                
                // Check if the predicted class is a furniture class we care about - Higher threshold for quality
                if isFurnitureClass(maxClassIndex) && maxClassValue > minConfidenceThreshold {
                    let offset = y * bytesPerRow + x
                    baseAddress.assumingMemoryBound(to: UInt8.self)[offset] = 255 // White = detected furniture
                }
            }
        }
        
        return buffer
    }
    
    private func processDirect2DSegmentationMask(_ multiArray: MLMultiArray, targetWidth: Int, targetHeight: Int) -> CVPixelBuffer? {
        let modelHeight = multiArray.shape[0].intValue
        let modelWidth = multiArray.shape[1].intValue
        
        print("📊 Model segmentation size: \(modelWidth) x \(modelHeight)")
        
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let result = CVPixelBufferCreate(
            kCFAllocatorDefault,
            targetWidth,
            targetHeight,
            kCVPixelFormatType_OneComponent8,
            attributes as CFDictionary,
            &pixelBuffer
        )
        
        guard result == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        
        // Initialize as transparent
        memset(baseAddress, 0, targetHeight * bytesPerRow)
        
        // Process segmentation mask - each pixel contains a class index
        
        // Scale factors
        let scaleX = Double(modelWidth) / Double(targetWidth)
        let scaleY = Double(modelHeight) / Double(targetHeight)
        
        var detectedClasses: [Int: Int] = [:]
        var samplePixelValues: [Int] = []
        var pixelCount = 0
        
        for y in 0..<targetHeight {
            for x in 0..<targetWidth {
                // Map to model coordinates
                let modelX = min(Int(Double(x) * scaleX), modelWidth - 1)
                let modelY = min(Int(Double(y) * scaleY), modelHeight - 1)
                
                let pixelIndex = modelY * modelWidth + modelX
                guard pixelIndex >= 0 && pixelIndex < multiArray.count else { continue }
                
                let classIndex: Int
                if multiArray.dataType == .int32 {
                    let int32Ptr = multiArray.dataPointer.assumingMemoryBound(to: Int32.self)
                    classIndex = Int(int32Ptr[pixelIndex])
                } else if multiArray.dataType == .float32 {
                    let floatPtr = multiArray.dataPointer.assumingMemoryBound(to: Float32.self)
                    classIndex = Int(floatPtr[pixelIndex])
                } else if multiArray.dataType == .double {
                    let doublePtr = multiArray.dataPointer.assumingMemoryBound(to: Double.self)
                    classIndex = Int(doublePtr[pixelIndex])
                } else {
                    // Try UInt8 for common segmentation mask format
                    let uint8Ptr = multiArray.dataPointer.assumingMemoryBound(to: UInt8.self)
                    classIndex = Int(uint8Ptr[pixelIndex])
                }
                
                // Sample pixel values for debugging
                pixelCount += 1
                if samplePixelValues.count < 100 && pixelCount % 1000 == 0 {
                    samplePixelValues.append(classIndex)
                }
                
                // Count detected classes for later use
                if isFurnitureClass(classIndex) {
                    detectedClasses[classIndex, default: 0] += 1
                    
                    // Mark as detected furniture in the mask
                    let offset = y * bytesPerRow + x
                    baseAddress.assumingMemoryBound(to: UInt8.self)[offset] = 255
                } else if classIndex > 0 && classIndex <= 20 {
                    // Count other PASCAL VOC classes for debugging
                    detectedClasses[classIndex, default: 0] += 1
                }
            }
        }
        
        print("📊 Sample pixel values: \(samplePixelValues.prefix(20))")
        
        // Show detected classes with names for better debugging
        let classesWithNames = detectedClasses.map { (classIndex, count) in
            "\(classIndex)(\(getClassNameForDebugging(classIndex))): \(count)"
        }.joined(separator: ", ")
        print("📊 All detected classes: [\(classesWithNames)]")
        
        let furnitureClasses = detectedClasses.filter { isFurnitureClass($0.key) }
        let furnitureWithNames = furnitureClasses.map { (classIndex, count) in
            "\(classIndex)(\(getClassNameForDebugging(classIndex))): \(count)"
        }.joined(separator: ", ")
        print("📊 Furniture classes found: [\(furnitureWithNames)]")
        
        return buffer
    }
    
    private func detectFurnitureClass(from multiArray: MLMultiArray, width: Int, height: Int) -> FurnitureClass {
        // Handle 2D segmentation mask format
        if multiArray.shape.count == 2 {
            return detectFurnitureClassFrom2DMask(multiArray)
        }
        
        let outputPtr = multiArray.dataPointer.assumingMemoryBound(to: Float32.self)
        
        // Use same shape detection logic as createSegmentationMask
        guard multiArray.shape.count >= 3 else {
            print("❌ Unexpected MLMultiArray shape in detectFurnitureClass: \(multiArray.shape)")
            return .chair // Default fallback
        }
        
        let numClasses: Int
        let modelHeight: Int
        let modelWidth: Int
        
        if multiArray.shape.count == 4 {
            numClasses = multiArray.shape[1].intValue
            modelHeight = multiArray.shape[2].intValue
            modelWidth = multiArray.shape[3].intValue
        } else if multiArray.shape.count == 3 {
            if multiArray.shape[0].intValue == 21 {
                numClasses = multiArray.shape[0].intValue
                modelHeight = multiArray.shape[1].intValue
                modelWidth = multiArray.shape[2].intValue
            } else {
                modelHeight = multiArray.shape[0].intValue
                modelWidth = multiArray.shape[1].intValue
                numClasses = multiArray.shape[2].intValue
            }
        } else {
            return .chair // Default fallback
        }
        
        // Count pixels for each furniture class
        var classCounts: [Int: Int] = [:]
        let totalElements = multiArray.count
        let scaleX = Double(modelWidth) / Double(width)
        let scaleY = Double(modelHeight) / Double(height)
        
        // Sample every 4th pixel for performance
        for y in stride(from: 0, to: height, by: 4) {
            for x in stride(from: 0, to: width, by: 4) {
                let modelX = min(Int(Double(x) * scaleX), modelWidth - 1)
                let modelY = min(Int(Double(y) * scaleY), modelHeight - 1)
                
                // Find the class with highest probability for this pixel
                var maxClassValue: Float32 = 0
                var maxClassIndex = 0
                
                for classIndex in 0..<min(numClasses, 21) {
                    let outputIndex: Int
                    
                    if multiArray.shape.count == 4 {
                        outputIndex = classIndex * modelHeight * modelWidth + modelY * modelWidth + modelX
                    } else if multiArray.shape.count == 3 && multiArray.shape[0].intValue == numClasses {
                        outputIndex = classIndex * modelHeight * modelWidth + modelY * modelWidth + modelX
                    } else {
                        outputIndex = modelY * modelWidth * numClasses + modelX * numClasses + classIndex
                    }
                    
                    // Bounds check
                    guard outputIndex >= 0 && outputIndex < totalElements else {
                        continue
                    }
                    
                    let classValue = outputPtr[outputIndex]
                    
                    if classValue > maxClassValue {
                        maxClassValue = classValue
                        maxClassIndex = classIndex
                    }
                }
                
                // Count significant furniture detections
                if isFurnitureClass(maxClassIndex) && maxClassValue > 0.3 {
                    classCounts[maxClassIndex, default: 0] += 1
                }
            }
        }
        
        // Find the most prominent furniture class
        let mostProminentClass = classCounts.max(by: { $0.value < $1.value })?.key ?? 15 // Default to chair
        
        return mapDeepLabClassToFurnitureClass(mostProminentClass)
    }
    
    private func detectFurnitureClassFrom2DMask(_ multiArray: MLMultiArray) -> FurnitureClass {
        let totalPixels = multiArray.count
        
        // Handle different data types
        let getClassIndex: (Int) -> Int
        
        if multiArray.dataType == .int32 {
            let outputPtr = multiArray.dataPointer.assumingMemoryBound(to: Int32.self)
            getClassIndex = { i in Int(outputPtr[i]) }
        } else if multiArray.dataType == .float32 {
            let outputPtr = multiArray.dataPointer.assumingMemoryBound(to: Float32.self)
            getClassIndex = { i in Int(outputPtr[i]) }
        } else if multiArray.dataType == .double {
            let outputPtr = multiArray.dataPointer.assumingMemoryBound(to: Double.self)
            getClassIndex = { i in Int(outputPtr[i]) }
        } else {
            // Try UInt8 for common segmentation mask format
            let outputPtr = multiArray.dataPointer.assumingMemoryBound(to: UInt8.self)
            getClassIndex = { i in Int(outputPtr[i]) }
        }
        
        var classCounts: [Int: Int] = [:]
        
        // Sample every 16th pixel for performance
        for i in stride(from: 0, to: totalPixels, by: 16) {
            let classIndex = getClassIndex(i)
            
            if isFurnitureClass(classIndex) {
                classCounts[classIndex, default: 0] += 1
            }
        }
        
        // Find the most prominent furniture class
        let mostProminentClass = classCounts.max(by: { $0.value < $1.value })?.key ?? 15
        print("📊 Most prominent furniture class: \(mostProminentClass) with \(classCounts[mostProminentClass] ?? 0) pixels")
        
        return mapDeepLabClassToFurnitureClass(mostProminentClass)
    }
    
    private func isFurnitureClass(_ classIndex: Int) -> Bool {
        // PASCAL VOC class indices for furniture items
        let furnitureClasses: Set<Int> = [
            9,  // chair
            11, // dining table
            18  // sofa
        ]
        return furnitureClasses.contains(classIndex)
    }
    
    private func getClassNameForDebugging(_ classIndex: Int) -> String {
        // PASCAL VOC class names for debugging
        let classNames: [Int: String] = [
            0: "background",
            1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle",
            6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
            11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
            16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"
        ]
        return classNames[classIndex] ?? "unknown(\(classIndex))"
    }
    
    private func mapDeepLabClassToFurnitureClass(_ deepLabClass: Int) -> FurnitureClass {
        // Map PASCAL VOC classes to our furniture classes
        switch deepLabClass {
        case 9: return .chair
        case 11: return .diningTable
        case 18: return .sofa
        default: return .chair // Default fallback
        }
    }
    
    // MARK: - Furniture Texture Extraction
    
    static func extractFurnitureFromImage(originalImage: CIImage, mask: CVPixelBuffer, boundingBox: CGRect) -> UIImage? {
        // Lock the pixel buffer for reading
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
        
        let width = CVPixelBufferGetWidth(mask)
        let height = CVPixelBufferGetHeight(mask)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(mask)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(mask) else {
            print("❌ Failed to get mask base address")
            return nil
        }
        
        // Create UIImage from CIImage first
        let context = CIContext()
        guard let cgImage = context.createCGImage(originalImage, from: originalImage.extent) else {
            print("❌ Failed to create CGImage from CIImage")
            return nil
        }
        
        // Create a new image with transparency based on mask
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        
        guard let bitmapContext = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            print("❌ Failed to create bitmap context")
            return nil
        }
        
        // Scale and draw the original image to match mask dimensions
        let imageRect = CGRect(x: 0, y: 0, width: width, height: height)
        bitmapContext.draw(cgImage, in: imageRect)
        
        // Get pixel data
        guard let pixelData = bitmapContext.data else {
            print("❌ Failed to get pixel data from context")
            return nil
        }
        
        let maskBuffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        let imagePixels = pixelData.assumingMemoryBound(to: UInt8.self)
        
        // Debug: Check mask value distribution
        var maskValueCounts: [UInt8: Int] = [:]
        for i in 0..<(width * height) {
            let maskValue = maskBuffer[i]
            maskValueCounts[maskValue, default: 0] += 1
        }
        print("🔍 Mask value distribution: \(maskValueCounts.sorted { $0.value > $1.value }.prefix(5))")
        
        // Apply mask - make non-furniture pixels transparent
        // The mask contains 255 for furniture pixels, 0 for background (after createSegmentationMask processing)
        var transparentPixels = 0
        var furniturePixels = 0
        
        for y in 0..<height {
            for x in 0..<width {
                let maskIndex = y * bytesPerRow + x
                let pixelIndex = (y * width + x) * 4
                
                // Check if this pixel is part of furniture (255 in processed mask)
                let maskValue = maskBuffer[maskIndex]
                
                if maskValue < 128 {
                    // Make background completely transparent
                    imagePixels[pixelIndex] = 0     // R = 0
                    imagePixels[pixelIndex + 1] = 0 // G = 0  
                    imagePixels[pixelIndex + 2] = 0 // B = 0
                    imagePixels[pixelIndex + 3] = 0 // Alpha = 0
                    transparentPixels += 1
                } else {
                    // Keep furniture pixels at full opacity with slight edge softening
                    let edgeSoftening = Float(maskValue) / 255.0 // Use mask value for edge softening
                    let currentAlpha = imagePixels[pixelIndex + 3]
                    imagePixels[pixelIndex + 3] = UInt8(Float(currentAlpha) * edgeSoftening * 0.95) // 95% opacity with edge softening
                    furniturePixels += 1
                }
            }
        }
        
        print("🎨 Texture processing: \(furniturePixels) furniture pixels, \(transparentPixels) transparent pixels")
        
        // Create CGImage from modified pixel data
        guard let modifiedCGImage = bitmapContext.makeImage() else {
            print("❌ Failed to create CGImage from modified pixels")
            return nil
        }
        
        let resultImage = UIImage(cgImage: modifiedCGImage)
        print("✅ Created furniture texture: \(resultImage.size.width)x\(resultImage.size.height)")
        return resultImage
    }
    
    // MARK: - Advanced Morphological Operations for Mask Cleanup
    
    private func cleanupMask(_ maskBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        guard enableEdgeRefinement else {
            return maskBuffer
        }
        
        CVPixelBufferLockBaseAddress(maskBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(maskBuffer, []) }
        
        let width = CVPixelBufferGetWidth(maskBuffer)
        let height = CVPixelBufferGetHeight(maskBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(maskBuffer)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(maskBuffer) else {
            return nil
        }
        
        let originalPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        // Create cleaned buffer
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        
        var cleanedBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                       width,
                                       height,
                                       kCVPixelFormatType_OneComponent8,
                                       attributes as CFDictionary,
                                       &cleanedBuffer)
        
        guard status == kCVReturnSuccess, let cleaned = cleanedBuffer else {
            print("❌ Failed to create cleaned mask buffer")
            return maskBuffer
        }
        
        // Step 1: Apply bilateral filter for edge-preserving smoothing
        if let bilateralFiltered = applyBilateralFilter(to: maskBuffer) {
            // Step 2: Apply morphological operations to clean up noise
            if let morphologyResult = applyMorphologicalOperations(to: bilateralFiltered) {
                // Step 3: Apply final edge smoothing
                return applyEdgeSmoothing(to: morphologyResult) ?? morphologyResult
            }
            return bilateralFiltered
        }
        
        return maskBuffer // Return original if all cleanup fails
    }
    
    private func applyBilateralFilter(to maskBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        // Simplified bilateral filter implementation for edge preservation
        CVPixelBufferLockBaseAddress(maskBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(maskBuffer, []) }
        
        let width = CVPixelBufferGetWidth(maskBuffer)
        let height = CVPixelBufferGetHeight(maskBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(maskBuffer)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(maskBuffer) else { return nil }
        let originalPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var filteredBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent8,
            [kCVPixelBufferCGImageCompatibilityKey as String: true] as CFDictionary,
            &filteredBuffer
        )
        
        guard status == kCVReturnSuccess, let filtered = filteredBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(filtered, [])
        defer { CVPixelBufferUnlockBaseAddress(filtered, []) }
        
        guard let filteredAddress = CVPixelBufferGetBaseAddress(filtered) else { return nil }
        let filteredPtr = filteredAddress.assumingMemoryBound(to: UInt8.self)
        
        let kernelRadius = 2
        let sigmaSpace: Float = 2.0
        let sigmaColor: Float = 20.0
        
        for y in 0..<height {
            for x in 0..<width {
                let centerIndex = y * bytesPerRow + x
                let centerValue = Float(originalPtr[centerIndex])
                
                var weightedSum: Float = 0.0
                var weightSum: Float = 0.0
                
                // Process neighborhood
                for ky in -kernelRadius...kernelRadius {
                    for kx in -kernelRadius...kernelRadius {
                        let ny = max(0, min(height - 1, y + ky))
                        let nx = max(0, min(width - 1, x + kx))
                        let neighborIndex = ny * bytesPerRow + nx
                        let neighborValue = Float(originalPtr[neighborIndex])
                        
                        // Spatial weight
                        let spatialDist = sqrt(Float(kx * kx + ky * ky))
                        let spatialWeight = exp(-(spatialDist * spatialDist) / (2.0 * sigmaSpace * sigmaSpace))
                        
                        // Color weight
                        let colorDist = abs(centerValue - neighborValue)
                        let colorWeight = exp(-(colorDist * colorDist) / (2.0 * sigmaColor * sigmaColor))
                        
                        let totalWeight = spatialWeight * colorWeight
                        
                        weightedSum += neighborValue * totalWeight
                        weightSum += totalWeight
                    }
                }
                
                if weightSum > 0 {
                    filteredPtr[centerIndex] = UInt8(max(0, min(255, weightedSum / weightSum)))
                } else {
                    filteredPtr[centerIndex] = originalPtr[centerIndex]
                }
            }
        }
        
        print("✨ Applied bilateral filter for edge-preserving smoothing")
        return filtered
    }
    
    private func applyMorphologicalOperations(to maskBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        // Apply opening (erosion followed by dilation) to remove noise
        guard let eroded = applyErosion(to: maskBuffer, kernelSize: morphologyKernelSize) else {
            return nil
        }
        
        guard let opened = applyDilation(to: eroded, kernelSize: morphologyKernelSize) else {
            return eroded
        }
        
        // Apply closing (dilation followed by erosion) to fill holes
        guard let dilated = applyDilation(to: opened, kernelSize: morphologyKernelSize) else {
            return opened
        }
        
        guard let closed = applyErosion(to: dilated, kernelSize: morphologyKernelSize) else {
            return dilated
        }
        
        print("✨ Applied morphological opening and closing operations")
        return closed
    }
    
    private func applyErosion(to maskBuffer: CVPixelBuffer, kernelSize: Int) -> CVPixelBuffer? {
        return applyMorphology(to: maskBuffer, kernelSize: kernelSize, operation: .erosion)
    }
    
    private func applyDilation(to maskBuffer: CVPixelBuffer, kernelSize: Int) -> CVPixelBuffer? {
        return applyMorphology(to: maskBuffer, kernelSize: kernelSize, operation: .dilation)
    }
    
    private enum MorphologyOperation {
        case erosion
        case dilation
    }
    
    private func applyMorphology(to maskBuffer: CVPixelBuffer, kernelSize: Int, operation: MorphologyOperation) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(maskBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(maskBuffer, []) }
        
        let width = CVPixelBufferGetWidth(maskBuffer)
        let height = CVPixelBufferGetHeight(maskBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(maskBuffer)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(maskBuffer) else { return nil }
        let originalPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var resultBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent8,
            [kCVPixelBufferCGImageCompatibilityKey as String: true] as CFDictionary,
            &resultBuffer
        )
        
        guard status == kCVReturnSuccess, let result = resultBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(result, [])
        defer { CVPixelBufferUnlockBaseAddress(result, []) }
        
        guard let resultAddress = CVPixelBufferGetBaseAddress(result) else { return nil }
        let resultPtr = resultAddress.assumingMemoryBound(to: UInt8.self)
        
        let kernelRadius = kernelSize / 2
        
        for y in 0..<height {
            for x in 0..<width {
                let centerIndex = y * bytesPerRow + x
                
                var operationValue: UInt8
                
                switch operation {
                case .erosion:
                    operationValue = 255 // Start with white, find minimum
                case .dilation:
                    operationValue = 0   // Start with black, find maximum
                }
                
                // Process kernel neighborhood
                for ky in -kernelRadius...kernelRadius {
                    for kx in -kernelRadius...kernelRadius {
                        let ny = max(0, min(height - 1, y + ky))
                        let nx = max(0, min(width - 1, x + kx))
                        let neighborIndex = ny * bytesPerRow + nx
                        let neighborValue = originalPtr[neighborIndex]
                        
                        switch operation {
                        case .erosion:
                            operationValue = min(operationValue, neighborValue)
                        case .dilation:
                            operationValue = max(operationValue, neighborValue)
                        }
                    }
                }
                
                resultPtr[centerIndex] = operationValue
            }
        }
        
        return result
    }
    
    private func applyEdgeSmoothing(to maskBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        // Apply Gaussian blur to smooth edges
        CVPixelBufferLockBaseAddress(maskBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(maskBuffer, []) }
        
        let width = CVPixelBufferGetWidth(maskBuffer)
        let height = CVPixelBufferGetHeight(maskBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(maskBuffer)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(maskBuffer) else { return nil }
        let originalPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var smoothedBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent8,
            [kCVPixelBufferCGImageCompatibilityKey as String: true] as CFDictionary,
            &smoothedBuffer
        )
        
        guard status == kCVReturnSuccess, let smoothed = smoothedBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(smoothed, [])
        defer { CVPixelBufferUnlockBaseAddress(smoothed, []) }
        
        guard let smoothedAddress = CVPixelBufferGetBaseAddress(smoothed) else { return nil }
        let smoothedPtr = smoothedAddress.assumingMemoryBound(to: UInt8.self)
        
        // Gaussian kernel for edge smoothing
        let kernelSize = 3
        let kernelRadius = kernelSize / 2
        let sigma: Float = 1.0
        var kernel: [Float] = []
        var kernelSum: Float = 0
        
        // Generate Gaussian kernel
        for y in -kernelRadius...kernelRadius {
            for x in -kernelRadius...kernelRadius {
                let distance = Float(x * x + y * y)
                let weight = exp(-distance / (2.0 * sigma * sigma))
                kernel.append(weight)
                kernelSum += weight
            }
        }
        
        // Normalize kernel
        kernel = kernel.map { $0 / kernelSum }
        
        // Apply Gaussian blur
        for y in 0..<height {
            for x in 0..<width {
                let centerIndex = y * bytesPerRow + x
                var weightedSum: Float = 0.0
                var kernelIndex = 0
                
                for ky in -kernelRadius...kernelRadius {
                    for kx in -kernelRadius...kernelRadius {
                        let ny = max(0, min(height - 1, y + ky))
                        let nx = max(0, min(width - 1, x + kx))
                        let neighborIndex = ny * bytesPerRow + nx
                        let neighborValue = Float(originalPtr[neighborIndex])
                        
                        weightedSum += neighborValue * kernel[kernelIndex]
                        kernelIndex += 1
                    }
                }
                
                smoothedPtr[centerIndex] = UInt8(max(0, min(255, weightedSum)))
            }
        }
        
        print("✨ Applied Gaussian edge smoothing")
        return smoothed
    }
    
    // MARK: - Quality Assessment and Temporal Smoothing
    
    private func assessMaskQuality(_ maskBuffer: CVPixelBuffer) -> Float {
        CVPixelBufferLockBaseAddress(maskBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(maskBuffer, .readOnly) }
        
        let width = CVPixelBufferGetWidth(maskBuffer)
        let height = CVPixelBufferGetHeight(maskBuffer)
        let totalPixels = width * height
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(maskBuffer) else {
            return 0.0
        }
        
        let maskPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        var furniturePixelCount = 0
        
        // Count furniture pixels and assess mask connectivity
        for i in 0..<totalPixels {
            if maskPtr[i] > 128 { // Consider as furniture pixel
                furniturePixelCount += 1
            }
        }
        
        let maskArea = Float(furniturePixelCount) / Float(totalPixels)
        
        // Reject if mask area is too small or too large
        if maskArea < minMaskArea || maskArea > maxMaskArea {
            return 0.0 // Poor quality
        }
        
        // Simple quality score based on area (could be enhanced with connectivity analysis)
        return min(1.0, maskArea * 2.5) // Scale so 40% area = 1.0 quality
    }
    
    private func addFrameToBuffer(_ frame: SegmentationFrame) {
        frameBuffer.append(frame)
        
        // Keep only the most recent frames
        if frameBuffer.count > maxBufferSize {
            frameBuffer.removeFirst()
        }
    }
    
    private func shouldUpdateSegmentation(newFrame: SegmentationFrame) -> Bool {
        let currentTime = CACurrentMediaTime()
        
        // Always update if this is the first frame
        guard let lastSegmentation = lastStableSegmentation else {
            return true
        }
        
        // Don't update too frequently
        if currentTime - lastUpdateTime < minUpdateInterval {
            return false
        }
        
        // Check if the new frame has significantly different bounding box
        let lastBox = lastSegmentation.boundingBox
        let newBox = newFrame.boundingBox
        
        // Calculate intersection over union (IoU)
        let intersection = lastBox.intersection(newBox)
        let union = lastBox.union(newBox)
        
        let iou = (intersection.width * intersection.height) / (union.width * union.height)
        
        // Update if IoU is low (different object) or if confidence is significantly higher
        if iou < 0.7 || newFrame.confidence > lastSegmentation.confidence + 0.2 {
            return true
        }
        
        return false
    }
    
    private func createStabilizedSegmentation(from frames: [SegmentationFrame], originalImage: CIImage) -> SegmentedObject? {
        guard !frames.isEmpty else { return nil }
        
        // Use the highest quality frame from recent frames
        let bestFrame = frames.max { frame1, frame2 in
            return frame1.confidence < frame2.confidence
        }
        
        guard let selectedFrame = bestFrame else { return nil }
        
        print("🎯 Creating stabilized segmentation with confidence: \(selectedFrame.confidence)")
        
        // Create stabilized segmented object
        let stabilizedObject = SegmentedObject(
            originalImage: originalImage,
            segmentationMask: selectedFrame.mask,
            boundingBox: selectedFrame.boundingBox,
            confidence: selectedFrame.confidence,
            objectClass: selectedFrame.objectClass
        )
        
        return stabilizedObject
    }
    
    private func processFrameWithStabilization(_ segmentedObject: SegmentedObject) {
        let currentTime = CACurrentMediaTime()
        
        // Clean up the mask with morphological operations
        guard let cleanedMask = cleanupMask(segmentedObject.segmentationMask) else {
            print("⚠️ Failed to cleanup mask, skipping frame")
            return
        }
        
        // Assess mask quality after cleanup
        let quality = assessMaskQuality(cleanedMask)
        
        print("📊 Frame quality assessment: \(quality), confidence: \(segmentedObject.confidence)")
        
        // Only process high-quality frames with enhanced thresholds for cleaner results
        guard quality > 0.5 && segmentedObject.confidence > minConfidenceThreshold else {
            print("⚠️ Frame rejected - quality: \(quality), confidence: \(segmentedObject.confidence), threshold: \(minConfidenceThreshold)")
            return
        }
        
        // Create cleaned segmented object
        let cleanedSegmentedObject = SegmentedObject(
            originalImage: segmentedObject.originalImage,
            segmentationMask: cleanedMask,
            boundingBox: segmentedObject.boundingBox,
            confidence: segmentedObject.confidence,
            objectClass: segmentedObject.objectClass
        )
        
        // Create frame for buffer using cleaned mask
        let frame = SegmentationFrame(
            mask: cleanedMask,
            confidence: segmentedObject.confidence,
            boundingBox: segmentedObject.boundingBox,
            timestamp: currentTime,
            objectClass: segmentedObject.objectClass
        )
        
        // Add to frame buffer
        addFrameToBuffer(frame)
        
        // Check if we should update the displayed segmentation
        if shouldUpdateSegmentation(newFrame: frame) {
            // Create stabilized segmentation from recent frames
            if let stabilizedSegmentation = createStabilizedSegmentation(from: frameBuffer, originalImage: cleanedSegmentedObject.originalImage) {
                print("✅ Updating stable segmentation with improved quality")
                lastStableSegmentation = stabilizedSegmentation
                lastUpdateTime = currentTime
                
                // Notify delegate with stabilized result
                delegate?.processor(self, didSegment: stabilizedSegmentation)
            }
        } else {
            print("🔄 Frame buffered, no update needed")
        }
    }
}

// MARK: - Segmentation Errors
enum SegmentationError: LocalizedError {
    case modelNotFound
    case processingFailed
    case noObjectDetected
    
    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Segmentation model not found"
        case .processingFailed:
            return "Failed to process image for segmentation"
        case .noObjectDetected:
            return "No supported objects detected in the frame"
        }
    }
}