import SwiftUI
import CoreML
import Vision
import CoreImage
import SceneKit
import Metal
import MetalPerformanceShaders

// MARK: - Room Structure (with proper initialization)
struct RoomStructure {
    var wallLines: [Line] = []
    var floorRegion: CGRect?
    var ceilingRegion: CGRect?
    var vanishingPoint: CGPoint?
    
    struct Line {
        var start: CGPoint
        var end: CGPoint
        var confidence: Float
    }
}

// MARK: - Single Photo Room Reconstructor
class SinglePhotoRoomReconstructor: ObservableObject {
    @Published var isProcessing = false
    @Published var progress: Float = 0.0
    @Published var statusMessage = "Ready"
    @Published var estimatedDimensions: RoomDimensions?
    @Published var generatedRoomURL: URL?
    
    private let depthEstimator = MiDaSDepthEstimator()
    private let roomAnalyzer = RoomStructureAnalyzer()
    private let textureProcessor = TextureProcessor()
    
    struct RoomDimensions {
        var width: Float = 4.0
        var depth: Float = 4.0
        var height: Float = 2.8
        var doorHeight: Float = 2.1
        var confidence: Float = 0.6
    }
    
    init() {
        print("🏗️ [Reconstructor] Initialized")
    }
    
    // MARK: - Helper Methods
    private func updateProgress(_ value: Float, _ message: String) async {
        print("📊 [Reconstructor] Progress: \(Int(value * 100))% - \(message)")
        await MainActor.run {
            self.progress = value
            self.statusMessage = message
        }
    }
    
    private func setError(_ message: String) async {
        print("❌ [Reconstructor] ERROR: \(message)")
        await MainActor.run {
            self.isProcessing = false
            self.statusMessage = message
        }
    }
    
    // MARK: - Main Processing Pipeline
    func processPhoto(_ image: UIImage) async {
        print("🚀 [Reconstructor] ========== STARTING PHOTO PROCESSING ==========")
        print("📸 [Reconstructor] Image size: \(image.size)")
        print("📸 [Reconstructor] Image scale: \(image.scale)")
        
        await MainActor.run {
            isProcessing = true
            progress = 0.0
            statusMessage = "Analyzing photo..."
        }
        
        // Step 1: Generate depth map
        await updateProgress(0.2, "Extracting depth information...")
        print("🔍 [Reconstructor] Step 1: Starting depth estimation")
        
        guard let depthMap = await depthEstimator.estimateDepth(from: image) else {
            await setError("Failed to estimate depth")
            return
        }
        print("✅ [Reconstructor] Step 1: Depth map created - extent: \(depthMap.extent)")
        
        // Step 2: Detect room structure
        await updateProgress(0.4, "Finding walls and corners...")
        print("🔍 [Reconstructor] Step 2: Starting room structure analysis")
        let roomStructure = await roomAnalyzer.analyzeRoom(image: image, depthMap: depthMap)
        print("✅ [Reconstructor] Step 2: Room structure analyzed")
        print("   - Wall lines found: \(roomStructure.wallLines.count)")
        print("   - Floor region: \(roomStructure.floorRegion?.debugDescription ?? "nil")")
        print("   - Ceiling region: \(roomStructure.ceilingRegion?.debugDescription ?? "nil")")
        
        // Step 3: Estimate dimensions
        await updateProgress(0.6, "Calculating room dimensions...")
        print("🔍 [Reconstructor] Step 3: Starting dimension estimation")
        let dimensions = await estimateDimensions(from: roomStructure, image: image)
        print("✅ [Reconstructor] Step 3: Dimensions estimated")
        print("   - Width: \(dimensions.width)m")
        print("   - Depth: \(dimensions.depth)m")
        print("   - Height: \(dimensions.height)m")
        print("   - Confidence: \(Int(dimensions.confidence * 100))%")
        
        await MainActor.run {
            self.estimatedDimensions = dimensions
        }
        
        // Step 4: Build 3D room
        await updateProgress(0.8, "Building 3D model...")
        print("🔍 [Reconstructor] Step 4: Starting 3D room construction")
        let roomURL = await build3DRoom(
            dimensions: dimensions,
            structure: roomStructure,
            originalImage: image,
            depthMap: depthMap
        )
        
        if let url = roomURL {
            print("✅ [Reconstructor] Step 4: 3D room built successfully")
            print("   - URL: \(url)")
            print("   - File exists: \(FileManager.default.fileExists(atPath: url.path))")
        } else {
            print("❌ [Reconstructor] Step 4: Failed to build 3D room")
        }
        
        await MainActor.run {
            self.generatedRoomURL = roomURL
            self.isProcessing = false
            self.progress = 1.0
            self.statusMessage = "Room ready!"
        }
        
        print("🎉 [Reconstructor] ========== PHOTO PROCESSING COMPLETE ==========")
    }
    
    // MARK: - Dimension Estimation
    private func estimateDimensions(from structure: RoomStructure, image: UIImage) async -> RoomDimensions {
        print("📏 [DimensionEstimator] Starting dimension estimation")
        var dimensions = RoomDimensions()
        
        // Try to find doors for scale reference
        print("🚪 [DimensionEstimator] Attempting door detection...")
        if let doorRect = await detectDoor(in: image) {
            print("✅ [DimensionEstimator] DOOR DETECTED!")
            print("   - Door rect: \(doorRect)")
            
            // Door detected - use as scale reference
            let doorPixelHeight = doorRect.height * image.size.height
            let imageHeight = image.size.height
            
            print("   - Door pixel height: \(doorPixelHeight)")
            print("   - Image height: \(imageHeight)")
            
            // Assuming standard door height of 2.1m
            let pixelsPerMeter = doorPixelHeight / 2.1
            print("   - Pixels per meter: \(pixelsPerMeter)")
            
            // Estimate room height from image proportions
            dimensions.height = Float(imageHeight / pixelsPerMeter)
            dimensions.width = Float(image.size.width / pixelsPerMeter * 1.2)
            dimensions.depth = dimensions.width
            dimensions.confidence = 0.8
            
            print("   - Calculated height: \(dimensions.height)m")
            print("   - Calculated width: \(dimensions.width)m")
            print("   - Confidence: 80%")
            
        } else if let personRect = await detectPerson(in: image) {
            print("✅ [DimensionEstimator] PERSON DETECTED (fallback)")
            print("   - Person rect: \(personRect)")
            
            // Fallback: use person height (average 1.7m)
            let personPixelHeight = personRect.height * image.size.height
            let pixelsPerMeter = personPixelHeight / 1.7
            
            print("   - Person pixel height: \(personPixelHeight)")
            print("   - Pixels per meter: \(pixelsPerMeter)")
            
            dimensions.height = Float(image.size.height / pixelsPerMeter * 0.4)
            dimensions.width = Float(image.size.width / pixelsPerMeter * 1.5)
            dimensions.depth = dimensions.width * 0.9
            dimensions.confidence = 0.5
            
            print("   - Calculated height: \(dimensions.height)m")
            print("   - Calculated width: \(dimensions.width)m")
            print("   - Confidence: 50%")
            
        } else {
            print("⚠️ [DimensionEstimator] NO REFERENCE FOUND - Using defaults")
            // No reference found - use typical room proportions
            dimensions.width = 4.0
            dimensions.depth = 4.5
            dimensions.height = 2.8
            dimensions.confidence = 0.3
            
            print("   - Default width: \(dimensions.width)m")
            print("   - Default depth: \(dimensions.depth)m")
            print("   - Default height: \(dimensions.height)m")
            print("   - Confidence: 30%")
        }
        
        // Clamp to reasonable ranges
        let originalWidth = dimensions.width
        let originalDepth = dimensions.depth
        let originalHeight = dimensions.height
        
        dimensions.width = min(max(dimensions.width, 2.0), 8.0)
        dimensions.depth = min(max(dimensions.depth, 2.0), 8.0)
        dimensions.height = min(max(dimensions.height, 2.2), 4.0)
        
        if originalWidth != dimensions.width || originalDepth != dimensions.depth || originalHeight != dimensions.height {
            print("⚠️ [DimensionEstimator] Dimensions clamped:")
            print("   - Width: \(originalWidth) -> \(dimensions.width)")
            print("   - Depth: \(originalDepth) -> \(dimensions.depth)")
            print("   - Height: \(originalHeight) -> \(dimensions.height)")
        }
        
        print("📏 [DimensionEstimator] Final dimensions: W:\(dimensions.width)m D:\(dimensions.depth)m H:\(dimensions.height)m")
        return dimensions
    }
    
    // MARK: - Object Detection
    private func detectDoor(in image: UIImage) async -> CGRect? {
        print("🚪 [DoorDetector] Starting door detection")
        guard let cgImage = image.cgImage else {
            print("❌ [DoorDetector] Failed to get CGImage")
            return nil
        }
        
        let request = VNDetectRectanglesRequest()
        request.minimumAspectRatio = 0.3
        request.maximumAspectRatio = 0.5
        request.minimumSize = 0.1
        request.maximumObservations = 5
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        do {
            try handler.perform([request])
            print("✅ [DoorDetector] Vision request performed")
        } catch {
            print("❌ [DoorDetector] Vision request failed: \(error)")
            return nil
        }
        
        // Look for door-like rectangles (tall and narrow)
        if let results = request.results {
            print("🚪 [DoorDetector] Found \(results.count) rectangles")
            
            for (index, rect) in results.enumerated() {
                let aspectRatio = rect.boundingBox.width / rect.boundingBox.height
                print("   Rectangle \(index): aspect=\(aspectRatio), height=\(rect.boundingBox.height)")
                
                if aspectRatio > 0.35 && aspectRatio < 0.5 && rect.boundingBox.height > 0.3 {
                    print("✅ [DoorDetector] Door-like rectangle found at index \(index)!")
                    return rect.boundingBox
                }
            }
            print("⚠️ [DoorDetector] No rectangles matched door criteria")
        } else {
            print("⚠️ [DoorDetector] No rectangles detected")
        }
        
        return nil
    }
    
    private func detectPerson(in image: UIImage) async -> CGRect? {
        print("👤 [PersonDetector] Starting person detection")
        guard let cgImage = image.cgImage else {
            print("❌ [PersonDetector] Failed to get CGImage")
            return nil
        }
        
        let request = VNDetectHumanRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        do {
            try handler.perform([request])
            print("✅ [PersonDetector] Vision request performed")
        } catch {
            print("❌ [PersonDetector] Vision request failed: \(error)")
            return nil
        }
        
        if let results = request.results, !results.isEmpty {
            print("✅ [PersonDetector] Found \(results.count) person(s)")
            let firstPerson = results.first!.boundingBox
            print("   - First person rect: \(firstPerson)")
            return firstPerson
        } else {
            print("⚠️ [PersonDetector] No persons detected")
        }
        
        return nil
    }
    
    // MARK: - 3D Room Building
    private func build3DRoom(dimensions: RoomDimensions, structure: RoomStructure, originalImage: UIImage, depthMap: CIImage) async -> URL? {
        print("🏗️ [RoomBuilder] Starting 3D room construction")
        print("   - Dimensions: W:\(dimensions.width) D:\(dimensions.depth) H:\(dimensions.height)")
        
        // Generate textures
        print("🎨 [RoomBuilder] Generating textures...")
        let floorTexture = generateFloorTexture(from: originalImage, structure: structure)
        let wallTexture = generateWallTexture(from: originalImage)
        
        let scene = SCNScene()
        print("✅ [RoomBuilder] SCNScene created")
        
        // Create room box - USE BOX GEOMETRY FOR PROPER INSIDE VIEW
        let roomBox = SCNBox(width: CGFloat(dimensions.width),
                            height: CGFloat(dimensions.height),
                            length: CGFloat(dimensions.depth),
                            chamferRadius: 0)
        
        // Apply textures to each face
        print("🎨 [RoomBuilder] Applying textures to room box...")
        
        // Front face (index 0) - your original photo
        let frontMaterial = SCNMaterial()
        frontMaterial.diffuse.contents = originalImage
        frontMaterial.isDoubleSided = true
        frontMaterial.lightingModel = .constant
        print("   - Front: Original photo")
        
        // Right face (index 1)
        let rightMaterial = SCNMaterial()
        rightMaterial.diffuse.contents = wallTexture
        rightMaterial.isDoubleSided = true
        rightMaterial.lightingModel = .constant
        print("   - Right: Wall texture")
        
        // Back face (index 2)
        let backMaterial = SCNMaterial()
        backMaterial.diffuse.contents = wallTexture
        backMaterial.isDoubleSided = true
        backMaterial.lightingModel = .constant
        print("   - Back: Wall texture")
        
        // Left face (index 3)
        let leftMaterial = SCNMaterial()
        leftMaterial.diffuse.contents = wallTexture
        leftMaterial.isDoubleSided = true
        leftMaterial.lightingModel = .constant
        print("   - Left: Wall texture")
        
        // Top face (index 4) - ceiling
        let topMaterial = SCNMaterial()
        topMaterial.diffuse.contents = UIColor(white: 0.95, alpha: 1.0)
        topMaterial.isDoubleSided = true
        topMaterial.lightingModel = .constant
        print("   - Top: Ceiling")
        
        // Bottom face (index 5) - floor
        let bottomMaterial = SCNMaterial()
        bottomMaterial.diffuse.contents = floorTexture
        bottomMaterial.isDoubleSided = true
        bottomMaterial.lightingModel = .constant
        print("   - Bottom: Floor texture")
        
        // Assign all materials to box
        roomBox.materials = [frontMaterial, rightMaterial, backMaterial, leftMaterial, topMaterial, bottomMaterial]
        
        let roomNode = SCNNode(geometry: roomBox)
        roomNode.position = SCNVector3(0, Float(dimensions.height) / 2, 0)
        print("✅ [RoomBuilder] Room box created and positioned")
        
        // Add camera INSIDE the room, looking straight ahead at the front wall
        print("📷 [RoomBuilder] Setting up camera...")
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.fieldOfView = 70
        cameraNode.camera?.zNear = 0.1
        cameraNode.camera?.zFar = 100
        
        // Position camera at eye level, centered, looking at front face
        let camX: Float = 0
        let camY = dimensions.height * 0.5  // Eye level
        let camZ: Float = 0  // Center of room
        cameraNode.position = SCNVector3(camX, camY, camZ)
        
        // Look straight ahead (towards negative Z, which is the front face)
        cameraNode.eulerAngles = SCNVector3(0, 0, 0)
        
        scene.rootNode.addChildNode(cameraNode)
        
        print("✅ [RoomBuilder] Camera positioned INSIDE room at: (\(camX), \(camY), \(camZ))")
        print("   - Looking straight ahead at front wall")
        print("   - Field of view: 70°")
        
        // Add lighting
        print("💡 [RoomBuilder] Setting up lighting...")
        
        // Bright ambient light since we're using constant materials
        let ambientLight = SCNNode()
        ambientLight.light = SCNLight()
        ambientLight.light?.type = .ambient
        ambientLight.light?.intensity = 1000
        ambientLight.light?.color = UIColor.white
        scene.rootNode.addChildNode(ambientLight)
        print("✅ [RoomBuilder] Ambient light added (intensity: 1000)")
        
        scene.rootNode.addChildNode(roomNode)
        print("✅ [RoomBuilder] Room node added to scene root")
        
        // Export to USDZ
        print("💾 [RoomBuilder] Exporting to USDZ...")
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("room_\(UUID().uuidString).usdz")
        print("   - Target URL: \(tempURL)")
        
        do {
            try scene.write(to: tempURL, options: nil, delegate: nil, progressHandler: nil)
            print("✅ [RoomBuilder] Successfully exported to USDZ")
            
            let fileSize = try FileManager.default.attributesOfItem(atPath: tempURL.path)[.size] as? Int ?? 0
            print("   - File size: \(fileSize) bytes")
            
            return tempURL
        } catch {
            print("❌ [RoomBuilder] Error exporting room: \(error)")
            return nil
        }
    }
    
    // MARK: - Texture Generation
    private func generateFloorTexture(from image: UIImage, structure: RoomStructure) -> UIImage {
        print("🎨 [TextureGen] Generating floor texture")
        
        guard let cgImage = image.cgImage else {
            print("⚠️ [TextureGen] No CGImage, using solid color")
            return createSolidColorTexture(color: UIColor(white: 0.85, alpha: 1.0))
        }
        
        let imageHeight = CGFloat(cgImage.height)
        let imageWidth = CGFloat(cgImage.width)
        
        // Use bottom 25% of image for floor
        let cropRect = CGRect(
            x: 0,
            y: imageHeight * 0.75,
            width: imageWidth,
            height: imageHeight * 0.25
        )
        
        print("   - Cropping bottom 25% for floor")
        print("   - Crop rect: \(cropRect)")
        
        if let croppedImage = cgImage.cropping(to: cropRect) {
            print("✅ [TextureGen] Floor texture extracted")
            return UIImage(cgImage: croppedImage)
        }
        
        print("⚠️ [TextureGen] Failed to crop, using solid color")
        return createSolidColorTexture(color: UIColor(white: 0.85, alpha: 1.0))
    }
    
    private func generateWallTexture(from image: UIImage) -> UIImage {
        print("🎨 [TextureGen] Generating wall texture")
        
        guard let cgImage = image.cgImage else {
            print("⚠️ [TextureGen] No CGImage, using solid color")
            return createSolidColorTexture(color: UIColor(white: 0.9, alpha: 1.0))
        }
        
        let ciImage = CIImage(cgImage: cgImage)
        let extent = ciImage.extent
        let centerRect = CGRect(
            x: extent.midX - 50,
            y: extent.midY - 50,
            width: 100,
            height: 100
        )
        
        print("   - Sampling center region for average color")
        print("   - Sample rect: \(centerRect)")
        
        let averageColor = ciImage.averageColor(in: centerRect) ?? UIColor(white: 0.9, alpha: 1.0)
        print("✅ [TextureGen] Wall color sampled")
        return createSolidColorTexture(color: averageColor)
    }
    
    private func createSolidColorTexture(color: UIColor) -> UIImage {
        let size = CGSize(width: 512, height: 512)
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { context in
            color.setFill()
            context.fill(CGRect(origin: .zero, size: size))
        }
    }
}

// MARK: - CIImage Extension for Color Averaging
extension CIImage {
    func averageColor(in rect: CGRect) -> UIColor? {
        let extentVector = CIVector(x: rect.origin.x, y: rect.origin.y, z: rect.size.width, w: rect.size.height)
        
        guard let filter = CIFilter(name: "CIAreaAverage", parameters: [kCIInputImageKey: self, kCIInputExtentKey: extentVector]) else {
            return nil
        }
        
        guard let outputImage = filter.outputImage else { return nil }
        
        var bitmap = [UInt8](repeating: 0, count: 4)
        let context = CIContext(options: [.workingColorSpace: kCFNull as Any])
        context.render(outputImage, toBitmap: &bitmap, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 1), format: .RGBA8, colorSpace: nil)
        
        return UIColor(red: CGFloat(bitmap[0]) / 255, green: CGFloat(bitmap[1]) / 255, blue: CGFloat(bitmap[2]) / 255, alpha: CGFloat(bitmap[3]) / 255)
    }
}

// MARK: - MiDaS Depth Estimator
class MiDaSDepthEstimator {
    private var model: VNCoreMLModel?
    
    init() {
        print("🧠 [DepthEstimator] Initializing")
        loadModel()
    }
    
    private func loadModel() {
        print("📦 [DepthEstimator] Attempting to load MiDaS model")
        // Placeholder - in real implementation, load MiDaS CoreML model
        print("⚠️ [DepthEstimator] MiDaS model not available, will use fallback")
    }
    
    func estimateDepth(from image: UIImage) async -> CIImage? {
        print("🔬 [DepthEstimator] Estimating depth from image")
        guard let cgImage = image.cgImage else {
            print("❌ [DepthEstimator] Failed to get CGImage")
            return nil
        }
        
        let ciImage = CIImage(cgImage: cgImage)
        
        // If MiDaS model is available, use it
        if let model = model {
            print("   - Using MiDaS model")
            // This won't run since model is nil
            return nil
        }
        
        // Fallback: create synthetic depth map
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

// MARK: - Room Structure Analyzer
class RoomStructureAnalyzer {
    func analyzeRoom(image: UIImage, depthMap: CIImage) async -> RoomStructure {
        print("🔍 [RoomAnalyzer] Analyzing room structure")
        var structure = RoomStructure()
        
        // Detect edges/lines in image
        print("   - Detecting wall lines...")
        structure.wallLines = await detectWallLines(in: image)
        
        // Find floor and ceiling regions
        print("   - Detecting floor region...")
        structure.floorRegion = detectFloorRegion(in: image, depthMap: depthMap)
        print("   - Detecting ceiling region...")
        structure.ceilingRegion = detectCeilingRegion(in: image, depthMap: depthMap)
        
        // Calculate vanishing point
        print("   - Calculating vanishing point...")
        structure.vanishingPoint = calculateVanishingPoint(from: structure.wallLines)
        
        print("✅ [RoomAnalyzer] Analysis complete")
        return structure
    }
    
    private func detectWallLines(in image: UIImage) async -> [RoomStructure.Line] {
        print("📐 [WallLineDetector] Detecting wall lines")
        guard let cgImage = image.cgImage else {
            print("❌ [WallLineDetector] Failed to get CGImage")
            return []
        }
        
        // Use Vision to detect edges
        let request = VNDetectContoursRequest()
        request.contrastAdjustment = 1.0
        request.detectsDarkOnLight = true
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        do {
            try handler.perform([request])
            print("✅ [WallLineDetector] Vision request performed")
        } catch {
            print("❌ [WallLineDetector] Vision request failed: \(error)")
            return []
        }
        
        var lines: [RoomStructure.Line] = []
        
        if let observations = request.results {
            print("   - Found \(observations.count) contours")
            
            for (index, contour) in observations.enumerated() {
                let path = contour.normalizedPath
                let bounds = path.boundingBox
                
                // Top edge
                lines.append(RoomStructure.Line(
                    start: CGPoint(x: bounds.minX, y: bounds.maxY),
                    end: CGPoint(x: bounds.maxX, y: bounds.maxY),
                    confidence: Float(contour.confidence)
                ))
                
                // Bottom edge
                lines.append(RoomStructure.Line(
                    start: CGPoint(x: bounds.minX, y: bounds.minY),
                    end: CGPoint(x: bounds.maxX, y: bounds.minY),
                    confidence: Float(contour.confidence)
                ))
            }
            
            print("✅ [WallLineDetector] Extracted \(lines.count) lines from contours")
        } else {
            print("⚠️ [WallLineDetector] No contours found")
        }
        
        return lines
    }
    
    private func detectFloorRegion(in image: UIImage, depthMap: CIImage) -> CGRect? {
        // Simple heuristic: floor is bottom 20% of image
        let region = CGRect(x: 0, y: 0.8, width: 1.0, height: 0.2)
        print("   - Floor region: \(region)")
        return region
    }
    
    private func detectCeilingRegion(in image: UIImage, depthMap: CIImage) -> CGRect? {
        // Simple heuristic: ceiling is top 15% of image
        let region = CGRect(x: 0, y: 0, width: 1.0, height: 0.15)
        print("   - Ceiling region: \(region)")
        return region
    }
    
    private func calculateVanishingPoint(from lines: [RoomStructure.Line]) -> CGPoint? {
        // Simplified: return center point
        let point = CGPoint(x: 0.5, y: 0.4)
        print("   - Vanishing point: \(point)")
        return point
    }
}

// MARK: - Texture Processor
class TextureProcessor {
    func inpaintMissingRegions(_ image: UIImage, mask: UIImage) -> UIImage {
        print("🎨 [TextureProcessor] Inpainting missing regions")
        
        guard let cgImage = image.cgImage else {
            print("⚠️ [TextureProcessor] No CGImage available")
            return image
        }
        
        let ciImage = CIImage(cgImage: cgImage)
        let blurred = ciImage.applyingGaussianBlur(sigma: 10.0)
        
        let filter = CIFilter(name: "CIBlendWithMask")!
        filter.setValue(blurred, forKey: "inputImage")
        filter.setValue(ciImage, forKey: "inputBackgroundImage")
        filter.setValue(CIImage(cgImage: mask.cgImage!), forKey: "inputMaskImage")
        
        guard let output = filter.outputImage,
              let cgOutput = CIContext().createCGImage(output, from: output.extent) else {
            print("⚠️ [TextureProcessor] Inpainting failed")
            return image
        }
        
        print("✅ [TextureProcessor] Inpainting complete")
        return UIImage(cgImage: cgOutput)
    }
}

// MARK: - SwiftUI View
struct SinglePhotoRoomView: View {
    @StateObject private var reconstructor = SinglePhotoRoomReconstructor()
    @State private var selectedImage: UIImage?
    @State private var showImagePicker = false
    @State private var adjustedWidth: Float = 4.0
    @State private var adjustedDepth: Float = 4.0
    @State private var adjustedHeight: Float = 2.8
    
    var body: some View {
        VStack {
            if let image = selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 300)
                    .cornerRadius(12)
                    .padding()
                    .onAppear {
                        print("🖼️ [View] Displaying selected image")
                    }
            } else {
                Button(action: {
                    print("🖼️ [View] Select photo button tapped")
                    showImagePicker = true
                }) {
                    VStack {
                        Image(systemName: "photo.on.rectangle")
                            .font(.system(size: 60))
                        Text("Select Room Photo")
                            .font(.headline)
                            .padding(.top, 8)
                    }
                    .frame(width: 250, height: 200)
                    .foregroundColor(.blue)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.blue.opacity(0.5), lineWidth: 2)
                    )
                }
                .padding()
            }
            
            if reconstructor.isProcessing {
                VStack {
                    ProgressView(value: reconstructor.progress)
                        .progressViewStyle(LinearProgressViewStyle())
                        .padding(.horizontal)
                    
                    Text(reconstructor.statusMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .onAppear {
                    print("⏳ [View] Processing view appeared")
                }
            }
            
            if let dimensions = reconstructor.estimatedDimensions {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Estimated Dimensions")
                        .font(.headline)
                    
                    HStack {
                        Image(systemName: "arrow.left.and.right")
                        Text("Width: \(String(format: "%.1f", adjustedWidth))m")
                        Slider(value: $adjustedWidth, in: 2...8, step: 0.1)
                            .onChange(of: adjustedWidth) { oldValue, newValue in
                                print("📏 [View] Width adjusted: \(oldValue) -> \(newValue)")
                            }
                    }
                    
                    HStack {
                        Image(systemName: "arrow.up.and.down")
                        Text("Depth: \(String(format: "%.1f", adjustedDepth))m")
                        Slider(value: $adjustedDepth, in: 2...8, step: 0.1)
                            .onChange(of: adjustedDepth) { oldValue, newValue in
                                print("📏 [View] Depth adjusted: \(oldValue) -> \(newValue)")
                            }
                    }
                    
                    HStack {
                        Image(systemName: "arrow.up.to.line")
                        Text("Height: \(String(format: "%.1f", adjustedHeight))m")
                        Slider(value: $adjustedHeight, in: 2.2...4, step: 0.1)
                            .onChange(of: adjustedHeight) { oldValue, newValue in
                                print("📏 [View] Height adjusted: \(oldValue) -> \(newValue)")
                            }
                    }
                    
                    Text("Confidence: \(Int(dimensions.confidence * 100))%")
                        .font(.caption)
                        .foregroundColor(confidenceColor(dimensions.confidence))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(confidenceColor(dimensions.confidence).opacity(0.1))
                        .cornerRadius(6)
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(12)
                .padding()
                .onAppear {
                    print("📊 [View] Dimensions view appeared")
                }
                
                Button("Rebuild with Adjusted Dimensions") {
                    print("🔄 [View] Rebuild button tapped")
                    rebuildRoom()
                }
                .buttonStyle(.borderedProminent)
            }
            
            if let roomURL = reconstructor.generatedRoomURL {
                NavigationLink(destination: SceneKitViewer(url: roomURL)) {
                    Text("View 3D Room")
                }
                .buttonStyle(.borderedProminent)
                .padding()
                .onAppear {
                    print("🎯 [View] View 3D Room button appeared")
                }
            }
            
            Spacer()
        }
        .navigationTitle("Photo to 3D Room")
        .sheet(isPresented: $showImagePicker) {
            PhotoPickerView(selectedImage: $selectedImage)
                .onDisappear {
                    print("📱 [View] Image picker dismissed")
                    if let image = selectedImage {
                        print("✅ [View] Image selected, starting processing...")
                        Task {
                            await reconstructor.processPhoto(image)
                            if let dims = reconstructor.estimatedDimensions {
                                adjustedWidth = dims.width
                                adjustedDepth = dims.depth
                                adjustedHeight = dims.height
                                print("📏 [View] Sliders updated with estimated dimensions")
                            }
                        }
                    } else {
                        print("⚠️ [View] No image selected")
                    }
                }
        }
        .onAppear {
            print("👁️ [View] SinglePhotoRoomView appeared")
            adjustedWidth = reconstructor.estimatedDimensions?.width ?? 4.0
            adjustedDepth = reconstructor.estimatedDimensions?.depth ?? 4.0
            adjustedHeight = reconstructor.estimatedDimensions?.height ?? 2.8
        }
    }
    
    private func rebuildRoom() {
        print("🔄 [View] Rebuilding room with adjusted dimensions")
        var updatedDimensions = reconstructor.estimatedDimensions ?? SinglePhotoRoomReconstructor.RoomDimensions()
        updatedDimensions.width = adjustedWidth
        updatedDimensions.depth = adjustedDepth
        updatedDimensions.height = adjustedHeight
        
        print("   - New dimensions: W:\(adjustedWidth) D:\(adjustedDepth) H:\(adjustedHeight)")
        
        Task {
            await MainActor.run {
                reconstructor.estimatedDimensions = updatedDimensions
            }
            
            if let image = selectedImage {
                await reconstructor.processPhoto(image)
            }
        }
    }
    
    private func confidenceColor(_ confidence: Float) -> Color {
        switch confidence {
        case 0.7...1.0:
            return .green
        case 0.4..<0.7:
            return .orange
        default:
            return .red
        }
    }
}

// MARK: - Photo Picker View
struct PhotoPickerView: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        print("📱 [PhotoPicker] Creating UIImagePickerController")
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: PhotoPickerView
        
        init(_ parent: PhotoPickerView) {
            self.parent = parent
            print("📱 [PhotoPicker] Coordinator initialized")
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            print("📱 [PhotoPicker] Image picked from library")
            if let image = info[.originalImage] as? UIImage {
                print("✅ [PhotoPicker] Got UIImage: \(image.size)")
                parent.selectedImage = image
            } else {
                print("❌ [PhotoPicker] Failed to get UIImage")
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            print("❌ [PhotoPicker] User cancelled")
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

// MARK: - SceneKit Viewer
struct SceneKitViewer: View {
    let url: URL
    @State private var showControls = true
    
    var body: some View {
        ZStack {
            SceneView(
                scene: try? SCNScene(url: url),
                options: [.allowsCameraControl, .autoenablesDefaultLighting]
            )
            .onAppear {
                print("🎬 [Viewer] SceneKit viewer appeared")
                print("   - URL: \(url)")
                print("   - File exists: \(FileManager.default.fileExists(atPath: url.path))")
                
                if let scene = try? SCNScene(url: url) {
                    print("✅ [Viewer] Scene loaded successfully")
                    print("   - Root node children: \(scene.rootNode.childNodes.count)")
                } else {
                    print("❌ [Viewer] Failed to load scene")
                }
            }
            
            // Control hints overlay
            if showControls {
                VStack {
                    Spacer()
                    HStack {
                        Image(systemName: "hand.draw")
                        Text("Drag to rotate • Pinch to zoom • Two fingers to pan")
                            .font(.caption)
                    }
                    .padding(8)
                    .background(.ultraThinMaterial)
                    .cornerRadius(8)
                    .padding()
                }
                .onAppear {
                    print("ℹ️ [Viewer] Controls hint displayed")
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                        withAnimation {
                            showControls = false
                        }
                    }
                }
            }
        }
        .navigationTitle("3D Room View")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showControls.toggle()
                    print("ℹ️ [Viewer] Controls hint toggled: \(showControls)")
                } label: {
                    Image(systemName: "questionmark.circle")
                }
            }
        }
    }
}
