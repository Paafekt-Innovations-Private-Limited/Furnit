import SwiftUI
import CoreML
import Vision
import CoreImage
import SceneKit
import Metal
import MetalPerformanceShaders
import ModelIO
import SceneKit.ModelIO

// MARK: - Room Structure (with proper initialization)
struct RoomStructure: Equatable {
    var wallLines: [Line] = []
    var floorRegion: CGRect?
    var ceilingRegion: CGRect?
    var vanishingPoint: CGPoint?
    
    // ✅ Boundary values from manual adjustment
    var floorY: CGFloat = 0.85
    var ceilingY: CGFloat = 0.15
    var leftX: CGFloat = 0.12
    var rightX: CGFloat = 0.88
    var vanishingX: CGFloat = 0.5
    var vanishingY: CGFloat = 0.45
    
    struct Line: Equatable {
        var start: CGPoint
        var end: CGPoint
        var confidence: Float
    }
}

// ✅ UIImage Extension for Orientation Fix
extension UIImage {
    func fixedOrientation() -> UIImage {
        guard imageOrientation != .up else { return self }
        
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        draw(in: CGRect(origin: .zero, size: size))
        let normalizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return normalizedImage ?? self
    }
}

// MARK: - Single Photo Room Reconstructor
class SinglePhotoRoomReconstructor: ObservableObject {
    @Published var isProcessing = false
    @Published var progress: Float = 0.0
    @Published var statusMessage = "Ready"
    @Published var estimatedDimensions: RoomDimensions?
    @Published var generatedRoomScene: SCNScene? // ✅ CHANGED: from URL to SCNScene
    
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
        
        // ✅ Fix image orientation FIRST
        let fixedImage = image.fixedOrientation()
        print("✅ [Reconstructor] Image orientation fixed")
        
        // Step 1: Generate depth map
        await updateProgress(0.2, "Extracting depth information...")
        print("🔍 [Reconstructor] Step 1: Starting depth estimation")
        
        guard let depthMap = await depthEstimator.estimateDepth(from: fixedImage) else{
            await setError("Failed to estimate depth")
            return
        }
        print("✅ [Reconstructor] Step 1: Depth map created - extent: \(depthMap.extent)")
        
        // Step 2: Detect room structure
        await updateProgress(0.4, "Finding walls and corners...")
        print("🔍 [Reconstructor] Step 2: Starting room structure analysis")
        let roomStructure = await roomAnalyzer.analyzeRoom(image: fixedImage, depthMap: depthMap)
        print("✅ [Reconstructor] Step 2: Room structure analyzed")
        print("   - Wall lines found: \(roomStructure.wallLines.count)")
        print("   - Floor region: \(roomStructure.floorRegion?.debugDescription ?? "nil")")
        print("   - Ceiling region: \(roomStructure.ceilingRegion?.debugDescription ?? "nil")")
        
        // Step 3: Estimate dimensions
        await updateProgress(0.6, "Calculating room dimensions...")
        print("🔍 [Reconstructor] Step 3: Starting dimension estimation")
        let dimensions = await estimateDimensions(from: roomStructure, image: fixedImage)
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
        let roomScene = await build3DRoom(
            dimensions: dimensions,
            structure: roomStructure,
            originalImage: fixedImage,
            depthMap: depthMap
        )
        
        if let scene = roomScene {
            print("✅ [Reconstructor] Step 4: 3D room built successfully")
            print("   - Scene nodes: \(scene.rootNode.childNodes.count)")
        } else {
            print("❌ [Reconstructor] Step 4: Failed to build 3D room")
        }
        
        await MainActor.run {
            self.generatedRoomScene = roomScene // ✅ CHANGED
            self.isProcessing = false
            self.progress = 1.0
            self.statusMessage = "Room ready!"
        }
        
        print("🎉 [Reconstructor] ========== PHOTO PROCESSING COMPLETE ==========")
    }

    
    // ✅ NEW: Process with Adjusted Boundaries
    func processPhotoWithBoundaries(_ image: UIImage, boundaries: RoomStructure) async {
        print("🚀 [Reconstructor] ========== PROCESSING WITH ADJUSTED BOUNDARIES ==========")
        print("   Floor: \(boundaries.floorY), Ceiling: \(boundaries.ceilingY)")
        print("   Left: \(boundaries.leftX), Right: \(boundaries.rightX)")
        print("   VP: (\(boundaries.vanishingX), \(boundaries.vanishingY))")
        
        await MainActor.run {
            isProcessing = true
            progress = 0.0
            statusMessage = "Rebuilding with adjusted boundaries..."
        }
        
        // Fix image orientation FIRST
        let fixedImage = image.fixedOrientation()
        print("✅ [Reconstructor] Image orientation fixed")
        
        await updateProgress(0.2, "Extracting depth information...")
        guard let depthMap = await depthEstimator.estimateDepth(from: fixedImage) else {
            await setError("Failed to estimate depth")
            return
        }
        
        await updateProgress(0.4, "Using your adjusted boundaries...")
        
        // Use the adjusted boundaries instead of detecting new ones
        var roomStructure = boundaries
        roomStructure.floorRegion = CGRect(x: 0, y: boundaries.floorY, width: 1.0, height: 1.0 - boundaries.floorY)
        roomStructure.ceilingRegion = CGRect(x: 0, y: 0, width: 1.0, height: boundaries.ceilingY)
        roomStructure.vanishingPoint = CGPoint(x: boundaries.vanishingX, y: boundaries.vanishingY)
        
        print("✅ [Reconstructor] Using adjusted boundaries:")
        print("   - Floor region: \(roomStructure.floorRegion!)")
        print("   - Ceiling region: \(roomStructure.ceilingRegion!)")
        
        await updateProgress(0.6, "Calculating dimensions with boundaries...")
        let dimensions = await estimateDimensions(from: roomStructure, image: fixedImage)
        
        await MainActor.run {
            self.estimatedDimensions = dimensions
        }
        
        await updateProgress(0.8, "Building 3D model...")
        let roomScene = await build3DRoom(
            dimensions: dimensions,
            structure: roomStructure,
            originalImage: fixedImage,
            depthMap: depthMap
        )
        
        await MainActor.run {
            self.generatedRoomScene = roomScene // ✅ CHANGED
            self.isProcessing = false
            self.progress = 1.0
            self.statusMessage = "Room ready with your boundaries!"
        }
        
        print("🎉 [Reconstructor] ========== PROCESSING COMPLETE WITH BOUNDARIES ==========")
    }
    // MARK: - Dimension Estimation
    private func estimateDimensions(from structure: RoomStructure, image: UIImage) async -> RoomDimensions {
        print("📏 [DimensionEstimator] Starting dimension estimation")
        var dimensions = RoomDimensions()
        
        print("🚪 [DimensionEstimator] Attempting door detection...")
        if let doorRect = await detectDoor(in: image) {
            print("✅ [DimensionEstimator] DOOR DETECTED!")
            print("   - Door rect: \(doorRect)")
            
            let doorPixelHeight = doorRect.height * image.size.height
            let imageHeight = image.size.height
            
            print("   - Door pixel height: \(doorPixelHeight)")
            print("   - Image height: \(imageHeight)")
            
            let pixelsPerMeter = doorPixelHeight / 2.1
            print("   - Pixels per meter: \(pixelsPerMeter)")
            
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
            dimensions.width = 4.0
            dimensions.depth = 4.5
            dimensions.height = 2.8
            dimensions.confidence = 0.3
            
            print("   - Default width: \(dimensions.width)m")
            print("   - Default depth: \(dimensions.depth)m")
            print("   - Default height: \(dimensions.height)m")
            print("   - Confidence: 30%")
        }
        
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
        
        // results is already [VNRectangleObservation]?
        guard let rects = request.results, !rects.isEmpty else {
            print("⚠️ [DoorDetector] No rectangles detected")
            return nil
        }
        
        print("🚪 [DoorDetector] Found \(rects.count) rectangles")
        for (index, rect) in rects.enumerated() {
            let aspectRatio = rect.boundingBox.width / rect.boundingBox.height
            print("   Rectangle \(index): aspect=\(aspectRatio), height=\(rect.boundingBox.height)")
            if aspectRatio > 0.35 && aspectRatio < 0.5 && rect.boundingBox.height > 0.3 {
                print("✅ [DoorDetector] Door-like rectangle found at index \(index)!")
                return rect.boundingBox
            }
        }
        print("⚠️ [DoorDetector] No rectangles matched door criteria")
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
        
        // results is already [VNHumanObservation]?
        guard let persons = request.results, !persons.isEmpty else {
            print("⚠️ [PersonDetector] No persons detected")
            return nil
        }
        
        print("✅ [PersonDetector] Found \(persons.count) person(s)")
        return persons.first!.boundingBox
    }
    
    // MARK: - 3D Room Building - WITH PHOTO TEXTURES
    private func build3DRoom(dimensions: RoomDimensions, structure: RoomStructure, originalImage: UIImage, depthMap: CIImage) async -> SCNScene? {
        print("🏗️ [RoomBuilder] Starting TEXTURED room construction")
        print("   - Dimensions: W:\(dimensions.width) D:\(dimensions.depth) H:\(dimensions.height)")
        
        let scene = SCNScene()
        let roomNode = SCNNode()
        
        // Generate textures from photo
        print("🎨 [RoomBuilder] Generating textures from photo...")
        let floorTexture = generateFloorTexture(from: originalImage, structure: structure)
        let wallTexture = originalImage // Use full photo for front wall
        let leftWallTexture = generateLeftWallTexture(from: originalImage, structure: structure)
        let rightWallTexture = generateRightWallTexture(from: originalImage, structure: structure)
        let backWallColor = generateWallTexture(from: originalImage) // Just color for back
        
        // FLOOR - With texture from photo
        print("🔨 Creating FLOOR with texture...")
        let floor = SCNBox(width: CGFloat(dimensions.width),
                           height: 0.01,
                           length: CGFloat(dimensions.depth),
                           chamferRadius: 0)
        let floorMaterial = SCNMaterial()
        floorMaterial.diffuse.contents = floorTexture
        floorMaterial.isDoubleSided = true
        floor.materials = [floorMaterial]
        
        let floorNode = SCNNode(geometry: floor)
        floorNode.position = SCNVector3(0, 0, 0)
        roomNode.addChildNode(floorNode)
        print("✅ FLOOR at y=0")
        
        // CEILING
        print("🔨 Creating CEILING...")
        let ceiling = SCNBox(width: CGFloat(dimensions.width),
                             height: 0.01,
                             length: CGFloat(dimensions.depth),
                             chamferRadius: 0)
        let ceilingMaterial = SCNMaterial()
        ceilingMaterial.diffuse.contents = UIColor.white
        ceilingMaterial.isDoubleSided = true
        ceiling.materials = [ceilingMaterial]
        
        let ceilingNode = SCNNode(geometry: ceiling)
        ceilingNode.position = SCNVector3(0, Float(dimensions.height), 0)
        roomNode.addChildNode(ceilingNode)
        print("✅ CEILING at y=\(dimensions.height)")
        
        // FRONT WALL - With YOUR PHOTO texture
        print("🔨 Creating FRONT WALL with photo texture...")
        let frontWall = SCNBox(width: CGFloat(dimensions.width),
                               height: CGFloat(dimensions.height),
                               length: 0.01,
                               chamferRadius: 0)
        let frontMaterial = SCNMaterial()
        frontMaterial.diffuse.contents = wallTexture
        frontMaterial.isDoubleSided = true
        frontWall.materials = [frontMaterial]
        
        let frontNode = SCNNode(geometry: frontWall)
        frontNode.position = SCNVector3(0, Float(dimensions.height) / 2, -Float(dimensions.depth) / 2)
        roomNode.addChildNode(frontNode)
        print("✅ FRONT WALL with photo at z=-\(dimensions.depth/2)")
        
        // BACK WALL
        print("🔨 Creating BACK WALL...")
        let backWall = SCNBox(width: CGFloat(dimensions.width),
                              height: CGFloat(dimensions.height),
                              length: 0.01,
                              chamferRadius: 0)
        let backMaterial = SCNMaterial()
        backMaterial.diffuse.contents = backWallColor
        backMaterial.isDoubleSided = true
        backWall.materials = [backMaterial]
        
        let backNode = SCNNode(geometry: backWall)
        backNode.position = SCNVector3(0, Float(dimensions.height) / 2, Float(dimensions.depth) / 2)
        roomNode.addChildNode(backNode)
        print("✅ BACK WALL at z=+\(dimensions.depth/2)")
        
        // LEFT WALL
        print("🔨 Creating LEFT WALL...")
        let leftWall = SCNBox(width: 0.01,
                              height: CGFloat(dimensions.height),
                              length: CGFloat(dimensions.depth),
                              chamferRadius: 0)
        let leftMaterial = SCNMaterial()
        leftMaterial.diffuse.contents = leftWallTexture
        leftMaterial.isDoubleSided = true
        leftWall.materials = [leftMaterial]
        
        let leftNode = SCNNode(geometry: leftWall)
        leftNode.position = SCNVector3(-Float(dimensions.width) / 2, Float(dimensions.height) / 2, 0)
        roomNode.addChildNode(leftNode)
        print("✅ LEFT WALL at x=-\(dimensions.width/2)")
        
        // RIGHT WALL
        print("🔨 Creating RIGHT WALL...")
        let rightWall = SCNBox(width: 0.01,
                               height: CGFloat(dimensions.height),
                               length: CGFloat(dimensions.depth),
                               chamferRadius: 0)
        let rightMaterial = SCNMaterial()
        rightMaterial.diffuse.contents = rightWallTexture
        rightMaterial.isDoubleSided = true
        rightWall.materials = [rightMaterial]
        
        let rightNode = SCNNode(geometry: rightWall)
        rightNode.position = SCNVector3(Float(dimensions.width) / 2, Float(dimensions.height) / 2, 0)
        roomNode.addChildNode(rightNode)
        print("✅ RIGHT WALL at x=+\(dimensions.width/2)")
        
        scene.rootNode.addChildNode(roomNode)
        
        // CAMERA - Looking at photo wall
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.fieldOfView = 70
        cameraNode.position = SCNVector3(0, Float(dimensions.height) * 0.5, 0)
        cameraNode.look(at: SCNVector3(0, Float(dimensions.height) * 0.5, -Float(dimensions.depth) / 2))
        scene.rootNode.addChildNode(cameraNode)
        print("✅ Camera looking at photo wall")
        
        // LIGHTING
        let ambientLight = SCNNode()
        ambientLight.light = SCNLight()
        ambientLight.light?.type = .ambient
        ambientLight.light?.intensity = 1000
        scene.rootNode.addChildNode(ambientLight)
        
        let directionalLight = SCNNode()
        directionalLight.light = SCNLight()
        directionalLight.light?.type = .directional
        directionalLight.light?.intensity = 500
        directionalLight.position = SCNVector3(0, Float(dimensions.height), 0)
        directionalLight.eulerAngles = SCNVector3(-Float.pi / 4, 0, 0)
        scene.rootNode.addChildNode(directionalLight)
        
        // ✅ Return scene directly (no file export!)
        print("✅ Room scene created successfully in memory")
        return scene
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
        
        // ✅ Use adjusted floor boundary
        let floorYPos = structure.floorY
        let floorHeight = 1.0 - floorYPos
        
        let cropRect = CGRect(
            x: 0,
            y: imageHeight * floorYPos,
            width: imageWidth,
            height: imageHeight * floorHeight
        )
        
        print("   - Using adjusted boundary: floorY=\(floorYPos)")
        print("   - Crop rect: \(cropRect)")
        
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
    
    // ✅ NEW: Extract LEFT wall texture from adjusted boundary
    private func generateLeftWallTexture(from image: UIImage, structure: RoomStructure) -> UIImage {
        print("🎨 [TextureGen] Generating LEFT wall texture from boundary")
        
        guard let cgImage = image.cgImage else {
            print("⚠️ [TextureGen] No CGImage, using solid color")
            return createSolidColorTexture(color: UIColor(white: 0.9, alpha: 1.0))
        }
        
        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)
        
        // Extract vertical strip from left boundary
        let stripWidth = imageWidth * 0.1 // 10% width strip
        let leftXPos = structure.leftX * imageWidth
        
        let cropRect = CGRect(
            x: max(0, leftXPos - stripWidth/2),
            y: structure.ceilingY * imageHeight,
            width: min(stripWidth, imageWidth),
            height: (structure.floorY - structure.ceilingY) * imageHeight
        )
        
        print("   - Left boundary: \(structure.leftX)")
        print("   - Crop rect: \(cropRect)")
        
        if let croppedImage = cgImage.cropping(to: cropRect) {
            print("✅ [TextureGen] Left wall texture extracted")
            return UIImage(cgImage: croppedImage)
        }
        
        print("⚠️ [TextureGen] Failed to crop, using solid color")
        return createSolidColorTexture(color: UIColor(white: 0.9, alpha: 1.0))
    }
    
    // ✅ NEW: Extract RIGHT wall texture from adjusted boundary
    private func generateRightWallTexture(from image: UIImage, structure: RoomStructure) -> UIImage {
        print("🎨 [TextureGen] Generating RIGHT wall texture from boundary")
        
        guard let cgImage = image.cgImage else {
            print("⚠️ [TextureGen] No CGImage, using solid color")
            return createSolidColorTexture(color: UIColor(white: 0.9, alpha: 1.0))
        }
        
        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)
        
        // Extract vertical strip from right boundary
        let stripWidth = imageWidth * 0.1 // 10% width strip
        let rightXPos = structure.rightX * imageWidth
        
        let cropRect = CGRect(
            x: max(0, rightXPos - stripWidth/2),
            y: structure.ceilingY * imageHeight,
            width: min(stripWidth, imageWidth - (rightXPos - stripWidth/2)),
            height: (structure.floorY - structure.ceilingY) * imageHeight
        )
        
        print("   - Right boundary: \(structure.rightX)")
        print("   - Crop rect: \(cropRect)")
        
        if let croppedImage = cgImage.cropping(to: cropRect) {
            print("✅ [TextureGen] Right wall texture extracted")
            return UIImage(cgImage: croppedImage)
        }
        
        print("⚠️ [TextureGen] Failed to crop, using solid color")
        return createSolidColorTexture(color: UIColor(white: 0.9, alpha: 1.0))
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
        context.render(outputImage, toBitmap: &bitmap, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 1), format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        
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

// MARK: - Room Structure Analyzer
class RoomStructureAnalyzer {
    func analyzeRoom(image: UIImage, depthMap: CIImage) async -> RoomStructure {
        print("🔍 [RoomAnalyzer] Analyzing room structure")
        var structure = RoomStructure()
        
        print("   - Detecting wall lines...")
        structure.wallLines = await detectWallLines(in: image)
        
        print("   - Detecting floor region...")
        structure.floorRegion = detectFloorRegion(in: image, depthMap: depthMap)
        print("   - Detecting ceiling region...")
        structure.ceilingRegion = detectCeilingRegion(in: image, depthMap: depthMap)
        
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
        
        // results is already [VNContoursObservation]?
        guard let observations = request.results else {
            print("⚠️ [WallLineDetector] No contours found")
            return []
        }
        
        var lines: [RoomStructure.Line] = []
        print("   - Found \(observations.count) contours")
        
        for contour in observations {
            let path = contour.normalizedPath
            let bounds = path.boundingBox
            let conf = contour.confidence
            // Two axis-aligned lines from bounds (simple placeholder)
            lines.append(RoomStructure.Line(
                start: CGPoint(x: bounds.minX, y: bounds.maxY),
                end: CGPoint(x: bounds.maxX, y: bounds.maxY),
                confidence: Float(conf)
            ))
            lines.append(RoomStructure.Line(
                start: CGPoint(x: bounds.minX, y: bounds.minY),
                end: CGPoint(x: bounds.maxX, y: bounds.minY),
                confidence: Float(conf)
            ))
        }
        
        print("✅ [WallLineDetector] Extracted \(lines.count) lines from contours")
        return lines
    }
    
    private func detectFloorRegion(in image: UIImage, depthMap: CIImage) -> CGRect? {
        let region = CGRect(x: 0, y: 0.8, width: 1.0, height: 0.2)
        print("   - Floor region: \(region)")
        return region
    }
    
    private func detectCeilingRegion(in image: UIImage, depthMap: CIImage) -> CGRect? {
        let region = CGRect(x: 0, y: 0, width: 1.0, height: 0.15)
        print("   - Ceiling region: \(region)")
        return region
    }
    
    private func calculateVanishingPoint(from lines: [RoomStructure.Line]) -> CGPoint? {
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

// MARK: - Room Boundary Detection View with DRAGGABLE boundaries
struct RoomBoundaryDetectionView: View {
    let originalImage: UIImage
    @Binding var savedBoundaries: RoomStructure? // ✅ NEW: Binding to save adjusted boundaries
    @State private var detectedBoundariesImage: UIImage?
    @State private var scale: CGFloat = 1.0
    @State private var lastScale: CGFloat = 1.0
    @State private var offset: CGSize = .zero
    @State private var lastOffset: CGSize = .zero
    @Environment(\.dismiss) var dismiss
    
    // Boundary positions (as percentages of image dimensions)
    @State private var floorY: CGFloat = 0.85
    @State private var ceilingY: CGFloat = 0.15
    @State private var leftX: CGFloat = 0.12
    @State private var rightX: CGFloat = 0.88
    @State private var vanishingX: CGFloat = 0.5
    @State private var vanishingY: CGFloat = 0.45
    
    @State private var showAdjustmentMode = false
    
    // Custom magenta color
    private let magentaColor = Color(red: 1.0, green: 0.0, blue: 1.0)
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                if showAdjustmentMode {
                    // Interactive adjustment view
                    GeometryReader { geometry in
                        ZStack {
                            // Background image
                            Image(uiImage: originalImage)
                                .resizable()
                                .scaledToFit()
                                .frame(width: geometry.size.width)
                            
                            // Overlay with draggable boundaries
                            BoundaryLinesCanvas(
                                imageSize: originalImage.size,
                                floorY: floorY,
                                ceilingY: ceilingY,
                                leftX: leftX,
                                rightX: rightX,
                                vanishingX: vanishingX,
                                vanishingY: vanishingY
                            )
                            .frame(width: geometry.size.width)
                            
                            // Draggable handles
                            DraggableHandlesOverlay(
                                geometry: geometry,
                                imageSize: originalImage.size,
                                floorY: $floorY,
                                ceilingY: $ceilingY,
                                leftX: $leftX,
                                rightX: $rightX,
                                vanishingX: $vanishingX,
                                vanishingY: $vanishingY,
                                magentaColor: magentaColor
                            )
                        }
                    }
                    
                    // Adjustment instructions
                    VStack(spacing: 12) {
                        Text("Drag the circles to adjust boundaries")
                            .font(.headline)
                            .padding(.top, 8)
                        
                        HStack(spacing: 16) {
                            Label("Floor", systemImage: "arrow.down")
                                .foregroundColor(.green)
                                .font(.caption)
                            Label("Ceiling", systemImage: "arrow.up")
                                .foregroundColor(.cyan)
                                .font(.caption)
                            Label("Walls", systemImage: "arrow.left.and.right")
                                .foregroundColor(.red)
                                .font(.caption)
                            Label("Vanish", systemImage: "scope")
                                .foregroundColor(magentaColor)
                                .font(.caption)
                        }
                        .padding(.horizontal)
                        
                        HStack(spacing: 20) {
                            Button("Reset") {
                                withAnimation {
                                    floorY = 0.85
                                    ceilingY = 0.15
                                    leftX = 0.12
                                    rightX = 0.88
                                    vanishingX = 0.5
                                    vanishingY = 0.45
                                }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Done Adjusting") {
                                // ✅ SAVE BOUNDARIES HERE
                                var boundaries = RoomStructure()
                                boundaries.floorY = floorY
                                boundaries.ceilingY = ceilingY
                                boundaries.leftX = leftX
                                boundaries.rightX = rightX
                                boundaries.vanishingX = vanishingX
                                boundaries.vanishingY = vanishingY
                                
                                savedBoundaries = boundaries
                                print("✅ Saved adjusted boundaries:")
                                print("   Floor: \(floorY), Ceiling: \(ceilingY)")
                                print("   Left: \(leftX), Right: \(rightX)")
                                print("   VP: (\(vanishingX), \(vanishingY))")
                                
                                showAdjustmentMode = false
                                generateFinalImage()
                            }
                            .buttonStyle(.borderedProminent)
                        }
                        .padding(.bottom, 8)
                    }
                    .background(.ultraThinMaterial)
                    
                } else if let boundariesImage = detectedBoundariesImage {
                    // View mode with zoom controls
                    GeometryReader { geometry in
                        ScrollView([.horizontal, .vertical], showsIndicators: true) {
                            Image(uiImage: boundariesImage)
                                .resizable()
                                .scaledToFit()
                                .frame(width: geometry.size.width)
                                .scaleEffect(scale)
                                .offset(offset)
                                .gesture(
                                    MagnificationGesture()
                                        .onChanged { value in
                                            let delta = value / lastScale
                                            lastScale = value
                                            scale *= delta
                                        }
                                        .onEnded { _ in
                                            lastScale = 1.0
                                            scale = min(max(scale, 0.5), 5.0)
                                        }
                                )
                                .gesture(
                                    DragGesture()
                                        .onChanged { value in
                                            offset = CGSize(
                                                width: lastOffset.width + value.translation.width,
                                                height: lastOffset.height + value.translation.height
                                            )
                                        }
                                        .onEnded { _ in
                                            lastOffset = offset
                                        }
                                )
                        }
                    }
                    
                    // Zoom controls
                    HStack(spacing: 20) {
                        Button(action: {
                            withAnimation { scale = max(0.5, scale - 0.5) }
                        }) {
                            Image(systemName: "minus.magnifyingglass")
                                .font(.title2)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .clipShape(Circle())
                        }
                        
                        Button("Adjust Boundaries") {
                            showAdjustmentMode = true
                        }
                        .font(.headline)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(Color.orange)
                        .foregroundColor(.white)
                        .cornerRadius(20)
                        
                        Button(action: {
                            withAnimation {
                                scale = 1.0
                                offset = .zero
                                lastOffset = .zero
                            }
                        }) {
                            Text("Reset")
                                .font(.headline)
                                .padding(.horizontal, 20)
                                .padding(.vertical, 10)
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(20)
                        }
                        
                        Button(action: {
                            withAnimation { scale = min(5.0, scale + 0.5) }
                        }) {
                            Image(systemName: "plus.magnifyingglass")
                                .font(.title2)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .clipShape(Circle())
                        }
                    }
                    .padding()
                    
                } else {
                    ProgressView("Detecting room boundaries...")
                        .padding()
                }
            }
            .navigationTitle("Room Boundaries")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .onAppear { generateFinalImage() }
    }
    
    func generateFinalImage() {
        Task {
            let result = await drawBoundariesOnImage()
            await MainActor.run { detectedBoundariesImage = result }
        }
    }
    
    func drawBoundariesOnImage() async -> UIImage {
        let width = originalImage.size.width
        let height = originalImage.size.height
        
        let renderer = UIGraphicsImageRenderer(size: originalImage.size)
        return renderer.image { context in
            // Draw original image
            originalImage.draw(at: .zero)
            
            let cgContext = context.cgContext
            
            // Draw floor boundary in GREEN
            cgContext.setStrokeColor(UIColor.green.cgColor)
            cgContext.setLineWidth(15.0)
            let floorYPos = floorY * height
            cgContext.move(to: CGPoint(x: 0, y: floorYPos))
            cgContext.addLine(to: CGPoint(x: width, y: floorYPos))
            cgContext.strokePath()
            
            let floorLabel = "FLOOR"
            let floorAttrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 60),
                .foregroundColor: UIColor.green,
                .strokeColor: UIColor.black,
                .strokeWidth: -4.0
            ]
            floorLabel.draw(at: CGPoint(x: 50, y: floorYPos - 80), withAttributes: floorAttrs)
            
            // Draw ceiling boundary in CYAN
            cgContext.setStrokeColor(UIColor.cyan.cgColor)
            cgContext.setLineWidth(15.0)
            let ceilingYPos = ceilingY * height
            cgContext.move(to: CGPoint(x: 0, y: ceilingYPos))
            cgContext.addLine(to: CGPoint(x: width, y: ceilingYPos))
            cgContext.strokePath()
            
            let ceilingLabel = "CEILING"
            let ceilingAttrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 60),
                .foregroundColor: UIColor.cyan,
                .strokeColor: UIColor.black,
                .strokeWidth: -4.0
            ]
            ceilingLabel.draw(at: CGPoint(x: 50, y: ceilingYPos + 30), withAttributes: ceilingAttrs)
            
            // Draw left wall in RED
            cgContext.setStrokeColor(UIColor.red.cgColor)
            cgContext.setLineWidth(12.0)
            let leftXPos = leftX * width
            cgContext.move(to: CGPoint(x: leftXPos, y: 0))
            cgContext.addLine(to: CGPoint(x: leftXPos, y: height))
            cgContext.strokePath()
            
            let leftLabel = "LEFT"
            let leftAttrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 50),
                .foregroundColor: UIColor.red,
                .strokeColor: UIColor.white,
                .strokeWidth: -3.0
            ]
            leftLabel.draw(at: CGPoint(x: leftXPos + 30, y: height / 2), withAttributes: leftAttrs)
            
            // Draw right wall in YELLOW
            cgContext.setStrokeColor(UIColor.yellow.cgColor)
            cgContext.setLineWidth(12.0)
            let rightXPos = rightX * width
            cgContext.move(to: CGPoint(x: rightXPos, y: 0))
            cgContext.addLine(to: CGPoint(x: rightXPos, y: height))
            cgContext.strokePath()
            
            let rightLabel = "RIGHT"
            let rightAttrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 50),
                .foregroundColor: UIColor.yellow,
                .strokeColor: UIColor.black,
                .strokeWidth: -3.0
            ]
            rightLabel.draw(at: CGPoint(x: rightXPos - 150, y: height / 2), withAttributes: rightAttrs)
            
            // Draw vanishing point in MAGENTA
            cgContext.setFillColor(UIColor.magenta.cgColor)
            let vpX = vanishingX * width
            let vpY = vanishingY * height
            let vpRadius: CGFloat = 40
            let vpRect = CGRect(x: vpX - vpRadius, y: vpY - vpRadius, width: vpRadius * 2, height: vpRadius * 2)
            cgContext.fillEllipse(in: vpRect)
            
            // Crosshair
            cgContext.setStrokeColor(UIColor.white.cgColor)
            cgContext.setLineWidth(5.0)
            cgContext.move(to: CGPoint(x: vpX - 80, y: vpY))
            cgContext.addLine(to: CGPoint(x: vpX + 80, y: vpY))
            cgContext.move(to: CGPoint(x: vpX, y: vpY - 80))
            cgContext.addLine(to: CGPoint(x: vpX, y: vpY + 80))
            cgContext.strokePath()
            
            let vpLabel = "VP"
            let vpAttrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 40),
                .foregroundColor: UIColor.magenta,
                .strokeColor: UIColor.white,
                .strokeWidth: -3.0
            ]
            vpLabel.draw(at: CGPoint(x: vpX - 30, y: vpY - 100), withAttributes: vpAttrs)
        }
    }
}

// MARK: - Boundary Lines Canvas (fixed: no top-level `let` in ViewBuilder)
struct BoundaryLinesCanvas: View {
    let imageSize: CGSize
    let floorY: CGFloat
    let ceilingY: CGFloat
    let leftX: CGFloat
    let rightX: CGFloat
    let vanishingX: CGFloat
    let vanishingY: CGFloat

    var body: some View {
        GeometryReader { geometry in
            BoundaryLinesCanvasInner(
                calc: calculateImageBounds(size: geometry.size),
                floorY: floorY,
                ceilingY: ceilingY,
                leftX: leftX,
                rightX: rightX,
                vanishingX: vanishingX,
                vanishingY: vanishingY
            )
        }
    }

    private func calculateImageBounds(size: CGSize) -> (imageWidth: CGFloat, imageHeight: CGFloat, offsetX: CGFloat, offsetY: CGFloat) {
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = size.width / size.height

        var imageWidth: CGFloat
        var imageHeight: CGFloat
        var offsetX: CGFloat = 0
        var offsetY: CGFloat = 0

        if imageAspect > viewAspect {
            imageWidth = size.width
            imageHeight = size.width / imageAspect
            offsetY = (size.height - imageHeight) / 2
        } else {
            imageHeight = size.height
            imageWidth = size.height * imageAspect
            offsetX = (size.width - imageWidth) / 2
        }

        return (imageWidth, imageHeight, offsetX, offsetY)
    }
}

private struct BoundaryLinesCanvasInner: View {
    let calc: (imageWidth: CGFloat, imageHeight: CGFloat, offsetX: CGFloat, offsetY: CGFloat)
    let floorY: CGFloat
    let ceilingY: CGFloat
    let leftX: CGFloat
    let rightX: CGFloat
    let vanishingX: CGFloat
    let vanishingY: CGFloat

    var body: some View {
        ZStack {
            // Floor line (GREEN)
            Path { path in
                let y = calc.offsetY + floorY * calc.imageHeight
                path.move(to: CGPoint(x: calc.offsetX, y: y))
                path.addLine(to: CGPoint(x: calc.offsetX + calc.imageWidth, y: y))
            }
            .stroke(Color.green, lineWidth: 8)

            // Ceiling line (CYAN)
            Path { path in
                let y = calc.offsetY + ceilingY * calc.imageHeight
                path.move(to: CGPoint(x: calc.offsetX, y: y))
                path.addLine(to: CGPoint(x: calc.offsetX + calc.imageWidth, y: y))
            }
            .stroke(Color.cyan, lineWidth: 8)

            // Left wall line (RED)
            Path { path in
                let x = calc.offsetX + leftX * calc.imageWidth
                path.move(to: CGPoint(x: x, y: calc.offsetY))
                path.addLine(to: CGPoint(x: x, y: calc.offsetY + calc.imageHeight))
            }
            .stroke(Color.red, lineWidth: 6)

            // Right wall line (YELLOW)
            Path { path in
                let x = calc.offsetX + rightX * calc.imageWidth
                path.move(to: CGPoint(x: x, y: calc.offsetY))
                path.addLine(to: CGPoint(x: x, y: calc.offsetY + calc.imageHeight))
            }
            .stroke(Color.yellow, lineWidth: 6)

            // Vanishing point (MAGENTA)
            Circle()
                .fill(Color(red: 1.0, green: 0.0, blue: 1.0))
                .frame(width: 30, height: 30)
                .position(
                    x: calc.offsetX + vanishingX * calc.imageWidth,
                    y: calc.offsetY + vanishingY * calc.imageHeight
                )

            // Crosshair
            Path { path in
                let vpX = calc.offsetX + vanishingX * calc.imageWidth
                let vpY = calc.offsetY + vanishingY * calc.imageHeight
                path.move(to: CGPoint(x: vpX - 30, y: vpY))
                path.addLine(to: CGPoint(x: vpX + 30, y: vpY))
                path.move(to: CGPoint(x: vpX, y: vpY - 30))
                path.addLine(to: CGPoint(x: vpX, y: vpY + 30))
            }
            .stroke(Color.white, lineWidth: 3)
        }
    }
}



// MARK: - Draggable Handles Overlay (fixed: no top-level lets in ViewBuilder)
struct DraggableHandlesOverlay: View {
    let geometry: GeometryProxy
    let imageSize: CGSize
    @Binding var floorY: CGFloat
    @Binding var ceilingY: CGFloat
    @Binding var leftX: CGFloat
    @Binding var rightX: CGFloat
    @Binding var vanishingX: CGFloat
    @Binding var vanishingY: CGFloat
    let magentaColor: Color

    var body: some View {
        DraggableHandlesOverlayInner(
            calc: computeBounds(),
            floorY: $floorY,
            ceilingY: $ceilingY,
            leftX: $leftX,
            rightX: $rightX,
            vanishingX: $vanishingX,
            vanishingY: $vanishingY,
            magentaColor: magentaColor
        )
    }

    private func computeBounds() -> (imageWidth: CGFloat, imageHeight: CGFloat, offsetX: CGFloat, offsetY: CGFloat) {
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = geometry.size.width / geometry.size.height

        var imageWidth: CGFloat
        var imageHeight: CGFloat
        var offsetX: CGFloat = 0
        var offsetY: CGFloat = 0

        if imageAspect > viewAspect {
            imageWidth = geometry.size.width
            imageHeight = geometry.size.width / imageAspect
            offsetY = (geometry.size.height - imageHeight) / 2
        } else {
            imageHeight = geometry.size.height
            imageWidth = geometry.size.height * imageAspect
            offsetX = (geometry.size.width - imageWidth) / 2
        }

        return (imageWidth, imageHeight, offsetX, offsetY)
    }
}

private struct DraggableHandlesOverlayInner: View {
    let calc: (imageWidth: CGFloat, imageHeight: CGFloat, offsetX: CGFloat, offsetY: CGFloat)
    @Binding var floorY: CGFloat
    @Binding var ceilingY: CGFloat
    @Binding var leftX: CGFloat
    @Binding var rightX: CGFloat
    @Binding var vanishingX: CGFloat
    @Binding var vanishingY: CGFloat
    let magentaColor: Color

    var body: some View {
        ZStack {
            // Floor handle (GREEN)
            DraggableHandle(color: .green, icon: "arrow.down.circle.fill")
                .position(
                    x: calc.offsetX + calc.imageWidth / 2,
                    y: calc.offsetY + floorY * calc.imageHeight
                )
                .gesture(
                    DragGesture().onChanged { value in
                        let newY = value.location.y - calc.offsetY
                        floorY = min(max(newY / calc.imageHeight, 0.5), 0.95)
                    }
                )

            // Ceiling handle (CYAN)
            DraggableHandle(color: .cyan, icon: "arrow.up.circle.fill")
                .position(
                    x: calc.offsetX + calc.imageWidth / 2,
                    y: calc.offsetY + ceilingY * calc.imageHeight
                )
                .gesture(
                    DragGesture().onChanged { value in
                        let newY = value.location.y - calc.offsetY
                        ceilingY = min(max(newY / calc.imageHeight, 0.05), 0.5)
                    }
                )

            // Left wall handle (RED)
            DraggableHandle(color: .red, icon: "arrow.left.circle.fill")
                .position(
                    x: calc.offsetX + leftX * calc.imageWidth,
                    y: calc.offsetY + calc.imageHeight / 2
                )
                .gesture(
                    DragGesture().onChanged { value in
                        let newX = value.location.x - calc.offsetX
                        leftX = min(max(newX / calc.imageWidth, 0.02), 0.4)
                    }
                )

            // Right wall handle (YELLOW)
            DraggableHandle(color: .yellow, icon: "arrow.right.circle.fill")
                .position(
                    x: calc.offsetX + rightX * calc.imageWidth,
                    y: calc.offsetY + calc.imageHeight / 2
                )
                .gesture(
                    DragGesture().onChanged { value in
                        let newX = value.location.x - calc.offsetX
                        rightX = min(max(newX / calc.imageWidth, 0.6), 0.98)
                    }
                )

            // Vanishing point handle (MAGENTA)
            DraggableHandle(color: magentaColor, icon: "scope", size: 50)
                .position(
                    x: calc.offsetX + vanishingX * calc.imageWidth,
                    y: calc.offsetY + vanishingY * calc.imageHeight
                )
                .gesture(
                    DragGesture().onChanged { value in
                        let newX = value.location.x - calc.offsetX
                        let newY = value.location.y - calc.offsetY
                        vanishingX = min(max(newX / calc.imageWidth, 0.1), 0.9)
                        vanishingY = min(max(newY / calc.imageHeight, 0.1), 0.9)
                    }
                )
        }
    }
}


// MARK: - Draggable Handle Component
struct DraggableHandle: View {
    let color: Color
    let icon: String
    var size: CGFloat = 44
    
    var body: some View {
        Image(systemName: icon)
            .font(.system(size: size))
            .foregroundColor(color)
            .shadow(color: .black.opacity(0.5), radius: 3, x: 0, y: 2)
            .shadow(color: color.opacity(0.5), radius: 10, x: 0, y: 0)
    }
}

// MARK: - SwiftUI View
struct SinglePhotoRoomView: View {
    @StateObject private var reconstructor = SinglePhotoRoomReconstructor()
    @State private var selectedImage: UIImage?
    @State private var showImagePicker = false
    @State private var showRoomBoundaries = false
    @State private var adjustedBoundaries: RoomStructure? // ✅ NEW: Store adjusted boundaries
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
                    .onAppear { print("🖼️ [View] Displaying selected image") }
                
                Button("Show Room Boundaries") {
                    print("🏠 [View] Room boundaries button tapped")
                    showRoomBoundaries = true
                }
                .buttonStyle(.borderedProminent)
                .padding()
                
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
                .onAppear { print("⏳ [View] Processing view appeared") }
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
                .onAppear { print("📊 [View] Dimensions view appeared") }
                
                Button("Rebuild with Adjusted Dimensions") {
                    print("🔄 [View] Rebuild button tapped")
                    rebuildRoom()
                }
                .buttonStyle(.borderedProminent)
            }
            
            // ✅ CHANGED: Using scene instead of URL
            if let roomScene = reconstructor.generatedRoomScene {
                NavigationLink(destination: SceneKitViewer(scene: roomScene)) {
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
        // Room boundaries sheet: always returns a View via wrapper
        .sheet(isPresented: $showRoomBoundaries) {
            RoomBoundarySheetView(
                image: selectedImage,
                savedBoundaries: $adjustedBoundaries
            )
        }
        .onAppear {
            print("👁️ [View] SinglePhotoRoomView appeared")
            adjustedWidth = reconstructor.estimatedDimensions?.width ?? 4.0
            adjustedDepth = reconstructor.estimatedDimensions?.depth ?? 4.0
            adjustedHeight = reconstructor.estimatedDimensions?.height ?? 2.8
        }
        // ✅ NEW: Watch for boundary changes and rebuild automatically
        .onChange(of: adjustedBoundaries) { oldValue, newValue in
            if let boundaries = newValue, let image = selectedImage {
                print("🔄 [View] Boundaries adjusted, rebuilding room...")
                Task {
                    await reconstructor.processPhotoWithBoundaries(image, boundaries: boundaries)
                }
            }
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
                if let boundaries = adjustedBoundaries {
                    await reconstructor.processPhotoWithBoundaries(image, boundaries: boundaries)
                } else {
                    await reconstructor.processPhoto(image)
                }
            }
        }
    }
    
    private func confidenceColor(_ confidence: Float) -> Color {
        switch confidence {
        case 0.7...1.0: return .green
        case 0.4..<0.7: return .orange
        default: return .red
        }
    }
}

// Wrapper view to guarantee the sheet always returns a View
private struct RoomBoundarySheetView: View {
    let image: UIImage?
    @Binding var savedBoundaries: RoomStructure?
    
    var body: some View {
        Group {
            if let image {
                RoomBoundaryDetectionView(
                    originalImage: image,
                    savedBoundaries: $savedBoundaries
                )
            } else {
                EmptyView()
            }
        }
    }
}

// MARK: - Photo Picker View
struct PhotoPickerView: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Environment(\.dismiss) var dismiss
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        print("📱 [PhotoPicker] Creating UIImagePickerController")
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator { Coordinator(self) }
    
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
            parent.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            print("❌ [PhotoPicker] User cancelled")
            parent.dismiss()
        }
    }
}

// MARK: - SceneKit Viewer
struct SceneKitViewer: View {
    let scene: SCNScene // ✅ CHANGED: Accept scene directly
    @State private var showControls = true
    
    var body: some View {
        ZStack {
            SceneView(
                scene: scene, // ✅ CHANGED
                options: [.allowsCameraControl, .autoenablesDefaultLighting]
            )
            .onAppear {
                print("🎬 [Viewer] SceneKit viewer appeared")
                print("   - Scene nodes: \(scene.rootNode.childNodes.count)")
            }
            
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
                        withAnimation { showControls = false }
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
