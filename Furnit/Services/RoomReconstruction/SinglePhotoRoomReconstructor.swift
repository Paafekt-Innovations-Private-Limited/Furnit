import SwiftUI
import CoreML
import Vision
import CoreImage
import SceneKit
import Accelerate

// MARK: - Single Photo Room Reconstructor
class SinglePhotoRoomReconstructor: ObservableObject {
    @Published var isProcessing = false
    @Published var progress: Float = 0.0
    @Published var statusMessage = L10n.PhotoRoom.reconstructorReady
    @Published var estimatedDimensions: RoomDimensions?
    @Published var generatedRoomScene: SCNScene? // ✅ CHANGED: from URL to SCNScene
    
    private let depthEstimator = MiDaSDepthEstimator()
    private let roomAnalyzer = RoomStructureAnalyzer()
    private let textureProcessor = TextureProcessor()
    
    struct RoomDimensions {
        var width: Float = 4.0
        var depth: Float = 4.5
        var height: Float = 2.8
        var doorHeight: Float = 2.1
        var confidence: Float = 0.6
    }
    
    // ✅ Max image dimension to prevent memory crashes
    private let maxImageDimension: CGFloat = 1600

    init() {
        logDebug("🏗️ [Reconstructor] Initialized")
    }

    // ✅ vImage-accelerated downscaling (GPU/NEON SIMD)
    private func downscaleIfNeeded(_ image: UIImage) -> UIImage {
        let maxDim = max(image.size.width, image.size.height)
        guard maxDim > maxImageDimension else { return image }

        let scale = maxImageDimension / maxDim
        guard let cgImage = image.cgImage else { return image }

        let newWidth = Int(CGFloat(cgImage.width) * scale)
        let newHeight = Int(CGFloat(cgImage.height) * scale)

        var format = vImage_CGImageFormat(
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            colorSpace: nil,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            version: 0,
            decode: nil,
            renderingIntent: .defaultIntent
        )

        var sourceBuffer = vImage_Buffer()
        var error = vImageBuffer_InitWithCGImage(&sourceBuffer, &format, nil, cgImage, vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else { return image }
        defer { free(sourceBuffer.data) }

        var destBuffer = vImage_Buffer()
        error = vImageBuffer_Init(&destBuffer, vImagePixelCount(newHeight), vImagePixelCount(newWidth), 32, vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else { return image }
        defer { free(destBuffer.data) }

        error = vImageScale_ARGB8888(&sourceBuffer, &destBuffer, nil, vImage_Flags(kvImageHighQualityResampling))
        guard error == kvImageNoError else { return image }

        guard let scaledCGImage = vImageCreateCGImageFromBuffer(&destBuffer, &format, nil, nil, vImage_Flags(kvImageNoFlags), &error)?.takeRetainedValue() else {
            return image
        }

        logDebug("🚀 [Reconstructor] Downscaled \(Int(image.size.width))x\(Int(image.size.height)) → \(newWidth)x\(newHeight)")
        return UIImage(cgImage: scaledCGImage)
    }
    
    // MARK: - Helper Methods
    private func updateProgress(_ value: Float, _ message: String) async {
        logDebug("📊 [Reconstructor] Progress: \(Int(value * 100))% - \(message)")
        await MainActor.run {
            self.progress = value
            self.statusMessage = message
        }
    }
    
    private func setError(_ message: String) async {
        logDebug("❌ [Reconstructor] ERROR: \(message)")
        await MainActor.run {
            self.isProcessing = false
            self.statusMessage = message
        }
    }
    
    // MARK: - Main Processing Pipeline
    func processPhoto(_ image: UIImage) async {
        logDebug("🚀 [Reconstructor] ========== STARTING PHOTO PROCESSING ==========")
        logDebug("📸 [Reconstructor] Image size: \(image.size)")
        logDebug("📸 [Reconstructor] Image scale: \(image.scale)")
        
        await MainActor.run {
            isProcessing = true
            progress = 0.0
            statusMessage = L10n.PhotoRoom.reconstructorAnalyzingPhoto
        }
        
        // ✅ Fix image orientation FIRST
        let orientedImage = image.fixedOrientation()
        logDebug("✅ [Reconstructor] Image orientation fixed")

        // ✅ OPTIMIZATION: Downscale large images to prevent memory crashes
        let fixedImage = downscaleIfNeeded(orientedImage)

        // Step 1: Generate depth map
        await updateProgress(0.2, L10n.PhotoRoom.reconstructorExtractingDepth)
        logDebug("🔍 [Reconstructor] Step 1: Starting depth estimation")
        
        guard let depthMap = await depthEstimator.estimateDepth(from: fixedImage) else{
            await setError(L10n.PhotoRoom.reconstructorDepthFailed)
            return
        }
        logDebug("✅ [Reconstructor] Step 1: Depth map created - extent: \(depthMap.extent)")
        
        // Step 2: Detect room structure
        await updateProgress(0.4, L10n.PhotoRoom.reconstructorFindingWalls)
        logDebug("🔍 [Reconstructor] Step 2: Starting room structure analysis")
        let roomStructure = await roomAnalyzer.analyzeRoom(image: fixedImage, depthMap: depthMap)
        logDebug("✅ [Reconstructor] Step 2: Room structure analyzed")
        logDebug("   - Wall lines found: \(roomStructure.wallLines.count)")
        logDebug("   - Floor region: \(roomStructure.floorRegion?.debugDescription ?? "nil")")
        logDebug("   - Ceiling region: \(roomStructure.ceilingRegion?.debugDescription ?? "nil")")
        
        // Step 3: Estimate dimensions
        await updateProgress(0.6, L10n.PhotoRoom.reconstructorCalculatingDimensions)
        logDebug("🔍 [Reconstructor] Step 3: Starting dimension estimation")
        let dimensions = await estimateDimensions(from: roomStructure, image: fixedImage)
        logDebug("✅ [Reconstructor] Step 3: Dimensions estimated")
        logDebug("   - Width: \(dimensions.width)m")
        logDebug("   - Depth: \(dimensions.depth)m")
        logDebug("   - Height: \(dimensions.height)m")
        logDebug("   - Confidence: \(Int(dimensions.confidence * 100))%")
        
        await MainActor.run {
            self.estimatedDimensions = dimensions
        }
        
        // Step 4: Build 3D room
        await updateProgress(0.8, L10n.PhotoRoom.reconstructorBuilding3D)
        logDebug("🔍 [Reconstructor] Step 4: Starting 3D room construction")
        let roomScene = await build3DRoom(
            dimensions: dimensions,
            structure: roomStructure,
            originalImage: fixedImage,
            depthMap: depthMap
        )
        
        if let scene = roomScene {
            logDebug("✅ [Reconstructor] Step 4: 3D room built successfully")
            logDebug("   - Scene nodes: \(scene.rootNode.childNodes.count)")
        } else {
            logDebug("❌ [Reconstructor] Step 4: Failed to build 3D room")
        }
        
        await MainActor.run {
            self.generatedRoomScene = roomScene // ✅ CHANGED
            self.isProcessing = false
            self.progress = 1.0
            self.statusMessage = L10n.PhotoRoom.reconstructorRoomReady
        }
        
        logDebug("🎉 [Reconstructor] ========== PHOTO PROCESSING COMPLETE ==========")
    }

    
    // ✅ NEW: Process with Adjusted Boundaries
    func processPhotoWithBoundaries(_ image: UIImage, boundaries: RoomStructure) async {
        logDebug("🚀 [Reconstructor] ========== PROCESSING WITH ADJUSTED BOUNDARIES ==========")
        logDebug("   Floor: \(boundaries.floorY), Ceiling: \(boundaries.ceilingY)")
        logDebug("   Left: \(boundaries.leftX), Right: \(boundaries.rightX)")
        logDebug("   VP: (\(boundaries.vanishingX), \(boundaries.vanishingY))")

        await MainActor.run {
            isProcessing = true
            progress = 0.0
            statusMessage = L10n.PhotoRoom.reconstructorStarting
        }

        // Small delay for UI feedback
        try? await Task.sleep(nanoseconds: 200_000_000) // 0.2 sec

        await updateProgress(0.1, L10n.PhotoRoom.reconstructorPreparingImage)

        // Fix image orientation FIRST
        let orientedImage = image.fixedOrientation()
        logDebug("✅ [Reconstructor] Image orientation fixed")

        // ✅ OPTIMIZATION: Downscale large images
        let fixedImage = downscaleIfNeeded(orientedImage)

        try? await Task.sleep(nanoseconds: 200_000_000) // 0.2 sec

        await updateProgress(0.25, L10n.PhotoRoom.reconstructorExtractingDepth)
        guard let depthMap = await depthEstimator.estimateDepth(from: fixedImage) else {
            await setError(L10n.PhotoRoom.reconstructorDepthFailed)
            return
        }

        try? await Task.sleep(nanoseconds: 200_000_000) // 0.2 sec

        await updateProgress(0.4, L10n.PhotoRoom.reconstructorAnalyzingBoundaries)

        // Use the adjusted boundaries instead of detecting new ones
        var roomStructure = boundaries
        roomStructure.floorRegion = CGRect(x: 0, y: boundaries.floorY, width: 1.0, height: 1.0 - boundaries.floorY)
        roomStructure.ceilingRegion = CGRect(x: 0, y: 0, width: 1.0, height: boundaries.ceilingY)
        roomStructure.vanishingPoint = CGPoint(x: boundaries.vanishingX, y: boundaries.vanishingY)

        logDebug("✅ [Reconstructor] Using adjusted boundaries:")
        logDebug("   - Floor region: \(roomStructure.floorRegion!)")
        logDebug("   - Ceiling region: \(roomStructure.ceilingRegion!)")

        try? await Task.sleep(nanoseconds: 200_000_000) // 0.2 sec

        await updateProgress(0.55, L10n.PhotoRoom.reconstructorCalculatingDimensions)
        let dimensions = await estimateDimensions(from: roomStructure, image: fixedImage)

        await MainActor.run {
            self.estimatedDimensions = dimensions
        }

        try? await Task.sleep(nanoseconds: 200_000_000) // 0.2 sec

        await updateProgress(0.7, L10n.PhotoRoom.reconstructorCreatingTextures)

        try? await Task.sleep(nanoseconds: 200_000_000) // 0.2 sec

        await updateProgress(0.85, L10n.PhotoRoom.reconstructorBuilding3D)
        let roomScene = await build3DRoom(
            dimensions: dimensions,
            structure: roomStructure,
            originalImage: fixedImage,
            depthMap: depthMap
        )

        try? await Task.sleep(nanoseconds: 200_000_000) // 0.2 sec

        await updateProgress(0.95, L10n.PhotoRoom.reconstructorFinalizing)

        try? await Task.sleep(nanoseconds: 150_000_000) // 0.15 sec

        await MainActor.run {
            self.generatedRoomScene = roomScene
            self.isProcessing = false
            self.progress = 1.0
            self.statusMessage = L10n.PhotoRoom.reconstructorRoomReady
        }

        logDebug("🎉 [Reconstructor] ========== BOUNDARY PROCESSING COMPLETE ==========")
    }
    
    // MARK: - Dimension Estimation
    private func estimateDimensions(from structure: RoomStructure, image: UIImage) async -> RoomDimensions {
        logDebug("📏 [DimensionEstimator] Estimating room dimensions")
        var dimensions = RoomDimensions()
        
        if let doorRect = await detectDoor(in: image) {
            logDebug("✅ [DimensionEstimator] DOOR DETECTED (best reference)")
            logDebug("   - Door rect: \(doorRect)")
            
            let doorPixelHeight = doorRect.height * image.size.height
            let imageHeight = image.size.height
            
            logDebug("   - Door pixel height: \(doorPixelHeight)")
            logDebug("   - Image height: \(imageHeight)")
            
            let pixelsPerMeter = doorPixelHeight / 2.1
            logDebug("   - Pixels per meter: \(pixelsPerMeter)")
            
            dimensions.height = Float(imageHeight / pixelsPerMeter)
            dimensions.width = Float(image.size.width / pixelsPerMeter * 1.2)
            dimensions.depth = dimensions.width
            dimensions.confidence = 0.8
            
            logDebug("   - Calculated height: \(dimensions.height)m")
            logDebug("   - Calculated width: \(dimensions.width)m")
            logDebug("   - Confidence: 80%")
            
        } else if let personRect = await detectPerson(in: image) {
            logDebug("✅ [DimensionEstimator] PERSON DETECTED (fallback)")
            logDebug("   - Person rect: \(personRect)")
            
            let personPixelHeight = personRect.height * image.size.height
            let pixelsPerMeter = personPixelHeight / 1.7
            
            logDebug("   - Person pixel height: \(personPixelHeight)")
            logDebug("   - Pixels per meter: \(pixelsPerMeter)")
            
            dimensions.height = Float(image.size.height / pixelsPerMeter * 0.4)
            dimensions.width = Float(image.size.width / pixelsPerMeter * 1.5)
            dimensions.depth = dimensions.width * 0.9
            dimensions.confidence = 0.5
            
            logDebug("   - Calculated height: \(dimensions.height)m")
            logDebug("   - Calculated width: \(dimensions.width)m")
            logDebug("   - Confidence: 50%")
            
        } else {
            logDebug("⚠️ [DimensionEstimator] NO REFERENCE FOUND - Using defaults")
            dimensions.width = 4.0
            dimensions.depth = 4.5
            dimensions.height = 2.8
            dimensions.confidence = 0.3
            
            logDebug("   - Default width: \(dimensions.width)m")
            logDebug("   - Default depth: \(dimensions.depth)m")
            logDebug("   - Default height: \(dimensions.height)m")
            logDebug("   - Confidence: 30%")
        }
        
        let originalWidth = dimensions.width
        let originalDepth = dimensions.depth
        let originalHeight = dimensions.height
        
        dimensions.width = min(max(dimensions.width, 2.0), 8.0)
        dimensions.depth = min(max(dimensions.depth, 2.0), 8.0)
        dimensions.height = min(max(dimensions.height, 2.2), 4.0)
        
        if originalWidth != dimensions.width || originalDepth != dimensions.depth || originalHeight != dimensions.height {
            logDebug("⚠️ [DimensionEstimator] Dimensions clamped:")
            logDebug("   - Width: \(originalWidth) -> \(dimensions.width)")
            logDebug("   - Depth: \(originalDepth) -> \(dimensions.depth)")
            logDebug("   - Height: \(originalHeight) -> \(dimensions.height)")
        }
        
        logDebug("📏 [DimensionEstimator] Final dimensions: W:\(dimensions.width)m D:\(dimensions.depth)m H:\(dimensions.height)m")
        return dimensions
    }
    
    // MARK: - Object Detection
    private func detectDoor(in image: UIImage) async -> CGRect? {
        logDebug("🚪 [DoorDetector] Starting door detection")
        guard let cgImage = image.cgImage else {
            logDebug("❌ [DoorDetector] Failed to get CGImage")
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
            logDebug("✅ [DoorDetector] Vision request performed")
        } catch {
            logDebug("❌ [DoorDetector] Vision request failed: \(error)")
            return nil
        }
        
        // results is already [VNRectangleObservation]?
        guard let rects = request.results, !rects.isEmpty else {
            logDebug("⚠️ [DoorDetector] No rectangles detected")
            return nil
        }
        
        logDebug("🚪 [DoorDetector] Found \(rects.count) rectangles")
        for (index, rect) in rects.enumerated() {
            let aspectRatio = rect.boundingBox.width / rect.boundingBox.height
            logDebug("   Rectangle \(index): aspect=\(aspectRatio), height=\(rect.boundingBox.height)")
            if aspectRatio > 0.35 && aspectRatio < 0.5 && rect.boundingBox.height > 0.3 {
                logDebug("✅ [DoorDetector] Door-like rectangle found at index \(index)!")
                return rect.boundingBox
            }
        }
        logDebug("⚠️ [DoorDetector] No rectangles matched door criteria")
        return nil
    }
    
    private func detectPerson(in image: UIImage) async -> CGRect? {
        logDebug("👤 [PersonDetector] Starting person detection")
        guard let cgImage = image.cgImage else {
            logDebug("❌ [PersonDetector] Failed to get CGImage")
            return nil
        }
        
        let request = VNDetectHumanRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        do {
            try handler.perform([request])
            logDebug("✅ [PersonDetector] Vision request performed")
        } catch {
            logDebug("❌ [PersonDetector] Vision request failed: \(error)")
            return nil
        }
        
        // results is already [VNHumanObservation]?
        guard let persons = request.results, !persons.isEmpty else {
            logDebug("⚠️ [PersonDetector] No persons detected")
            return nil
        }
        
        logDebug("✅ [PersonDetector] Found \(persons.count) person(s)")
        return persons.first!.boundingBox
    }
    
    // ✅ NEW: Configure material for USDZ export with self-illumination
    private func configureMaterialForUSDZExport(_ material: SCNMaterial, texture: Any) {
        // Set the main texture
        material.diffuse.contents = texture
        
        // ✅ ADD EMISSION - Makes material self-lit (doesn't need external lights!)
        material.emission.contents = texture
        material.emission.intensity = 0.5  // 50% emission = bright and visible
        
        // ✅ Better lighting response
        material.lightingModel = .physicallyBased
        
        // ✅ Brightness boost
        material.multiply.contents = UIColor(white: 1.2, alpha: 1.0)
        
        // ✅ Proper USDZ settings
        material.isDoubleSided = true
        material.writesToDepthBuffer = true
        
        logDebug("   ✅ Material configured with self-illumination for USDZ export")
    }
    
    // MARK: - 3D Room Building - WITH PHOTO TEXTURES
    private func build3DRoom(dimensions: RoomDimensions, structure: RoomStructure, originalImage: UIImage, depthMap: CIImage) async -> SCNScene? {
        logDebug("🏗️ [RoomBuilder] Starting TEXTURED room construction")
        logDebug("   - Dimensions: W:\(dimensions.width) D:\(dimensions.depth) H:\(dimensions.height)")
        
        let scene = SCNScene()
        let roomNode = SCNNode()
        
        // Generate textures from photo
        logDebug("🎨 [RoomBuilder] Generating textures from photo...")
        let floorTexture = generateFloorTexture(from: originalImage, structure: structure)
        let ceilingTexture = generateCeilingTexture(from: originalImage, structure: structure) // ✅ Use boundaries
        let frontWallTexture = generateFrontWallTexture(from: originalImage, structure: structure) // ✅ Use boundaries
        let leftWallTexture = generateLeftWallTexture(from: originalImage, structure: structure)
        let rightWallTexture = generateRightWallTexture(from: originalImage, structure: structure)
        let backWallColor = generateWallTexture(from: originalImage) // Just color for back
        
        // FLOOR - With texture from photo
        logDebug("🔨 Creating FLOOR with texture...")
        let floor = SCNBox(width: CGFloat(dimensions.width),
                           height: 0.01,
                           length: CGFloat(dimensions.depth),
                           chamferRadius: 0)
        let floorMaterial = SCNMaterial()
        configureMaterialForUSDZExport(floorMaterial, texture: floorTexture)
        floor.materials = [floorMaterial]
        
        let floorNode = SCNNode(geometry: floor)
        floorNode.position = SCNVector3(0, 0, 0)
        roomNode.addChildNode(floorNode)
        logDebug("✅ FLOOR at y=0")
        
        // CEILING
        logDebug("🔨 Creating CEILING...")
        let ceiling = SCNBox(width: CGFloat(dimensions.width),
                             height: 0.01,
                             length: CGFloat(dimensions.depth),
                             chamferRadius: 0)
        let ceilingMaterial = SCNMaterial()
        configureMaterialForUSDZExport(ceilingMaterial, texture: ceilingTexture)
        ceiling.materials = [ceilingMaterial]
        
        let ceilingNode = SCNNode(geometry: ceiling)
        ceilingNode.position = SCNVector3(0, Float(dimensions.height), 0)
        roomNode.addChildNode(ceilingNode)
        logDebug("✅ CEILING at y=\(dimensions.height)")
        
        // FRONT WALL - With YOUR PHOTO texture
        logDebug("🔨 Creating FRONT WALL with photo texture...")
        let frontWall = SCNBox(width: CGFloat(dimensions.width),
                               height: CGFloat(dimensions.height),
                               length: 0.01,
                               chamferRadius: 0)
        let frontMaterial = SCNMaterial()
        configureMaterialForUSDZExport(frontMaterial, texture: frontWallTexture)
        frontWall.materials = [frontMaterial]
        
        let frontNode = SCNNode(geometry: frontWall)
        frontNode.position = SCNVector3(0, Float(dimensions.height) / 2, -Float(dimensions.depth) / 2)
        roomNode.addChildNode(frontNode)
        logDebug("✅ FRONT WALL with photo at z=-\(dimensions.depth/2)")
        
        // BACK WALL - REMOVED so camera can be positioned outside looking in
        // The camera will be placed at MAX Z looking toward FRONT wall (MIN Z)
        logDebug("⏭️ BACK WALL SKIPPED - camera will view from outside")
        _ = backWallColor // Suppress unused variable warning
        
        // LEFT WALL
        logDebug("🔨 Creating LEFT WALL...")
        let leftWall = SCNBox(width: 0.01,
                              height: CGFloat(dimensions.height),
                              length: CGFloat(dimensions.depth),
                              chamferRadius: 0)
        let leftMaterial = SCNMaterial()
        configureMaterialForUSDZExport(leftMaterial, texture: leftWallTexture)
        leftWall.materials = [leftMaterial]
        
        let leftNode = SCNNode(geometry: leftWall)
        leftNode.position = SCNVector3(-Float(dimensions.width) / 2, Float(dimensions.height) / 2, 0)
        roomNode.addChildNode(leftNode)
        logDebug("✅ LEFT WALL at x=-\(dimensions.width/2)")
        
        // RIGHT WALL
        logDebug("🔨 Creating RIGHT WALL...")
        let rightWall = SCNBox(width: 0.01,
                               height: CGFloat(dimensions.height),
                               length: CGFloat(dimensions.depth),
                               chamferRadius: 0)
        let rightMaterial = SCNMaterial()
        configureMaterialForUSDZExport(rightMaterial, texture: rightWallTexture)
        rightWall.materials = [rightMaterial]
        
        let rightNode = SCNNode(geometry: rightWall)
        rightNode.position = SCNVector3(Float(dimensions.width) / 2, Float(dimensions.height) / 2, 0)
        roomNode.addChildNode(rightNode)
        logDebug("✅ RIGHT WALL at x=+\(dimensions.width/2)")
        
        scene.rootNode.addChildNode(roomNode)
        
        // CAMERA - Looking at photo wall
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.fieldOfView = 70
        cameraNode.position = SCNVector3(0, Float(dimensions.height) * 0.5, 0)
        cameraNode.look(at: SCNVector3(0, Float(dimensions.height) * 0.5, -Float(dimensions.depth) / 2))
        scene.rootNode.addChildNode(cameraNode)
        logDebug("✅ Camera looking at photo wall")
        
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
        logDebug("✅ Room scene created successfully in memory")
        return scene
    }
    
    // MARK: - Texture Generation
    private func generateFloorTexture(from image: UIImage, structure: RoomStructure) -> UIImage {
        logDebug("🎨 [TextureGen] Generating floor texture")
        
        guard let cgImage = image.cgImage else {
            logDebug("⚠️ [TextureGen] No CGImage, using solid color")
            return createSolidColorTexture(color: UIColor(white: 0.85, alpha: 1.0))
        }
        
        let imageHeight = CGFloat(cgImage.height)
        let imageWidth = CGFloat(cgImage.width)
        
        // ✅ Use adjusted boundaries - LEFT to RIGHT, FLOOR to BOTTOM
        let leftXPos = structure.leftX * imageWidth
        let rightXPos = structure.rightX * imageWidth
        let floorYPos = structure.floorY * imageHeight
        
        let cropRect = CGRect(
            x: leftXPos,
            y: floorYPos,
            width: rightXPos - leftXPos,
            height: imageHeight - floorYPos
        )
        
        logDebug("   - Boundaries: L:\(structure.leftX) R:\(structure.rightX) F:\(structure.floorY)")
        logDebug("   - Crop rect: \(cropRect)")
        
        if let croppedImage = cgImage.cropping(to: cropRect) {
            logDebug("✅ [TextureGen] Floor texture extracted from boundaries")
            return UIImage(cgImage: croppedImage)
        }
        
        logDebug("⚠️ [TextureGen] Failed to crop, using solid color")
        return createSolidColorTexture(color: UIColor(white: 0.85, alpha: 1.0))
    }
    
    private func generateWallTexture(from image: UIImage) -> UIImage {
        logDebug("🎨 [TextureGen] Generating wall texture")
        
        guard let cgImage = image.cgImage else {
            logDebug("⚠️ [TextureGen] No CGImage, using solid color")
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
        
        logDebug("   - Sampling center region for average color")
        logDebug("   - Sample rect: \(centerRect)")
        
        let averageColor = ciImage.averageColor(in: centerRect) ?? UIColor(white: 0.9, alpha: 1.0)
        logDebug("✅ [TextureGen] Wall color sampled")
        return createSolidColorTexture(color: averageColor)
    }
    
    // ✅ NEW: Extract CEILING texture from adjusted boundaries
    private func generateCeilingTexture(from image: UIImage, structure: RoomStructure) -> UIImage {
        logDebug("🎨 [TextureGen] Generating CEILING texture from boundaries")
        
        guard let cgImage = image.cgImage else {
            logDebug("⚠️ [TextureGen] No CGImage, using white")
            return createSolidColorTexture(color: .white)
        }
        
        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)
        
        // Extract the ceiling region BETWEEN boundaries, from TOP to CEILING line
        let leftXPos = structure.leftX * imageWidth
        let rightXPos = structure.rightX * imageWidth
        let ceilingYPos = structure.ceilingY * imageHeight
        
        let cropRect = CGRect(
            x: leftXPos,
            y: 0,
            width: rightXPos - leftXPos,
            height: ceilingYPos
        )
        
        logDebug("   - Boundaries: L:\(structure.leftX) R:\(structure.rightX) C:\(structure.ceilingY)")
        logDebug("   - Crop rect: \(cropRect)")
        
        if let croppedImage = cgImage.cropping(to: cropRect) {
            logDebug("✅ [TextureGen] Ceiling texture extracted from boundaries")
            return UIImage(cgImage: croppedImage)
        }
        
        logDebug("⚠️ [TextureGen] Failed to crop, using white")
        return createSolidColorTexture(color: .white)
    }
    
    // ✅ NEW: Extract FRONT wall texture within adjusted boundaries
    private func generateFrontWallTexture(from image: UIImage, structure: RoomStructure) -> UIImage {
        logDebug("🎨 [TextureGen] Generating FRONT wall texture from boundaries")
        
        guard let cgImage = image.cgImage else {
            logDebug("⚠️ [TextureGen] No CGImage, using original")
            return image
        }
        
        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)
        
        // Extract the rectangle WITHIN the boundaries
        let leftXPos = structure.leftX * imageWidth
        let rightXPos = structure.rightX * imageWidth
        let ceilingYPos = structure.ceilingY * imageHeight
        let floorYPos = structure.floorY * imageHeight
        
        let cropRect = CGRect(
            x: leftXPos,
            y: ceilingYPos,
            width: rightXPos - leftXPos,
            height: floorYPos - ceilingYPos
        )
        
        logDebug("   - Boundaries: L:\(structure.leftX) R:\(structure.rightX) C:\(structure.ceilingY) F:\(structure.floorY)")
        logDebug("   - Crop rect: \(cropRect)")
        
        if let croppedImage = cgImage.cropping(to: cropRect) {
            logDebug("✅ [TextureGen] Front wall texture extracted from boundaries")
            return UIImage(cgImage: croppedImage)
        }
        
        logDebug("⚠️ [TextureGen] Failed to crop, using original")
        return image
    }
    
    // ✅ NEW: Extract LEFT wall texture from adjusted boundary
    private func generateLeftWallTexture(from image: UIImage, structure: RoomStructure) -> UIImage {
        logDebug("🎨 [TextureGen] Generating LEFT wall texture from boundary")
        
        guard let cgImage = image.cgImage else {
            logDebug("⚠️ [TextureGen] No CGImage, using solid color")
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
        
        logDebug("   - Left boundary: \(structure.leftX)")
        logDebug("   - Crop rect: \(cropRect)")
        
        if let croppedImage = cgImage.cropping(to: cropRect) {
            logDebug("✅ [TextureGen] Left wall texture extracted")
            return UIImage(cgImage: croppedImage)
        }
        
        logDebug("⚠️ [TextureGen] Failed to crop, using solid color")
        return createSolidColorTexture(color: UIColor(white: 0.9, alpha: 1.0))
    }
    
    // ✅ NEW: Extract RIGHT wall texture from adjusted boundary
    private func generateRightWallTexture(from image: UIImage, structure: RoomStructure) -> UIImage {
        logDebug("🎨 [TextureGen] Generating RIGHT wall texture from boundary")
        
        guard let cgImage = image.cgImage else {
            logDebug("⚠️ [TextureGen] No CGImage, using solid color")
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
        
        logDebug("   - Right boundary: \(structure.rightX)")
        logDebug("   - Crop rect: \(cropRect)")
        
        if let croppedImage = cgImage.cropping(to: cropRect) {
            logDebug("✅ [TextureGen] Right wall texture extracted")
            return UIImage(cgImage: croppedImage)
        }
        
        logDebug("⚠️ [TextureGen] Failed to crop, using solid color")
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
