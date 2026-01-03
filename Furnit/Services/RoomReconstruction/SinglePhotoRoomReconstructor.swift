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
    @Published var statusMessage = "Ready"
    @Published var estimatedDimensions: RoomDimensions?
    @Published var generatedRoomScene: SCNScene? // ✅ CHANGED: from URL to SCNScene
    @Published var splatCount: Int = 0  // Track splat count

    private let depthEstimator = MiDaSDepthEstimator()
    private let roomAnalyzer = RoomStructureAnalyzer()
    private let textureProcessor = TextureProcessor()
    private let splatService = SHARPSplatService()  // Gaussian splat service

    // Enable/disable splat generation
    var enableSplats: Bool = true
    var splatDepthThreshold: Float = 0.7  // Keep objects in front 70% of depth range
    
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
        print("🏗️🏗️🏗️ [SinglePhotoRoomReconstructor] INIT CALLED 🏗️🏗️🏗️")
        print("   splatService.isModelLoaded = \(splatService.isModelLoaded)")
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
            statusMessage = "Analyzing photo..."
        }
        
        // ✅ Fix image orientation FIRST
        let orientedImage = image.fixedOrientation()
        logDebug("✅ [Reconstructor] Image orientation fixed")

        // ✅ OPTIMIZATION: Downscale large images to prevent memory crashes
        let fixedImage = downscaleIfNeeded(orientedImage)

        // Step 1: Generate depth map
        await updateProgress(0.2, "Extracting depth information...")
        logDebug("🔍 [Reconstructor] Step 1: Starting depth estimation")
        
        guard let depthMap = await depthEstimator.estimateDepth(from: fixedImage) else{
            await setError("Failed to estimate depth")
            return
        }
        logDebug("✅ [Reconstructor] Step 1: Depth map created - extent: \(depthMap.extent)")
        
        // Step 2: Detect room structure
        await updateProgress(0.4, "Finding walls and corners...")
        logDebug("🔍 [Reconstructor] Step 2: Starting room structure analysis")
        let roomStructure = await roomAnalyzer.analyzeRoom(image: fixedImage, depthMap: depthMap)
        logDebug("✅ [Reconstructor] Step 2: Room structure analyzed")
        logDebug("   - Wall lines found: \(roomStructure.wallLines.count)")
        logDebug("   - Floor region: \(roomStructure.floorRegion?.debugDescription ?? "nil")")
        logDebug("   - Ceiling region: \(roomStructure.ceilingRegion?.debugDescription ?? "nil")")
        
        // Step 3: Estimate dimensions
        await updateProgress(0.6, "Calculating room dimensions...")
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
        await updateProgress(0.8, "Building 3D model...")
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
            self.statusMessage = "Room ready!"
        }
        
        logDebug("🎉 [Reconstructor] ========== PHOTO PROCESSING COMPLETE ==========")
    }

    
    // ✅ NEW: Process with Adjusted Boundaries
    func processPhotoWithBoundaries(_ image: UIImage, boundaries: RoomStructure) async {
        print("🚀🚀🚀 [Reconstructor] processPhotoWithBoundaries CALLED 🚀🚀🚀")
        logDebug("🚀 [Reconstructor] ========== PROCESSING WITH ADJUSTED BOUNDARIES ==========")
        logDebug("   Floor: \(boundaries.floorY), Ceiling: \(boundaries.ceilingY)")
        logDebug("   Left: \(boundaries.leftX), Right: \(boundaries.rightX)")
        logDebug("   VP: (\(boundaries.vanishingX), \(boundaries.vanishingY))")

        await MainActor.run {
            isProcessing = true
            progress = 0.0
            statusMessage = "Rebuilding with adjusted boundaries..."
        }

        // Wrap in autoreleasepool to release original 4284x5712 image immediately after downscaling
        let fixedImage: UIImage = autoreleasepool {
            // Fix image orientation FIRST
            let orientedImage = image.fixedOrientation()
            logDebug("✅ [Reconstructor] Image orientation fixed")

            // ✅ OPTIMIZATION: Downscale large images - releases original after this
            return downscaleIfNeeded(orientedImage)
        }
        // Original image and orientedImage are now released

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
        
        logDebug("✅ [Reconstructor] Using adjusted boundaries:")
        logDebug("   - Floor region: \(roomStructure.floorRegion!)")
        logDebug("   - Ceiling region: \(roomStructure.ceilingRegion!)")
        
        await updateProgress(0.6, "Calculating dimensions with boundaries...")
        let dimensions = await estimateDimensions(from: roomStructure, image: fixedImage)
        
        await MainActor.run {
            self.estimatedDimensions = dimensions
        }
        
        await updateProgress(0.7, "Building 3D model...")
        let roomScene = await build3DRoom(
            dimensions: dimensions,
            structure: roomStructure,
            originalImage: fixedImage,
            depthMap: depthMap
        )

        // Add Gaussian splats for foreground objects
        print("🔷🔷🔷 [Reconstructor] Checking splats: enableSplats=\(enableSplats), modelLoaded=\(splatService.isModelLoaded), isLoading=\(splatService.isModelLoading) 🔷🔷🔷")
        if let scene = roomScene, enableSplats {
            // Wait for model to finish loading if still in progress
            if splatService.isModelLoading {
                print("⏳ [Reconstructor] SHARP model still loading, waiting...")
                await updateProgress(0.8, "Loading SHARP model...")
                // Wait up to 60 seconds for model to load
                for _ in 0..<120 {
                    try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 sec
                    if !splatService.isModelLoading { break }
                }
            }

            if splatService.isModelLoaded {
                print("🔷🔷🔷 [Reconstructor] Adding splats to scene... 🔷🔷🔷")
                await updateProgress(0.85, "Adding detail with Gaussian splats...")
                await addSplatsToScene(scene, image: fixedImage, boundaries: boundaries, dimensions: dimensions)
            } else {
                print("⚠️⚠️⚠️ [Reconstructor] SHARP model NOT loaded - skipping splats ⚠️⚠️⚠️")
            }
        }

        await MainActor.run {
            self.generatedRoomScene = roomScene
            self.isProcessing = false
            self.progress = 1.0
            self.statusMessage = splatCount > 0 ? "Room ready with \(splatCount) splats!" : "Room rebuilt!"
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

    // MARK: - Add Gaussian Splats to Scene
    /// Adds foreground Gaussian splats to an existing room scene
    func addSplatsToScene(_ scene: SCNScene, image: UIImage, boundaries: RoomStructure, dimensions: RoomDimensions) async {
        guard enableSplats else {
            logDebug("⏭️ [Reconstructor] Splats disabled, skipping")
            return
        }

        logDebug("🔷 [Reconstructor] Adding Gaussian splats for foreground objects...")

        // Generate foreground splats
        let splats = await splatService.generateForegroundSplats(
            from: image,
            boundaries: boundaries,
            depthThreshold: splatDepthThreshold
        )

        guard !splats.isEmpty else {
            logDebug("⚠️ [Reconstructor] No foreground splats generated")
            return
        }

        // Create splat geometry and add to scene
        let roomDims = (width: dimensions.width, depth: dimensions.depth, height: dimensions.height)

        // Use point cloud for efficiency with many splats
        if let splatNode = splatService.createPointCloudGeometry(from: splats, roomDimensions: roomDims) {
            // Position splats relative to room center
            splatNode.position = SCNVector3(0, 0, 0)

            // Add to the room node (first child of root)
            if let roomNode = scene.rootNode.childNodes.first {
                roomNode.addChildNode(splatNode)
            } else {
                scene.rootNode.addChildNode(splatNode)
            }

            await MainActor.run {
                self.splatCount = splats.count
            }

            logDebug("✅ [Reconstructor] Added \(splats.count) splats to scene")
        }
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

// MARK: - SHARP Gaussian Splat Service
/// Generates Gaussian splats from a single photo using the SHARP CoreML model
class SHARPSplatService: ObservableObject {
    @Published var isProcessing = false
    @Published var progress: Float = 0.0
    @Published var statusMessage = "Ready"
    @Published var splatCount: Int = 0

    private var sharpModel: MLModel?
    private let maxImageDimension: CGFloat = 1536  // Original size - CVPixelBuffer path is memory efficient

    struct GaussianSplat {
        let position: SIMD3<Float>
        let scale: SIMD3<Float>
        let quaternion: SIMD4<Float>
        let color: SIMD3<Float>
        let opacity: Float
        let depth: Float
    }

    @Published var isModelLoading = false

    init() {
        print("🔷🔷🔷 [SHARPSplatService] INIT CALLED 🔷🔷🔷")
        // Load model in background thread
        loadModelAsync()
    }

    private func loadModelAsync() {
        isModelLoading = true
        print("🔷🔷🔷 [SHARPSplatService] Starting background model load... 🔷🔷🔷")

        Task.detached(priority: .userInitiated) { [weak self] in
            await self?.loadModel()
        }
    }

    private func loadModel() async {
        print("🔷🔷🔷 [SHARPSplatService] loadModel() called on background thread 🔷🔷🔷")

        // Try different model names - prioritize FP16 version (50% smaller, less memory)
        let modelNames = ["SHARP_fp16", "SHARP_image_input", "SHARP 2", "SHARP", "SHARP_2"]

        for modelName in modelNames {
            print("🔷 [SHARPSplatService] Checking for model: \(modelName)...")

            // Try compiled model first
            if let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") {
                print("   ✅ Found \(modelName).mlmodelc in bundle")
                print("   URL: \(modelURL)")

                // Try CPU first (most reliable), then GPU, then ANE
                let computeOptions: [MLComputeUnits] = [.cpuOnly, .cpuAndGPU, .all]

                for computeUnit in computeOptions {
                    print("   Trying computeUnits: \(computeUnit)...")
                    let startTime = CFAbsoluteTimeGetCurrent()
                    do {
                        let config = MLModelConfiguration()
                        config.computeUnits = computeUnit
                        print("   Loading model (this can take 30-60 seconds for large models)...")
                        let model = try MLModel(contentsOf: modelURL, configuration: config)
                        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                        print("   ⏱️ Model load took \(String(format: "%.1f", elapsed)) seconds")
                        await MainActor.run {
                            self.sharpModel = model
                            self.isModelLoading = false
                        }
                        print("✅✅✅ [SHARPSplatService] SHARP model loaded with \(computeUnit) ✅✅✅")
                        return
                    } catch {
                        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                        print("❌ [SHARPSplatService] Failed with \(computeUnit) after \(String(format: "%.1f", elapsed))s: \(error)")
                    }
                }
            }

            // Try mlpackage
            if let packageURL = Bundle.main.url(forResource: modelName, withExtension: "mlpackage") {
                print("   ✅ Found \(modelName).mlpackage - compiling (this may take a while)...")
                do {
                    let startTime = CFAbsoluteTimeGetCurrent()
                    let compiledURL = try await MLModel.compileModel(at: packageURL)
                    print("   ⏱️ Compilation took \(CFAbsoluteTimeGetCurrent() - startTime) seconds")

                    let config = MLModelConfiguration()
                    config.computeUnits = .all  // Use ANE
                    let model = try MLModel(contentsOf: compiledURL, configuration: config)
                    await MainActor.run {
                        self.sharpModel = model
                        self.isModelLoading = false
                    }
                    print("✅✅✅ [SHARPSplatService] SHARP model compiled and loaded ✅✅✅")
                    return
                } catch {
                    print("❌ [SHARPSplatService] Failed to compile \(modelName): \(error)")
                }
            } else {
                print("   ❌ \(modelName).mlpackage not in bundle")
            }
        }

        await MainActor.run {
            self.isModelLoading = false
        }
        print("❌❌❌ [SHARPSplatService] SHARP model NOT FOUND! Tried: \(modelNames) ❌❌❌")
    }

    var isModelLoaded: Bool { sharpModel != nil }

    func generateForegroundSplats(from image: UIImage, boundaries: RoomStructure, depthThreshold: Float = 0.7) async -> [GaussianSplat] {
        print("🔷🔷🔷 [SHARPSplatService] generateForegroundSplats called 🔷🔷🔷")

        guard sharpModel != nil else {
            print("❌❌❌ [SHARPSplatService] Model not loaded - cannot generate splats ❌❌❌")
            return []
        }

        await MainActor.run {
            isProcessing = true
            progress = 0.0
        }

        let resizedImage = resizeImage(image, to: CGSize(width: maxImageDimension, height: maxImageDimension))

        await MainActor.run { progress = 0.2 }

        guard let allSplats = await runInference(on: resizedImage) else {
            await MainActor.run { isProcessing = false }
            return []
        }

        print("🔷 [SHARPSplatService] Generated \(allSplats.count) total splats")

        let foregroundSplats = filterForegroundSplats(splats: allSplats, depthThreshold: depthThreshold)
        print("🔷 [SHARPSplatService] Kept \(foregroundSplats.count) foreground splats")

        await MainActor.run {
            progress = 1.0
            splatCount = foregroundSplats.count
            isProcessing = false
        }

        return foregroundSplats
    }

    private func runInference(on image: UIImage) async -> [GaussianSplat]? {
        guard let model = sharpModel else {
            print("❌ [SHARPSplatService] Model not loaded")
            return nil
        }

        // Get model description
        let modelDescription = model.modelDescription
        print("🔷 [SHARPSplatService] Model inputs:")
        for (name, desc) in modelDescription.inputDescriptionsByName {
            print("   - \(name): \(desc.type), constraint: \(String(describing: desc.multiArrayConstraint))")
        }

        // Find input name and check type
        guard let inputName = modelDescription.inputDescriptionsByName.keys.first,
              let inputDesc = modelDescription.inputDescriptionsByName[inputName] else {
            print("❌ [SHARPSplatService] No input features found")
            return nil
        }

        do {
            // Wrap in autoreleasepool to release CGImage and temp buffers immediately
            let inputValue: MLFeatureValue? = autoreleasepool {
                if inputDesc.type == .multiArray {
                    // Model expects MultiArray - convert image to array
                    guard let multiArray = image.toMLMultiArray(size: Int(maxImageDimension)) else {
                        print("❌ [SHARPSplatService] Failed to convert image to MLMultiArray")
                        return nil
                    }
                    print("🔷 [SHARPSplatService] Created MLMultiArray shape: \(multiArray.shape)")
                    return MLFeatureValue(multiArray: multiArray)
                } else if inputDesc.type == .image {
                    // Model expects image - use CVPixelBuffer (ANE handles normalization in hardware)
                    print("✅ [SHARPSplatService] Using CVPixelBuffer path - ANE will handle normalization")
                    guard let pixelBuffer = image.toSHARPPixelBuffer(size: CGSize(width: maxImageDimension, height: maxImageDimension)) else {
                        print("❌ [SHARPSplatService] Failed to create pixel buffer")
                        return nil
                    }
                    print("✅ [SHARPSplatService] CVPixelBuffer created: \(CVPixelBufferGetWidth(pixelBuffer))x\(CVPixelBufferGetHeight(pixelBuffer))")
                    return MLFeatureValue(pixelBuffer: pixelBuffer)
                } else {
                    print("❌ [SHARPSplatService] Unsupported input type: \(inputDesc.type)")
                    return nil
                }
            }

            guard let inputValue = inputValue else { return nil }

            print("🔷 [SHARPSplatService] Creating feature provider...")
            let inputFeature = try MLDictionaryFeatureProvider(dictionary: [inputName: inputValue])

            print("🔷 [SHARPSplatService] Starting model.prediction()...")
            print("🔷 [SHARPSplatService] Memory before inference: \(getMemoryUsage()) MB")

            // Run prediction
            let output = try await model.prediction(from: inputFeature)

            print("✅ [SHARPSplatService] Inference completed!")
            print("🔷 [SHARPSplatService] Memory after inference: \(getMemoryUsage()) MB")

            // Parse output in autoreleasepool to free buffers
            return autoreleasepool { parseModelOutput(output) }
        } catch {
            print("❌ [SHARPSplatService] Inference error: \(error)")
            print("❌ [SHARPSplatService] Error type: \(type(of: error))")
            return nil
        }
    }

    private func parseModelOutput(_ output: MLFeatureProvider) -> [GaussianSplat]? {
        // Log all output features
        print("🔷 [SHARPSplatService] Output features: \(output.featureNames)")
        for name in output.featureNames {
            if let feature = output.featureValue(for: name), let array = feature.multiArrayValue {
                let shape = array.shape.map { $0.intValue }
                print("   - \(name): \(shape)")
            }
        }

        // Try new SHARP_fp16 format (separate outputs)
        if let meanVectors = output.featureValue(for: "mean_vectors")?.multiArrayValue,
           let singularValues = output.featureValue(for: "singular_values")?.multiArrayValue,
           let quaternions = output.featureValue(for: "quaternions")?.multiArrayValue,
           let colors = output.featureValue(for: "colors")?.multiArrayValue,
           let opacities = output.featureValue(for: "opacities")?.multiArrayValue {
            print("✅ [SHARPSplatService] Parsing SHARP_fp16 format (separate outputs)")
            return parseSeparateOutputs(meanVectors: meanVectors, singularValues: singularValues,
                                        quaternions: quaternions, colors: colors, opacities: opacities)
        }

        // Try alternative output names (var_XXXX format from CoreML)
        // CoreML scrambles output order, so identify by shape:
        // - [N, 4] = quaternions
        // - [N, 1] or [N] = opacities
        // - [N, 3] = positions, scales, colors
        let featureNames = Array(output.featureNames)
        if featureNames.count >= 5 {
            let arrays = featureNames.compactMap { output.featureValue(for: $0)?.multiArrayValue }
            if arrays.count >= 5 {
                print("✅ [SHARPSplatService] Identifying outputs by shape...")

                var quaternions: MLMultiArray?
                var opacities: MLMultiArray?
                var threeComponentArrays: [MLMultiArray] = []

                for (i, arr) in arrays.enumerated() {
                    let shape = arr.shape.map { $0.intValue }
                    let lastDim = shape.last ?? 0
                    print("   [\(i)] shape=\(shape), lastDim=\(lastDim)")

                    if shape.count == 2 || (shape.count == 3 && lastDim == 1) {
                        // [batch, N] or [batch, N, 1] = opacities
                        opacities = arr
                    } else if lastDim == 4 {
                        quaternions = arr
                    } else if lastDim == 3 {
                        threeComponentArrays.append(arr)
                    }
                }

                guard let quat = quaternions, let opac = opacities, threeComponentArrays.count >= 3 else {
                    print("❌ Could not identify all required outputs")
                    return nil
                }

                // Identify which array is which by looking at value ranges:
                // - Colors (SH): can be negative, range roughly -3 to +3
                // - Positions: normalized 0 to 1
                // - Scales: very small values (log space or raw)
                // Sample first element of each to determine order
                var posArray: MLMultiArray = threeComponentArrays[0]
                var scaleArray: MLMultiArray = threeComponentArrays[1]
                var colorArray: MLMultiArray = threeComponentArrays[2]

                // Check value ranges to identify arrays
                for (idx, arr) in threeComponentArrays.enumerated() {
                    let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
                    let v0 = Float(ptr[0])
                    let v1 = Float(ptr[1])
                    let v2 = Float(ptr[2])
                    print("   Array[\(idx)] sample: (\(v0), \(v1), \(v2))")

                    // Colors have negative values (SH coefficients)
                    if v0 < -0.5 || v1 < -0.5 || v2 < -0.5 {
                        colorArray = arr
                        print("     → Identified as COLORS (has negative values)")
                    }
                    // Positions are in 0-1 range (normalized)
                    else if v0 > 0.1 && v0 < 1.0 && v1 > 0.1 && v1 < 1.0 {
                        posArray = arr
                        print("     → Identified as POSITIONS (0-1 range)")
                    }
                    // Scales are very small
                    else if abs(v0) < 0.1 && abs(v1) < 0.1 {
                        scaleArray = arr
                        print("     → Identified as SCALES (small values)")
                    }
                }

                print("✅ [SHARPSplatService] Parsing with auto-identified outputs")
                return parseSeparateOutputs(meanVectors: posArray,
                                            singularValues: scaleArray,
                                            quaternions: quat,
                                            colors: colorArray,
                                            opacities: opac)
            }
        }

        // Fallback: try old combined format
        for name in output.featureNames {
            if let feature = output.featureValue(for: name), let array = feature.multiArrayValue {
                let shape = array.shape.map { $0.intValue }
                if shape.count >= 2 && shape[1] >= 14 {
                    print("✅ [SHARPSplatService] Parsing combined format")
                    return parseCombinedOutput(array)
                }
            }
        }

        print("❌ [SHARPSplatService] Could not parse model output")
        return nil
    }

    private func parseSeparateOutputs(meanVectors: MLMultiArray, singularValues: MLMultiArray,
                                      quaternions: MLMultiArray, colors: MLMultiArray,
                                      opacities: MLMultiArray) -> [GaussianSplat]? {
        let posShape = meanVectors.shape.map { $0.intValue }

        print("🔷 [SHARPSplatService] Array data types:")
        print("   positions: \(meanVectors.dataType.rawValue) (0=double, 1=float32, 2=int32, 65552=float16)")
        print("   scales: \(singularValues.dataType.rawValue)")
        print("   quaternions: \(quaternions.dataType.rawValue)")
        print("   colors: \(colors.dataType.rawValue)")
        print("   opacities: \(opacities.dataType.rawValue)")

        // Shape could be [batch, num_splats, channels] or [num_splats, channels]
        let numSplats: Int
        if posShape.count == 3 {
            numSplats = posShape[1]
        } else if posShape.count == 2 {
            numSplats = posShape[0]
        } else {
            print("❌ Unexpected positions shape: \(posShape)")
            return nil
        }

        print("🔷 [SHARPSplatService] Parsing \(numSplats) splats...")

        var splats: [GaussianSplat] = []
        let maxSplats = min(numSplats, 50000)
        splats.reserveCapacity(maxSplats)

        // Check if Float16 (dataType.rawValue == 65552)
        let isFloat16 = meanVectors.dataType.rawValue == 65552

        // Sigmoid function for color activation
        func sigmoid(_ x: Float) -> Float {
            return 1.0 / (1.0 + exp(-x))
        }

        if isFloat16 {
            print("✅ [SHARPSplatService] Reading Float16 data")

            // Get strides for proper data access - MLMultiArray may have padding!
            let posStrides = meanVectors.strides.map { $0.intValue }
            let scaleStrides = singularValues.strides.map { $0.intValue }
            let quatStrides = quaternions.strides.map { $0.intValue }
            let colorStrides = colors.strides.map { $0.intValue }
            let opacStrides = opacities.strides.map { $0.intValue }

            // For [1, N, C] arrays, stride[1] is the step between splats
            let posStep = posStrides.count >= 2 ? posStrides[1] : 3
            let scaleStep = scaleStrides.count >= 2 ? scaleStrides[1] : 3
            let quatStep = quatStrides.count >= 2 ? quatStrides[1] : 4
            let colorStep = colorStrides.count >= 2 ? colorStrides[1] : 3
            let opacStep = opacStrides.count >= 2 ? opacStrides[1] : 1

            print("   Strides - pos:\(posStep) scale:\(scaleStep) quat:\(quatStep) color:\(colorStep) opac:\(opacStep)")

            let meansPtr = meanVectors.dataPointer.assumingMemoryBound(to: Float16.self)
            let scalesPtr = singularValues.dataPointer.assumingMemoryBound(to: Float16.self)
            let quatsPtr = quaternions.dataPointer.assumingMemoryBound(to: Float16.self)
            let colorsPtr = colors.dataPointer.assumingMemoryBound(to: Float16.self)
            let opacPtr = opacities.dataPointer.assumingMemoryBound(to: Float16.self)

            // Log sample values from first splat (using correct strides)
            print("🔷 [SHARPSplatService] Sample values (splat 0):")
            print("   pos: (\(Float(meansPtr[0])), \(Float(meansPtr[1])), \(Float(meansPtr[2])))")
            print("   scale: (\(Float(scalesPtr[0])), \(Float(scalesPtr[1])), \(Float(scalesPtr[2])))")
            print("   quat: (\(Float(quatsPtr[0])), \(Float(quatsPtr[1])), \(Float(quatsPtr[2])), \(Float(quatsPtr[3])))")
            let rawColor = SIMD3<Float>(Float(colorsPtr[0]), Float(colorsPtr[1]), Float(colorsPtr[2]))
            print("   color (raw): \(rawColor)")
            // SHARP uses sigmoid activation for colors
            let rgbColor = SIMD3<Float>(sigmoid(rawColor.x), sigmoid(rawColor.y), sigmoid(rawColor.z))
            print("   color (sigmoid RGB): \(rgbColor)")
            print("   opacity: \(Float(opacPtr[0]))")

            for i in 0..<maxSplats {
                // Use actual strides to access data (handles padding)
                let posIdx = i * posStep
                let scaleIdx = i * scaleStep
                let quatIdx = i * quatStep
                let colorIdx = i * colorStep
                let opacIdx = i * opacStep

                let position = SIMD3<Float>(Float(meansPtr[posIdx]), Float(meansPtr[posIdx + 1]), Float(meansPtr[posIdx + 2]))
                let scale = SIMD3<Float>(Float(scalesPtr[scaleIdx]), Float(scalesPtr[scaleIdx + 1]), Float(scalesPtr[scaleIdx + 2]))
                let quaternion = SIMD4<Float>(Float(quatsPtr[quatIdx]), Float(quatsPtr[quatIdx + 1]), Float(quatsPtr[quatIdx + 2]), Float(quatsPtr[quatIdx + 3]))
                // SHARP outputs raw logits - apply sigmoid for RGB
                let rawR = Float(colorsPtr[colorIdx])
                let rawG = Float(colorsPtr[colorIdx + 1])
                let rawB = Float(colorsPtr[colorIdx + 2])
                let color = SIMD3<Float>(sigmoid(rawR), sigmoid(rawG), sigmoid(rawB))
                let opacity = Float(opacPtr[opacIdx])

                splats.append(GaussianSplat(
                    position: position,
                    scale: scale,
                    quaternion: quaternion,
                    color: color,
                    opacity: opacity,
                    depth: position.z
                ))
            }
        } else {
            print("✅ [SHARPSplatService] Reading Float32 data")
            let meansPtr = meanVectors.dataPointer.assumingMemoryBound(to: Float.self)
            let scalesPtr = singularValues.dataPointer.assumingMemoryBound(to: Float.self)
            let quatsPtr = quaternions.dataPointer.assumingMemoryBound(to: Float.self)
            let colorsPtr = colors.dataPointer.assumingMemoryBound(to: Float.self)
            let opacPtr = opacities.dataPointer.assumingMemoryBound(to: Float.self)

            // Log sample values from first splat
            print("🔷 [SHARPSplatService] Sample values (splat 0):")
            print("   pos: (\(meansPtr[0]), \(meansPtr[1]), \(meansPtr[2]))")
            print("   scale: (\(scalesPtr[0]), \(scalesPtr[1]), \(scalesPtr[2]))")
            print("   quat: (\(quatsPtr[0]), \(quatsPtr[1]), \(quatsPtr[2]), \(quatsPtr[3]))")
            print("   color: (\(colorsPtr[0]), \(colorsPtr[1]), \(colorsPtr[2]))")
            print("   opacity: \(opacPtr[0])")

            for i in 0..<maxSplats {
                let position = SIMD3<Float>(meansPtr[i * 3], meansPtr[i * 3 + 1], meansPtr[i * 3 + 2])
                let scale = SIMD3<Float>(scalesPtr[i * 3], scalesPtr[i * 3 + 1], scalesPtr[i * 3 + 2])
                let quaternion = SIMD4<Float>(quatsPtr[i * 4], quatsPtr[i * 4 + 1], quatsPtr[i * 4 + 2], quatsPtr[i * 4 + 3])
                let color = SIMD3<Float>(colorsPtr[i * 3], colorsPtr[i * 3 + 1], colorsPtr[i * 3 + 2])
                let opacity = opacPtr[i]

                splats.append(GaussianSplat(
                    position: position,
                    scale: scale,
                    quaternion: quaternion,
                    color: color,
                    opacity: opacity,
                    depth: position.z
                ))
            }
        }

        // Log value ranges
        let positions = splats.map { $0.position }
        let splatOpacities = splats.map { $0.opacity }
        let splatColors = splats.map { $0.color }
        print("🔷 [SHARPSplatService] Value ranges (first \(maxSplats)):")
        print("   X: [\(positions.map{$0.x}.min()!), \(positions.map{$0.x}.max()!)]")
        print("   Y: [\(positions.map{$0.y}.min()!), \(positions.map{$0.y}.max()!)]")
        print("   Z: [\(positions.map{$0.z}.min()!), \(positions.map{$0.z}.max()!)]")
        print("   Opacity: [\(splatOpacities.min()!), \(splatOpacities.max()!)]")
        print("   Color R: [\(splatColors.map{$0.x}.min()!), \(splatColors.map{$0.x}.max()!)]")

        print("✅ [SHARPSplatService] Parsed \(splats.count) splats")
        return splats
    }

    private func parseCombinedOutput(_ combined: MLMultiArray) -> [GaussianSplat]? {
        let shape = combined.shape.map { $0.intValue }
        guard shape.count >= 2 else { return nil }

        let numSplats = shape[0]
        let channels = shape[1]
        guard channels >= 14 else { return nil }

        var splats: [GaussianSplat] = []
        splats.reserveCapacity(numSplats)

        let pointer = combined.dataPointer.assumingMemoryBound(to: Float.self)
        let sh0Coeff = Float(sqrt(1.0 / (4.0 * .pi)))

        for i in 0..<numSplats {
            let offset = i * channels
            let position = SIMD3<Float>(pointer[offset], pointer[offset + 1], pointer[offset + 2])
            let scale = SIMD3<Float>(exp(pointer[offset + 3]), exp(pointer[offset + 4]), exp(pointer[offset + 5]))
            let quaternion = SIMD4<Float>(pointer[offset + 6], pointer[offset + 7], pointer[offset + 8], pointer[offset + 9])
            let color = SIMD3<Float>(
                pointer[offset + 10] * sh0Coeff + 0.5,
                pointer[offset + 11] * sh0Coeff + 0.5,
                pointer[offset + 12] * sh0Coeff + 0.5
            )
            let opacity = 1.0 / (1.0 + exp(-pointer[offset + 13]))

            splats.append(GaussianSplat(position: position, scale: scale, quaternion: quaternion, color: color, opacity: opacity, depth: position.z))
        }

        return splats
    }

    private func filterForegroundSplats(splats: [GaussianSplat], depthThreshold: Float) -> [GaussianSplat] {
        guard !splats.isEmpty else { return [] }

        // No filtering - boundaries already define the room structure
        // Just return top splats by opacity for performance (limit to 20000)
        let sorted = splats.sorted { $0.opacity > $1.opacity }
        let result = Array(sorted.prefix(20000))

        print("🔷 [SHARPSplatService] Returning \(result.count) splats (top by opacity)")
        return result
    }

    func createPointCloudGeometry(from splats: [GaussianSplat], roomDimensions: (width: Float, depth: Float, height: Float)) -> SCNNode? {
        guard !splats.isEmpty else { return nil }

        // Find actual position ranges from splats
        let xs = splats.map { $0.position.x }
        let ys = splats.map { $0.position.y }
        let zs = splats.map { $0.position.z }
        let minX = xs.min() ?? 0, maxX = xs.max() ?? 1
        let minY = ys.min() ?? 0, maxY = ys.max() ?? 1
        let minZ = zs.min() ?? 0, maxZ = zs.max() ?? 1

        let rangeX = max(maxX - minX, 0.001)
        let rangeY = max(maxY - minY, 0.001)
        let rangeZ = max(maxZ - minZ, 0.001)

        print("🔷 [SHARPSplatService] Position ranges for scaling:")
        print("   X: [\(minX), \(maxX)] range=\(rangeX)")
        print("   Y: [\(minY), \(maxY)] range=\(rangeY)")
        print("   Z: [\(minZ), \(maxZ)] range=\(rangeZ)")

        var vertices: [SCNVector3] = []
        var colors: [SCNVector4] = []

        for splat in splats {
            guard splat.opacity > 0.005 else { continue }

            // Normalize positions to [0, 1] then map to room dimensions
            let normX = (splat.position.x - minX) / rangeX  // 0 to 1
            let normY = (splat.position.y - minY) / rangeY  // 0 to 1
            let normZ = (splat.position.z - minZ) / rangeZ  // 0 to 1

            // Map to room: X centered, Y from floor up, Z from front to back
            let worldX = (normX - 0.5) * roomDimensions.width   // -width/2 to +width/2
            let worldY = normY * roomDimensions.height           // 0 to height
            let worldZ = (normZ - 0.5) * roomDimensions.depth   // -depth/2 to +depth/2

            vertices.append(SCNVector3(worldX, worldY, worldZ))

            // Ensure colors are in valid range
            let r = max(0, min(1, splat.color.x))
            let g = max(0, min(1, splat.color.y))
            let b = max(0, min(1, splat.color.z))
            let a = max(0.5, min(1, splat.opacity))  // Minimum 0.5 for visibility
            colors.append(SCNVector4(r, g, b, a))
        }

        guard !vertices.isEmpty else { return nil }

        let vertexSource = SCNGeometrySource(vertices: vertices)
        let colorData = Data(bytes: colors, count: colors.count * MemoryLayout<SCNVector4>.stride)
        let colorSource = SCNGeometrySource(data: colorData, semantic: .color, vectorCount: colors.count, usesFloatComponents: true, componentsPerVector: 4, bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0, dataStride: MemoryLayout<SCNVector4>.stride)

        let indices: [Int32] = Array(0..<Int32(vertices.count))
        let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
        let element = SCNGeometryElement(data: indexData, primitiveType: .point, primitiveCount: vertices.count, bytesPerIndex: MemoryLayout<Int32>.size)
        element.pointSize = 3.0
        element.minimumPointScreenSpaceRadius = 1.0
        element.maximumPointScreenSpaceRadius = 10.0

        let geometry = SCNGeometry(sources: [vertexSource, colorSource], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.isDoubleSided = true
        geometry.materials = [material]

        let node = SCNNode(geometry: geometry)
        node.name = "GaussianSplatPointCloud"
        print("✅ [SHARPSplatService] Created point cloud with \(vertices.count) points")
        return node
    }

    private func getMemoryUsage() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.resident_size / 1024 / 1024) : 0
    }

    private func resizeImage(_ image: UIImage, to size: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext() ?? image
        UIGraphicsEndImageContext()
        return resized
    }
}

// MARK: - UIImage Extension for SHARP Pixel Buffer
extension UIImage {
    /// Create CVPixelBuffer for CoreML image input (ANE handles normalization)
    func toSHARPPixelBuffer(size: CGSize) -> CVPixelBuffer? {
        // Use autoreleasepool to release CGImage immediately
        return autoreleasepool {
            guard let cgImage = self.cgImage else { return nil }

            // Use 32BGRA - CoreML image input handles color conversion
            let attrs: [CFString: Any] = [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                kCVPixelBufferMetalCompatibilityKey: true  // Enable Metal/ANE acceleration
            ]

            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(
                kCFAllocatorDefault,
                Int(size.width),
                Int(size.height),
                kCVPixelFormatType_32BGRA,
                attrs as CFDictionary,
                &pixelBuffer
            )
            guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

            CVPixelBufferLockBaseAddress(buffer, [])
            defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

            guard let context = CGContext(
                data: CVPixelBufferGetBaseAddress(buffer),
                width: Int(size.width),
                height: Int(size.height),
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            ) else { return nil }

            context.draw(cgImage, in: CGRect(origin: .zero, size: size))
            return buffer
        }
    }

    /// Convert UIImage to MLMultiArray for SHARP model input using Accelerate/vDSP/CBLAS
    /// Shape: [1, 3, size, size] - batch, channels (RGB), height, width
    func toMLMultiArray(size: Int) -> MLMultiArray? {
        // Use autoreleasepool to release CGImage immediately after use
        return autoreleasepool {
            guard let cgImage = self.cgImage else { return nil }
            return convertCGImageToMLMultiArray(cgImage, size: size)
        }
    }

    private func convertCGImageToMLMultiArray(_ cgImage: CGImage, size: Int) -> MLMultiArray? {
        // Use vImage for efficient resizing
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
        guard error == kvImageNoError else { return nil }

        // Create destination buffer at target size
        var destBuffer = vImage_Buffer()
        error = vImageBuffer_Init(&destBuffer, vImagePixelCount(size), vImagePixelCount(size), 32, vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else {
            free(sourceBuffer.data)
            return nil
        }
        defer { free(destBuffer.data) }

        // Scale image then immediately free source to reduce peak memory
        error = vImageScale_ARGB8888(&sourceBuffer, &destBuffer, nil, vImage_Flags(kvImageHighQualityResampling))
        free(sourceBuffer.data)  // Free immediately after scaling - no longer needed
        guard error == kvImageNoError else { return nil }

        // Create MLMultiArray with Float16 to reduce memory (half the size of Float32)
        guard let multiArray = try? MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float16) else {
            return nil
        }

        let pixelCount = size * size
        let srcPtr = destBuffer.data.assumingMemoryBound(to: UInt8.self)

        // For Float16, we need to convert through Float32 temp buffer (process in chunks to save memory)
        let chunkSize = 65536  // Process 64K pixels at a time
        var tempFloat = [Float](repeating: 0, count: min(chunkSize, pixelCount))
        let dstPtr = multiArray.dataPointer.assumingMemoryBound(to: Float16.self)
        let scale: Float = 1.0 / 255.0

        // Process each channel in chunks
        for channel in 0..<3 {
            let channelOffset = channel * pixelCount
            var processed = 0

            while processed < pixelCount {
                let remaining = pixelCount - processed
                let currentChunk = min(chunkSize, remaining)

                // Convert UInt8 to Float32
                vDSP_vfltu8(srcPtr + channel + (processed * 4), 4, &tempFloat, 1, vDSP_Length(currentChunk))

                // Normalize in-place
                cblas_sscal(Int32(currentChunk), scale, &tempFloat, 1)

                // Convert Float32 to Float16 and write to destination
                for i in 0..<currentChunk {
                    dstPtr[channelOffset + processed + i] = Float16(tempFloat[i])
                }

                processed += currentChunk
            }
        }

        return multiArray
    }
}
