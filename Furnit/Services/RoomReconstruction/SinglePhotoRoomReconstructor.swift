import SwiftUI
import CoreML
import Vision
import CoreImage
import SceneKit
import Accelerate
import AVFoundation

// MARK: - Single Photo Room Reconstructor
class SinglePhotoRoomReconstructor: ObservableObject {
    @Published var isProcessing = false
    @Published var progress: Float = 0.0
    @Published var statusMessage = "Ready"
    @Published var estimatedDimensions: RoomDimensions?
    @Published var generatedRoomScene: SCNScene? // ✅ CHANGED: from URL to SCNScene
    @Published var splatCount: Int = 0  // Track splat count
    @Published var rawSplats: [GaussianSplat] = []  // Furniture-band splats for mask generation
    @Published var rawSplatsFullRoom: [GaussianSplat] = []  // Full-room splats for splat viewer

    // MARK: - Room Textures (for Metal splat renderer)
    @Published var floorTexture: UIImage?
    @Published var ceilingTexture: UIImage?
    @Published var frontWallTexture: UIImage?
    @Published var leftWallTexture: UIImage?
    @Published var rightWallTexture: UIImage?

    // MARK: - Dependencies
    private let depthEstimator = DepthEstimator()
    private let roomAnalyzer = RoomStructureAnalyzer()
    private let dimensionEstimator = DimensionEstimator()
    private let sceneBuilder = RoomSceneBuilder()

    // MARK: - Room Dimensions
    struct RoomDimensions {
        var width: Float = 4.0
        var depth: Float = 4.5
        var height: Float = 2.8
        var confidence: Float = 0.3
    }

    // MARK: - Room Build Mode
    enum RoomBuildMode {
        case fullPlanes           // Layer-1: all 5 planes (floor, ceiling, front, left, right)
        case frontWallPlusSplats  // Hybrid: only front wall plane, SHARP handles the rest
    }

    // MARK: - Surface Classification
    enum SurfaceID: UInt8 {
        case none = 0       // Unclassified / outlier
        case floor = 1
        case ceiling = 2
        case frontWall = 3
        case leftWall = 4
        case rightWall = 5
    }

    // MARK: - Gaussian Splat Representation
    struct GaussianSplat {
        var position: SIMD3<Float>      // Normalized [0, 1] in SHARP space
        var scale: SIMD3<Float>         // Anisotropic scale
        var quaternion: SIMD4<Float>    // Orientation as quaternion
        var color: SIMD3<Float>         // Linear RGB in [0, 1]
        var opacity: Float              // [0, 1]
        var surfaceId: SurfaceID = .none  // Which room surface this splat belongs to
    }

    // MARK: - SHARP Splat Service
    private let splatService = SHARPSplatService()

    // MARK: - Image Downscaling (Performance Critical)
    /// Downscale very large images to avoid huge intermediate buffers.
    private func downscaleIfNeeded(_ image: UIImage, maxImageDimension: CGFloat = 1600) -> UIImage {
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
        error = vImageBuffer_Init(
            &destBuffer,
            UInt(newHeight),
            UInt(newWidth),
            format.bitsPerPixel,
            vImage_Flags(kvImageNoFlags)
        )
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

    // MARK: - Image Preprocessing
    /// Fixes image orientation and downsizes very large images inside an autoreleasepool
    /// to promptly release temporary CoreGraphics buffers.
    private func preprocessImage(_ image: UIImage) -> UIImage {
        return autoreleasepool {
            let orientedImage = image.fixedOrientation()
            logDebug("✅ [Reconstructor] Image orientation fixed")
            return downscaleIfNeeded(orientedImage)
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

        // ✅ Fix orientation + downscale with scoped autoreleasepool
        let fixedImage = preprocessImage(image)

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
            depthMap: depthMap,
            mode: .fullPlanes  // ✅ Build floor + 4 walls + ceiling
        )

        logDebug("✅ [Reconstructor] Step 4: 3D room built (SceneKit scene created)")

        if let roomScene = roomScene {
            // 🚫 DISABLED: Don't render splats as balls - just use textured planes
            /*
            if splatService.isModelLoaded && !splatService.isModelLoading {
                await updateProgress(0.9, "Adding detail with Gaussian splats...")
                await addSplatsToScene(roomScene, image: fixedImage, boundaries: roomStructure, dimensions: dimensions)
            } else {
                logDebug("⚠️ [Reconstructor] SHARP model not loaded yet, skipping splats for now")
            }
            */
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

        // ✅ Fix orientation + downscale with scoped autoreleasepool
        let fixedImage = preprocessImage(image)
        // Original image and orientedImage are now released

        await updateProgress(0.2, "Extracting depth information...")
        guard let depthMap = await depthEstimator.estimateDepth(from: fixedImage) else {
            await setError("Failed to estimate depth")
            return
        }

        await updateProgress(0.4, "Using your adjusted boundaries...")
        logDebug("✅ [Reconstructor] Using adjusted boundaries:")
        logDebug("   Floor: \(boundaries.floorY), Ceiling: \(boundaries.ceilingY)")
        logDebug("   Left: \(boundaries.leftX), Right: \(boundaries.rightX)")

        // Use the adjusted boundaries instead of detecting new ones
        var roomStructure = boundaries
        roomStructure.floorRegion = CGRect(x: 0, y: boundaries.floorY, width: 1.0, height: 1.0 - boundaries.floorY)
        roomStructure.ceilingRegion = CGRect(x: 0, y: 0, width: 1.0, height: boundaries.ceilingY)
        roomStructure.vanishingPoint = CGPoint(x: boundaries.vanishingX, y: boundaries.vanishingY)

        logDebug("✅ [Reconstructor] Using adjusted boundaries:")
        logDebug("   - Floor region: \(roomStructure.floorRegion!)")
        logDebug("   - Ceiling region: \(roomStructure.ceilingRegion!)")

        await updateProgress(0.5, "Calculating dimensions with boundaries...")
        let dimensions = await estimateDimensions(from: roomStructure, image: fixedImage)

        await MainActor.run {
            self.estimatedDimensions = dimensions
        }

        // ✅ Run SHARP to generate full-room splats (furniture band DISABLED)
        let foregroundMask: CGImage? = nil  // 🔕 DISABLED: no furniture removal
        var splatsForScene: [GaussianSplat] = []

        if splatService.isModelLoaded && !splatService.isModelLoading {
            await updateProgress(0.55, "Generating room splats...")
            logDebug("🔷 [Reconstructor] Running SHARP (furniture band DISABLED)...")

            // 🔕 DISABLED: furniture band splats for mask generation
            // splatsForScene = await splatService.generateForegroundSplats(
            //     from: fixedImage,
            //     boundaries: boundaries,
            //     roomDimensions: (width: dimensions.width, depth: dimensions.depth, height: dimensions.height),
            //     mode: .furnitureBand
            // )

            // Generate full-room splats for the splat viewer (this is the main cloud)
            let fullRoomSplats = await splatService.generateForegroundSplats(
                from: fixedImage,
                boundaries: boundaries,
                roomDimensions: (width: dimensions.width, depth: dimensions.depth, height: dimensions.height),
                mode: .fullRoomCloud
            )

            await MainActor.run {
                self.splatCount = fullRoomSplats.count
                self.rawSplats = []  // 🔕 No furniture-band splats
                self.rawSplatsFullRoom = fullRoomSplats  // Full room for viewer
            }
            logDebug("🔷 [Reconstructor] Generated \(fullRoomSplats.count) full-room splats (furniture band disabled)")

            // 🔕 DISABLED: furniture mask generation
            // if !splatsForScene.isEmpty {
            //     await updateProgress(0.6, "Creating furniture mask...")
            //     foregroundMask = splatService.generateForegroundMask(
            //         from: splatsForScene,
            //         imageSize: fixedImage.size,
            //         boundaries: boundaries
            //     )
            //     logDebug("🔷 [Reconstructor] Foreground mask generated: \(foregroundMask != nil)")
            // }
        } else {
            logDebug("⚠️ [Reconstructor] SHARP model not loaded - no splats")
        }

        await updateProgress(0.7, "Building 3D model...")
        let roomScene = await build3DRoom(
            dimensions: dimensions,
            structure: roomStructure,
            originalImage: fixedImage,
            depthMap: depthMap,
            foregroundMask: foregroundMask,  // ✅ Pass SHARP mask for texture inpainting
            mode: .fullPlanes                // ✅ Build floor + 4 walls + ceiling
        )

        if let scene = roomScene {
            logDebug("✅ [Reconstructor] Room scene created successfully in memory")

            // 🚫 DISABLED: Don't render splats as balls - just use textured planes
            // Splats are only used for mask generation, not visual rendering
            /*
            if !splatsForScene.isEmpty {
                await updateProgress(0.85, "Adding Gaussian splats...")
                await addSplatsToSceneFromCache(scene, splats: splatsForScene, boundaries: boundaries, dimensions: dimensions)
            }
            */
        }

        await MainActor.run {
            self.generatedRoomScene = roomScene
            self.isProcessing = false
            self.progress = 1.0
            self.statusMessage = splatCount > 0 ? "Room ready with \(splatCount) splats!" : "Room rebuilt!"
        }

        logDebug("🎉 [Reconstructor] ========== BOUNDARY PROCESSING COMPLETE ==========")
    }

    // Helper to add pre-generated splats to scene
    private func addSplatsToSceneFromCache(_ scene: SCNScene, splats: [GaussianSplat], boundaries: RoomStructure, dimensions: RoomDimensions) async {
        logDebug("🔷 [Reconstructor] Adding \(splats.count) cached splats to scene...")

        let width = dimensions.width
        let height = dimensions.height
        let depth = dimensions.depth

        if let roomRoot = scene.rootNode.childNode(withName: "RoomRoot", recursively: false) {
            if let splatNode = splatService.createPointCloudGeometry(from: splats, roomDimensions: (width, depth, height), boundaries: boundaries) {
                splatNode.name = "SHARPSplats"
                roomRoot.addChildNode(splatNode)
                logDebug("✅ [Reconstructor] Added \(splats.count) splats to scene")

                // DEBUG: Log RoomRoot children hierarchy
                logDebug("🔍 [DEBUG] RoomRoot children count: \(roomRoot.childNodes.count)")
                for child in roomRoot.childNodes {
                    let childCount = child.childNodes.count
                    logDebug("   - \(child.name ?? "<no name>") (\(childCount) children)")
                }
            } else {
                logDebug("⚠️ [Reconstructor] Failed to create point cloud node from splats")
            }
        } else {
            logDebug("⚠️ [Reconstructor] RoomRoot node not found - cannot attach splats")
        }
    }

    // MARK: - Dimension Estimation
    private func estimateDimensions(from structure: RoomStructure, image: UIImage) async -> RoomDimensions {
        logDebug("📏 [DimensionEstimator] Estimating room dimensions")

        var dimensions = RoomDimensions()

        // Use detected door to estimate scale if available
        if let doorRect = dimensionEstimator.detectDoor(in: image) {
            logDebug("🚪 [DimensionEstimator] Using door detection for scale")
            let doorPixelHeight = doorRect.height * image.size.height
            let pixelsPerMeter = doorPixelHeight / 2.0 // Assume 2m tall door

            logDebug("   - Door pixel height: \(doorPixelHeight)")
            logDebug("   - Pixels per meter: \(pixelsPerMeter)")

            dimensions.height = Float(image.size.height / pixelsPerMeter * 0.8)
            dimensions.width = Float(image.size.width / pixelsPerMeter * 1.5)
            dimensions.depth = dimensions.width * 1.1
            dimensions.confidence = 0.7

            logDebug("   - Calculated height: \(dimensions.height)m")
            logDebug("   - Calculated width: \(dimensions.width)m")
            logDebug("   - Calculated depth: \(dimensions.depth)m")
            logDebug("   - Confidence: 70%")

        } else if let personRect = dimensionEstimator.detectPerson(in: image) {
            logDebug("👤 [DimensionEstimator] Using person detection for scale")
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
            // Fall back to defaults when no reliable reference is found
            logDebug("⚠️ [DimensionEstimator] NO REFERENCE FOUND - Using defaults")
            dimensions.width = 4.0
            dimensions.depth = 4.5
            dimensions.height = 2.8
            dimensions.confidence = 0.3

            logDebug("   - Default width: 4.0m")
            logDebug("   - Default depth: 4.5m")
            logDebug("   - Default height: 2.8m")
            logDebug("   - Confidence: 30%")
        }

        logDebug("📏 [DimensionEstimator] Final dimensions: W:\(dimensions.width)m D:\(dimensions.depth)m H:\(dimensions.height)m")
        return dimensions
    }

    // MARK: - 3D Room Construction

    /// Inpaint furniture from the entire image using the foreground mask.
    /// Returns a "clean" image with furniture regions filled from nearby pixels.
    private func inpaintFurnitureOnImage(_ image: UIImage, mask: CGImage) -> UIImage {
        logDebug("🎨 [Inpaint] Removing furniture from full image...")
        logDebug("🧪 [Inpaint] Incoming mask size: \(mask.width)x\(mask.height)")

        guard let cgImage = image.cgImage else {
            logDebug("⚠️ [Inpaint] No CGImage, returning original")
            return image
        }

        let width = cgImage.width
        let height = cgImage.height
        logDebug("🧪 [Inpaint] Target image size: \(width)x\(height)")

        // Scale mask to match image size
        guard let scaledMask = scaleImage(mask, to: CGSize(width: width, height: height)) else {
            logDebug("⚠️ [Inpaint] Failed to scale mask, returning original")
            return image
        }
        logDebug("🧪 [Inpaint] Scaled mask size: \(scaledMask.width)x\(scaledMask.height)")

        // Create context for the result
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            logDebug("⚠️ [Inpaint] Failed to create context, returning original")
            return image
        }

        // Draw the original image
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Create mask context
        guard let maskContext = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            logDebug("⚠️ [Inpaint] Failed to create mask context, returning original")
            return image
        }
        maskContext.draw(scaledMask, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let imageData = context.data, let maskData = maskContext.data else {
            logDebug("⚠️ [Inpaint] Failed to get data pointers")
            return image
        }

        let imagePtr = imageData.bindMemory(to: UInt8.self, capacity: width * height * 4)
        let maskPtr = maskData.bindMemory(to: UInt8.self, capacity: width * height)

        // Count non-zero pixels in the mask AFTER scaling and drawing
        var maskNonZero = 0
        var maskAboveThreshold = 0
        let foregroundThreshold: UInt8 = 32
        for i in 0..<(width * height) {
            if maskPtr[i] > 0 { maskNonZero += 1 }
            if maskPtr[i] >= foregroundThreshold { maskAboveThreshold += 1 }
        }
        logDebug("🧪 [Inpaint] Mask in context: \(maskNonZero) non-zero, \(maskAboveThreshold) above threshold \(foregroundThreshold)")

        // Simple horizontal inpaint: for masked pixels, copy from nearest unmasked pixel to the left
        var inpaintedCount = 0

        for y in 0..<height {
            var lastGoodR: UInt8? = nil
            var lastGoodG: UInt8? = nil
            var lastGoodB: UInt8? = nil

            for x in 0..<width {
                let maskIdx = y * width + x
                let imgIdx = maskIdx * 4
                let maskValue = maskPtr[maskIdx]

                if maskValue < foregroundThreshold {
                    // Background pixel - save as "good" color
                    lastGoodR = imagePtr[imgIdx + 0]
                    lastGoodG = imagePtr[imgIdx + 1]
                    lastGoodB = imagePtr[imgIdx + 2]
                } else {
                    // Foreground pixel - replace with last good color if we have one
                    if let goodR = lastGoodR, let goodG = lastGoodG, let goodB = lastGoodB {
                        imagePtr[imgIdx + 0] = goodR
                        imagePtr[imgIdx + 1] = goodG
                        imagePtr[imgIdx + 2] = goodB
                        inpaintedCount += 1
                    }
                }
            }
        }

        logDebug("✅ [Inpaint] Inpainted \(inpaintedCount) pixels from full image")

        guard let result = context.makeImage() else {
            return image
        }

        return UIImage(cgImage: result)
    }

    private func build3DRoom(
        dimensions: RoomDimensions,
        structure: RoomStructure,
        originalImage: UIImage,
        depthMap: CIImage,
        foregroundMask: CGImage? = nil,
        mode: RoomBuildMode = .fullPlanes
    ) async -> SCNScene? {
        logDebug("🏗️ [RoomBuilder] Starting TEXTURED room construction")
        logDebug("   - Mode: \(mode == .fullPlanes ? "fullPlanes (5 walls)" : "frontWallPlusSplats (front wall only + SHARP)")")
        logDebug("   - Dimensions: W:\(dimensions.width) D:\(dimensions.depth) H:\(dimensions.height)")
        logDebug("   - Foreground mask: \(foregroundMask != nil ? "provided (furniture will be removed from ALL surfaces)" : "none")")

        let scene = SCNScene()
        let roomNode = SCNNode()
        roomNode.name = "RoomRoot"
        scene.rootNode.addChildNode(roomNode)

        // Create a "clean" image with furniture removed FIRST
        // This is used for ALL wall/floor/ceiling textures
        let cleanImage: UIImage
        if let mask = foregroundMask {
            logDebug("🎨 [RoomBuilder] Creating furniture-free base image...")
            cleanImage = inpaintFurnitureOnImage(originalImage, mask: mask)
        } else {
            logDebug("⚠️ [RoomBuilder] No mask, using original image for textures")
            cleanImage = originalImage
        }

        // Generate textures from the CLEAN image (furniture already removed)
        logDebug("🎨 [RoomBuilder] Generating textures from clean image...")

        let floorTex = generateFloorTexture(from: cleanImage, structure: structure)
        let ceilingTex = generateCeilingTexture(from: cleanImage, structure: structure)
        let frontWallTex = generateFrontWallTexture(from: cleanImage, structure: structure, foregroundMask: nil)  // No mask needed, already clean
        let leftWallTex = generateLeftWallTexture(from: cleanImage, structure: structure)
        let rightWallTex = generateRightWallTexture(from: cleanImage, structure: structure)

        // Store textures for Metal splat renderer
        await MainActor.run {
            self.floorTexture = floorTex
            self.ceilingTexture = ceilingTex
            self.frontWallTexture = frontWallTex
            self.leftWallTexture = leftWallTex
            self.rightWallTexture = rightWallTex
            logDebug("🎨 [RoomBuilder] Stored textures for Metal renderer")
        }

        let wallColor = sampleWallColor(from: originalImage) ?? UIColor(white: 0.92, alpha: 1.0)

        // FLOOR - With texture from photo (skip in frontWallPlusSplats mode)
        if mode == .fullPlanes {
            logDebug("🔨 Creating FLOOR with texture...")
            let floor = SCNBox(width: CGFloat(dimensions.width),
                               height: 0.01,
                               length: CGFloat(dimensions.depth),
                               chamferRadius: 0)
            let floorMaterial = SCNMaterial()
            configureMaterialForUSDZExport(floorMaterial, texture: floorTex)
            floor.materials = [floorMaterial]

            let floorNode = SCNNode(geometry: floor)
            floorNode.position = SCNVector3(0, 0, 0)
            roomNode.addChildNode(floorNode)
            logDebug("✅ FLOOR at y=0")
        }

        // CEILING - With texture from photo (skip in frontWallPlusSplats mode)
        if mode == .fullPlanes {
            logDebug("🔨 Creating CEILING...")
            let ceiling = SCNBox(width: CGFloat(dimensions.width),
                                 height: 0.01,
                                 length: CGFloat(dimensions.depth),
                                 chamferRadius: 0)
            let ceilingMaterial = SCNMaterial()
            configureMaterialForUSDZExport(ceilingMaterial, texture: ceilingTex)
            ceiling.materials = [ceilingMaterial]

            let ceilingNode = SCNNode(geometry: ceiling)
            ceilingNode.position = SCNVector3(0, dimensions.height, 0)
            roomNode.addChildNode(ceilingNode)
            logDebug("✅ CEILING at y=\(dimensions.height)")
        }

        // FRONT WALL - With YOUR PHOTO texture
        logDebug("🔨 Creating FRONT WALL with photo texture...")
        let frontWall = SCNBox(width: CGFloat(dimensions.width),
                               height: CGFloat(dimensions.height),
                               length: 0.01,
                               chamferRadius: 0)
        let frontMaterial = SCNMaterial()
        configureMaterialForUSDZExport(frontMaterial, texture: frontWallTex)
        frontWall.materials = [frontMaterial]

        let frontNode = SCNNode(geometry: frontWall)
        frontNode.position = SCNVector3(0, Float(dimensions.height) / 2, -Float(dimensions.depth) / 2)
        roomNode.addChildNode(frontNode)
        logDebug("✅ FRONT WALL with photo at z=-\(dimensions.depth/2)")

        // LEFT WALL - With texture from left strip of image (skip in frontWallPlusSplats mode)
        if mode == .fullPlanes {
            logDebug("🔨 Creating LEFT WALL...")
            let leftWall = SCNBox(width: CGFloat(dimensions.depth),
                                  height: CGFloat(dimensions.height),
                                  length: 0.01,
                                  chamferRadius: 0)
            let leftMaterial = SCNMaterial()
            configureMaterialForUSDZExport(leftMaterial, texture: leftWallTex)
            leftWall.materials = [leftMaterial]

            let leftWallNode = SCNNode(geometry: leftWall)
            leftWallNode.position = SCNVector3(-Float(dimensions.width) / 2, Float(dimensions.height) / 2, 0)
            leftWallNode.eulerAngles = SCNVector3(0, Float.pi / 2, 0)
            roomNode.addChildNode(leftWallNode)
            logDebug("✅ LEFT WALL at x=-\(dimensions.width/2)")
        }

        // RIGHT WALL - With texture from right strip of image (skip in frontWallPlusSplats mode)
        if mode == .fullPlanes {
            logDebug("🔨 Creating RIGHT WALL...")
            let rightWall = SCNBox(width: CGFloat(dimensions.depth),
                                   height: CGFloat(dimensions.height),
                                   length: 0.01,
                                   chamferRadius: 0)
            let rightMaterial = SCNMaterial()
            configureMaterialForUSDZExport(rightMaterial, texture: rightWallTex)
            rightWall.materials = [rightMaterial]

            let rightWallNode = SCNNode(geometry: rightWall)
            rightWallNode.position = SCNVector3(Float(dimensions.width) / 2, Float(dimensions.height) / 2, 0)
            rightWallNode.eulerAngles = SCNVector3(0, -Float.pi / 2, 0)
            roomNode.addChildNode(rightWallNode)
            logDebug("✅ RIGHT WALL at x=+\(dimensions.width/2)")
        }

        // Basic lighting so room is visible in SceneKit
        let ambientLight = SCNLight()
        ambientLight.type = .ambient
        ambientLight.color = UIColor(white: 0.4, alpha: 1.0)
        let ambientNode = SCNNode()
        ambientNode.light = ambientLight
        roomNode.addChildNode(ambientNode)

        let directionalLight = SCNLight()
        directionalLight.type = .directional
        directionalLight.intensity = 1000
        let directionalNode = SCNNode()
        directionalNode.light = directionalLight
        directionalNode.position = SCNVector3(0, dimensions.height, Float(dimensions.depth))
        directionalNode.eulerAngles = SCNVector3(-Float.pi / 4, 0, 0)
        roomNode.addChildNode(directionalNode)

        logDebug("✅ Room scene created successfully in memory")
        return scene
    }

    // MARK: - Add Gaussian Splats to Scene
    func addSplatsToScene(_ scene: SCNScene, image: UIImage, boundaries: RoomStructure, dimensions: RoomDimensions) async {
        logDebug("🔷 [Reconstructor] Adding Gaussian splats for foreground objects...")

        // Convert image + boundaries + dimensions to SHARP’s normalized coordinates
        let height = dimensions.height
        let width = dimensions.width
        let depth = dimensions.depth

        // Run SHARP model to generate splats
        let splats = await splatService.generateForegroundSplats(
            from: image,
            boundaries: boundaries,
            roomDimensions: (width: width, depth: depth, height: height)
        )

        await MainActor.run {
            self.splatCount = splats.count
        }

        guard !splats.isEmpty else {
            logDebug("⚠️ [Reconstructor] No splats returned from SHARP - skipping")
            return
        }

        // Create a node containing all splats and add to the room root
        if let roomRoot = scene.rootNode.childNode(withName: "RoomRoot", recursively: false) {
            if let splatNode = splatService.createPointCloudGeometry(from: splats, roomDimensions: (width, depth, height), boundaries: boundaries) {
                splatNode.name = "SHARPSplats"
                roomRoot.addChildNode(splatNode)
                logDebug("✅ [Reconstructor] Added \(splats.count) splats to scene")

                // DEBUG: Log RoomRoot children hierarchy
                logDebug("🔍 [DEBUG] RoomRoot children count: \(roomRoot.childNodes.count)")
                for child in roomRoot.childNodes {
                    let childCount = child.childNodes.count
                    logDebug("   - \(child.name ?? "<no name>") (\(childCount) children)")
                }
            } else {
                logDebug("⚠️ [Reconstructor] Failed to create point cloud node from splats")
            }
        } else {
            logDebug("⚠️ [Reconstructor] RoomRoot node not found - cannot attach splats")
        }
    }
}

// MARK: - Dimension Estimator & Room Scene Builder
// (These are placeholders for your existing implementations.)

class DimensionEstimator {
    func detectDoor(in image: UIImage) -> CGRect? {
        logDebug("🚪 [DoorDetector] Starting door detection")

        guard let cgImage = image.cgImage else { return nil }

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
        guard let rectangles = request.results, !rectangles.isEmpty else {
            logDebug("⚠️ [DoorDetector] No rectangles detected")
            return nil
        }

        logDebug("✅ [DoorDetector] Found \(rectangles.count) rectangle(s)")
        return rectangles.first!.boundingBox
    }

    func detectPerson(in image: UIImage) -> CGRect? {
        logDebug("👤 [PersonDetector] Starting person detection")

        guard let cgImage = image.cgImage else { return nil }

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
}

class RoomSceneBuilder {
    // Placeholder if you had a separate builder; most of the logic is now in build3DRoom.
}

// MARK: - Room Structure Model
struct RoomStructure: Equatable {
    var wallLines: [Line] = []
    var floorRegion: CGRect?
    var ceilingRegion: CGRect?
    var vanishingPoint: CGPoint?
    var floorY: CGFloat = 0.75
    var ceilingY: CGFloat = 0.25
    var leftX: CGFloat = 0.1
    var rightX: CGFloat = 0.9
    var vanishingX: CGFloat = 0.5
    var vanishingY: CGFloat = 0.45
}

struct Line: Equatable {
    var start: CGPoint
    var end: CGPoint
}

// MARK: - Room Structure Analyzer
class RoomStructureAnalyzer {
    func analyzeRoom(image: UIImage, depthMap: CIImage?) async -> RoomStructure {
        // For now, we keep your manual boundary inputs as primary.
        // This can be extended later to use depth/edges to auto-detect.
        return RoomStructure()
    }
}

// MARK: - Depth Estimator (MiDaS or fallback)
class DepthEstimator {
    private var isInitialized = false

    init() {
        logDebug("🧠 [DepthEstimator] Initializing")
        // Attempt to load MiDaS (or any depth model you have)
        logDebug("📦 [DepthEstimator] Attempting to load MiDaS model")
        // For now, assume model is not available -> synthetic depth
        logDebug("⚠️ [DepthEstimator] MiDaS model not available, will use fallback")
    }

    func estimateDepth(from image: UIImage) async -> CIImage? {
        logDebug("🔬 [DepthEstimator] Estimating depth from image")
        // Fallback: synthetic depth map with simple radial gradient
        logDebug("🎨 [DepthEstimator] Generating synthetic depth map")

        guard let cgImage = image.cgImage else { return nil }

        let width = cgImage.width
        let height = cgImage.height
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitsPerComponent = 8
        let bytesPerRow = width
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            return nil
        }

        let centerX = CGFloat(width) / 2.0
        let centerY = CGFloat(height) / 2.0
        let maxDistance = hypot(centerX, centerY)

        for y in 0..<height {
            for x in 0..<width {
                let dx = CGFloat(x) - centerX
                let dy = CGFloat(y) - centerY
                let distance = hypot(dx, dy)
                let depth = UInt8((1.0 - min(distance / maxDistance, 1.0)) * 255.0)
                context.data?.storeBytes(of: depth, toByteOffset: y * bytesPerRow + x, as: UInt8.self)
            }
        }

        guard let depthCGImage = context.makeImage() else { return nil }
        let ciImage = CIImage(cgImage: depthCGImage)
        logDebug("✅ [DepthEstimator] Synthetic depth map created")
        return ciImage
    }
}

// MARK: - SHARP output modes

/// Controls how SHARP splats are filtered for different use cases
enum SHARPSplatMode {
    /// Current behaviour: focus on furniture near the floor,
    /// drop curtains / high Y and far-depth "walls" - for mask/inpaint pipeline
    case furnitureBand

    /// Keep *all* room splats (no Y / depth culling) for a pure "room cloud" viewer
    case fullRoomCloud
}

// MARK: - SHARPSplatService
final class SHARPSplatService {
    private(set) var sharpModel: MLModel?
    private(set) var isModelLoaded: Bool = false
    private(set) var isModelLoading: Bool = false

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

            // 1. Try compiled .mlmodelc in bundle
            if let compiledURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") {
                print("   ✅ Found \(modelName).mlmodelc in bundle")
                print("   URL: \(compiledURL)")

                // Try CPU first (most reliable), then GPU, then ANE
                let computeOptions: [MLComputeUnits] = [.cpuOnly, .cpuAndGPU, .all]
                for computeUnit in computeOptions {
                    print("   Trying computeUnits: \(computeUnit)...")
                    do {
                        let config = MLModelConfiguration()
                        config.computeUnits = computeUnit
                        let model = try MLModel(contentsOf: compiledURL, configuration: config)
                        await MainActor.run {
                            self.sharpModel = model
                            self.isModelLoaded = true
                            self.isModelLoading = false
                        }
                        print("✅✅✅ [SHARPSplatService] SHARP model loaded with \(computeUnit) ✅✅✅")
                        return
                    } catch {
                        print("❌ [SHARPSplatService] Failed with \(computeUnit): \(error)")
                    }
                }
            }

            // 2. Try .mlpackage and compile on device
            if let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlpackage") {
                print("   ✅ Found \(modelName).mlpackage in bundle")
                print("   URL: \(modelURL)")

                do {
                    // Compile and load - may take a while
                    let compiledURL = try await MLModel.compileModel(at: modelURL)
                    let config = MLModelConfiguration()
                    config.computeUnits = .all  // Use ANE
                    let model = try MLModel(contentsOf: compiledURL, configuration: config)
                    await MainActor.run {
                        self.sharpModel = model
                        self.isModelLoaded = true
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
            self.isModelLoaded = false
            self.isModelLoading = false
        }
        print("❌❌❌ [SHARPSplatService] No SHARP model could be loaded ❌❌❌")
    }

    // MARK: - Normalization Stats (for consistent UV mapping)

    /// Stores the bounding box of ALL splats for consistent UV normalization
    struct SplatNormalizationStats {
        let minX: Float, maxX: Float
        let minY: Float, maxY: Float
        let minZ: Float, maxZ: Float

        var xRange: Float { max(maxX - minX, 1e-6) }
        var yRange: Float { max(maxY - minY, 1e-6) }
        var zRange: Float { max(maxZ - minZ, 1e-6) }
    }

    /// Cached normalization stats from the most recent SHARP run
    private var cachedNormStats: SplatNormalizationStats?

    // MARK: - Gaussian Splat Generation

    /// Generate Gaussian splats for foreground objects in the room.
    /// This is where we interpret SHARP outputs and map them into room space.
    /// - Parameter mode: `.fullRoomCloud` for pure splat viewer (default), `.furnitureBand` for mask/inpaint
    func generateForegroundSplats(
        from image: UIImage,
        boundaries: RoomStructure,
        roomDimensions: (width: Float, depth: Float, height: Float),
        mode: SHARPSplatMode = .fullRoomCloud,
        depthThreshold: Float = 0.7
    ) async -> [SinglePhotoRoomReconstructor.GaussianSplat] {
        let isFullRoomCloud = (mode == .fullRoomCloud)
        print("🔷🔷🔷 [SHARPSplatService] generateForegroundSplats called (mode=\(mode)) 🔷🔷🔷")

        guard let model = sharpModel else {
            print("❌ [SHARPSplatService] Model not loaded")
            return []
        }

        // SHARP expects square input; we’ll pad/crop to 1536x1536 (as in Apple’s example)
        let maxImageDimension: CGFloat = 1536

        // Use model description to decide input type (multiArray or image)
        let modelDescription = model.modelDescription
        guard let inputName = modelDescription.inputDescriptionsByName.keys.first,
              let inputDesc = modelDescription.inputDescriptionsByName[inputName] else {
            print("❌ [SHARPSplatService] No input features found")
            return []
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
                    guard let pixelBuffer = image.toSHARPPixelBufferNormalized(size: CGSize(width: maxImageDimension, height: maxImageDimension)) else {
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

            guard let inputValue = inputValue else { return [] }

            print("🔷 [SHARPSplatService] Creating feature provider...")
            let inputFeature = try MLDictionaryFeatureProvider(dictionary: [inputName: inputValue])

            print("🔷 [SHARPSplatService] Starting model.prediction()...")
            print("🔷 [SHARPSplatService] Memory before inference: \(getMemoryUsage()) MB")

            // Run prediction
            let output = try await model.prediction(from: inputFeature)

            print("✅ [SHARPSplatService] Inference completed!")
            print("🔷 [SHARPSplatService] Memory after inference: \(getMemoryUsage()) MB")

            // Parse output in autoreleasepool to free buffers
            return autoreleasepool { parseModelOutput(output, roomDimensions: roomDimensions, boundaries: boundaries, isFullRoomCloud: isFullRoomCloud) }

        } catch {
            print("❌ [SHARPSplatService] Error during prediction: \(error)")
            return []
        }
    }

    // MARK: - SHARP Output Parsing

    /// Parses SHARP outputs to Gaussian splats.  This expects the model to produce:
    /// - 1xN x 4  -> quaternions
    /// - 1xN      -> opacities
    /// - 1xN x 3  -> positions
    /// - 1xN x 3  -> scales
    /// - 1xN x 3  -> colors
    private func parseModelOutput(
        _ output: MLFeatureProvider,
        roomDimensions: (width: Float, depth: Float, height: Float),
        boundaries: RoomStructure,
        isFullRoomCloud: Bool
    ) -> [SinglePhotoRoomReconstructor.GaussianSplat] {
        let featureNames = Array(output.featureNames)
        print("🔷 [SHARPSplatService] Output features: \(featureNames)")

        // Collect and identify the five arrays by shape and value ranges
        struct ArrayInfo {
            let name: String
            let multiArray: MLMultiArray
        }

        var arrays: [ArrayInfo] = []

        for name in featureNames {
            guard let value = output.featureValue(for: name), value.type == .multiArray,
                  let array = value.multiArrayValue else { continue }

            let shape = array.shape.map { $0.intValue }
            print("   - \(name): \(shape)")

            arrays.append(ArrayInfo(name: name, multiArray: array))
        }

        guard arrays.count == 5 else {
            print("❌ [SHARPSplatService] Expected 5 output arrays, got \(arrays.count)")
            return []
        }

        var positionsArray: MLMultiArray?
        var scalesArray: MLMultiArray?
        var quaternionsArray: MLMultiArray?
        var colorsArray: MLMultiArray?
        var opacitiesArray: MLMultiArray?

        // SmallCandidate to collect arrays with small values for deferred identification
        struct SmallCandidate {
            let info: ArrayInfo
            let sample: (Float, Float, Float)
            let maxAbs: Float
        }
        var smallCandidates: [SmallCandidate] = []

        for info in arrays {
            let array = info.multiArray
            let shape = array.shape.map { $0.intValue }
            let lastDim = shape.last ?? 0

            if shape.count == 3 && lastDim == 4 {
                print("   → \(info.name) identified as QUATERNIONS")
                quaternionsArray = array
            } else if shape.count == 2 {
                print("   → \(info.name) identified as OPACITIES")
                opacitiesArray = array
            } else if shape.count == 3 && lastDim == 3 {
                // Sample a few elements to distinguish position / scale / color
                let sample = sampleTriple(from: array)
                let maxAbs = max(abs(sample.0), abs(sample.1), abs(sample.2))
                print("   Array \(info.name) sample: \(sample), maxAbs: \(maxAbs)")

                if sample.0 < 0 || sample.1 < 0 || sample.2 < 0 {
                    print("     → Identified as COLORS (has negative values)")
                    colorsArray = array
                } else if maxAbs < 0.05 {
                    // Defer small value decisions - collect as candidate
                    print("     → Small values, deferring identification...")
                    smallCandidates.append(SmallCandidate(info: info, sample: sample, maxAbs: maxAbs))
                } else {
                    print("     → Identified as POSITIONS (0-1 range)")
                    positionsArray = array
                }
            }
        }

        // After loop: sort small candidates by magnitude to identify positions vs scales
        // Larger small values → positions, smaller small values → scales
        if !smallCandidates.isEmpty {
            let sorted = smallCandidates.sorted { $0.maxAbs > $1.maxAbs }
            print("   Sorted small candidates by maxAbs:")
            for (i, c) in sorted.enumerated() {
                print("     [\(i)] \(c.info.name): maxAbs=\(c.maxAbs)")
            }

            if positionsArray == nil && sorted.count >= 1 {
                positionsArray = sorted[0].info.multiArray
                print("     → \(sorted[0].info.name) assigned as POSITIONS (largest small)")
            }
            if scalesArray == nil && sorted.count >= 2 {
                scalesArray = sorted[1].info.multiArray
                print("     → \(sorted[1].info.name) assigned as SCALES (smallest small)")
            } else if scalesArray == nil && sorted.count == 1 && positionsArray != nil {
                // Only one small candidate and positions already set elsewhere
                scalesArray = sorted[0].info.multiArray
                print("     → \(sorted[0].info.name) assigned as SCALES (only small candidate)")
            }
        }

        guard let positions = positionsArray,
              let scales = scalesArray,
              let quaternions = quaternionsArray,
              let colors = colorsArray,
              let opacities = opacitiesArray else {
            print("❌ [SHARPSplatService] Failed to identify all arrays")
            return []
        }

        // Log data types
        print("🔷 [SHARPSplatService] Array data types:")
        print("   positions: \(positions.dataType.rawValue)")
        print("   scales: \(scales.dataType.rawValue)")
        print("   quaternions: \(quaternions.dataType.rawValue)")
        print("   colors: \(colors.dataType.rawValue)")
        print("   opacities: \(opacities.dataType.rawValue)")

        // Convert Float16 to Float and parse N splats
        let count = positions.shape[1].intValue
        print("🔷 [SHARPSplatService] Parsing \(count) splats...")

        let isFloat16: Bool = (positions.dataType == .float16)

        func float16Ptr(_ array: MLMultiArray) -> UnsafePointer<UInt16> {
            return UnsafePointer<UInt16>(OpaquePointer(array.dataPointer))
        }

        func floatPtr(_ array: MLMultiArray) -> UnsafePointer<Float> {
            return UnsafePointer<Float>(OpaquePointer(array.dataPointer))
        }

        let posPtr16 = isFloat16 ? float16Ptr(positions) : nil
        let scalePtr16 = isFloat16 ? float16Ptr(scales) : nil
        let quatPtr16 = isFloat16 ? float16Ptr(quaternions) : nil
        let colorPtr16 = isFloat16 ? float16Ptr(colors) : nil
        let opacPtr16 = isFloat16 ? float16Ptr(opacities) : nil

        let posPtr32 = !isFloat16 ? floatPtr(positions) : nil
        let scalePtr32 = !isFloat16 ? floatPtr(scales) : nil
        let quatPtr32 = !isFloat16 ? floatPtr(quaternions) : nil
        let colorPtr32 = !isFloat16 ? floatPtr(colors) : nil
        let opacPtr32 = !isFloat16 ? floatPtr(opacities) : nil

        // Strides: we saw 32 because of padding, but index by stride[1]
        let posStride = positions.strides[1].intValue
        let scaleStride = scales.strides[1].intValue
        let quatStride = quaternions.strides[1].intValue
        let colorStride = colors.strides[1].intValue
        let opacStride = opacities.strides.count > 1 ? opacities.strides[1].intValue : 1

        print("   Strides - pos:\(posStride) scale:\(scaleStride) quat:\(quatStride) color:\(colorStride) opac:\(opacStride)")

        var splats: [SinglePhotoRoomReconstructor.GaussianSplat] = []
        splats.reserveCapacity(count)   // 🔄 use full count

        func f16(_ v: UInt16) -> Float {
            return Float(half: v)
        }

        var minX: Float = .greatestFiniteMagnitude
        var maxX: Float = -.greatestFiniteMagnitude
        var minY: Float = .greatestFiniteMagnitude
        var maxY: Float = -.greatestFiniteMagnitude
        var minZ: Float = .greatestFiniteMagnitude
        var maxZ: Float = -.greatestFiniteMagnitude

        var minOpacity: Float = 1.0
        var maxOpacity: Float = 0.0

        var minColorR: Float = 1.0
        var maxColorR: Float = 0.0

        // 🔄 process ALL splats, no 50k cap
        let sampleCount = count

        for i in 0..<sampleCount {
            let posIndex = i * posStride
            let scaleIndex = i * scaleStride
            let quatIndex = i * quatStride
            let colorIndex = i * colorStride
            let opacIndex = i * opacStride

            let px: Float
            let py: Float
            let pz: Float
            let sx: Float
            let sy: Float
            let sz: Float
            let qx: Float
            let qy: Float
            let qz: Float
            let qw: Float
            let cr: Float
            let cg: Float
            let cb: Float
            let opacity: Float

            if isFloat16 {
                let pos = posPtr16!
                let sc = scalePtr16!
                let qt = quatPtr16!
                let col = colorPtr16!
                let op = opacPtr16!

                px = f16(pos[posIndex + 0])
                py = f16(pos[posIndex + 1])
                pz = f16(pos[posIndex + 2])

                sx = f16(sc[scaleIndex + 0])
                sy = f16(sc[scaleIndex + 1])
                sz = f16(sc[scaleIndex + 2])

                qx = f16(qt[quatIndex + 0])
                qy = f16(qt[quatIndex + 1])
                qz = f16(qt[quatIndex + 2])
                qw = f16(qt[quatIndex + 3])

                cr = f16(col[colorIndex + 0])
                cg = f16(col[colorIndex + 1])
                cb = f16(col[colorIndex + 2])

                opacity = f16(op[opacIndex])
            } else {
                let pos = posPtr32!
                let sc = scalePtr32!
                let qt = quatPtr32!
                let col = colorPtr32!
                let op = opacPtr32!

                px = pos[posIndex + 0]
                py = pos[posIndex + 1]
                pz = pos[posIndex + 2]

                sx = sc[scaleIndex + 0]
                sy = sc[scaleIndex + 1]
                sz = sc[scaleIndex + 2]

                qx = qt[quatIndex + 0]
                qy = qt[quatIndex + 1]
                qz = qt[quatIndex + 2]
                qw = qt[quatIndex + 3]

                cr = col[colorIndex + 0]
                cg = col[colorIndex + 1]
                cb = col[colorIndex + 2]

                opacity = op[opacIndex]
            }

            if i == 0 {
                let rawColor = SIMD3<Float>(cr, cg, cb)
                let sigmoidColor = SIMD3<Float>(
                    1.0 / (1.0 + exp(-cr)),
                    1.0 / (1.0 + exp(-cg)),
                    1.0 / (1.0 + exp(-cb))
                )
                print("🔷 [SHARPSplatService] Sample values (splat 0):")
                print("   pos: (\(px), \(py), \(pz))")
                print("   scale: (\(sx), \(sy), \(sz))")
                print("   quat: (\(qx), \(qy), \(qz), \(qw))")
                print("   color (raw): \(rawColor)")
                print("   color (sigmoid RGB): \(sigmoidColor)")
                print("   opacity: \(opacity)")
            }

            minX = min(minX, px)
            maxX = max(maxX, px)
            minY = min(minY, py)
            maxY = max(maxY, py)
            minZ = min(minZ, pz)
            maxZ = max(maxZ, pz)

            minOpacity = min(minOpacity, opacity)
            maxOpacity = max(maxOpacity, opacity)

            let sigmoidR = 1.0 / (1.0 + exp(-cr))
            let sigmoidG = 1.0 / (1.0 + exp(-cg))
            let sigmoidB = 1.0 / (1.0 + exp(-cb))

            minColorR = min(minColorR, sigmoidR)
            maxColorR = max(maxColorR, sigmoidR)

            let splat = SinglePhotoRoomReconstructor.GaussianSplat(
                position: SIMD3<Float>(px, py, pz),
                scale: SIMD3<Float>(sx, sy, sz),
                quaternion: SIMD4<Float>(qx, qy, qz, qw),
                color: SIMD3<Float>(sigmoidR, sigmoidG, sigmoidB),
                opacity: opacity
            )

            splats.append(splat)
        }

        print("🔷 [SHARPSplatService] Parsed \(splats.count) splats")
        print("🔷 [SHARPSplatService] Position ranges for scaling:")
        print("   X: [\(minX), \(maxX)] range=\(maxX - minX)")
        print("   Y: [\(minY), \(maxY)] range=\(maxY - minY)")
        print("   Z: [\(minZ), \(maxZ)] range=\(maxZ - minZ)")
        print("   Opacity: [\(minOpacity), \(maxOpacity)]")
        print("   Color R: [\(minColorR), \(maxColorR)]")

        // Cache normalization stats from ALL splats for consistent UV mapping
        cachedNormStats = SplatNormalizationStats(
            minX: minX, maxX: maxX,
            minY: minY, maxY: maxY,
            minZ: minZ, maxZ: maxZ
        )
        print("🔷 [SHARPSplatService] Cached normalization stats for mask generation")

        // Filter using room boundaries and depth-biased scoring
        let filtered = filterForegroundSplats(splats: splats, depthThreshold: 0.7, boundaries: boundaries, isFullRoomCloud: isFullRoomCloud)
        if isFullRoomCloud {
            print("🔷 [SHARPSplatService] Full-room cloud: \(filtered.count) splats")
        } else {
            print("🔷 [SHARPSplatService] Kept \(filtered.count) foreground splats")
        }

        return filtered
    }

    private func sampleTriple(from array: MLMultiArray) -> (Float, Float, Float) {
        let shape = array.shape.map { $0.intValue }
        guard shape.count == 3 else { return (0, 0, 0) }

        let stride = array.strides[1].intValue
        let baseIndex = 0 * stride

        if array.dataType == .float16 {
            let ptr = UnsafePointer<UInt16>(OpaquePointer(array.dataPointer))
            return (
                Float(half: ptr[baseIndex + 0]),
                Float(half: ptr[baseIndex + 1]),
                Float(half: ptr[baseIndex + 2])
            )
        } else {
            let ptr = UnsafePointer<Float>(OpaquePointer(array.dataPointer))
            return (
                ptr[baseIndex + 0],
                ptr[baseIndex + 1],
                ptr[baseIndex + 2]
            )
        }
    }

    private func filterForegroundSplats(
        splats: [SinglePhotoRoomReconstructor.GaussianSplat],
        depthThreshold: Float,
        boundaries: RoomStructure? = nil,
        isFullRoomCloud: Bool = false
    ) -> [SinglePhotoRoomReconstructor.GaussianSplat] {
        guard !splats.isEmpty else { return [] }

        // Compute stats for normalization
        let xs = splats.map { $0.position.x }
        let ys = splats.map { $0.position.y }
        let zs = splats.map { $0.position.z }

        let xMin = xs.min() ?? 0, xMax = xs.max() ?? 1
        let yMin = ys.min() ?? 0, yMax = ys.max() ?? 1
        let zMin = zs.min() ?? 0, zMax = zs.max() ?? 1

        let xRange = max(xMax - xMin, 1e-6)
        let yRange = max(yMax - yMin, 1e-6)
        let zRange = max(zMax - zMin, 1e-6)

        // Room-aware filter using actual boundaries
        // In fullRoomCloud mode, we keep everything inside the room band
        func isRoomForeground(u: Float, v: Float) -> Bool {
            guard let b = boundaries else { return true }

            let floorY = Float(b.floorY)
            let ceilingY = Float(b.ceilingY)
            let leftX = Float(b.leftX)
            let rightX = Float(b.rightX)

            // Safety margins
            let marginX: Float = 0.03
            let floorMargin: Float = 0.05
            let ceilingMargin: Float = 0.03

            // Kill floor (anything below floor line)
            if v > floorY - floorMargin { return false }

            // Kill ceiling band - only in furniture mode
            if !isFullRoomCloud && v < ceilingY + ceilingMargin { return false }

            // Kill side walls - only in furniture mode
            if !isFullRoomCloud {
                if u < leftX + marginX { return false }
                if u > rightX - marginX { return false }
            }

            return true
        }

        // Score each splat: opacity + depth bias
        struct ScoredSplat {
            let splat: SinglePhotoRoomReconstructor.GaussianSplat
            let score: Float
        }

        var scoredSplats: [ScoredSplat] = []

        for splat in splats {
            // Normalize to [0,1] in SHARP space
            let u = (splat.position.x - xMin) / xRange
            let v = (splat.position.y - yMin) / yRange
            let zNorm = (splat.position.z - zMin) / zRange

            // Skip low opacity splats (always enforce - even in fullRoomCloud)
            if splat.opacity < 0.05 { continue }

            // Room-aware filter: in fullRoomCloud mode, keeps most splats
            if !isRoomForeground(u: u, v: v) { continue }

            // Score: opacity × (0.5 + 0.5 × depth)
            // Larger Z = closer to camera = higher score
            let depthBoost = 0.5 + 0.5 * zNorm
            let score = splat.opacity * depthBoost

            scoredSplats.append(ScoredSplat(splat: splat, score: score))
        }

        if isFullRoomCloud {
            print("🔷 [SHARPSplatService] Full-room mode: \(scoredSplats.count) candidates (no ceiling/wall culling)")
        } else {
            print("🔷 [SHARPSplatService] After room-aware filter: \(scoredSplats.count) candidates")
            if let b = boundaries {
                print("   Boundaries: floor=\(b.floorY), ceil=\(b.ceilingY), L=\(b.leftX), R=\(b.rightX)")
            }
        }

        // Thin the cloud in depth - skip in fullRoomCloud mode
        if isFullRoomCloud {
            // Classify each splat by surface using 3D world positions (not 2D image boundaries)
            // This gives us proper room geometry: floor/ceiling based on Y, walls based on X/Z
            let sorted = scoredSplats.sorted { $0.score > $1.score }

            // 3D surface classification helper - uses normalized world positions
            // margin = how thick the "shell" of each surface is (12% of range)
            let margin: Float = 0.12

            func classifySurface3D(xNorm: Float, yNorm: Float, zNorm: Float) -> SinglePhotoRoomReconstructor.SurfaceID {
                // SHARP coordinates: Y is vertical (0=bottom, 1=top in normalized space)
                // Check floor/ceiling first (horizontal surfaces)
                if yNorm < margin {
                    return .floor      // Bottom of the space
                }
                if yNorm > (1.0 - margin) {
                    return .ceiling    // Top of the space
                }
                // Then check walls (vertical surfaces)
                if zNorm < margin {
                    return .frontWall  // Front (closest to camera, small Z)
                }
                if xNorm < margin {
                    return .leftWall   // Left side
                }
                if xNorm > (1.0 - margin) {
                    return .rightWall  // Right side
                }
                // Interior points - assign to front wall as default
                return .frontWall
            }

            var surfaceCounts: [SinglePhotoRoomReconstructor.SurfaceID: Int] = [:]

            let classified: [SinglePhotoRoomReconstructor.GaussianSplat] = sorted.map { scored in
                var splat = scored.splat

                // Normalize to [0,1] in SHARP 3D space
                let xNorm = (splat.position.x - xMin) / xRange
                let yNorm = (splat.position.y - yMin) / yRange
                let zNorm = (splat.position.z - zMin) / zRange

                splat.surfaceId = classifySurface3D(xNorm: xNorm, yNorm: yNorm, zNorm: zNorm)
                surfaceCounts[splat.surfaceId, default: 0] += 1
                return splat
            }

            print("🔷 [SHARPSplatService] Full-room mode (3D classifier): \(classified.count) splats")
            print("   Surface counts: floor=\(surfaceCounts[.floor] ?? 0), ceiling=\(surfaceCounts[.ceiling] ?? 0), front=\(surfaceCounts[.frontWall] ?? 0), left=\(surfaceCounts[.leftWall] ?? 0), right=\(surfaceCounts[.rightWall] ?? 0)")
            return classified
        }

        // Normal furniture mode: thin the cloud in depth
        let zValues = scoredSplats.map { ($0.splat.position.z - zMin) / zRange }.sorted()
        guard !zValues.isEmpty else { return [] }

        let medianZNorm = zValues[zValues.count / 2]
        let thickness: Float = 0.60  // Keep 60% of depth range (was 25%)
        let zMinKeep = medianZNorm - thickness / 2
        let zMaxKeep = medianZNorm + thickness / 2

        let depthFiltered = scoredSplats.filter { scored in
            let zNorm = (scored.splat.position.z - zMin) / zRange
            return zNorm >= zMinKeep && zNorm <= zMaxKeep
        }

        print("🔷 [SHARPSplatService] After depth thinning: \(depthFiltered.count) (median Z=\(medianZNorm), range=[\(zMinKeep), \(zMaxKeep)])")

        // 🔴 NO TOP-N CAP: keep all depthFiltered splats, sorted by score
        let sorted = depthFiltered.sorted { $0.score > $1.score }
        let result = sorted.map { $0.splat }

        print("🔷 [SHARPSplatService] Returning \(result.count) splats (full set after filters)")
        return result
    }

    // MARK: - Foreground Mask Generation

    /// Generate a foreground mask from SHARP splats.
    /// White = furniture/foreground, Black = background.
    /// This version is deliberately generous so that layer-1 furniture
    /// like the chair actually gets removed from the front wall texture.
    func generateForegroundMask(
        from splats: [SinglePhotoRoomReconstructor.GaussianSplat],
        imageSize: CGSize,
        boundaries: RoomStructure
    ) -> CGImage? {
        guard !splats.isEmpty else {
            print("🔷 [SHARPSplatService] No splats, skipping mask generation")
            return nil
        }

        let width  = Int(imageSize.width)
        let height = Int(imageSize.height)

        // Use cached normalization stats from ALL splats (computed in generateForegroundSplats)
        // This ensures consistent UV mapping between filtering and mask generation
        let xMin: Float, xMax: Float, yMin: Float, yMax: Float, xRange: Float, yRange: Float
        let zMin: Float, zMax: Float, zRange: Float

        if let stats = cachedNormStats {
            xMin = stats.minX
            xMax = stats.maxX
            yMin = stats.minY
            yMax = stats.maxY
            zMin = stats.minZ
            zMax = stats.maxZ
            xRange = stats.xRange
            yRange = stats.yRange
            zRange = stats.zRange
            print("🔷 [SHARPSplatService] Using cached normalization stats for mask")
            print("   X: [\(xMin), \(xMax)] range=\(xRange)")
            print("   Y: [\(yMin), \(yMax)] range=\(yRange)")
            print("   Z: [\(zMin), \(zMax)] range=\(zRange)")
        } else {
            // Fallback: compute from filtered splats (less accurate but better than nothing)
            print("⚠️ [SHARPSplatService] No cached stats, computing from filtered splats")
            let xs = splats.map { $0.position.x }
            let ys = splats.map { $0.position.y }
            let zs = splats.map { $0.position.z }
            xMin = xs.min() ?? 0
            xMax = xs.max() ?? 1
            yMin = ys.min() ?? 0
            yMax = ys.max() ?? 1
            zMin = zs.min() ?? 0
            zMax = zs.max() ?? 1
            xRange = max(xMax - xMin, 1e-6)
            yRange = max(yMax - yMin, 1e-6)
            zRange = max(zMax - zMin, 1e-6)
        }

        // Compute median Z for depth filtering (furniture is closer to camera than walls)
        let sortedZ = splats.map { $0.position.z }.sorted()
        let medianZ = sortedZ.isEmpty ? zMin : sortedZ[sortedZ.count / 2]
        print("   Median Z: \(medianZ)")

        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            print("❌ [SHARPSplatService] Failed to create mask context")
            return nil
        }

        // Start fully background
        context.setFillColor(gray: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        context.setFillColor(gray: 1, alpha: 1)

        let floorY   = Float(boundaries.floorY)
        let ceilingY = Float(boundaries.ceilingY)
        let leftX    = Float(boundaries.leftX)
        let rightX   = Float(boundaries.rightX)

        // 🔧 NEW: much taller furniture band.
        // We only shave off a small cap at the top, so tall chairs / sofas are fully covered.
        let roomHeight = floorY - ceilingY
        let furnitureBandTop = ceilingY + 0.10 * roomHeight  // keep bottom ~90% of the room

        // Slightly larger radius so overlapping splats fill the whole object
        let splatRadius: CGFloat = 18
        var foregroundCount = 0
        var skippedLowOpacity = 0
        var skippedHighY = 0
        let skippedFarDepth = 0  // Depth filtering disabled; kept for logging

        // Depth threshold: keep splats closer to camera than median
        // Furniture sticks OUT from the wall, so it has smaller Z values
        let depthThreshold = medianZ - 0.02 * zRange  // Slightly in front of median

        // NOTE: we *do not* enforce an extra depth threshold anymore.
        // filterForegroundSplats already did depth-based pruning; re-doing it here
        // was causing the chair back to disappear from the mask.

        print("🔷 [SHARPSplatService] Mapping \(splats.count) splats to mask...")
        print("   Room band: X=[\(leftX), \(rightX)] Y=[\(ceilingY), \(floorY)]")
        print("   Furniture band (bottom ~90%): Y >= \(furnitureBandTop)")
        print("   Depth threshold (not enforced): Z < \(depthThreshold) (median=\(medianZ))")

        for splat in splats {
            // 1) Ignore extremely transparent dust
            if splat.opacity < 0.02 {
                skippedLowOpacity += 1
                continue
            }

            // 2) Normalize SHARP position to [0,1] within the SHARP cluster
            let clusterX = (splat.position.x - xMin) / xRange
            let clusterY = (splat.position.y - yMin) / yRange

            // 3) Map cluster [0,1] to IMAGE [0,1] within the room band
            // This places the SHARP output INTO the visible room area
            let imgXNorm = leftX + clusterX * (rightX - leftX)
            let imgYNorm = ceilingY + clusterY * (floorY - ceilingY)

            // 4) Y-gating: only drop *very* high stuff (tiny band near ceiling)
            if imgYNorm < furnitureBandTop {
                skippedHighY += 1
                continue
            }

            // 5) Depth filtering – DISABLED here.
            // We already used depth when selecting foreground splats.
            // Leaving this as a commented block for reference:
            /*
            if splat.position.z > depthThreshold {
                skippedFarDepth += 1
                continue
            }
            */

            // Clamp to valid range
            let clampedX = max(0, min(1, imgXNorm))
            let clampedY = max(0, min(1, imgYNorm))

            // Convert image-normalized coords → pixel coords
            let pixelX = CGFloat(clampedX) * CGFloat(width - 1)
            let pixelY = CGFloat(clampedY) * CGFloat(height - 1)

            let rect = CGRect(
                x: pixelX - splatRadius,
                y: pixelY - splatRadius,
                width: splatRadius * 2,
                height: splatRadius * 2
            )

            context.fillEllipse(in: rect)
            foregroundCount += 1
        }

        print("🔷 [SHARPSplatService] Foreground mask seeds: \(foregroundCount) (skipped \(skippedLowOpacity) low opacity, \(skippedHighY) high Y/curtains, \(skippedFarDepth) far depth/walls)")

        // Fallback: if SHARP gave us effectively nothing, assume there is
        // furniture in the right-bottom "chair band" and mask that out
        if foregroundCount == 0 {
            let fbRightStart = max(rightX - 0.22, 0.0)   // last ~22% width
            let fbBottomStart = min(floorY + 0.02, 1.0)  // just below floor line
            let fbHeightFrac: Float = 0.28               // ~bottom 28% of image

            let fallbackRect = CGRect(
                x: CGFloat(fbRightStart)  * CGFloat(width),
                y: CGFloat(fbBottomStart) * CGFloat(height),
                width: CGFloat(rightX - fbRightStart) * CGFloat(width),
                height: CGFloat(fbHeightFrac) * CGFloat(height)
            ).intersection(CGRect(x: 0, y: 0, width: width, height: height))

            if !fallbackRect.isNull {
                context.fill(fallbackRect)
                print("⚠️ [SHARPSplatService] SHARP mask empty – drew fallback right-bottom band: \(fallbackRect)")
            } else {
                print("⚠️ [SHARPSplatService] SHARP mask empty and fallback rect was null")
            }
        }

        guard let rawMask = context.makeImage() else {
            print("❌ [SHARPSplatService] Failed to create raw mask image")
            return nil
        }

        // Count non-zero pixels in raw mask for debugging
        if let maskData = context.data {
            let maskPtr = maskData.bindMemory(to: UInt8.self, capacity: width * height)
            var nonZeroCount = 0
            for i in 0..<(width * height) {
                if maskPtr[i] > 0 { nonZeroCount += 1 }
            }
            print("🔍 [SHARPSplatService] Raw mask non-zero pixels: \(nonZeroCount) / \(width * height)")
        }

        // Apply light blur to thicken/smooth the seed points into a continuous region
        let ciImage = CIImage(cgImage: rawMask)

        let blurFilter = CIFilter(name: "CIGaussianBlur")
        blurFilter?.setValue(ciImage, forKey: kCIInputImageKey)
        blurFilter?.setValue(5.0, forKey: kCIInputRadiusKey)

        guard let blurred = blurFilter?.outputImage else {
            print("⚠️ [SHARPSplatService] Blur failed, returning raw mask")
            return rawMask
        }

        let finalCG = CIContext().createCGImage(
            blurred,
            from: CGRect(x: 0, y: 0, width: width, height: height)
        )

        if let result = finalCG {
            print("✅ [SHARPSplatService] Mask smoothed with blur radius 5")
            return result
        } else {
            print("⚠️ [SHARPSplatService] Failed to create blurred mask, returning raw")
            return rawMask
        }
    }

    // MARK: - SHARP Scene Statistics
    struct SharpStats {
        let minX: Float, maxX: Float
        let minY: Float, maxY: Float
        let minZ: Float, maxZ: Float
        let rangeX: Float, rangeY: Float, rangeZ: Float
    }

    private func computeSharpStats(from splats: [SinglePhotoRoomReconstructor.GaussianSplat]) -> SharpStats {
        guard !splats.isEmpty else {
            return SharpStats(minX: 0, maxX: 1, minY: 0, maxY: 1, minZ: 0, maxZ: 1,
                              rangeX: 1, rangeY: 1, rangeZ: 1)
        }

        let xs = splats.map { $0.position.x }
        let ys = splats.map { $0.position.y }
        let zs = splats.map { $0.position.z }

        let minX = xs.min() ?? 0
        let maxX = xs.max() ?? 1
        let minY = ys.min() ?? 0
        let maxY = ys.max() ?? 1
        let minZ = zs.min() ?? 0
        let maxZ = zs.max() ?? 1

        let rangeX = max(maxX - minX, 1e-5)
        let rangeY = max(maxY - minY, 1e-5)
        let rangeZ = max(maxZ - minZ, 1e-5)

        print("🔷 [SHARPSplatService] Scene stats:")
        print("   X: [\(minX), \(maxX)] range=\(rangeX)")
        print("   Y: [\(minY), \(maxY)] range=\(rangeY)")
        print("   Z: [\(minZ), \(maxZ)] range=\(rangeZ)")

        return SharpStats(minX: minX, maxX: maxX, minY: minY, maxY: maxY, minZ: minZ, maxZ: maxZ,
                          rangeX: rangeX, rangeY: rangeY, rangeZ: rangeZ)
    }

    /// Which room plane a splat belongs to based on its image UV
    private enum RoomPlane {
        case frontWall, floor, leftWall, rightWall, ceiling, discard
    }

    /// Classify splat to a room plane based on image UV and boundaries
    private func classifyPlane(u: Float, v: Float, boundaries: RoomStructure) -> RoomPlane {
        let floorY = Float(boundaries.floorY)
        let ceilingY = Float(boundaries.ceilingY)
        let leftX = Float(boundaries.leftX)
        let rightX = Float(boundaries.rightX)

        // v is image Y: 0=top, 1=bottom
        if v >= floorY {
            return .floor
        } else if v <= ceilingY {
            return .ceiling  // could discard these
        } else {
            // Middle band: walls
            if u <= leftX { return .leftWall }
            if u >= rightX { return .rightWall }
            return .frontWall  // between left & right → back wall (curtains)
        }
    }

    /// Map SHARP position to room plane using UV projection
    /// Treats SHARP XY as image UV, ignores Z for placement
    private func mapSharpToRoomPlane(
        _ p: SIMD3<Float>,
        room: (width: Float, depth: Float, height: Float),
        stats: SharpStats,
        boundaries: RoomStructure
    ) -> SIMD3<Float>? {
        func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float { a + (b - a) * t }

        // 1. Normalize SHARP XY to 0...1 (image UV)
        let nx = (p.x - stats.minX) / stats.rangeX
        let ny = (p.y - stats.minY) / stats.rangeY

        // Clamp to 0-1
        let u = max(0, min(1, nx))
        let v = max(0, min(1, ny))  // v: 0=top, 1=bottom in image space

        // Room geometry
        let minX = -room.width / 2
        let maxX = room.width / 2
        let floorYWorld: Float = 0.0
        let ceilYWorld: Float = room.height
        let frontZ: Float = -room.depth / 2  // front wall Z position

        // Boundary values
        let floorY = Float(boundaries.floorY)
        let ceilingY = Float(boundaries.ceilingY)
        let leftX = Float(boundaries.leftX)
        let rightX = Float(boundaries.rightX)

        // 2. Classify which plane this splat belongs to
        let plane = classifyPlane(u: u, v: v, boundaries: boundaries)

        // 3. Project onto the chosen plane (ignore SHARP Z)
        switch plane {
        case .frontWall:
            // Map u across wall width, v across wall height
            let tX = (u - leftX) / max(rightX - leftX, 0.001)
            let tY = (v - ceilingY) / max(floorY - ceilingY, 0.001)
            let worldX = lerp(minX, maxX, tX)
            let worldY = lerp(ceilYWorld, floorYWorld, tY)  // ceiling→floor as v increases
            let worldZ = frontZ + 0.03  // tiny offset to avoid z-fighting
            return SIMD3<Float>(worldX, worldY, worldZ)

        case .floor:
            // Stick to floor plane, map depth from floorY to bottom of image
            let tX = (u - leftX) / max(rightX - leftX, 0.001)
            let tZ = (v - floorY) / max(1.0 - floorY, 0.001)  // 0 at floor line, 1 at image bottom
            let worldX = lerp(minX, maxX, tX)
            let worldZ = lerp(frontZ, frontZ + room.depth * 0.5, tZ)  // front → mid room
            let worldY = floorYWorld + 0.02  // slightly above floor
            return SIMD3<Float>(worldX, worldY, worldZ)

        case .leftWall, .rightWall, .ceiling:
            // Discard for now - focus on front wall and floor
            return nil

        case .discard:
            return nil
        }
    }

    func createPointCloudGeometry(
        from splats: [SinglePhotoRoomReconstructor.GaussianSplat],
        roomDimensions: (width: Float, depth: Float, height: Float),
        boundaries: RoomStructure
    ) -> SCNNode? {
        guard !splats.isEmpty else { return nil }

        let parent = SCNNode()

        // Compute scene statistics for UV normalization
        let stats = computeSharpStats(from: splats)

        // Color mode: .sharpRadiance = rainbow colors, .neutralWhite = geometry debug, .magenta = visibility debug
        enum SplatColorMode { case sharpRadiance, neutralWhite, magenta }
        let colorMode: SplatColorMode = .neutralWhite  // 🔧 Change to see geometry without rainbow

        let sphereRadius: CGFloat = 0.02  // 2cm spheres

        var placedCount = 0
        var frontWallCount = 0
        var floorCount = 0

        for (index, splat) in splats.enumerated() {
            // Map using UV → plane projection (ignores SHARP Z for placement)
            guard let worldPos = mapSharpToRoomPlane(splat.position, room: roomDimensions, stats: stats, boundaries: boundaries) else {
                continue  // Skip discarded planes (ceiling, side walls)
            }

            let sphere = SCNSphere(radius: sphereRadius)
            let material = SCNMaterial()

            switch colorMode {
            case .magenta:
                // DEBUG: Bright magenta to spot against room
                material.diffuse.contents = UIColor.magenta
                material.transparency = 0.7
            case .neutralWhite:
                // Neutral white - shows geometry without distracting rainbow
                material.diffuse.contents = UIColor.white
                material.transparency = CGFloat(1.0 - min(max(splat.opacity, 0.1), 0.8))
            case .sharpRadiance:
                // Use actual SHARP colors (rainbow)
                material.diffuse.contents = UIColor(
                    red: CGFloat(splat.color.x),
                    green: CGFloat(splat.color.y),
                    blue: CGFloat(splat.color.z),
                    alpha: 1.0
                )
                material.transparency = 0.5  // Semi-transparent so underlying geometry shows
            }
            material.lightingModel = .constant
            material.isDoubleSided = true
            // Proper depth testing - splats can be occluded by walls/chair
            material.readsFromDepthBuffer = true
            material.writesToDepthBuffer = true
            sphere.materials = [material]

            let node = SCNNode(geometry: sphere)
            node.position = SCNVector3(worldPos.x, worldPos.y, worldPos.z)
            parent.addChildNode(node)

            placedCount += 1

            // Track plane distribution
            if abs(worldPos.z - (-roomDimensions.depth / 2 + 0.03)) < 0.1 {
                frontWallCount += 1
            } else {
                floorCount += 1
            }

            if index == 0 {
                print("🔷 [SHARPSplatService] First splat:")
                print("   SHARP pos: \(splat.position)")
                print("   World pos: \(worldPos)")
            }
        }

        // Default rendering order - splats participate in normal depth testing
        parent.renderingOrder = 0

        print("✅ [SHARPSplatService] Created point cloud with \(placedCount) points")
        print("   Front wall: \(frontWallCount), Floor: \(floorCount)")
        print("   Rendering order: 0 (normal depth testing)")
        return parent
    }

    // MARK: - Utilities

    private func getMemoryUsage() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<integer_t>.size)

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }

        if kerr == KERN_SUCCESS {
            return info.phys_footprint / 1_048_576
        } else {
            return 0
        }
    }
}

// MARK: - Float16 Helper
extension Float {
    init(half: UInt16) {
        self = Float(Float16(bitPattern: half))
    }
}

// MARK: - UIImage Helpers for SHARP
extension UIImage {
    /// Convert UIImage to MLMultiArray (C x H x W) normalized to [-1, 1] for SHARP
    func toMLMultiArray(size: Int) -> MLMultiArray? {
        guard let cgImage = self.cgImage else { return nil }

        let width = size
        let height = size
        let bytesPerRow = width * 4

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var bitmapInfo = CGBitmapInfo.byteOrder32Big
        bitmapInfo.formUnion(.init(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue))

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            return nil
        }

        let rect = AVMakeRect(aspectRatio: CGSize(width: cgImage.width, height: cgImage.height), insideRect: CGRect(x: 0, y: 0, width: width, height: height))
        context.draw(cgImage, in: rect)

        guard let data = context.data else { return nil }

        let count = 3 * width * height
        guard let array = try? MLMultiArray(shape: [3, height as NSNumber, width as NSNumber], dataType: .float32) else {
            return nil
        }

        let ptr = data.bindMemory(to: UInt8.self, capacity: bytesPerRow * height)
        let outPtr = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))

        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                let r = Float(ptr[offset + 0]) / 255.0
                let g = Float(ptr[offset + 1]) / 255.0
                let b = Float(ptr[offset + 2]) / 255.0

                let yr = y
                let xr = x

                outPtr[0 * height * width + yr * width + xr] = (r - 0.5) * 2.0
                outPtr[1 * height * width + yr * width + xr] = (g - 0.5) * 2.0
                outPtr[2 * height * width + yr * width + xr] = (b - 0.5) * 2.0
            }
        }

        return array
    }

    /// Convert UIImage to CVPixelBuffer normalized for SHARP (image input path)
    func toSHARPPixelBufferNormalized(size: CGSize) -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(size.width),
                                         Int(size.height),
                                         kCVPixelFormatType_32BGRA,
                                         attrs,
                                         &pixelBuffer)
        guard status == kCVReturnSuccess, let pb = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(pb, [])

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(pb),
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else {
            CVPixelBufferUnlockBaseAddress(pb, [])
            return nil
        }

        guard let cgImage = self.cgImage else {
            CVPixelBufferUnlockBaseAddress(pb, [])
            return nil
        }

        let rect = AVMakeRect(aspectRatio: CGSize(width: cgImage.width, height: cgImage.height),
                              insideRect: CGRect(origin: .zero, size: size))
        context.clear(CGRect(origin: .zero, size: size))
        context.draw(cgImage, in: rect)

        CVPixelBufferUnlockBaseAddress(pb, [])

        // Use 32BGRA - CoreML image input handles color conversion
        return pb
    }
}

// MARK: - Texture Generation Helpers
extension SinglePhotoRoomReconstructor {
    // Generate floor texture from bottom region defined by boundaries
    private func generateFloorTexture(from image: UIImage, structure: RoomStructure) -> UIImage {
        logDebug("🎨 [TextureGen] Generating floor texture")
        guard let cgImage = image.cgImage else { return createSolidColorTexture(color: .lightGray) }

        let width = CGFloat(cgImage.width)
        let height = CGFloat(cgImage.height)

        let floorMinY = structure.floorY
        let floorRect = CGRect(
            x: structure.leftX * width,
            y: floorMinY * height,
            width: (structure.rightX - structure.leftX) * width,
            height: (1.0 - floorMinY) * height
        )

        logDebug("   - Boundaries: L:\(structure.leftX) R:\(structure.rightX) F:\(structure.floorY)")
        logDebug("   - Crop rect: \(floorRect)")

        if let cropped = cgImage.cropping(to: floorRect) {
            logDebug("✅ [TextureGen] Floor texture extracted from boundaries")
            return UIImage(cgImage: cropped)
        }

        logDebug("⚠️ [TextureGen] Failed to crop floor, using solid color")
        return createSolidColorTexture(color: .lightGray)
    }

    private func generateCeilingTexture(from image: UIImage, structure: RoomStructure) -> UIImage {
        logDebug("🎨 [TextureGen] Generating CEILING texture from boundaries")
        guard let cgImage = image.cgImage else { return createSolidColorTexture(color: .white) }

        let width = CGFloat(cgImage.width)
        let height = CGFloat(cgImage.height)

        let ceilingRect = CGRect(
            x: structure.leftX * width,
            y: 0,
            width: (structure.rightX - structure.leftX) * width,
            height: structure.ceilingY * height
        )

        logDebug("   - Boundaries: L:\(structure.leftX) R:\(structure.rightX) C:\(structure.ceilingY)")
        logDebug("   - Crop rect: \(ceilingRect)")

        if let cropped = cgImage.cropping(to: ceilingRect) {
            logDebug("✅ [TextureGen] Ceiling texture extracted from boundaries")
            return UIImage(cgImage: cropped)
        }

        logDebug("⚠️ [TextureGen] Failed to crop ceiling, using solid color")
        return createSolidColorTexture(color: .white)
    }

    private func generateFrontWallTexture(from image: UIImage, structure: RoomStructure, foregroundMask: CGImage? = nil) -> UIImage {
        logDebug("🎨 [TextureGen] Generating FRONT wall texture from boundaries")
        guard let cgImage = image.cgImage else { return createSolidColorTexture(color: .white) }

        let width = CGFloat(cgImage.width)
        let height = CGFloat(cgImage.height)

        let wallTop = structure.ceilingY * height
        let wallBottom = structure.floorY * height

        let wallRect = CGRect(
            x: structure.leftX * width,
            y: wallTop,
            width: (structure.rightX - structure.leftX) * width,
            height: wallBottom - wallTop
        )

        logDebug("   - Boundaries: L:\(structure.leftX) R:\(structure.rightX) C:\(structure.ceilingY) F:\(structure.floorY)")
        logDebug("   - Foreground mask: \(foregroundMask != nil ? "provided" : "none")")
        logDebug("   - Crop rect: \(wallRect)")

        guard let cropped = cgImage.cropping(to: wallRect) else {
            logDebug("⚠️ [TextureGen] Failed to crop front wall, using solid color")
            return createSolidColorTexture(color: .white)
        }

        // If no mask provided, return the simple crop
        guard let mask = foregroundMask else {
            logDebug("✅ [TextureGen] Front wall texture extracted (no mask)")
            return UIImage(cgImage: cropped)
        }

        // Apply mask: inpaint foreground regions with nearby background colors
        logDebug("🎨 [TextureGen] Applying foreground mask to remove furniture...")

        let croppedWidth = cropped.width
        let croppedHeight = cropped.height

        // First, resize mask to match the original image size (in case it's different)
        // Then crop it with the same wallRect so mask and image pixels are aligned
        guard let resizedMask = scaleImage(mask, to: CGSize(width: width, height: height)) else {
            logDebug("⚠️ [TextureGen] Failed to resize mask to image size")
            return UIImage(cgImage: cropped)
        }

        guard let croppedMask = resizedMask.cropping(to: wallRect) else {
            logDebug("⚠️ [TextureGen] Failed to crop mask with wallRect")
            return UIImage(cgImage: cropped)
        }

        logDebug("   - Mask original size: \(mask.width)x\(mask.height)")
        logDebug("   - Mask resized to: \(Int(width))x\(Int(height))")
        logDebug("   - Mask cropped to: \(croppedMask.width)x\(croppedMask.height)")

        // Create a new context for the inpainted result
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: croppedWidth,
            height: croppedHeight,
            bitsPerComponent: 8,
            bytesPerRow: croppedWidth * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            logDebug("⚠️ [TextureGen] Failed to create inpaint context")
            return UIImage(cgImage: cropped)
        }

        // Draw the original cropped image
        context.draw(cropped, in: CGRect(x: 0, y: 0, width: croppedWidth, height: croppedHeight))

        // Get pixel data from both
        guard let imageData = context.data else {
            return UIImage(cgImage: cropped)
        }

        // Create context for mask to read its pixels
        guard let maskContext = CGContext(
            data: nil,
            width: croppedWidth,
            height: croppedHeight,
            bitsPerComponent: 8,
            bytesPerRow: croppedWidth,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return UIImage(cgImage: cropped)
        }
        maskContext.draw(croppedMask, in: CGRect(x: 0, y: 0, width: croppedWidth, height: croppedHeight))

        guard let maskData = maskContext.data else {
            return UIImage(cgImage: cropped)
        }

        let imagePtr = imageData.bindMemory(to: UInt8.self, capacity: croppedWidth * croppedHeight * 4)
        let maskPtr = maskData.bindMemory(to: UInt8.self, capacity: croppedWidth * croppedHeight)

        // Simple horizontal inpaint: for masked pixels, copy from nearest unmasked pixel to the left
        // Lower threshold (32 instead of 128) to catch more foreground pixels
        let foregroundThreshold: UInt8 = 32  // ~0.125, more forgiving
        var inpaintedCount = 0
        for y in 0..<croppedHeight {
            var lastGoodR: UInt8? = nil
            var lastGoodG: UInt8? = nil
            var lastGoodB: UInt8? = nil

            for x in 0..<croppedWidth {
                let maskIdx = y * croppedWidth + x
                let imgIdx = maskIdx * 4

                let maskValue = maskPtr[maskIdx]

                if maskValue < foregroundThreshold {
                    // Background pixel - save as "good" color
                    lastGoodR = imagePtr[imgIdx + 0]
                    lastGoodG = imagePtr[imgIdx + 1]
                    lastGoodB = imagePtr[imgIdx + 2]
                } else {
                    // Foreground pixel - replace with last good color if we have one
                    // If no good color yet (left edge of mask), keep original pixel
                    if let goodR = lastGoodR, let goodG = lastGoodG, let goodB = lastGoodB {
                        imagePtr[imgIdx + 0] = goodR
                        imagePtr[imgIdx + 1] = goodG
                        imagePtr[imgIdx + 2] = goodB
                        inpaintedCount += 1
                    }
                    // else: keep original pixel - don't fill with arbitrary grey
                }
            }
        }

        logDebug("✅ [TextureGen] Inpainted \(inpaintedCount) foreground pixels")

        guard let result = context.makeImage() else {
            return UIImage(cgImage: cropped)
        }

        return UIImage(cgImage: result)
    }

    private func scaleImage(_ image: CGImage, to size: CGSize) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: nil,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: Int(size.width),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return nil
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(origin: .zero, size: size))
        return context.makeImage()
    }

    // LEFT wall from a vertical strip at leftX
    private func generateLeftWallTexture(from image: UIImage, structure: RoomStructure) -> UIImage {
        logDebug("🎨 [TextureGen] Generating LEFT wall texture from boundary")

        guard let cgImage = image.cgImage else {
            logDebug("⚠️ [TextureGen] No CGImage, using solid color")
            return createSolidColorTexture(color: UIColor(white: 0.9, alpha: 1.0))
        }

        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)

        let stripWidth = imageWidth * 0.1
        let leftXPos = structure.leftX * imageWidth

        let wallTop = structure.ceilingY * imageHeight
        let wallBottom = structure.floorY * imageHeight

        let cropRect = CGRect(
            x: max(leftXPos - stripWidth * 0.5, 0),
            y: wallTop,
            width: stripWidth,
            height: wallBottom - wallTop
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

    // RIGHT wall from a vertical strip at rightX
    private func generateRightWallTexture(from image: UIImage, structure: RoomStructure) -> UIImage {
        logDebug("🎨 [TextureGen] Generating RIGHT wall texture from boundary")

        guard let cgImage = image.cgImage else {
            logDebug("⚠️ [TextureGen] No CGImage, using solid color")
            return createSolidColorTexture(color: UIColor(white: 0.9, alpha: 1.0))
        }

        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)

        let stripWidth = imageWidth * 0.1
        let rightXPos = structure.rightX * imageWidth

        let wallTop = structure.ceilingY * imageHeight
        let wallBottom = structure.floorY * imageHeight

        let cropRect = CGRect(
            x: min(rightXPos - stripWidth * 0.5, imageWidth - stripWidth),
            y: wallTop,
            width: stripWidth,
            height: wallBottom - wallTop
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

    private func sampleWallColor(from image: UIImage) -> UIColor? {
        logDebug("🎨 [TextureGen] Generating wall texture")

        guard let cgImage = image.cgImage else {
            logDebug("⚠️ [TextureGen] No CGImage for sampling")
            return nil
        }

        let width = CGFloat(cgImage.width)
        let height = CGFloat(cgImage.height)
        let rect = CGRect(x: width * 0.45, y: height * 0.35, width: width * 0.1, height: height * 0.1)
        logDebug("   - Sampling center region for average color")
        logDebug("   - Sample rect: \(rect)")

        let ciImage = CIImage(cgImage: cgImage)
        return ciImage.averageColor(in: rect)
    }

    private func configureMaterialForUSDZExport(_ material: SCNMaterial, texture: UIImage) {
        material.diffuse.contents = texture
        material.locksAmbientWithDiffuse = true
        material.isDoubleSided = false

        material.emission.contents = texture
        material.emission.intensity = 1.0

        material.lightingModel = .constant
        // Walls participate in normal depth testing
        // They can occlude splats that are behind them
        material.writesToDepthBuffer = true
        material.readsFromDepthBuffer = true
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

        guard let filter = CIFilter(name: "CIAreaAverage", parameters: [
            kCIInputImageKey: self,
            kCIInputExtentKey: extentVector
        ]) else {
            return nil
        }

        guard let outputImage = filter.outputImage else { return nil }

        var bitmap = [UInt8](repeating: 0, count: 4)
        let context = CIContext(options: [.workingColorSpace: NSNull()])

        context.render(outputImage,
                       toBitmap: &bitmap,
                       rowBytes: 4,
                       bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
                       format: .RGBA8,
                       colorSpace: nil)

        return UIColor(red: CGFloat(bitmap[0]) / 255.0,
                       green: CGFloat(bitmap[1]) / 255.0,
                       blue: CGFloat(bitmap[2]) / 255.0,
                       alpha: 1.0)
    }
}

