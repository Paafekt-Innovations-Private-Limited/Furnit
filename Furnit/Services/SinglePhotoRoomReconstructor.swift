import SwiftUI
import CoreML
import Vision
import CoreImage
import SceneKit
import Metal
import MetalPerformanceShaders

// MARK: - Room Structure (with proper initialization)
struct RoomStructure {
    var wallLines: [Line] = []  // Added default value
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
        var width: Float = 4.0  // meters
        var depth: Float = 4.0
        var height: Float = 2.8
        var doorHeight: Float = 2.1
        var confidence: Float = 0.6
    }
    
    // MARK: - Helper Methods
    private func updateProgress(_ value: Float, _ message: String) async {
        await MainActor.run {
            self.progress = value
            self.statusMessage = message
        }
    }
    
    private func setError(_ message: String) async {
        await MainActor.run {
            self.isProcessing = false
            self.statusMessage = message
        }
    }
    
    // MARK: - Main Processing Pipeline
    func processPhoto(_ image: UIImage) async {
        await MainActor.run {
            isProcessing = true
            progress = 0.0
            statusMessage = "Analyzing photo..."
        }
        
        // Step 1: Generate depth map
        await updateProgress(0.2, "Extracting depth information...")
        guard let depthMap = await depthEstimator.estimateDepth(from: image) else {
            await setError("Failed to estimate depth")
            return
        }
        
        // Step 2: Detect room structure
        await updateProgress(0.4, "Finding walls and corners...")
        let roomStructure = await roomAnalyzer.analyzeRoom(image: image, depthMap: depthMap)
        
        // Step 3: Estimate dimensions
        await updateProgress(0.6, "Calculating room dimensions...")
        let dimensions = await estimateDimensions(from: roomStructure, image: image)
        
        await MainActor.run {
            self.estimatedDimensions = dimensions
        }
        
        // Step 4: Build 3D room
        await updateProgress(0.8, "Building 3D model...")
        let roomURL = await build3DRoom(
            dimensions: dimensions,
            structure: roomStructure,
            originalImage: image,
            depthMap: depthMap
        )
        
        await MainActor.run {
            self.generatedRoomURL = roomURL
            self.isProcessing = false
            self.progress = 1.0
            self.statusMessage = "Room ready!"
        }
    }
    
    // MARK: - Dimension Estimation
    private func estimateDimensions(from structure: RoomStructure, image: UIImage) async -> RoomDimensions {
        var dimensions = RoomDimensions()
        
        // Try to find doors for scale reference
        if let doorRect = await detectDoor(in: image) {
            // Door detected - use as scale reference
            let doorPixelHeight = doorRect.height * image.size.height
            let imageHeight = image.size.height
            
            // Assuming standard door height of 2.1m
            let pixelsPerMeter = doorPixelHeight / 2.1
            
            // Estimate room height from image proportions
            dimensions.height = Float(imageHeight / pixelsPerMeter)
            dimensions.width = Float(image.size.width / pixelsPerMeter * 1.2) // Adjust for FOV
            dimensions.depth = dimensions.width // Initial guess, will be refined
            dimensions.confidence = 0.8
        } else if let personRect = await detectPerson(in: image) {
            // Fallback: use person height (average 1.7m)
            let personPixelHeight = personRect.height * image.size.height
            let pixelsPerMeter = personPixelHeight / 1.7
            
            dimensions.height = Float(image.size.height / pixelsPerMeter * 0.4) // Typical room ratio
            dimensions.width = Float(image.size.width / pixelsPerMeter * 1.5)
            dimensions.depth = dimensions.width * 0.9
            dimensions.confidence = 0.5
        } else {
            // No reference found - use typical room proportions
            dimensions.width = 4.0
            dimensions.depth = 4.5
            dimensions.height = 2.8
            dimensions.confidence = 0.3
        }
        
        // Clamp to reasonable ranges
        dimensions.width = min(max(dimensions.width, 2.0), 8.0)
        dimensions.depth = min(max(dimensions.depth, 2.0), 8.0)
        dimensions.height = min(max(dimensions.height, 2.2), 4.0)
        
        return dimensions
    }
    
    // MARK: - Object Detection
    private func detectDoor(in image: UIImage) async -> CGRect? {
        guard let cgImage = image.cgImage else { return nil }
        
        let request = VNDetectRectanglesRequest()
        request.minimumAspectRatio = 0.3
        request.maximumAspectRatio = 0.5
        request.minimumSize = 0.1
        request.maximumObservations = 5
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
        
        // Look for door-like rectangles (tall and narrow)
        if let results = request.results {
            for rect in results {
                let aspectRatio = rect.boundingBox.width / rect.boundingBox.height
                if aspectRatio > 0.35 && aspectRatio < 0.5 && rect.boundingBox.height > 0.3 {
                    return rect.boundingBox
                }
            }
        }
        return nil
    }
    
    private func detectPerson(in image: UIImage) async -> CGRect? {
        guard let cgImage = image.cgImage else { return nil }
        
        let request = VNDetectHumanRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
        
        return request.results?.first?.boundingBox
    }
    
    // MARK: - 3D Room Building
    private func build3DRoom(dimensions: RoomDimensions, structure: RoomStructure, originalImage: UIImage, depthMap: CIImage) async -> URL? {
        let scene = SCNScene()
        
        // Create room box
        let roomNode = SCNNode()
        
        // Floor
        let floor = SCNPlane(width: CGFloat(dimensions.width), height: CGFloat(dimensions.depth))
        let floorNode = SCNNode(geometry: floor)
        floorNode.eulerAngles.x = -.pi / 2
        floorNode.position.y = 0
        floor.firstMaterial?.diffuse.contents = generateFloorTexture(from: originalImage, structure: structure)
        roomNode.addChildNode(floorNode)
        
        // Ceiling
        let ceiling = SCNPlane(width: CGFloat(dimensions.width), height: CGFloat(dimensions.depth))
        let ceilingNode = SCNNode(geometry: ceiling)
        ceilingNode.eulerAngles.x = .pi / 2
        ceilingNode.position.y = Float(dimensions.height)
        ceiling.firstMaterial?.diffuse.contents = UIColor(white: 0.95, alpha: 1.0)
        roomNode.addChildNode(ceilingNode)
        
        // Front wall (visible in photo)
        let frontWall = SCNPlane(width: CGFloat(dimensions.width), height: CGFloat(dimensions.height))
        let frontNode = SCNNode(geometry: frontWall)
        frontNode.position.z = -Float(dimensions.depth) / 2
        frontWall.firstMaterial?.diffuse.contents = originalImage
        roomNode.addChildNode(frontNode)
        
        // Back wall (generated)
        let backWall = SCNPlane(width: CGFloat(dimensions.width), height: CGFloat(dimensions.height))
        let backNode = SCNNode(geometry: backWall)
        backNode.position.z = Float(dimensions.depth) / 2
        backNode.eulerAngles.y = .pi
        backWall.firstMaterial?.diffuse.contents = generateWallTexture(from: originalImage)
        roomNode.addChildNode(backNode)
        
        // Side walls
        let leftWall = SCNPlane(width: CGFloat(dimensions.depth), height: CGFloat(dimensions.height))
        let leftNode = SCNNode(geometry: leftWall)
        leftNode.position.x = -Float(dimensions.width) / 2
        leftNode.eulerAngles.y = .pi / 2
        leftWall.firstMaterial?.diffuse.contents = generateWallTexture(from: originalImage)
        roomNode.addChildNode(leftNode)
        
        let rightWall = SCNPlane(width: CGFloat(dimensions.depth), height: CGFloat(dimensions.height))
        let rightNode = SCNNode(geometry: rightWall)
        rightNode.position.x = Float(dimensions.width) / 2
        rightNode.eulerAngles.y = -.pi / 2
        rightWall.firstMaterial?.diffuse.contents = generateWallTexture(from: originalImage)
        roomNode.addChildNode(rightNode)
        
        // Add camera
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, dimensions.height / 2, dimensions.depth * 0.8)
        cameraNode.look(at: SCNVector3(0, dimensions.height / 2, 0))
        scene.rootNode.addChildNode(cameraNode)
        
        // Add lighting
        let lightNode = SCNNode()
        lightNode.light = SCNLight()
        lightNode.light?.type = .omni
        lightNode.position = SCNVector3(0, dimensions.height * 0.8, 0)
        scene.rootNode.addChildNode(lightNode)
        
        scene.rootNode.addChildNode(roomNode)
        
        // Export to USDZ
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("room_\(UUID().uuidString).usdz")
        
        do {
            try scene.write(to: tempURL, options: nil, delegate: nil, progressHandler: nil)
            return tempURL
        } catch {
            print("Error exporting room: \(error)")
            return nil
        }
    }
    
    // MARK: - Texture Generation
    private func generateFloorTexture(from image: UIImage, structure: RoomStructure) -> UIImage {
        guard let floorRegion = structure.floorRegion else {
            return createSolidColorTexture(color: UIColor(white: 0.8, alpha: 1.0))
        }
        
        // Extract floor region from image
        guard let cgImage = image.cgImage else {
            return createSolidColorTexture(color: UIColor(white: 0.8, alpha: 1.0))
        }
        
        let imageHeight = CGFloat(cgImage.height)
        let imageWidth = CGFloat(cgImage.width)
        
        let cropRect = CGRect(
            x: floorRegion.origin.x * imageWidth,
            y: (1.0 - floorRegion.origin.y - floorRegion.height) * imageHeight,
            width: floorRegion.width * imageWidth,
            height: floorRegion.height * imageHeight
        )
        
        if let croppedImage = cgImage.cropping(to: cropRect) {
            return UIImage(cgImage: croppedImage)
        }
        
        return createSolidColorTexture(color: UIColor(white: 0.8, alpha: 1.0))
    }
    
    private func generateWallTexture(from image: UIImage) -> UIImage {
        // Analyze wall color from image center
        guard let cgImage = image.cgImage else {
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
        
        let averageColor = ciImage.averageColor(in: centerRect) ?? UIColor(white: 0.9, alpha: 1.0)
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
        // Note: You need to download MiDaS v3.1 small CoreML model
        // from https://github.com/isl-org/MiDaS
        // Convert to CoreML format and add to project
        loadModel()
    }
    
    private func loadModel() {
        // Placeholder - in real implementation, load MiDaS CoreML model
        // For now, we'll use Vision's depth estimation if available
    }
    
    func estimateDepth(from image: UIImage) async -> CIImage? {
        guard let cgImage = image.cgImage else { return nil }
        let ciImage = CIImage(cgImage: cgImage)
        
        // If MiDaS model is available, use it
        if let model = model {
            // Run depth estimation with MiDaS
            // This is a placeholder
            return nil
        }
        
        // Fallback: create synthetic depth map based on image gradients
        return generateSyntheticDepthMap(from: ciImage)
    }
    
    private func generateSyntheticDepthMap(from image: CIImage) -> CIImage {
        // Simple gradient-based depth approximation
        // In real app, this would use a proper depth model
        guard let grayscale = CIFilter(name: "CIPhotoEffectMono")?.apply(image: image),
              let edges = CIFilter(name: "CIEdges")?.apply(image: grayscale, intensity: 2.0) else {
            return image
        }
        
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
        var structure = RoomStructure()  // Now works with default values
        
        // Detect edges/lines in image
        structure.wallLines = await detectWallLines(in: image)
        
        // Find floor and ceiling regions
        structure.floorRegion = detectFloorRegion(in: image, depthMap: depthMap)
        structure.ceilingRegion = detectCeilingRegion(in: image, depthMap: depthMap)
        
        // Calculate vanishing point
        structure.vanishingPoint = calculateVanishingPoint(from: structure.wallLines)
        
        return structure
    }
    
    private func detectWallLines(in image: UIImage) async -> [RoomStructure.Line] {
        guard let cgImage = image.cgImage else { return [] }
        
        // Use Vision to detect edges
        let request = VNDetectContoursRequest()
        request.contrastAdjustment = 1.0
        request.detectsDarkOnLight = true
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
        
        var lines: [RoomStructure.Line] = []
        
        if let observations = request.results {
            for contour in observations {
                // Extract significant straight lines from contours
                let path = contour.normalizedPath
                
                // Simplified: just get bounding edges
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
        }
        
        return lines
    }
    
    private func detectFloorRegion(in image: UIImage, depthMap: CIImage) -> CGRect? {
        // Simple heuristic: floor is bottom 20% of image
        return CGRect(x: 0, y: 0.8, width: 1.0, height: 0.2)
    }
    
    private func detectCeilingRegion(in image: UIImage, depthMap: CIImage) -> CGRect? {
        // Simple heuristic: ceiling is top 15% of image
        return CGRect(x: 0, y: 0, width: 1.0, height: 0.15)
    }
    
    private func calculateVanishingPoint(from lines: [RoomStructure.Line]) -> CGPoint? {
        // Simplified: return center point
        // In real implementation, find intersection of perspective lines
        return CGPoint(x: 0.5, y: 0.4)
    }
}

// MARK: - Texture Processor
class TextureProcessor {
    func inpaintMissingRegions(_ image: UIImage, mask: UIImage) -> UIImage {
        // Simple inpainting using CoreImage
        // In production, you'd use more sophisticated inpainting
        guard let cgImage = image.cgImage else { return image }
        let ciImage = CIImage(cgImage: cgImage)
        
        // Apply blur to create simple inpaint effect
        let blurred = ciImage.applyingGaussianBlur(sigma: 10.0)
        
        // Blend based on mask
        let filter = CIFilter(name: "CIBlendWithMask")!
        filter.setValue(blurred, forKey: "inputImage")
        filter.setValue(ciImage, forKey: "inputBackgroundImage")
        filter.setValue(CIImage(cgImage: mask.cgImage!), forKey: "inputMaskImage")
        
        guard let output = filter.outputImage,
              let cgOutput = CIContext().createCGImage(output, from: output.extent) else {
            return image
        }
        
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
    @State private var showDimensionAdjustment = false
    
    var body: some View {
        VStack {
            if let image = selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 300)
                    .cornerRadius(12)
                    .padding()
            } else {
                Button(action: { showImagePicker = true }) {
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
            }
            
            if let dimensions = reconstructor.estimatedDimensions {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Estimated Dimensions")
                        .font(.headline)
                    
                    HStack {
                        Image(systemName: "arrow.left.and.right")
                        Text("Width: \(String(format: "%.1f", adjustedWidth))m")
                        Slider(value: $adjustedWidth, in: 2...8, step: 0.1)
                    }
                    
                    HStack {
                        Image(systemName: "arrow.up.and.down")
                        Text("Depth: \(String(format: "%.1f", adjustedDepth))m")
                        Slider(value: $adjustedDepth, in: 2...8, step: 0.1)
                    }
                    
                    HStack {
                        Image(systemName: "arrow.up.to.line")
                        Text("Height: \(String(format: "%.1f", adjustedHeight))m")
                        Slider(value: $adjustedHeight, in: 2.2...4, step: 0.1)
                    }
                    
                    Text("Confidence: \(Int(dimensions.confidence * 100))%")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(12)
                .padding()
                
                Button("Rebuild with Adjusted Dimensions") {
                    rebuildRoom()
                }
                .buttonStyle(.borderedProminent)
            }
            
            if let roomURL = reconstructor.generatedRoomURL {
                NavigationLink("View 3D Room", destination: SceneKitViewer(url: roomURL))
                    .buttonStyle(.borderedProminent)
                    .padding()
            }
            
            Spacer()
        }
        .navigationTitle("Photo to 3D Room")
        .sheet(isPresented: $showImagePicker) {
            PhotoPickerView(selectedImage: $selectedImage)
                .onDisappear {
                    if let image = selectedImage {
                        Task {
                            await reconstructor.processPhoto(image)
                            if let dims = reconstructor.estimatedDimensions {
                                adjustedWidth = dims.width
                                adjustedDepth = dims.depth
                                adjustedHeight = dims.height
                            }
                        }
                    }
                }
        }
        .onAppear {
            adjustedWidth = reconstructor.estimatedDimensions?.width ?? 4.0
            adjustedDepth = reconstructor.estimatedDimensions?.depth ?? 4.0
            adjustedHeight = reconstructor.estimatedDimensions?.height ?? 2.8
        }
    }
    
    private func rebuildRoom() {
        // Update dimensions and rebuild
        var updatedDimensions = reconstructor.estimatedDimensions ?? SinglePhotoRoomReconstructor.RoomDimensions()
        updatedDimensions.width = adjustedWidth
        updatedDimensions.depth = adjustedDepth
        updatedDimensions.height = adjustedHeight
        
        Task {
            await MainActor.run {
                reconstructor.estimatedDimensions = updatedDimensions
            }
            
            if let image = selectedImage {
                await reconstructor.processPhoto(image)
            }
        }
    }
}

// MARK: - Photo Picker View (iOS 16+ PhotosUI)
struct PhotoPickerView: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
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
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.selectedImage = image
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

// MARK: - SceneKit Viewer
struct SceneKitViewer: View {
    let url: URL
    
    var body: some View {
        SceneView(
            scene: try? SCNScene(url: url),
            options: [.allowsCameraControl, .autoenablesDefaultLighting]
        )
        .navigationTitle("3D Room View")
        .navigationBarTitleDisplayMode(.inline)
    }
}
