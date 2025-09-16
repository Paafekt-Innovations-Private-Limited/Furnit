import Foundation
import UIKit
import SceneKit
import ModelIO

// AR 3D Model Processor - replaces ObjectSegmentationProcessor
// Handles backend API integration for 3D model generation from images
@MainActor
class AR3DModelProcessor: ObservableObject {
    // Published properties for UI updates
    @Published var isProcessing = false
    @Published var processingState: ARProcessingState = .idle
    @Published var generatedModel: SCNNode?
    @Published var errorMessage: String?
    
    // API client for backend communication
    private let apiClient = Stable3DAPIClient()
    
    // Quality settings reference
    private var qualitySettings: QualitySettings?
    
    // Generated model data for placement
    private var modelData: Data?
    
    init() {
        print("🤖 AR3DModelProcessor initialized")
    }
    
    // Set quality settings reference
    func setQualitySettings(_ settings: QualitySettings) {
        qualitySettings = settings
        print("🎨 Quality settings configured: \(settings.selectedQuality.displayName)")
    }
    
    // Check if processing should proceed based on quality settings
    private func shouldProcessImage() -> Bool {
        guard let settings = qualitySettings else {
            errorMessage = "Quality settings not configured"
            return false
        }
        
        // Only process when quality is set to standard
        if settings.selectedQuality != .standard {
            errorMessage = "3D generation only available with Standard quality setting"
            return false
        }
        
        return true
    }
    
    // Process image using backend API instead of local segmentation
    func processImage(_ image: UIImage) async -> SCNNode? {
        // Check if processing should proceed
        guard shouldProcessImage() else {
            return nil
        }
        
        // Reset state
        generatedModel = nil
        modelData = nil
        errorMessage = nil
        isProcessing = true
        
        // Update processing state
        processingState = .capturing
        
        print("📤 Starting 3D model generation from captured image")
        print("   Image size: \(image.size)")
        
        do {
            // Generate 3D model using backend API with status updates
            let modelFileData = try await apiClient.generateComplete3DModel(from: image) { [weak self] statusMessage, progress in
                Task { @MainActor in
                    if let progress = progress {
                        self?.processingState = .processing(progress: progress)
                    } else {
                        // Map status message to appropriate state
                        if statusMessage.contains("Uploading") {
                            self?.processingState = .uploading
                        } else if statusMessage.contains("Downloading") {
                            self?.processingState = .downloading
                        } else if statusMessage.contains("Baking") || statusMessage.contains("textures") {
                            self?.processingState = .baking
                        }
                    }
                    print("📊 Status: \(statusMessage)")
                }
            }
            
            print("✅ 3D model data received: \(modelFileData.count) bytes")
            
            // Store model data for placement
            modelData = modelFileData
            
            // Convert USDZ data to SceneKit node
            guard let sceneNode = await loadUSDZModel(from: modelFileData) else {
                errorMessage = "Failed to load 3D model from USDZ data"
                processingState = .error("Failed to load 3D model")
                isProcessing = false
                return nil
            }
            
            // Prepare model for AR placement
            let processedNode = preprocessModelForAR(sceneNode)
            
            // Update state
            generatedModel = processedNode
            processingState = .ready
            isProcessing = false
            
            print("✅ 3D model ready for AR placement")
            return processedNode
            
        } catch {
            print("⚠️ 3D model generation failed: \(error)")
            errorMessage = error.localizedDescription
            processingState = .error(error.localizedDescription)
            isProcessing = false
            return nil
        }
    }
    
    // Load USDZ model from data and convert to SceneKit node
    private func loadUSDZModel(from data: Data) async -> SCNNode? {
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    // Create temporary file for USDZ data
                    let tempURL = FileManager.default.temporaryDirectory
                        .appendingPathComponent("ar_model_\(UUID().uuidString).usdz")
                    
                    // Write USDZ data to temporary file with explicit permissions
                    try data.write(to: tempURL, options: [.atomicWrite])
                    
                    print("💾 USDZ data written to: \(tempURL)")
                    print("   File exists: \(FileManager.default.fileExists(atPath: tempURL.path))")
                    print("   File size: \(data.count) bytes")
                    
                    // Load using SceneKit directly with proper loading options
                    let loadingOptions: [SCNSceneSource.LoadingOption: Any] = [
                        .checkConsistency: true,
                        // .convertUnitsToMeters: true,  // Removed - was causing tiny models
                        .convertToYUp: true,
                        .createNormalsIfAbsent: true,
                        .flattenScene: false
                    ]
                    
                    // Load USDZ scene directly using SceneKit
                    let scene = try SCNScene(url: tempURL, options: loadingOptions)
                    
                    print("✅ USDZ scene loaded successfully using SceneKit")
                    print("   Root node child count: \(scene.rootNode.childNodes.count)")
                    
                    // Get the root node with all its children
                    let rootNode = scene.rootNode.clone()
                    
                    // Log node hierarchy for debugging
                    self.logNodeHierarchy(rootNode, indent: "")
                    
                    // Clean up temporary file
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        try? FileManager.default.removeItem(at: tempURL)
                    }
                    
                    continuation.resume(returning: rootNode)
                    
                } catch {
                    print("⚠️ Failed to load USDZ model: \(error)")
                    continuation.resume(returning: SCNNode())
                }
            }
        }
    }
    
    // Preprocess the loaded model for AR placement with proper container structure
    private func preprocessModelForAR(_ node: SCNNode) -> SCNNode {
        print("🔧 Preprocessing 3D model for AR placement (using original scale)")
        
        // Create a container node that will be positioned at the placement point
        let containerNode = SCNNode()
        containerNode.name = "AR_Model_Container"
        
        // Clone the original model to avoid modifying the original
        let modelNode = node.clone()
        
        // Calculate bounding box to understand model dimensions
        let (minBound, maxBound) = modelNode.boundingBox
        let size = SCNVector3(
            maxBound.x - minBound.x,
            maxBound.y - minBound.y,
            maxBound.z - minBound.z
        )
        
        print("   Original size: \(size)")
        print("   Original bounding box: min(\(minBound)), max(\(maxBound))")
        
        // Intelligent scaling: only scale if model is unreasonably small or large
        let maxDimension = max(size.x, max(size.y, size.z))
        print("   Max dimension: \(maxDimension) units")
        
        if maxDimension < 0.1 {
            // Model is tiny (less than 10cm) - likely wrong units, scale up to furniture size
            let targetSize: Float = 1.0  // 1 meter for typical furniture
            let scaleFactor = targetSize / maxDimension
            modelNode.scale = SCNVector3(scaleFactor, scaleFactor, scaleFactor)
            print("   Model too small (\(maxDimension)m), scaling up by \(scaleFactor)x")
        } else if maxDimension > 10.0 {
            // Model is huge (larger than 10 meters) - scale down to reasonable size
            let targetSize: Float = 2.0  // 2 meters max for furniture
            let scaleFactor = targetSize / maxDimension
            modelNode.scale = SCNVector3(scaleFactor, scaleFactor, scaleFactor)
            print("   Model too large (\(maxDimension)m), scaling down by \(scaleFactor)x")
        } else {
            // Model size is reasonable (0.1m to 10m) - use original scale
            print("   Model size reasonable (\(maxDimension)m), using original scale")
        }
        
        // Position model so its bottom sits at the container's origin (Y=0)
        // This ensures the furniture sits on the floor rather than floating
        let bottomY = minBound.y
        modelNode.position = SCNVector3(
            0,        // Center horizontally
            -bottomY, // Lift model so bottom is at Y=0 
            0         // Center depth-wise
        )
        
        print("   Model positioned with bottom at floor: \(modelNode.position)")
        
        // Add the model to the container
        containerNode.addChildNode(modelNode)
        
        // Enhance materials for AR display
        enhanceMaterialsForAR(containerNode)
        
        // Add subtle ambient lighting
        addAmbientLighting(to: containerNode)
        
        // Log container structure for debugging
        logNodeHierarchy(containerNode, indent: "   Container: ")
        
        return containerNode
    }
    
    // Enhance materials for better AR visualization
    private func enhanceMaterialsForAR(_ node: SCNNode) {
        node.enumerateChildNodes { childNode, _ in
            if let geometry = childNode.geometry {
                for material in geometry.materials {
                    // Ensure materials render well in AR environment
                    if material.lightingModel == .constant {
                        material.lightingModel = .blinn
                    }
                    
                    // Adjust transparency handling
                    if material.transparency < 1.0 {
                        material.blendMode = .alpha
                        material.writesToDepthBuffer = false
                    }
                    
                    // Enhance ambient reflection
                    if material.ambient.contents == nil {
                        material.ambient.contents = UIColor(white: 0.2, alpha: 1.0)
                    }
                }
            }
        }
        
        print("✨ Enhanced \(node.childNodes.count) child nodes for AR display")
    }
    
    // Add subtle ambient lighting to the model
    private func addAmbientLighting(to node: SCNNode) {
        let ambientLight = SCNLight()
        ambientLight.type = .ambient
        ambientLight.color = UIColor(white: 0.3, alpha: 1.0)
        ambientLight.intensity = 200
        
        let lightNode = SCNNode()
        lightNode.light = ambientLight
        node.addChildNode(lightNode)
        
        print("💡 Added ambient lighting to 3D model")
    }
    
    // Log node hierarchy for debugging
    private func logNodeHierarchy(_ node: SCNNode, indent: String) {
        print("\(indent)Node: \(node.name ?? "unnamed")")
        
        if let geometry = node.geometry {
            print("\(indent)  Geometry: \(type(of: geometry))")
            print("\(indent)  Materials: \(geometry.materials.count)")
        }
        
        if !node.childNodes.isEmpty {
            print("\(indent)  Children: \(node.childNodes.count)")
            for child in node.childNodes {
                logNodeHierarchy(child, indent: indent + "    ")
            }
        }
    }
    
    // Get the generated model for placement
    func getGeneratedModel() -> SCNNode? {
        return generatedModel
    }
    
    // Get model data for additional processing
    func getModelData() -> Data? {
        return modelData
    }
    
    // Reset processor state
    func reset() {
        isProcessing = false
        processingState = .idle
        generatedModel = nil
        modelData = nil
        errorMessage = nil
        
        print("🔄 AR3DModelProcessor reset")
    }
    
    // Check API health
    func checkAPIHealth() async -> Bool {
        do {
            let health = try await apiClient.checkHealth()
            print("🏥 API Health: \(health.status)")
            return health.status == "healthy" || health.status == "ok"
        } catch {
            print("⚠️ API Health check failed: \(error)")
            return false
        }
    }
}