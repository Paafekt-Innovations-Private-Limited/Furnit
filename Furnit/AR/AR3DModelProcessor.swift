import Foundation
import UIKit
import RealityKit
import simd

// AR 3D Model Processor - replaces ObjectSegmentationProcessor
// Handles backend API integration for 3D model generation from images
@MainActor
class AR3DModelProcessor: ObservableObject {
    // Published properties for UI updates
    @Published var isProcessing = false
    @Published var processingState: ARProcessingState = .idle
    @Published var generatedModel: Entity?
    @Published var errorMessage: String?
    
    // API client for backend communication
    private let apiClient = Stable3DAPIClient()
    
    // Quality settings reference
    private var qualitySettings: QualitySettings?
    
    // Generated model data for placement
    private var modelData: Data?
    
    init() {
        logDebug("🤖 AR3DModelProcessor initialized")
    }
    
    // Set quality settings reference
    func setQualitySettings(_ settings: QualitySettings) {
        qualitySettings = settings
        logDebug("🎨 Quality settings configured: \(settings.selectedQuality.displayName)")
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
    func processImage(_ image: UIImage) async -> Entity? {
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
        
        logDebug("📤 Starting 3D model generation from captured image")
        logDebug("   Image size: \(image.size)")
        
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
                    logDebug("📊 Status: \(statusMessage)")
                }
            }
            
            logDebug("✅ 3D model data received: \(modelFileData.count) bytes")
            
            // Store model data for placement
            modelData = modelFileData
            
            // Convert USDZ data to RealityKit Entity
            guard let modelEntity = await loadUSDZModel(from: modelFileData) else {
                errorMessage = "Failed to load 3D model from USDZ data"
                processingState = .error("Failed to load 3D model")
                isProcessing = false
                return nil
            }
            
            // Prepare model for AR placement
            let processedEntity = preprocessModelForAR(modelEntity)
            
            // Update state
            generatedModel = processedEntity
            processingState = .ready
            isProcessing = false
            
            logDebug("✅ 3D model ready for AR placement")
            return processedEntity
            
        } catch {
            logDebug("⚠️ 3D model generation failed: \(error)")
            errorMessage = error.localizedDescription
            processingState = .error(error.localizedDescription)
            isProcessing = false
            return nil
        }
    }
    
    // Load USDZ model from data and convert to RealityKit Entity
    private func loadUSDZModel(from data: Data) async -> Entity? {
        do {
            // Create temporary file for USDZ data
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("ar_model_\(UUID().uuidString).usdz")
            
            // Write USDZ data to temporary file
            try data.write(to: tempURL, options: [.atomicWrite])
            
            logDebug("💾 USDZ data written to: \(tempURL)")
            logDebug("   File exists: \(FileManager.default.fileExists(atPath: tempURL.path))")
            logDebug("   File size: \(data.count) bytes")
            
            // Load USDZ model using RealityKit's Entity loading
            let modelEntity = try await Entity.load(contentsOf: tempURL)
            
            logDebug("✅ USDZ model loaded successfully using RealityKit")
            
            // Log entity hierarchy for debugging
            logEntityHierarchy(modelEntity, indent: "")
            
            // Clean up temporary file
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                try? FileManager.default.removeItem(at: tempURL)
            }
            
            return modelEntity
            
        } catch {
            logDebug("⚠️ Failed to load USDZ model with RealityKit: \(error)")
            return nil
        }
    }
    
    // Preprocess the loaded model for AR placement with proper container structure
    private func preprocessModelForAR(_ entity: Entity) -> Entity {
        logDebug("🔧 Preprocessing 3D model for AR placement (using RealityKit)")
        
        // Create a container entity that will be positioned at the placement point
        let containerEntity = Entity()
        containerEntity.name = "AR_Model_Container"
        
        // Clone the original model to avoid modifying the original
        let modelEntity = entity.clone(recursive: true)
        
        // Calculate bounding box to understand model dimensions
        let bounds = modelEntity.components[ModelComponent.self]?.mesh.bounds
        let size: SIMD3<Float>
        let minBound: SIMD3<Float>
        let maxBound: SIMD3<Float>
        
        if let bounds = bounds {
            minBound = bounds.min
            maxBound = bounds.max
            size = maxBound - minBound
        } else {
            // Default bounds if no model component found
            minBound = SIMD3<Float>(-0.5, 0, -0.5)
            maxBound = SIMD3<Float>(0.5, 1, 0.5)
            size = SIMD3<Float>(1, 1, 1)
        }
        
        logDebug("   Original size: \(size)")
        logDebug("   Original bounding box: min(\(minBound)), max(\(maxBound))")
        
        // Intelligent scaling: only scale if model is unreasonably small or large
        let maxDimension = max(size.x, max(size.y, size.z))
        logDebug("   Max dimension: \(maxDimension) units")
        
        if maxDimension < 0.1 {
            // Model is tiny (less than 10cm) - likely wrong units, scale up to furniture size
            let targetSize: Float = 1.0  // 1 meter for typical furniture
            let scaleFactor = targetSize / maxDimension
            modelEntity.scale = SIMD3<Float>(scaleFactor, scaleFactor, scaleFactor)
            logDebug("   Model too small (\(maxDimension)m), scaling up by \(scaleFactor)x")
        } else if maxDimension > 10.0 {
            // Model is huge (larger than 10 meters) - scale down to reasonable size
            let targetSize: Float = 2.0  // 2 meters max for furniture
            let scaleFactor = targetSize / maxDimension
            modelEntity.scale = SIMD3<Float>(scaleFactor, scaleFactor, scaleFactor)
            logDebug("   Model too large (\(maxDimension)m), scaling down by \(scaleFactor)x")
        } else {
            // Model size is reasonable (0.1m to 10m) - use original scale
            logDebug("   Model size reasonable (\(maxDimension)m), using original scale")
        }
        
        // Position model so its bottom sits at the container's origin (Y=0)
        // This ensures the furniture sits on the floor rather than floating
        let bottomY = minBound.y
        modelEntity.position = SIMD3<Float>(
            0,        // Center horizontally
            -bottomY, // Lift model so bottom is at Y=0 
            0         // Center depth-wise
        )
        
        logDebug("   Model positioned with bottom at floor: \(modelEntity.position)")
        
        // Add the model to the container
        containerEntity.addChild(modelEntity)
        
        // Enhance materials for AR display
        enhanceMaterialsForAR(containerEntity)
        
        // Log container structure for debugging
        logEntityHierarchy(containerEntity, indent: "   Container: ")
        
        return containerEntity
    }
    
    // Enhance materials for better AR visualization
    private func enhanceMaterialsForAR(_ entity: Entity) {
        // Recursively enhance materials for all child entities
        enhanceMaterialsRecursively(entity)
        
        logDebug("✨ Enhanced entity hierarchy for AR display")
    }
    
    // Recursively enhance materials for entity and its children
    private func enhanceMaterialsRecursively(_ entity: Entity) {
        // Enhance materials for current entity
        if var modelComponent = entity.components[ModelComponent.self] {
            var materials = modelComponent.materials
            
            for i in 0..<materials.count {
                // RealityKit uses PBR materials - enhance for AR display
                if let material = materials[i] as? SimpleMaterial {
                    var enhancedMaterial = material
                    
                    // Set reasonable default values for better AR visualization
                    // SimpleMaterial uses MaterialScalarParameter for metallic and roughness
                    enhancedMaterial.metallic = MaterialScalarParameter(floatLiteral: 0.1)
                    enhancedMaterial.roughness = MaterialScalarParameter(floatLiteral: 0.3)
                    
                    materials[i] = enhancedMaterial
                } else if let material = materials[i] as? PhysicallyBasedMaterial {
                    var enhancedMaterial = material
                    
                    // Enhance PBR material properties
                    // Set reasonable default values for metallic and roughness
                    enhancedMaterial.metallic = PhysicallyBasedMaterial.Metallic(floatLiteral: 0.6)
                    enhancedMaterial.roughness = PhysicallyBasedMaterial.Roughness(floatLiteral: 0.2)
                    
                    materials[i] = enhancedMaterial
                }
            }
            
            modelComponent.materials = materials
            entity.components.set(modelComponent)
        }
        
        // Recursively process children
        for child in entity.children {
            enhanceMaterialsRecursively(child)
        }
    }
    
    // Note: Ambient lighting is handled at the ARView level in RealityKit
    
    // Log entity hierarchy for debugging
    private func logEntityHierarchy(_ entity: Entity, indent: String) {
        logDebug("\(indent)Entity: \(entity.name)")
        
        if let modelComponent = entity.components[ModelComponent.self] {
            logDebug("\(indent)  Model: \(type(of: modelComponent.mesh))")
            logDebug("\(indent)  Materials: \(modelComponent.materials.count)")
        }
        
        if !entity.children.isEmpty {
            logDebug("\(indent)  Children: \(entity.children.count)")
            for child in entity.children {
                logEntityHierarchy(child, indent: indent + "    ")
            }
        }
    }
    
    // Get the generated model for placement
    func getGeneratedModel() -> Entity? {
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
        
        logDebug("🔄 AR3DModelProcessor reset")
    }
    
    // Check API health
    func checkAPIHealth() async -> Bool {
        do {
            let health = try await apiClient.checkHealth()
            logDebug("🏥 API Health: \(health.status)")
            return health.status == "healthy" || health.status == "ok"
        } catch {
            logDebug("⚠️ API Health check failed: \(error)")
            return false
        }
    }
}
