import Foundation
import UIKit
import SceneKit

class USDZModelManager: ObservableObject {
    @Published var models: [USDZModel] = []
    
    // Directory for saved rooms
    private let documentsDirectory: URL = {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }()
    
    private var modelsDirectory: URL {
        documentsDirectory.appendingPathComponent("SavedRooms", isDirectory: true)
    }
    
    init() {
        print("📦 [USDZModelManager] Initializing...")
        createDirectoriesIfNeeded()
        loadModels()
        print("📦 [USDZModelManager] Initialization complete. Loaded \(models.count) models")
    }
    
    private func createDirectoriesIfNeeded() {
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: modelsDirectory.path) {
            do {
                try fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
                print("✅ [USDZModelManager] Created SavedRooms directory")
            } catch {
                print("❌ [USDZModelManager] Failed to create directory: \(error)")
            }
        }
    }
    
    private func loadModels() {
        print("📦 [USDZModelManager] Starting to load models...")
        
        let modelNames = [
            "vintage_living_room",
            "cozy_living_room_baked"
        ]
        
        print("📦 [USDZModelManager] Model names to load: \(modelNames)")
        
        // Check if files exist in bundle (for debugging)
        for name in modelNames {
            if let url = Bundle.main.url(forResource: name, withExtension: "usdz") {
                print("✅ [USDZModelManager] Found file in bundle: \(name).usdz at \(url)")
            } else {
                print("❌ [USDZModelManager] File NOT found in bundle: \(name).usdz")
            }
        }
        
        // Load bundle models
        var allModels = modelNames.compactMap { name in
            print("📦 [USDZModelManager] Attempting to create model: \(name)")
            let model = USDZModel(name: name, fileName: name, isSavedRoom: false)
            
            if model.dataAsset != nil {
                print("✅ [USDZModelManager] Model created successfully: \(name)")
                print("   - displayName: \(model.displayName)")
                print("   - fileName: \(model.fileName)")
                return model
            } else {
                print("❌ [USDZModelManager] Model creation failed (dataAsset is nil): \(name)")
                return nil
            }
        }
        
        // Load saved rooms
        do {
            let files = try FileManager.default.contentsOfDirectory(at: modelsDirectory, includingPropertiesForKeys: nil)
            let usdzFiles = files.filter { $0.pathExtension == "usdz" }
            
            print("📦 [USDZModelManager] Found \(usdzFiles.count) saved USDZ files")
            
            for fileURL in usdzFiles {
                let fileName = fileURL.deletingPathExtension().lastPathComponent
                print("📦 [USDZModelManager] Loading saved model: \(fileName)")
                
                let model = USDZModel(name: fileName, fileName: fileName, isSavedRoom: true)
                allModels.append(model)
                print("✅ [USDZModelManager] Added saved model: \(fileName)")
            }
        } catch {
            print("⚠️ [USDZModelManager] Error reading saved rooms: \(error)")
        }
        
        models = allModels
        print("📦 [USDZModelManager] Final models array: \(models.map { $0.fileName })")
    }
    
    func getModel(by fileName: String) -> USDZModel? {
        return models.first { $0.fileName == fileName }
    }
    
    func refreshModels() {
        print("🔄 [USDZModelManager] Refreshing models...")
        loadModels()
        print("✅ [USDZModelManager] Refresh complete. Now have \(models.count) models")
    }
    
    // ✅ FIXED: Save room with lighting and proper refresh
    func saveRoom(scene: SCNScene, name: String, completion: @escaping (Bool, String?) -> Void) {
        print("💾 [USDZModelManager] Starting to save room: \(name)")
        
        let fileName = sanitizeFileName(name)
        let fileURL = modelsDirectory.appendingPathComponent("\(fileName).usdz")
        
        // ✅ ADD LIGHTING TO THE SCENE BEFORE EXPORT
        addLightingToScene(scene)
        
        // Export scene to USDZ
        scene.write(to: fileURL, options: nil, delegate: nil) { progress, error, _ in
            if let error = error {
                print("❌ [USDZModelManager] Export failed: \(error)")
                DispatchQueue.main.async {
                    completion(false, error.localizedDescription)
                }
                return
            }
            
            if progress >= 1.0 {
                print("✅ [USDZModelManager] Export complete: \(fileURL.lastPathComponent)")
                
                // ✅ REFRESH MODELS LIST AFTER SAVE
                DispatchQueue.main.async {
                    self.refreshModels()
                    completion(true, nil)
                }
            }
        }
    }
    
    // ✅ NEW: Add proper lighting to exported scenes
    private func addLightingToScene(_ scene: SCNScene) {
        print("💡 [USDZModelManager] Adding lighting to scene...")
        
        // Remove any existing lights to avoid conflicts
        scene.rootNode.childNodes.filter { $0.light != nil }.forEach { $0.removeFromParentNode() }
        
        // Add ambient light (soft overall illumination)
        let ambientLight = SCNLight()
        ambientLight.type = .ambient
        ambientLight.color = UIColor.white
        ambientLight.intensity = 300
        let ambientNode = SCNNode()
        ambientNode.light = ambientLight
        scene.rootNode.addChildNode(ambientNode)
        print("   ✅ Added ambient light (intensity: 300)")
        
        // Add directional light from above (like sun)
        let directionalLight = SCNLight()
        directionalLight.type = .directional
        directionalLight.color = UIColor.white
        directionalLight.intensity = 800
        directionalLight.castsShadow = true
        directionalLight.shadowMode = .deferred
        let directionalNode = SCNNode()
        directionalNode.light = directionalLight
        directionalNode.position = SCNVector3(0, 5, 0)
        directionalNode.eulerAngles = SCNVector3(-Float.pi / 3, 0, 0) // Angle downward
        scene.rootNode.addChildNode(directionalNode)
        print("   ✅ Added directional light (intensity: 800)")
        
        // Add fill light from front (helps with shadows)
        let fillLight = SCNLight()
        fillLight.type = .omni
        fillLight.color = UIColor(white: 0.9, alpha: 1.0)
        fillLight.intensity = 400
        let fillNode = SCNNode()
        fillNode.light = fillLight
        fillNode.position = SCNVector3(0, 1.5, 3)
        scene.rootNode.addChildNode(fillNode)
        print("   ✅ Added fill light (intensity: 400)")
        
        print("💡 [USDZModelManager] Lighting setup complete")
    }
    
    private func sanitizeFileName(_ name: String) -> String {
        let invalidChars = CharacterSet(charactersIn: ":/\\?%*|\"<>")
        let sanitized = name.components(separatedBy: invalidChars).joined()
        return sanitized.replacingOccurrences(of: " ", with: "_")
    }
}
