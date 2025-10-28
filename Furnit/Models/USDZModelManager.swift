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
        
        print("📦 [USDZModelManager] Loading bundle models: \(modelNames)")
        
        // Load bundle models
        var bundleModels = modelNames.compactMap { name in
            let model = USDZModel(name: name, fileName: name, isSavedRoom: false)
            if model.dataAsset != nil {
                print("   ✅ Bundle model loaded: \(name)")
                return model
            } else {
                print("   ❌ Bundle model failed: \(name)")
                return nil
            }
        }
        
        // Load saved rooms and sort by date
        var savedRoomModels: [USDZModel] = []
        
        do {
            let files = try FileManager.default.contentsOfDirectory(at: modelsDirectory,
                                                                    includingPropertiesForKeys: [.creationDateKey, .fileSizeKey])
            let usdzFiles = files.filter { $0.pathExtension.lowercased() == "usdz" }
            
            print("📦 [USDZModelManager] Found \(usdzFiles.count) USDZ files in SavedRooms")
            
            // Get files with dates
            var filesWithDates: [(url: URL, date: Date)] = []
            for fileURL in usdzFiles {
                let attrs = try fileURL.resourceValues(forKeys: [.creationDateKey, .fileSizeKey])
                let date = attrs.creationDate ?? Date.distantPast
                let size = attrs.fileSize ?? 0
                filesWithDates.append((url: fileURL, date: date))
                
                let fileName = fileURL.deletingPathExtension().lastPathComponent
                print("   - \(fileName) (created: \(date), size: \(size / 1024) KB)")
            }
            
            // Sort by date - NEWEST FIRST
            filesWithDates.sort { $0.date > $1.date }
            
            print("📦 [USDZModelManager] Sorted by date (newest first):")
            
            // Create models in sorted order
            for (fileURL, date) in filesWithDates {
                let fileName = fileURL.deletingPathExtension().lastPathComponent
                print("   - \(fileName)")
                let model = USDZModel(name: fileName, fileName: fileName, isSavedRoom: true)
                savedRoomModels.append(model)
            }
            
        } catch {
            print("⚠️ [USDZModelManager] Error reading saved rooms: \(error)")
        }
        
        // Combine: saved rooms first (newest at top), then bundle models
        models = savedRoomModels + bundleModels
        
        print("📦 [USDZModelManager] Loading complete:")
        print("   - Total models: \(models.count)")
        print("   - Saved rooms: \(savedRoomModels.count)")
        print("   - Bundle models: \(bundleModels.count)")
        if !models.isEmpty {
            print("   - First 3: \(models.prefix(3).map { $0.fileName }.joined(separator: ", "))")
        }
    }
    
    func getModel(by fileName: String) -> USDZModel? {
        return models.first { $0.fileName == fileName }
    }
    
    func refreshModels() {
        print("🔄 [USDZModelManager] Refreshing models...")
        models.removeAll()
        loadModels()
        print("✅ [USDZModelManager] Refresh complete. Now have \(models.count) models")
    }
    
    func saveRoom(scene: SCNScene, name: String, completion: @escaping (Bool, String?) -> Void) {
        print("💾 [USDZModelManager] Starting to save room: \(name)")
        
        let fileName = sanitizeFileName(name)
        let fileURL = modelsDirectory.appendingPathComponent("\(fileName).usdz")
        
        // Add lighting to the scene before export
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
                
                // Refresh models list after save
                DispatchQueue.main.async {
                    self.refreshModels()
                    completion(true, nil)
                }
            }
        }
    }
    
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
