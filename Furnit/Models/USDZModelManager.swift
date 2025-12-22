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
        logDebug("📦 [USDZModelManager] Initializing...")
        createDirectoriesIfNeeded()
        loadModels()
        logDebug("📦 [USDZModelManager] Initialization complete. Loaded \(models.count) models")
    }
    
    private func createDirectoriesIfNeeded() {
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: modelsDirectory.path) {
            do {
                try fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
                logDebug("✅ [USDZModelManager] Created SavedRooms directory")
            } catch {
                logDebug("❌ [USDZModelManager] Failed to create directory: \(error)")
            }
        }
    }
    
    private func loadModels() {
        logDebug("📦 [USDZModelManager] Starting to load models...")
        
        let modelNames = [
            "vintage_living_room",
            "cozy_living_room_baked"
        ]
        
        logDebug("📦 [USDZModelManager] Loading bundle models: \(modelNames)")
        
        // Load bundle models
        var bundleModels = modelNames.compactMap { name in
            let model = USDZModel(name: name, fileName: name, isSavedRoom: false)
            if model.dataAsset != nil {
                logDebug("   ✅ Bundle model loaded: \(name)")
                return model
            } else {
                logDebug("   ❌ Bundle model failed: \(name)")
                return nil
            }
        }
        
        // Load saved rooms and sort by date
        var savedRoomModels: [USDZModel] = []
        
        do {
            let files = try FileManager.default.contentsOfDirectory(at: modelsDirectory,
                                                                    includingPropertiesForKeys: [.creationDateKey, .fileSizeKey])
            let usdzFiles = files.filter { $0.pathExtension.lowercased() == "usdz" }
            
            logDebug("📦 [USDZModelManager] Found \(usdzFiles.count) USDZ files in SavedRooms")
            
            // Get files with dates
            var filesWithDates: [(url: URL, date: Date)] = []
            for fileURL in usdzFiles {
                let attrs = try fileURL.resourceValues(forKeys: [.creationDateKey, .fileSizeKey])
                let date = attrs.creationDate ?? Date.distantPast
                let size = attrs.fileSize ?? 0
                filesWithDates.append((url: fileURL, date: date))
                
                let fileName = fileURL.deletingPathExtension().lastPathComponent
                logDebug("   - \(fileName) (created: \(date), size: \(size / 1024) KB)")
            }
            
            // Sort by date - NEWEST FIRST
            filesWithDates.sort { $0.date > $1.date }
            
            logDebug("📦 [USDZModelManager] Sorted by date (newest first):")
            
            // Create models in sorted order
            for (fileURL, date) in filesWithDates {
                let fileName = fileURL.deletingPathExtension().lastPathComponent
                logDebug("   - \(fileName)")
                let model = USDZModel(name: fileName, fileName: fileName, isSavedRoom: true)
                savedRoomModels.append(model)
            }
            
        } catch {
            logDebug("⚠️ [USDZModelManager] Error reading saved rooms: \(error)")
        }
        
        // Combine: saved rooms first (newest at top), then bundle models
        models = savedRoomModels + bundleModels
        
        logDebug("📦 [USDZModelManager] Loading complete:")
        logDebug("   - Total models: \(models.count)")
        logDebug("   - Saved rooms: \(savedRoomModels.count)")
        logDebug("   - Bundle models: \(bundleModels.count)")
        if !models.isEmpty {
            logDebug("   - First 3: \(models.prefix(3).map { $0.fileName }.joined(separator: ", "))")
        }
    }
    
    func getModel(by fileName: String) -> USDZModel? {
        return models.first { $0.fileName == fileName }
    }
    
    func refreshModels() {
        logDebug("🔄 [USDZModelManager] Refreshing models...")
        models.removeAll()
        loadModels()
        logDebug("✅ [USDZModelManager] Refresh complete. Now have \(models.count) models")
    }
    
    // ✅ NEW: Delete model functionality
    func deleteModel(id: UUID) {
        logDebug("🗑️ [USDZModelManager] Starting delete for model ID: \(id)")
        
        guard let modelIndex = models.firstIndex(where: { $0.id == id }) else {
            logDebug("❌ [USDZModelManager] Model not found with ID: \(id)")
            return
        }
        
        let model = models[modelIndex]
        logDebug("🗑️ [USDZModelManager] Found model: \(model.displayName) (isSavedRoom: \(model.isSavedRoom))")
        
        // Only delete file if it's a saved room (not bundle model)
        if model.isSavedRoom {
            let fileURL = modelsDirectory.appendingPathComponent("\(model.fileName).usdz")
            
            do {
                if FileManager.default.fileExists(atPath: fileURL.path) {
                    try FileManager.default.removeItem(at: fileURL)
                    logDebug("✅ [USDZModelManager] File deleted: \(fileURL.lastPathComponent)")
                } else {
                    logDebug("⚠️ [USDZModelManager] File not found: \(fileURL.path)")
                }
            } catch {
                logDebug("❌ [USDZModelManager] Failed to delete file: \(error)")
            }
        } else {
            logDebug("⚠️ [USDZModelManager] Skipping file deletion - this is a bundle model")
        }
        
        // Remove from models array
        models.remove(at: modelIndex)
        logDebug("✅ [USDZModelManager] Model removed from list. Remaining: \(models.count)")
    }
    
    func saveRoom(scene: SCNScene, name: String, completion: @escaping (Bool, String?) -> Void) {
        logDebug("💾 [USDZModelManager] Starting to save room: \(name)")
        
        let fileName = sanitizeFileName(name)
        let fileURL = modelsDirectory.appendingPathComponent("\(fileName).usdz")
        
        // Add lighting to the scene before export
        addLightingToScene(scene)
        
        // Export scene to USDZ
        scene.write(to: fileURL, options: nil, delegate: nil) { progress, error, _ in
            if let error = error {
                logDebug("❌ [USDZModelManager] Export failed: \(error)")
                DispatchQueue.main.async {
                    completion(false, error.localizedDescription)
                }
                return
            }
            
            if progress >= 1.0 {
                logDebug("✅ [USDZModelManager] Export complete: \(fileURL.lastPathComponent)")
                
                // Refresh models list after save
                DispatchQueue.main.async {
                    self.refreshModels()
                    completion(true, nil)
                }
            }
        }
    }
    
    private func addLightingToScene(_ scene: SCNScene) {
        logDebug("💡 [USDZModelManager] Adding lighting to scene...")
        
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
        logDebug("   ✅ Added ambient light (intensity: 300)")
        
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
        logDebug("   ✅ Added directional light (intensity: 800)")
        
        // Add fill light from front (helps with shadows)
        let fillLight = SCNLight()
        fillLight.type = .omni
        fillLight.color = UIColor(white: 0.9, alpha: 1.0)
        fillLight.intensity = 400
        let fillNode = SCNNode()
        fillNode.light = fillLight
        fillNode.position = SCNVector3(0, 1.5, 3)
        scene.rootNode.addChildNode(fillNode)
        logDebug("   ✅ Added fill light (intensity: 400)")
        
        logDebug("💡 [USDZModelManager] Lighting setup complete")
    }
    
    private func sanitizeFileName(_ name: String) -> String {
        let invalidChars = CharacterSet(charactersIn: ":/\\?%*|\"<>")
        let sanitized = name.components(separatedBy: invalidChars).joined()
        return sanitized.replacingOccurrences(of: " ", with: "_")
    }
}
