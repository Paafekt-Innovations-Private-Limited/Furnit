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
        if AppStateManager.shared.qualitySettings.debugMode {
            logDebug("📦 [USDZModelManager] Initializing...")
        }
        createDirectoriesIfNeeded()
        loadModels()
        if AppStateManager.shared.qualitySettings.debugMode {
            logDebug("📦 [USDZModelManager] Initialization complete. Loaded \(models.count) models")
        }
    }
    
    private func createDirectoriesIfNeeded() {
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: modelsDirectory.path) {
            do {
                try fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
                if AppStateManager.shared.qualitySettings.debugMode {
                    logDebug("✅ [USDZModelManager] Created SavedRooms directory")
                }
            } catch {
                if AppStateManager.shared.qualitySettings.debugMode {
                    logDebug("❌ [USDZModelManager] Failed to create directory: \(error)")
                }
            }
        }
    }
    
    private func loadModels() {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            logDebug("📦 [USDZModelManager] Starting to load models...")
        }
        
        let modelNames = [
            "vintage_living_room",
            "cozy_living_room_baked"
        ]
        
        if debugMode {
            logDebug("📦 [USDZModelManager] Loading bundle models: \(modelNames)")
        }
        
        // Load bundle models
        let bundleModels = modelNames.compactMap { name in
            let model = USDZModel(name: name, fileName: name, isSavedRoom: false)
            if model.dataAsset != nil {
                if debugMode {
                    logDebug("   ✅ Bundle model loaded: \(name)")
                }
                return model
            } else {
                if debugMode {
                    logDebug("   ❌ Bundle model failed: \(name)")
                }
                return nil
            }
        }
        
        // Load saved rooms (both USDZ and PLY) and sort by date
        var savedRoomModels: [USDZModel] = []

        // Supported file extensions
        let supportedExtensions = ["usdz", "ply"]

        do {
            let files = try FileManager.default.contentsOfDirectory(at: modelsDirectory,
                                                                    includingPropertiesForKeys: [.creationDateKey, .fileSizeKey])
            let modelFiles = files.filter { supportedExtensions.contains($0.pathExtension.lowercased()) }

            if debugMode {
                logDebug("📦 [USDZModelManager] Found \(modelFiles.count) model files in SavedRooms")
            }

            // Get files with dates and sizes
            var filesWithDates: [(url: URL, date: Date, size: UInt64)] = []
            for fileURL in modelFiles {
                let attrs = try fileURL.resourceValues(forKeys: [.creationDateKey, .fileSizeKey])
                let date = attrs.creationDate ?? Date.distantPast
                let size = UInt64(attrs.fileSize ?? 0)
                filesWithDates.append((url: fileURL, date: date, size: size))

                if debugMode {
                    let fileName = fileURL.deletingPathExtension().lastPathComponent
                    let ext = fileURL.pathExtension.lowercased()
                    logDebug("   - \(fileName).\(ext) (created: \(date), size: \(size / 1024) KB)")
                }
            }

            // Sort by date - NEWEST FIRST
            filesWithDates.sort { $0.date > $1.date }

            if debugMode {
                logDebug("📦 [USDZModelManager] Sorted by date (newest first):")
            }

            // Create models in sorted order with proper file type
            for (fileURL, _, size) in filesWithDates {
                let fileName = fileURL.deletingPathExtension().lastPathComponent
                let ext = fileURL.pathExtension.lowercased()
                let fileType: ModelFileType = (ext == "ply") ? .ply : .usdz

                // Load photo orientation from metadata for PLY files
                let orientation: PhotoOrientation = (fileType == .ply) ? loadPhotoOrientation(for: fileName) : .portrait

                if debugMode {
                    logDebug("   - \(fileName) (\(fileType.rawValue), orientation: \(orientation.rawValue))")
                }

                let model = USDZModel(
                    name: fileName,
                    fileName: fileName,
                    isSavedRoom: true,
                    fileType: fileType,
                    fileSize: size,
                    photoOrientation: orientation
                )
                savedRoomModels.append(model)
            }

        } catch {
            if debugMode {
                logDebug("⚠️ [USDZModelManager] Error reading saved rooms: \(error)")
            }
        }
        
        // Combine: saved rooms first (newest at top), then bundle models
        models = savedRoomModels + bundleModels
        
        if debugMode {
            logDebug("📦 [USDZModelManager] Loading complete:")
            logDebug("   - Total models: \(models.count)")
            logDebug("   - Saved rooms: \(savedRoomModels.count)")
            logDebug("   - Bundle models: \(bundleModels.count)")
            if !models.isEmpty {
                logDebug("   - First 3: \(models.prefix(3).map { $0.fileName }.joined(separator: ", "))")
            }
        }
    }
    
    func getModel(by fileName: String) -> USDZModel? {
        return models.first { $0.fileName == fileName }
    }
    
    func refreshModels() {
        if AppStateManager.shared.qualitySettings.debugMode {
            logDebug("🔄 [USDZModelManager] Refreshing models...")
        }
        models.removeAll()
        loadModels()
        if AppStateManager.shared.qualitySettings.debugMode {
            logDebug("✅ [USDZModelManager] Refresh complete. Now have \(models.count) models")
        }
    }
    
    // ✅ NEW: Delete model functionality
    func deleteModel(id: UUID) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            logDebug("🗑️ [USDZModelManager] Starting delete for model ID: \(id)")
        }
        
        guard let modelIndex = models.firstIndex(where: { $0.id == id }) else {
            if debugMode {
                logDebug("❌ [USDZModelManager] Model not found with ID: \(id)")
            }
            return
        }
        
        let model = models[modelIndex]
        if debugMode {
            logDebug("🗑️ [USDZModelManager] Found model: \(model.displayName) (isSavedRoom: \(model.isSavedRoom))")
        }
        
        // Only delete file if it's a saved room (not bundle model)
        if model.isSavedRoom {
            // Use appropriate file extension based on file type
            let fileExtension = model.fileType == .ply ? "ply" : "usdz"
            let fileURL = modelsDirectory.appendingPathComponent("\(model.fileName).\(fileExtension)")
            
            do {
                if FileManager.default.fileExists(atPath: fileURL.path) {
                    try FileManager.default.removeItem(at: fileURL)
                    if debugMode {
                        logDebug("✅ [USDZModelManager] File deleted: \(fileURL.lastPathComponent)")
                    }
                } else {
                    if debugMode {
                        logDebug("⚠️ [USDZModelManager] File not found: \(fileURL.path)")
                    }
                }
            } catch {
                if debugMode {
                    logDebug("❌ [USDZModelManager] Failed to delete file: \(error)")
                }
                CrashReporter.shared.report(error, context: "Deleting Room")
            }
        } else {
            if debugMode {
                logDebug("⚠️ [USDZModelManager] Skipping file deletion - this is a bundle model")
            }
        }
        
        // Remove from models array
        models.remove(at: modelIndex)
        if debugMode {
            logDebug("✅ [USDZModelManager] Model removed from list. Remaining: \(models.count)")
        }
    }
    
    func saveRoom(scene: SCNScene, name: String, completion: @escaping (Bool, String?) -> Void) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            logDebug("💾 [USDZModelManager] Starting to save room: \(name)")
        }
        
        let fileName = sanitizeFileName(name)
        let fileURL = modelsDirectory.appendingPathComponent("\(fileName).usdz")
        
        // Add lighting to the scene before export
        addLightingToScene(scene)
        
        // Export scene to USDZ
        scene.write(to: fileURL, options: nil, delegate: nil) { progress, error, _ in
            if let error = error {
                if debugMode {
                    logDebug("❌ [USDZModelManager] Export failed: \(error)")
                }
                DispatchQueue.main.async {
                    completion(false, error.localizedDescription)
                }
                return
            }
            
            if progress >= 1.0 {
                if debugMode {
                    logDebug("✅ [USDZModelManager] Export complete: \(fileURL.lastPathComponent)")
                }
                
                // Refresh models list after save
                DispatchQueue.main.async {
                    self.refreshModels()
                    completion(true, nil)
                }
            }
        }
    }
    
    private func addLightingToScene(_ scene: SCNScene) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            logDebug("💡 [USDZModelManager] Adding lighting to scene...")
        }
        
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
        if debugMode {
            logDebug("   ✅ Added ambient light (intensity: 300)")
        }
        
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
        if debugMode {
            logDebug("   ✅ Added directional light (intensity: 800)")
        }
        
        // Add fill light from front (helps with shadows)
        let fillLight = SCNLight()
        fillLight.type = .omni
        fillLight.color = UIColor(white: 0.9, alpha: 1.0)
        fillLight.intensity = 400
        let fillNode = SCNNode()
        fillNode.light = fillLight
        fillNode.position = SCNVector3(0, 1.5, 3)
        scene.rootNode.addChildNode(fillNode)
        if debugMode {
            logDebug("   ✅ Added fill light (intensity: 400)")
        }
        
        if debugMode {
            logDebug("💡 [USDZModelManager] Lighting setup complete")
        }
    }
    
    private func sanitizeFileName(_ name: String) -> String {
        let invalidChars = CharacterSet(charactersIn: ":/\\?%*|\"<>")
        let sanitized = name.components(separatedBy: invalidChars).joined()
        return sanitized.replacingOccurrences(of: " ", with: "_")
    }

    /// Save a PLY file to SavedRooms directory
    func savePLY(from sourceURL: URL, name: String, photoOrientation: PhotoOrientation = .portrait, completion: @escaping (Bool, String?) -> Void) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode

        if debugMode {
            logDebug("💾 [USDZModelManager] Starting to save PLY: \(name) (orientation: \(photoOrientation.rawValue))")
        }

        let fileName = sanitizeFileName(name)
        let destinationURL = modelsDirectory.appendingPathComponent("\(fileName).ply")
        let metadataURL = modelsDirectory.appendingPathComponent("\(fileName).ply.meta")

        // Check if source exists
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            if debugMode {
                logDebug("❌ [USDZModelManager] Source PLY not found: \(sourceURL.path)")
            }
            completion(false, "Source file not found")
            return
        }

        do {
            // Remove existing file if it exists
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            if FileManager.default.fileExists(atPath: metadataURL.path) {
                try FileManager.default.removeItem(at: metadataURL)
            }

            // Copy PLY file
            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)

            // Save metadata (orientation)
            let metadata: [String: String] = ["photoOrientation": photoOrientation.rawValue]
            let metadataData = try JSONEncoder().encode(metadata)
            try metadataData.write(to: metadataURL)

            if debugMode {
                logDebug("✅ [USDZModelManager] PLY saved to: \(destinationURL.path)")
                logDebug("✅ [USDZModelManager] Metadata saved to: \(metadataURL.path)")
            }

            // Reload models to include the new one
            DispatchQueue.main.async {
                self.loadModels()
                completion(true, nil)
            }
        } catch {
            if debugMode {
                logDebug("❌ [USDZModelManager] Failed to save PLY: \(error)")
            }
            completion(false, error.localizedDescription)
        }
    }

    /// Load photo orientation from metadata file
    private func loadPhotoOrientation(for fileName: String) -> PhotoOrientation {
        let metadataURL = modelsDirectory.appendingPathComponent("\(fileName).ply.meta")

        guard FileManager.default.fileExists(atPath: metadataURL.path),
              let data = try? Data(contentsOf: metadataURL),
              let metadata = try? JSONDecoder().decode([String: String].self, from: data),
              let orientationStr = metadata["photoOrientation"],
              let orientation = PhotoOrientation(rawValue: orientationStr) else {
            return .portrait  // Default to portrait if no metadata
        }

        return orientation
    }
}
