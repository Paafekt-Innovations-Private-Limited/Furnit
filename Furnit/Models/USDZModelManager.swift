import Foundation
import UIKit

class USDZModelManager: ObservableObject {
    @Published var models: [USDZModel] = []
    
    init() {
        loadModels()
    }
    
    private func loadModels() {
        // Load existing bundle-based room models (don't disturb these)
        let modelNames = [
            "vintage_living_room",
            "cozy_living_room_baked"
        ]
        
        models = modelNames.compactMap { name in
            let model = USDZModel(name: name, fileName: name)
            return model.dataAsset != nil ? model : nil
        }
        
        // Load 2.5D dollhouse models from Documents directory
        loadDollhouseModels()
    }
    
    private func loadDollhouseModels() {
        print("🏠 Loading 2.5D dollhouse models from Documents directory...")
        
        let documentsURL = FileManager.default.urls(
            for: .documentDirectory, 
            in: .userDomainMask
        ).first!
        
        do {
            let fileURLs = try FileManager.default.contentsOfDirectory(
                at: documentsURL,
                includingPropertiesForKeys: [.fileSizeKey],
                options: []
            )
            
            // Filter for dollhouse USDZ files
            let dollhouseFiles = fileURLs.filter { url in
                return url.pathExtension.lowercased() == "usdz" && 
                       url.lastPathComponent.contains("dollhouse_")
            }
            
            print("📁 Found \(dollhouseFiles.count) dollhouse files in Documents")
            
            // Create USDZModel objects for each dollhouse file
            for fileURL in dollhouseFiles {
                let fileName = fileURL.lastPathComponent
                let displayName = fileName
                    .replacingOccurrences(of: "dollhouse_", with: "")
                    .replacingOccurrences(of: ".usdz", with: "")
                    .replacingOccurrences(of: "_", with: " ")
                    .capitalized
                
                let dollhouseModel = USDZModel(
                    name: "2.5D " + displayName,
                    fileName: fileName
                )
                
                // Verify the file exists before adding
                if FileManager.default.fileExists(atPath: fileURL.path) {
                    models.append(dollhouseModel)
                    print("✅ Added 2.5D dollhouse: \(dollhouseModel.displayName)")
                } else {
                    print("❌ File not found: \(fileName)")
                }
            }
            
            print("📊 Total models loaded: \(models.count)")
            
        } catch {
            print("❌ Error loading dollhouse models: \(error)")
        }
    }
    
    func getModel(by fileName: String) -> USDZModel? {
        return models.first { $0.fileName == fileName }
    }
    
    // Method to refresh models (call this when new dollhouse models are created)
    func refreshModels() {
        print("🔄 Refreshing model list...")
        listDocumentsDirectory() // Debug: show what files exist
        loadModels()
    }
    
    // Method to add a new dollhouse model dynamically
    func addDollhouseModel(fileName: String, displayName: String? = nil) {
        let documentsURL = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!
        let fileURL = documentsURL.appendingPathComponent(fileName)
        
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            print("❌ Cannot add dollhouse model - file not found: \(fileName)")
            return
        }
        
        // Check if model already exists
        if models.contains(where: { $0.fileName == fileName }) {
            print("⚠️ Dollhouse model already exists: \(fileName)")
            return
        }
        
        let modelDisplayName = displayName ?? fileName
            .replacingOccurrences(of: "dollhouse_", with: "")
            .replacingOccurrences(of: ".usdz", with: "")
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
        
        let dollhouseModel = USDZModel(
            name: "2.5D " + modelDisplayName,
            fileName: fileName
        )
        
        models.append(dollhouseModel)
        print("✅ Added new 2.5D dollhouse model: \(dollhouseModel.displayName)")
    }
    
    // Debug method to list all files in Documents directory
    func listDocumentsDirectory() {
        print("📂 Current Documents Directory Contents:")
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        
        do {
            let fileURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: [.fileSizeKey])
            
            if fileURLs.isEmpty {
                print("  (No files found)")
            } else {
                for fileURL in fileURLs {
                    let fileName = fileURL.lastPathComponent
                    let fileSize = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                    let isDollhouse = fileName.contains("dollhouse_") && fileName.hasSuffix(".usdz")
                    
                    print("  - \(fileName) (\(fileSize) bytes) \(isDollhouse ? "🏠 DOLLHOUSE" : "")")
                }
            }
        } catch {
            print("  ❌ Error listing files: \(error)")
        }
    }
}
