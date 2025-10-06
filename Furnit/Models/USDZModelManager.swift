import Foundation
import UIKit
import Combine

class USDZModelManager: ObservableObject {
    @Published var models: [USDZModel] = []
    
    // Notification for when new dollhouse models are added
    static let didAddModelNotification = Notification.Name("USDZModelManager.didAddModel")
    
    init() {
        loadModels()
        
        // Listen for notifications when new dollhouse models are created
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleNewDollhouseModel(_:)),
            name: Self.didAddModelNotification,
            object: nil
        )
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
    
    @objc private func handleNewDollhouseModel(_ notification: Notification) {
        if let fileName = notification.userInfo?["fileName"] as? String {
            print("🔔 Received notification for new dollhouse model: \(fileName)")
            DispatchQueue.main.async {
                self.addDollhouseModel(fileName: fileName)
            }
        }
    }
    
    private func loadModels() {
        print("📚 Loading all models...")
        
        // Load existing bundle-based room models (don't disturb these)
        let modelNames = [
            "vintage_living_room",
            "cozy_living_room_baked"
        ]
        
        models = modelNames.compactMap { name in
            let model = USDZModel(name: name, fileName: name)
            return model.dataAsset != nil ? model : nil
        }
        
        print("✅ Loaded \(models.count) bundle models")
        
        // Load 2.5D dollhouse models from Documents directory
        loadDollhouseModels()
    }
    
    private func loadDollhouseModels() {
        print("🏠 Loading 2.5D dollhouse models from Documents directory...")
        
        guard let documentsURL = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first else {
            print("❌ Could not access Documents directory")
            return
        }
        
        do {
            let fileURLs = try FileManager.default.contentsOfDirectory(
                at: documentsURL,
                includingPropertiesForKeys: [.fileSizeKey, .creationDateKey],
                options: []
            )
            
            // Filter for dollhouse USDZ files
            let dollhouseFiles = fileURLs.filter { url in
                return url.pathExtension.lowercased() == "usdz" &&
                       url.lastPathComponent.contains("dollhouse_")
            }
            
            print("📁 Found \(dollhouseFiles.count) dollhouse files in Documents")
            
            // Sort by creation date (newest first)
            let sortedFiles = dollhouseFiles.sorted { url1, url2 in
                let date1 = (try? url1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
                let date2 = (try? url2.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
                return date1 > date2
            }
            
            // Create USDZModel objects for each dollhouse file
            for fileURL in sortedFiles {
                let fileName = fileURL.lastPathComponent
                
                // Skip if already in models array
                if models.contains(where: { $0.fileName == fileName }) {
                    print("⏭️ Skipping duplicate: \(fileName)")
                    continue
                }
                
                let displayName = formatDisplayName(from: fileName)
                
                let dollhouseModel = USDZModel(
                    name: "2.5D " + displayName,
                    fileName: fileName
                )
                
                // Verify the file exists and has content
                if FileManager.default.fileExists(atPath: fileURL.path) {
                    let fileSize = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                    
                    if fileSize > 0 {
                        models.append(dollhouseModel)
                        print("✅ Added 2.5D dollhouse: \(dollhouseModel.displayName) (\(fileSize) bytes)")
                    } else {
                        print("⚠️ Skipping empty file: \(fileName)")
                    }
                } else {
                    print("❌ File not found: \(fileName)")
                }
            }
            
            print("📊 Total models loaded: \(models.count)")
            
        } catch {
            print("❌ Error loading dollhouse models: \(error.localizedDescription)")
        }
    }
    
    private func formatDisplayName(from fileName: String) -> String {
        return fileName
            .replacingOccurrences(of: "dollhouse_", with: "")
            .replacingOccurrences(of: ".usdz", with: "")
            .replacingOccurrences(of: "_", with: " ")
            .split(separator: " ")
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }
    
    func getModel(by fileName: String) -> USDZModel? {
        return models.first { $0.fileName == fileName }
    }
    
    // Method to refresh models (call this when new dollhouse models are created)
    func refreshModels() {
        print("🔄 Refreshing model list...")
        
        // Store current bundle models
        let bundleModels = models.filter { $0.dataAsset != nil }
        
        // Clear and reload everything
        models.removeAll()
        models = bundleModels
        
        // Reload dollhouse models
        loadDollhouseModels()
        
        print("✅ Refresh complete - total models: \(models.count)")
        listDocumentsDirectory()
    }
    
    // Method to add a new dollhouse model dynamically (main thread safe)
    @MainActor
    func addDollhouseModel(fileName: String, displayName: String? = nil) {
        print("➕ Attempting to add dollhouse model: \(fileName)")
        
        guard let documentsURL = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first else {
            print("❌ Could not access Documents directory")
            return
        }
        
        let fileURL = documentsURL.appendingPathComponent(fileName)
        
        // Wait briefly for file to be fully written
        var attempts = 0
        let maxAttempts = 10
        
        while attempts < maxAttempts {
            if FileManager.default.fileExists(atPath: fileURL.path) {
                if let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize,
                   fileSize > 0 {
                    break
                }
            }
            attempts += 1
            Thread.sleep(forTimeInterval: 0.1)
        }
        
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            print("❌ Cannot add dollhouse model - file not found: \(fileName)")
            print("   Expected path: \(fileURL.path)")
            listDocumentsDirectory()
            return
        }
        
        // Check if model already exists
        if models.contains(where: { $0.fileName == fileName }) {
            print("⚠️ Dollhouse model already exists: \(fileName)")
            return
        }
        
        let modelDisplayName = displayName ?? formatDisplayName(from: fileName)
        
        let dollhouseModel = USDZModel(
            name: "2.5D " + modelDisplayName,
            fileName: fileName
        )
        
        // Add at the beginning so newest scans appear first
        models.insert(dollhouseModel, at: 0)
        
        let fileSize = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        print("✅ Added new 2.5D dollhouse model: \(dollhouseModel.displayName) (\(fileSize) bytes)")
        print("📊 Total models now: \(models.count)")
    }
    
    // Debug method to list all files in Documents directory
    func listDocumentsDirectory() {
        print("📂 Current Documents Directory Contents:")
        
        guard let documentsURL = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first else {
            print("  ❌ Could not access Documents directory")
            return
        }
        
        print("  Path: \(documentsURL.path)")
        
        do {
            let fileURLs = try FileManager.default.contentsOfDirectory(
                at: documentsURL,
                includingPropertiesForKeys: [.fileSizeKey, .creationDateKey]
            )
            
            if fileURLs.isEmpty {
                print("  (No files found)")
            } else {
                for fileURL in fileURLs {
                    let fileName = fileURL.lastPathComponent
                    let fileSize = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                    let creationDate = (try? fileURL.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
                    let isDollhouse = fileName.contains("dollhouse_") && fileName.hasSuffix(".usdz")
                    
                    let dateFormatter = DateFormatter()
                    dateFormatter.dateFormat = "MM/dd HH:mm:ss"
                    let dateString = dateFormatter.string(from: creationDate)
                    
                    print("  - \(fileName)")
                    print("    Size: \(fileSize) bytes | Created: \(dateString) \(isDollhouse ? "🏠 DOLLHOUSE" : "")")
                }
            }
        } catch {
            print("  ❌ Error listing files: \(error.localizedDescription)")
        }
    }
    
    // Method to remove a model (useful for cleanup)
    @MainActor
    func removeModel(_ model: USDZModel) {
        guard let index = models.firstIndex(where: { $0.id == model.id }) else {
            return
        }
        
        models.remove(at: index)
        
        // If it's a dollhouse model, optionally delete the file
        if model.fileName.contains("dollhouse_"),
           let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = documentsURL.appendingPathComponent(model.fileName)
            try? FileManager.default.removeItem(at: fileURL)
            print("🗑️ Removed dollhouse model: \(model.displayName)")
        }
    }
}
