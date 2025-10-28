import Foundation
import RoomPlan
import SwiftUI

// MARK: - Room Capture Manager
@available(iOS 16.0, *)
class RoomCaptureManager: ObservableObject {
    @Published var capturedRoom: CapturedRoom?
    @Published var isProcessing = false
    @Published var statusMessage = "Ready to scan"
    @Published var exportProgress: Float = 0.0
    @Published var finalScene: URL?
    @Published var captureSession: RoomCaptureSession?  // Made @Published
    @Published var isSessionRunning = false
    @Published var instruction = ""
    @Published var feedbackText = ""
    
    init() {
        print("📂 [RoomCaptureManager] Initialized")
        _ = checkDeviceSupport()
    }
    
    // MARK: - Device Support Check
    func checkDeviceSupport() -> Bool {
        let isSupported = RoomCaptureSession.isSupported
        print("📱 [RoomCaptureManager] Device support: \(isSupported ? "✅ Supported" : "❌ Not supported")")
        return isSupported
    }
    
    // MARK: - Start Capture Session
    func startCaptureSession() -> RoomCaptureSession? {
        guard checkDeviceSupport() else {
            print("❌ [RoomCaptureManager] Device doesn't support RoomPlan")
            statusMessage = "RoomPlan not supported on this device"
            return nil
        }
        
        print("🚀 [RoomCaptureManager] Starting room capture session")
        let session = RoomCaptureSession()
        self.captureSession = session
        
        // Configure and run the session
        var configuration = RoomCaptureSession.Configuration()
        configuration.isCoachingEnabled = true  // Enable coaching overlay
        session.run(configuration: configuration)
        
        isSessionRunning = true
        statusMessage = "Point at the floor to start scanning"
        instruction = "Move slowly and steadily"
        print("✅ [RoomCaptureManager] Capture session running")
        
        return session
    }
    
    // MARK: - Stop Capture Session
    func stopCaptureSession() {
        print("🛑 [RoomCaptureManager] Stopping capture session")
        captureSession?.stop()
        captureSession = nil
        isSessionRunning = false
        statusMessage = "Scan stopped"
    }
    
    // MARK: - Update Instruction (called during scanning)
    func updateInstruction(_ text: String) {
        DispatchQueue.main.async {
            self.instruction = text
        }
    }
    
    // MARK: - Process Captured Room
    func processCapturedRoom(_ room: CapturedRoom) async {
        print("🏠 [RoomCaptureManager] Processing captured room")
        print("   - Walls: \(room.walls.count)")
        print("   - Objects: \(room.objects.count)")
        print("   - Doors: \(room.doors.count)")
        print("   - Windows: \(room.windows.count)")
        print("   - Openings: \(room.openings.count)")
        
        await MainActor.run {
            self.capturedRoom = room
            self.isProcessing = true
            self.statusMessage = "Processing room data..."
            self.feedbackText = "Found \(room.walls.count) walls, \(room.objects.count) objects"
        }
        
        // Export to USDZ
        await exportToUSDZ(room)
    }
    
    // MARK: - Export to USDZ
    private func exportToUSDZ(_ room: CapturedRoom) async {
        print("📦 [RoomCaptureManager] Exporting room to USDZ")
        
        await MainActor.run {
            self.exportProgress = 0.2
            self.statusMessage = "Creating 3D model..."
        }
        
        do {
            // Create temporary file URL
            let fileName = "RoomScan_\(UUID().uuidString).usdz"
            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
            
            print("   - Export path: \(tempURL.path)")
            
            await MainActor.run {
                self.exportProgress = 0.5
                self.statusMessage = "Generating USDZ file..."
            }
            
            // Export room to USDZ
            try await room.export(to: tempURL, exportOptions: .model)
            
            print("✅ [RoomCaptureManager] Room exported successfully")
            
            await MainActor.run {
                self.exportProgress = 1.0
                self.statusMessage = "Room scan complete!"
                self.finalScene = tempURL
                self.isProcessing = false
            }
            
        } catch {
            print("❌ [RoomCaptureManager] Export failed: \(error.localizedDescription)")
            await MainActor.run {
                self.statusMessage = "Export failed: \(error.localizedDescription)"
                self.isProcessing = false
            }
        }
    }
    
    // MARK: - Save Room to Library
    func saveRoomToLibrary(name: String, completion: @escaping (Bool, String?) -> Void) {
        print("💾 [RoomCaptureManager] ========== SAVE ROOM DEBUG START ==========")
        print("💾 [RoomCaptureManager] Room name to save: '\(name)'")
        
        guard let sceneURL = finalScene else {
            print("❌ [RoomCaptureManager] No scene available to save")
            print("   - finalScene is nil")
            completion(false, "No room scan available")
            return
        }
        
        print("💾 [RoomCaptureManager] Source file: \(sceneURL.path)")
        print("   - File exists: \(FileManager.default.fileExists(atPath: sceneURL.path))")
        
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: sceneURL.path)
            let fileSize = attributes[.size] as? UInt64 ?? 0
            print("   - File size: \(fileSize) bytes")
        } catch {
            print("   - Could not get file attributes: \(error)")
        }
        
        do {
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            print("💾 [RoomCaptureManager] Documents path: \(documentsPath.path)")
            
            let roomsFolder = documentsPath.appendingPathComponent("SavedRooms", isDirectory: true)
            print("💾 [RoomCaptureManager] SavedRooms folder: \(roomsFolder.path)")
            print("   - Folder exists: \(FileManager.default.fileExists(atPath: roomsFolder.path))")
            
            // Create SavedRooms folder if needed
            if !FileManager.default.fileExists(atPath: roomsFolder.path) {
                print("💾 [RoomCaptureManager] Creating SavedRooms folder...")
                try FileManager.default.createDirectory(at: roomsFolder, withIntermediateDirectories: true)
                print("   ✅ Folder created")
            }
            
            // Sanitize filename
            let sanitizedName = name.replacingOccurrences(of: " ", with: "_")
                                   .replacingOccurrences(of: "/", with: "_")
                                   .replacingOccurrences(of: ":", with: "_")
            print("💾 [RoomCaptureManager] Sanitized name: '\(sanitizedName)'")
            
            let permanentURL = roomsFolder.appendingPathComponent("\(sanitizedName).usdz")
            print("💾 [RoomCaptureManager] Target path: \(permanentURL.path)")
            
            // Copy file
            if FileManager.default.fileExists(atPath: permanentURL.path) {
                print("⚠️ [RoomCaptureManager] File already exists, removing old version...")
                try FileManager.default.removeItem(at: permanentURL)
                print("   ✅ Old file removed")
            }
            
            print("💾 [RoomCaptureManager] Copying file...")
            print("   - From: \(sceneURL.path)")
            print("   - To: \(permanentURL.path)")
            try FileManager.default.copyItem(at: sceneURL, to: permanentURL)
            print("   ✅ File copied successfully")
            
            // Verify the copy
            let copiedFileExists = FileManager.default.fileExists(atPath: permanentURL.path)
            print("💾 [RoomCaptureManager] Verification:")
            print("   - File exists at destination: \(copiedFileExists)")
            
            if copiedFileExists {
                let copiedAttributes = try FileManager.default.attributesOfItem(atPath: permanentURL.path)
                let copiedSize = copiedAttributes[.size] as? UInt64 ?? 0
                print("   - Copied file size: \(copiedSize) bytes")
            }
            
            // List all files in SavedRooms after saving
            let allFiles = try FileManager.default.contentsOfDirectory(at: roomsFolder, includingPropertiesForKeys: nil)
            print("💾 [RoomCaptureManager] All files in SavedRooms after save:")
            for file in allFiles {
                print("   - \(file.lastPathComponent)")
            }
            
            // Save metadata to UserDefaults (simple approach)
            var savedRooms = UserDefaults.standard.dictionary(forKey: "SavedRooms") as? [String: [String: Any]] ?? [:]
            savedRooms[sanitizedName] = [
                "filePath": permanentURL.path,
                "originalName": name,
                "createdAt": Date().timeIntervalSince1970
            ]
            UserDefaults.standard.set(savedRooms, forKey: "SavedRooms")
            print("💾 [RoomCaptureManager] Metadata saved to UserDefaults")
            print("   - Key: '\(sanitizedName)'")
            print("   - Path: \(permanentURL.path)")
            
            // Log all saved rooms in UserDefaults
            print("💾 [RoomCaptureManager] All saved rooms in UserDefaults:")
            for (key, value) in savedRooms {
                if let dict = value as? [String: Any] {
                    print("   - \(key): \(dict["filePath"] ?? "unknown")")
                }
            }
            
            print("✅ [RoomCaptureManager] Room saved successfully")
            print("💾 [RoomCaptureManager] ========== SAVE ROOM DEBUG END ==========")
            completion(true, nil)
            
        } catch {
            print("❌ [RoomCaptureManager] Save failed: \(error.localizedDescription)")
            print("   - Error details: \(error)")
            print("💾 [RoomCaptureManager] ========== SAVE ROOM DEBUG END (ERROR) ==========")
            completion(false, error.localizedDescription)
        }
    }
    
    // MARK: - Get Room Dimensions
    func getRoomDimensions() -> (width: Float, depth: Float, height: Float)? {
        guard let room = capturedRoom else { return nil }
        
        // Calculate room bounds from walls using their dimensions and transforms
        var minX: Float = .infinity
        var maxX: Float = -.infinity
        var minY: Float = .infinity
        var maxY: Float = -.infinity
        var minZ: Float = .infinity
        var maxZ: Float = -.infinity
        
        // Check walls
        for wall in room.walls {
            // Get the transform matrix to find wall position
            let transform = wall.transform
            let position = SIMD3<Float>(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
            
            // Get wall dimensions
            let dimensions = wall.dimensions
            let width = dimensions.x
            let height = dimensions.y
            
            // Calculate bounds considering wall position and size
            minX = min(minX, position.x - width / 2)
            maxX = max(maxX, position.x + width / 2)
            minY = min(minY, position.y - height / 2)
            maxY = max(maxY, position.y + height / 2)
            minZ = min(minZ, position.z)
            maxZ = max(maxZ, position.z)
        }
        
        let roomWidth = maxX - minX
        let roomDepth = maxZ - minZ
        let roomHeight = maxY - minY
        
        print("📏 [RoomCaptureManager] Room dimensions:")
        print("   - Width: \(roomWidth)m")
        print("   - Depth: \(roomDepth)m")
        print("   - Height: \(roomHeight)m")
        
        return (roomWidth, roomDepth, roomHeight)
    }
    
    // MARK: - Get Room Statistics
    func getRoomStatistics() -> String {
        guard let room = capturedRoom else { return "No room data available" }
        
        var stats = "Room Scan Statistics\n\n"
        stats += "Walls: \(room.walls.count)\n"
        stats += "Objects: \(room.objects.count)\n"
        stats += "Doors: \(room.doors.count)\n"
        stats += "Windows: \(room.windows.count)\n"
        stats += "Openings: \(room.openings.count)\n"
        
        if let dims = getRoomDimensions() {
            stats += "\nDimensions:\n"
            stats += "Width: \(String(format: "%.2f", dims.width))m\n"
            stats += "Depth: \(String(format: "%.2f", dims.depth))m\n"
            stats += "Height: \(String(format: "%.2f", dims.height))m\n"
        }
        
        return stats
    }
}
