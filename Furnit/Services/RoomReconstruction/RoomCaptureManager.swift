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
    @Published var captureSession: RoomCaptureSession?
    @Published var isSessionRunning = false
    @Published var instruction = ""
    @Published var feedbackText = ""
    @Published var isSessionInitialized = false  // ✅ NEW: Track initialization state
    
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
    
    // MARK: - Initialize Session (called by view when ready)
    func initializeSession(from captureView: RoomCaptureSession?) {
        guard captureSession == nil else {
            print("⚠️ [RoomCaptureManager] Session already initialized")
            return
        }
        
        self.captureSession = captureView
        self.isSessionInitialized = true
        print("✅ [RoomCaptureManager] Session initialized from capture view")
        
        // If we were waiting to start, start now
        if isSessionRunning && captureSession != nil {
            print("🚀 [RoomCaptureManager] Auto-starting session that was waiting")
        }
    }
    
    // MARK: - Start Capture Session
    func startCaptureSession() -> RoomCaptureSession? {
        guard checkDeviceSupport() else {
            print("❌ [RoomCaptureManager] Device doesn't support RoomPlan")
            statusMessage = "RoomPlan not supported on this device"
            return nil
        }
        
        print("🚀 [RoomCaptureManager] Preparing to start room capture session")
        
        // Set flag that we want to start scanning
        isSessionRunning = true
        statusMessage = "Point at the floor to start scanning"
        instruction = "Move slowly and steadily"
        
        // Return the session if it exists (will be nil initially, that's ok)
        return captureSession
    }
    
    // MARK: - Stop Capture Session
    func stopCaptureSession() {
        print("🛑 [RoomCaptureManager] Stopping capture session")
        
        if let session = captureSession {
            // DON'T clear delegate - let it receive the final processed result
            session.stop()
            
            // Update UI immediately
            isSessionRunning = false
            statusMessage = "Processing final scan data..."
            
            print("⏳ [RoomCaptureManager] Waiting for final room data from delegate...")
        }
    }
    
    // MARK: - Cleanup Session
    func cleanupSession() {
        print("🧹 [RoomCaptureManager] Cleaning up session resources")
        
        if let session = captureSession {
            session.delegate = nil
            
            // Delayed cleanup to ensure delegate callbacks are done
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.captureSession = nil
                self?.isSessionInitialized = false  // Reset initialization flag
                print("✅ [RoomCaptureManager] Session resources cleaned up")
            }
        }
        
        captureSession = nil
        capturedRoom = nil
        isSessionRunning = false
        isSessionInitialized = false
    }
    
    // MARK: - Update Instruction
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
            // Create file in Documents directory
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let tempFolder = documentsPath.appendingPathComponent("TempScans", isDirectory: true)
            
            // Create temp folder if needed
            if !FileManager.default.fileExists(atPath: tempFolder.path) {
                try FileManager.default.createDirectory(at: tempFolder, withIntermediateDirectories: true)
                print("✅ [RoomCaptureManager] Created TempScans directory")
            }
            
            let fileName = "RoomScan_\(UUID().uuidString).usdz"
            let exportURL = tempFolder.appendingPathComponent(fileName)
            
            print("   - Export path: \(exportURL.path)")
            
            await MainActor.run {
                self.exportProgress = 0.5
                self.statusMessage = "Generating USDZ file..."
            }
            
            // Export room to USDZ
            try await room.export(to: exportURL, exportOptions: .model)
            
            print("✅ [RoomCaptureManager] Room exported successfully")
            
            await MainActor.run {
                self.exportProgress = 1.0
                self.statusMessage = "Room scan complete!"
                self.finalScene = exportURL
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
        guard let sceneURL = finalScene else {
            print("❌ [RoomCaptureManager] No scene available to save")
            completion(false, "No room scan available")
            return
        }
        
        print("💾 [RoomCaptureManager] Saving room to library: \(name)")
        
        // Check if source file exists
        guard FileManager.default.fileExists(atPath: sceneURL.path) else {
            print("❌ [RoomCaptureManager] Source file no longer exists at: \(sceneURL.path)")
            completion(false, "Source file was deleted")
            return
        }
        
        do {
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let roomsFolder = documentsPath.appendingPathComponent("SavedRooms", isDirectory: true)
            
            // Create SavedRooms folder if needed
            if !FileManager.default.fileExists(atPath: roomsFolder.path) {
                try FileManager.default.createDirectory(at: roomsFolder, withIntermediateDirectories: true)
                print("✅ [RoomCaptureManager] Created SavedRooms directory")
            }
            
            let permanentURL = roomsFolder.appendingPathComponent("\(name).usdz")
            
            // Remove existing file if present
            if FileManager.default.fileExists(atPath: permanentURL.path) {
                try FileManager.default.removeItem(at: permanentURL)
            }
            
            // Copy the file to permanent location
            try FileManager.default.copyItem(at: sceneURL, to: permanentURL)
            
            // Verify the copy succeeded
            guard FileManager.default.fileExists(atPath: permanentURL.path) else {
                print("❌ [RoomCaptureManager] Copy failed - file not found at destination")
                completion(false, "Failed to copy file to permanent storage")
                return
            }
            
            print("✅ [RoomCaptureManager] Successfully copied to: \(permanentURL.path)")
            
            // Save metadata to UserDefaults
            var savedRooms = UserDefaults.standard.dictionary(forKey: "SavedRooms") as? [String: [String: Any]] ?? [:]
            savedRooms[name] = [
                "filePath": permanentURL.path,
                "createdAt": Date().timeIntervalSince1970
            ]
            UserDefaults.standard.set(savedRooms, forKey: "SavedRooms")
            
            // Clean up temp file after successful save
            try? FileManager.default.removeItem(at: sceneURL)
            print("🧹 [RoomCaptureManager] Cleaned up temp file")
            
            print("✅ [RoomCaptureManager] Room saved successfully")
            completion(true, nil)
            
        } catch {
            print("❌ [RoomCaptureManager] Save failed: \(error.localizedDescription)")
            completion(false, error.localizedDescription)
        }
    }
    
    // MARK: - Get Room Dimensions
    func getRoomDimensions() -> (width: Float, depth: Float, height: Float)? {
        guard let room = capturedRoom else { return nil }
        
        // Calculate room bounds from walls
        var minX: Float = .infinity
        var maxX: Float = -.infinity
        var minY: Float = .infinity
        var maxY: Float = -.infinity
        var minZ: Float = .infinity
        var maxZ: Float = -.infinity
        
        for wall in room.walls {
            let transform = wall.transform
            let position = SIMD3<Float>(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
            
            let dimensions = wall.dimensions
            let width = dimensions.x
            let height = dimensions.y
            
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
