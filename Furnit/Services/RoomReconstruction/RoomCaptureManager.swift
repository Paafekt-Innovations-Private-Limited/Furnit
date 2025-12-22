import Foundation
import RoomPlan
import SwiftUI

// MARK: - Room Capture Manager (Complete Production Version with All Fixes)
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
    @Published var isSessionInitialized = false
    
    // Track if we're waiting for final data
    private var isWaitingForFinalData = false
    private var hasReceivedFinalData = false
    
    init() {
        logDebug("📂 [RoomCaptureManager] Initialized")
        _ = checkDeviceSupport()
    }
    
    // MARK: - Device Support Check
    func checkDeviceSupport() -> Bool {
        let isSupported = RoomCaptureSession.isSupported
        logDebug("📱 [RoomCaptureManager] Device support: \(isSupported ? "✅ Supported" : "❌ Not supported")")
        return isSupported
    }
    
    // MARK: - Initialize Session
    func initializeSession(from captureView: RoomCaptureSession?) {
        guard captureSession == nil else {
            logDebug("⚠️ [RoomCaptureManager] Session already initialized")
            return
        }
        
        self.captureSession = captureView
        self.isSessionInitialized = true
        logDebug("✅ [RoomCaptureManager] Session initialized from capture view")
        
        if isSessionRunning && captureSession != nil {
            logDebug("🚀 [RoomCaptureManager] Auto-starting session that was waiting")
        }
    }
    
    // MARK: - Start Capture Session
    func startCaptureSession() -> RoomCaptureSession? {
        guard checkDeviceSupport() else {
            logDebug("❌ [RoomCaptureManager] Device doesn't support RoomPlan")
            statusMessage = "RoomPlan not supported on this device"
            return nil
        }
        
        logDebug("🚀 [RoomCaptureManager] Preparing to start room capture session")
        
        // Reset flags
        hasReceivedFinalData = false
        isWaitingForFinalData = false
        
        isSessionRunning = true
        statusMessage = "Point at the floor to start scanning"
        instruction = "Move slowly and steadily"
        
        return captureSession
    }
    
    // MARK: - Stop Capture Session (Fixed - no unnecessary try)
    func stopCaptureSession() {
        logDebug("🛑 [RoomCaptureManager] Stopping capture session")
        
        guard let session = captureSession else {
            logDebug("⚠️ [RoomCaptureManager] No session to stop")
            return
        }
        
        // Mark that we're waiting for final data
        isWaitingForFinalData = true
        hasReceivedFinalData = false
        
        // Update UI
        isSessionRunning = false
        statusMessage = "Processing scan data..."
        
        // Check if we already have room data
        if let currentRoom = self.capturedRoom {
            logDebug("📦 [RoomCaptureManager] Already have room data: \(currentRoom.walls.count) walls")
        }
        
        // Now stop the session
        logDebug("⏸️ [RoomCaptureManager] Calling session.stop() - delegate should receive final data")
        session.stop()
        
        // If we have a room but delegate doesn't fire, process it anyway
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            guard let self = self else { return }
            
            if let room = self.capturedRoom, self.finalScene == nil {
                logDebug("🔧 [RoomCaptureManager] Forcing room processing after stop")
                Task {
                    await self.processCapturedRoom(room)
                }
            }
        }
    }
    
    // MARK: - Force Process Current Data
    func forceProcessCurrentData() {
        logDebug("⚠️ [RoomCaptureManager] Forcing processing of current data")
        
        if let room = capturedRoom {
            logDebug("📦 [RoomCaptureManager] Processing existing captured room")
            Task {
                await processCapturedRoom(room)
            }
        } else {
            logDebug("❌ [RoomCaptureManager] No room data to process")
            statusMessage = "No room data captured"
        }
    }
    
    // MARK: - Cleanup Session
    func cleanupSession() {
        logDebug("🧹 [RoomCaptureManager] Cleaning up session resources")
        
        if let session = captureSession {
            session.delegate = nil
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.captureSession = nil
                self?.isSessionInitialized = false
                logDebug("✅ [RoomCaptureManager] Session resources cleaned up")
            }
        }
        
        captureSession = nil
        capturedRoom = nil
        isSessionRunning = false
        isSessionInitialized = false
        isWaitingForFinalData = false
        hasReceivedFinalData = false
        finalScene = nil
        exportProgress = 0
    }
    
    // MARK: - Update Instruction
    func updateInstruction(_ text: String) {
        DispatchQueue.main.async {
            self.instruction = text
        }
    }
    
    // MARK: - Process Captured Room (Called by Delegate)
    func processCapturedRoom(_ room: CapturedRoom) async {
        logDebug("🏠 [RoomCaptureManager] Processing captured room")
        logDebug("   - Walls: \(room.walls.count)")
        logDebug("   - Objects: \(room.objects.count)")
        logDebug("   - Doors: \(room.doors.count)")
        logDebug("   - Windows: \(room.windows.count)")
        logDebug("   - Openings: \(room.openings.count)")
        
        // Mark that we received final data
        hasReceivedFinalData = true
        
        await MainActor.run {
            self.capturedRoom = room
            self.isProcessing = true
            self.statusMessage = "Processing room data..."
            self.feedbackText = "Found \(room.walls.count) walls, \(room.objects.count) objects"
        }
        
        // Only export if we have valid room data
        if room.walls.count > 0 || room.objects.count > 0 {
            await exportToUSDZ(room)
        } else {
            logDebug("⚠️ [RoomCaptureManager] Room has no content, skipping export")
            await MainActor.run {
                self.statusMessage = "No room content captured"
                self.isProcessing = false
            }
        }
    }
    
    // MARK: - Export to USDZ
    private func exportToUSDZ(_ room: CapturedRoom) async {
        logDebug("📦 [RoomCaptureManager] Exporting room to USDZ")
        
        await MainActor.run {
            self.exportProgress = 0.2
            self.statusMessage = "Creating 3D model..."
        }
        
        do {
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let tempFolder = documentsPath.appendingPathComponent("TempScans", isDirectory: true)
            
            // Create directory if it doesn't exist
            if !FileManager.default.fileExists(atPath: tempFolder.path) {
                try FileManager.default.createDirectory(at: tempFolder, withIntermediateDirectories: true)
                logDebug("✅ [RoomCaptureManager] Created TempScans directory")
            }
            
            let fileName = "RoomScan_\(UUID().uuidString).usdz"
            let exportURL = tempFolder.appendingPathComponent(fileName)
            
            logDebug("   - Export path: \(exportURL.path)")
            
            await MainActor.run {
                self.exportProgress = 0.5
                self.statusMessage = "Generating USDZ file..."
            }
            
            // Export room to USDZ
            try await room.export(to: exportURL, exportOptions: .model)
            
            logDebug("✅ [RoomCaptureManager] Room exported successfully")
            
            // Check file size
            if let attributes = try? FileManager.default.attributesOfItem(atPath: exportURL.path),
               let fileSize = attributes[.size] as? Int {
                logDebug("   - File size: \(fileSize) bytes")
            }
            
            await MainActor.run {
                self.exportProgress = 1.0
                self.statusMessage = "Room scan complete!"
                self.finalScene = exportURL
                self.isProcessing = false
            }
            
        } catch {
            logDebug("❌ [RoomCaptureManager] Export failed: \(error.localizedDescription)")
            logDebug("   - Error details: \(error)")
            await MainActor.run {
                self.statusMessage = "Export failed: \(error.localizedDescription)"
                self.isProcessing = false
            }
        }
    }
    
    // MARK: - Save Room to Library
    func saveRoomToLibrary(name: String, completion: @escaping (Bool, String?) -> Void) {
        guard let sceneURL = finalScene else {
            logDebug("❌ [RoomCaptureManager] No scene available to save")
            completion(false, "No room scan available")
            return
        }
        
        logDebug("💾 [RoomCaptureManager] Saving room to library: \(name)")
        
        // Verify source file exists
        guard FileManager.default.fileExists(atPath: sceneURL.path) else {
            logDebug("❌ [RoomCaptureManager] Source file no longer exists at: \(sceneURL.path)")
            completion(false, "Source file was deleted")
            return
        }
        
        do {
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let roomsFolder = documentsPath.appendingPathComponent("SavedRooms", isDirectory: true)
            
            // Create SavedRooms directory if it doesn't exist
            if !FileManager.default.fileExists(atPath: roomsFolder.path) {
                try FileManager.default.createDirectory(at: roomsFolder, withIntermediateDirectories: true)
                logDebug("✅ [RoomCaptureManager] Created SavedRooms directory")
            }
            
            let permanentURL = roomsFolder.appendingPathComponent("\(name).usdz")
            
            // Remove existing file if it exists
            if FileManager.default.fileExists(atPath: permanentURL.path) {
                try FileManager.default.removeItem(at: permanentURL)
            }
            
            // Copy to permanent location
            try FileManager.default.copyItem(at: sceneURL, to: permanentURL)
            
            // Verify copy succeeded
            guard FileManager.default.fileExists(atPath: permanentURL.path) else {
                logDebug("❌ [RoomCaptureManager] Copy failed - file not found at destination")
                completion(false, "Failed to copy file to permanent storage")
                return
            }
            
            logDebug("✅ [RoomCaptureManager] Successfully copied to: \(permanentURL.path)")
            
            // Save metadata to UserDefaults
            var savedRooms = UserDefaults.standard.dictionary(forKey: "SavedRooms") as? [String: [String: Any]] ?? [:]
            savedRooms[name] = [
                "filePath": permanentURL.path,
                "createdAt": Date().timeIntervalSince1970,
                "wallCount": capturedRoom?.walls.count ?? 0,
                "objectCount": capturedRoom?.objects.count ?? 0
            ]
            UserDefaults.standard.set(savedRooms, forKey: "SavedRooms")
            
            // Clean up temp file
            try? FileManager.default.removeItem(at: sceneURL)
            logDebug("🧹 [RoomCaptureManager] Cleaned up temp file")
            
            logDebug("✅ [RoomCaptureManager] Room saved successfully")
            completion(true, nil)
            
        } catch {
            logDebug("❌ [RoomCaptureManager] Save failed: \(error.localizedDescription)")
            completion(false, error.localizedDescription)
        }
    }
    
    // MARK: - Get Room Dimensions
    func getRoomDimensions() -> (width: Float, depth: Float, height: Float)? {
        guard let room = capturedRoom else { return nil }
        
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
        
        // Check if we got valid bounds
        if minX == .infinity || maxX == -.infinity {
            return nil
        }
        
        let roomWidth = maxX - minX
        let roomDepth = maxZ - minZ
        let roomHeight = maxY - minY
        
        logDebug("📏 [RoomCaptureManager] Room dimensions:")
        logDebug("   - Width: \(roomWidth)m")
        logDebug("   - Depth: \(roomDepth)m")
        logDebug("   - Height: \(roomHeight)m")
        
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
            
            let area = dims.width * dims.depth
            let volume = area * dims.height
            stats += "\nArea: \(String(format: "%.2f", area))m²\n"
            stats += "Volume: \(String(format: "%.2f", volume))m³\n"
        }
        
        return stats
    }
    
    // MARK: - Check if has valid data
    func hasValidRoomData() -> Bool {
        return capturedRoom != nil && (capturedRoom!.walls.count > 0 || capturedRoom!.objects.count > 0)
    }
    
    // MARK: - Manual Capture Current State (Fixed - no unnecessary try)
    func manuallyCaptureCurrent() {
        logDebug("🔧 [RoomCaptureManager] Manually capturing current state")
        
        guard captureSession != nil else {
            logDebug("❌ No session available")
            return
        }
        
        // Check if we have a captured room already (no try needed)
        if let room = capturedRoom {
            logDebug("✅ Using existing room: \(room.walls.count) walls, \(room.objects.count) objects")
            Task {
                await processCapturedRoom(room)
            }
        } else {
            logDebug("❌ No room data available")
        }
    }
    
    // MARK: - Get Saved Rooms List
    static func getSavedRooms() -> [(name: String, date: Date, wallCount: Int, objectCount: Int)] {
        guard let savedRooms = UserDefaults.standard.dictionary(forKey: "SavedRooms") as? [String: [String: Any]] else {
            return []
        }
        
        var rooms: [(name: String, date: Date, wallCount: Int, objectCount: Int)] = []
        
        for (name, data) in savedRooms {
            if let timestamp = data["createdAt"] as? TimeInterval,
               let wallCount = data["wallCount"] as? Int,
               let objectCount = data["objectCount"] as? Int {
                let date = Date(timeIntervalSince1970: timestamp)
                rooms.append((name, date, wallCount, objectCount))
            }
        }
        
        // Sort by date, newest first
        rooms.sort { $0.date > $1.date }
        
        return rooms
    }
    
    // MARK: - Load Saved Room
    static func loadSavedRoom(name: String) -> URL? {
        guard let savedRooms = UserDefaults.standard.dictionary(forKey: "SavedRooms") as? [String: [String: Any]],
              let roomData = savedRooms[name],
              let filePath = roomData["filePath"] as? String else {
            logDebug("❌ [RoomCaptureManager] Room '\(name)' not found in saved rooms")
            return nil
        }
        
        let url = URL(fileURLWithPath: filePath)
        
        guard FileManager.default.fileExists(atPath: url.path) else {
            logDebug("❌ [RoomCaptureManager] File no longer exists at: \(filePath)")
            return nil
        }
        
        logDebug("✅ [RoomCaptureManager] Loaded room: \(name) from \(filePath)")
        return url
    }
    
    // MARK: - Delete Saved Room
    static func deleteSavedRoom(name: String) -> Bool {
        var savedRooms = UserDefaults.standard.dictionary(forKey: "SavedRooms") as? [String: [String: Any]] ?? [:]
        
        guard let roomData = savedRooms[name],
              let filePath = roomData["filePath"] as? String else {
            logDebug("❌ [RoomCaptureManager] Room '\(name)' not found")
            return false
        }
        
        // Delete the file
        let url = URL(fileURLWithPath: filePath)
        do {
            try FileManager.default.removeItem(at: url)
            logDebug("✅ [RoomCaptureManager] Deleted file: \(filePath)")
        } catch {
            logDebug("⚠️ [RoomCaptureManager] Could not delete file: \(error)")
        }
        
        // Remove from UserDefaults
        savedRooms.removeValue(forKey: name)
        UserDefaults.standard.set(savedRooms, forKey: "SavedRooms")
        
        logDebug("✅ [RoomCaptureManager] Removed room '\(name)' from saved rooms")
        return true
    }
}
