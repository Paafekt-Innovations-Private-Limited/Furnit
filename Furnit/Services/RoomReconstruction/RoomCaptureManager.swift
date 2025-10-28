import Foundation
import RoomPlan
import SwiftUI

// MARK: - Room Capture Manager (Fixed Compilation Errors)
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
        print("📂 [RoomCaptureManager] Initialized")
        _ = checkDeviceSupport()
    }
    
    // MARK: - Device Support Check
    func checkDeviceSupport() -> Bool {
        let isSupported = RoomCaptureSession.isSupported
        print("📱 [RoomCaptureManager] Device support: \(isSupported ? "✅ Supported" : "❌ Not supported")")
        return isSupported
    }
    
    // MARK: - Initialize Session
    func initializeSession(from captureView: RoomCaptureSession?) {
        guard captureSession == nil else {
            print("⚠️ [RoomCaptureManager] Session already initialized")
            return
        }
        
        self.captureSession = captureView
        self.isSessionInitialized = true
        print("✅ [RoomCaptureManager] Session initialized from capture view")
        
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
        
        // Reset flags
        hasReceivedFinalData = false
        isWaitingForFinalData = false
        
        isSessionRunning = true
        statusMessage = "Point at the floor to start scanning"
        instruction = "Move slowly and steadily"
        
        return captureSession
    }
    
    // MARK: - Stop Capture Session (Fixed)
    func stopCaptureSession() {
        print("🛑 [RoomCaptureManager] Stopping capture session")
        
        guard let session = captureSession else {
            print("⚠️ [RoomCaptureManager] No session to stop")
            return
        }
        
        // Mark that we're waiting for final data
        isWaitingForFinalData = true
        hasReceivedFinalData = false
        
        // Update UI
        isSessionRunning = false
        statusMessage = "Processing scan data..."
        
        // Note: capturedRoom is a property, not a throwing function
        // So we don't need try here
        if let currentRoom = self.capturedRoom {
            print("📦 [RoomCaptureManager] Already have room data: \(currentRoom.walls.count) walls")
        }
        
        // Now stop the session
        print("⏸️ [RoomCaptureManager] Calling session.stop() - delegate should receive final data")
        session.stop()
        
        // If we have a room but delegate doesn't fire, process it anyway
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            if let room = self?.capturedRoom, self?.finalScene == nil {
                print("🔧 [RoomCaptureManager] Forcing room processing after stop")
                Task {
                    if let self = self {
                        await self.processCapturedRoom(room)
                    }
                }
            }
        }
    }
    
    // MARK: - Force Process Current Data
    func forceProcessCurrentData() {
        print("⚠️ [RoomCaptureManager] Forcing processing of current data")
        
        if let room = capturedRoom {
            print("📦 [RoomCaptureManager] Processing existing captured room")
            Task {
                await processCapturedRoom(room)
            }
        } else {
            print("❌ [RoomCaptureManager] No room data to process")
            statusMessage = "No room data captured"
        }
    }
    
    // MARK: - Cleanup Session
    func cleanupSession() {
        print("🧹 [RoomCaptureManager] Cleaning up session resources")
        
        if let session = captureSession {
            session.delegate = nil
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.captureSession = nil
                self?.isSessionInitialized = false
                print("✅ [RoomCaptureManager] Session resources cleaned up")
            }
        }
        
        captureSession = nil
        capturedRoom = nil
        isSessionRunning = false
        isSessionInitialized = false
        isWaitingForFinalData = false
        hasReceivedFinalData = false
    }
    
    // MARK: - Update Instruction
    func updateInstruction(_ text: String) {
        DispatchQueue.main.async {
            self.instruction = text
        }
    }
    
    // MARK: - Process Captured Room (Called by Delegate)
    func processCapturedRoom(_ room: CapturedRoom) async {
        print("🏠 [RoomCaptureManager] Processing captured room")
        print("   - Walls: \(room.walls.count)")
        print("   - Objects: \(room.objects.count)")
        print("   - Doors: \(room.doors.count)")
        print("   - Windows: \(room.windows.count)")
        print("   - Openings: \(room.openings.count)")
        
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
            print("⚠️ [RoomCaptureManager] Room has no content, skipping export")
            await MainActor.run {
                self.statusMessage = "No room content captured"
                self.isProcessing = false
            }
        }
    }
    
    // MARK: - Export to USDZ
    private func exportToUSDZ(_ room: CapturedRoom) async {
        print("📦 [RoomCaptureManager] Exporting room to USDZ")
        
        await MainActor.run {
            self.exportProgress = 0.2
            self.statusMessage = "Creating 3D model..."
        }
        
        do {
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let tempFolder = documentsPath.appendingPathComponent("TempScans", isDirectory: true)
            
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
            
            // Check file size
            if let attributes = try? FileManager.default.attributesOfItem(atPath: exportURL.path),
               let fileSize = attributes[.size] as? Int {
                print("   - File size: \(fileSize) bytes")
            }
            
            await MainActor.run {
                self.exportProgress = 1.0
                self.statusMessage = "Room scan complete!"
                self.finalScene = exportURL
                self.isProcessing = false
            }
            
        } catch {
            print("❌ [RoomCaptureManager] Export failed: \(error.localizedDescription)")
            print("   - Error details: \(error)")
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
        
        guard FileManager.default.fileExists(atPath: sceneURL.path) else {
            print("❌ [RoomCaptureManager] Source file no longer exists at: \(sceneURL.path)")
            completion(false, "Source file was deleted")
            return
        }
        
        do {
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let roomsFolder = documentsPath.appendingPathComponent("SavedRooms", isDirectory: true)
            
            if !FileManager.default.fileExists(atPath: roomsFolder.path) {
                try FileManager.default.createDirectory(at: roomsFolder, withIntermediateDirectories: true)
                print("✅ [RoomCaptureManager] Created SavedRooms directory")
            }
            
            let permanentURL = roomsFolder.appendingPathComponent("\(name).usdz")
            
            if FileManager.default.fileExists(atPath: permanentURL.path) {
                try FileManager.default.removeItem(at: permanentURL)
            }
            
            try FileManager.default.copyItem(at: sceneURL, to: permanentURL)
            
            guard FileManager.default.fileExists(atPath: permanentURL.path) else {
                print("❌ [RoomCaptureManager] Copy failed - file not found at destination")
                completion(false, "Failed to copy file to permanent storage")
                return
            }
            
            print("✅ [RoomCaptureManager] Successfully copied to: \(permanentURL.path)")
            
            var savedRooms = UserDefaults.standard.dictionary(forKey: "SavedRooms") as? [String: [String: Any]] ?? [:]
            savedRooms[name] = [
                "filePath": permanentURL.path,
                "createdAt": Date().timeIntervalSince1970,
                "wallCount": capturedRoom?.walls.count ?? 0,
                "objectCount": capturedRoom?.objects.count ?? 0
            ]
            UserDefaults.standard.set(savedRooms, forKey: "SavedRooms")
            
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
    
    // MARK: - Manual Capture Current State (Fixed)
    func manuallyCaptureCurrent() {
        print("🔧 [RoomCaptureManager] Manually capturing current state")
        
        guard captureSession != nil else {
            print("❌ No session available")
            return
        }
        
        // Check if we have a captured room already
        if let room = capturedRoom {
            print("✅ Using existing room: \(room.walls.count) walls, \(room.objects.count) objects")
            Task {
                await processCapturedRoom(room)
            }
        } else {
            print("❌ No room data available")
        }
    }
}
