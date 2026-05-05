import Foundation

/// Manages room creation limits (20 room limit only, no payments)
@MainActor
class RoomLimitManager: ObservableObject {
    static let shared = RoomLimitManager()

    // MARK: - Published Properties
    @Published var roomCount: Int = 0

    // MARK: - Constants
    let roomLimit = 20
    
    private init() {
        updateRoomCount()
    }
    
    // MARK: - Room Limit Checks
    
    /// Check if user can create more rooms
    func canCreateMoreRooms() -> Bool {
        return roomCount < roomLimit
    }
    
    /// Get remaining free rooms
    func remainingRooms() -> Int {
        return max(0, roomLimit - roomCount)
    }
    
    /// Update the current room count
    func updateRoomCount() {
        // Count saved rooms (excluding bundle models) - both USDZ and PLY files
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let modelsDirectory = documentsDirectory.appendingPathComponent("SavedRooms", isDirectory: true)

        // Supported file extensions
        let supportedExtensions = ["usdz", "ply"]

        do {
            let files = try FileManager.default.contentsOfDirectory(at: modelsDirectory,
                                                                    includingPropertiesForKeys: nil)
            let modelFiles = files.filter {
                supportedExtensions.contains($0.pathExtension.lowercased()) &&
                !$0.deletingPathExtension().lastPathComponent.hasSuffix("_classic")
            }
            roomCount = modelFiles.count

            if AppStateManager.shared.qualitySettings.debugMode {
                logDebug("📊 [RoomLimitManager] Updated room count: \(roomCount)/\(roomLimit)")
            }
        } catch {
            roomCount = 0
            if AppStateManager.shared.qualitySettings.debugMode {
                logDebug("⚠️ [RoomLimitManager] Error counting rooms: \(error)")
            }
        }
    }
}
