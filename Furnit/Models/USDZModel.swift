import Foundation
import UIKit

struct USDZModel: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let fileName: String
    let dataAsset: NSDataAsset?
    let isSavedRoom: Bool  // ✅ NEW: Track if this is a saved room
    
    init(name: String, fileName: String, isSavedRoom: Bool = false) {
        self.name = name
        self.fileName = fileName
        self.isSavedRoom = isSavedRoom
        // Only load NSDataAsset for bundle rooms
        self.dataAsset = isSavedRoom ? nil : NSDataAsset(name: fileName)
    }
    
    var displayName: String {
        return name.replacingOccurrences(of: "_", with: " ").capitalized
    }
    
    // ✅ UPDATED: Handle both bundle and saved rooms
    var temporaryURL: URL? {
        if isSavedRoom {
            // For saved rooms, return the actual file path
            let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let savedURL = documentsDirectory
                .appendingPathComponent("SavedRooms", isDirectory: true)
                .appendingPathComponent("\(fileName).usdz")
            
            if FileManager.default.fileExists(atPath: savedURL.path) {
                print("✅ [USDZModel] Found saved room at: \(savedURL.lastPathComponent)")
                return savedURL
            } else {
                print("❌ [USDZModel] Saved room not found: \(fileName)")
                return nil
            }
        } else {
            // For bundle rooms, use existing logic
            guard let data = dataAsset?.data else {
                print("❌ [USDZModel] No data for \(fileName)")
                return nil
            }
            
            let tempDirectory = FileManager.default.temporaryDirectory
            let tempURL = tempDirectory.appendingPathComponent("\(fileName)_\(id.uuidString).usdz")
            
            do {
                try data.write(to: tempURL)
                print("✅ [USDZModel] Wrote temporary file for \(fileName) at \(tempURL)")
                return tempURL
            } catch {
                print("❌ [USDZModel] Failed to write temporary file for \(fileName): \(error)")
                return nil
            }
        }
    }
}
