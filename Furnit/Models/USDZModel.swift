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
    
    // ✅ UPDATED: Handle both bundle and saved rooms with EXTENSIVE LOGGING
    var temporaryURL: URL? {
        print("🔍 [USDZModel.temporaryURL] ========================================")
        print("🔍 [USDZModel.temporaryURL] Called for: \(fileName)")
        print("🔍 [USDZModel.temporaryURL] isSavedRoom: \(isSavedRoom)")
        print("🔍 [USDZModel.temporaryURL] displayName: \(displayName)")
        
        if isSavedRoom {
            // For saved rooms, return the actual file path
            let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            print("🔍 [USDZModel.temporaryURL] Documents directory: \(documentsDirectory.path)")
            
            let savedRoomsDir = documentsDirectory.appendingPathComponent("SavedRooms", isDirectory: true)
            print("🔍 [USDZModel.temporaryURL] SavedRooms directory: \(savedRoomsDir.path)")
            
            let savedURL = savedRoomsDir.appendingPathComponent("\(fileName).usdz")
            print("🔍 [USDZModel.temporaryURL] Looking for file at: \(savedURL.path)")
            
            let fileExists = FileManager.default.fileExists(atPath: savedURL.path)
            print("🔍 [USDZModel.temporaryURL] File exists: \(fileExists)")
            
            if fileExists {
                do {
                    let attributes = try FileManager.default.attributesOfItem(atPath: savedURL.path)
                    let fileSize = attributes[.size] as? UInt64 ?? 0
                    print("✅ [USDZModel.temporaryURL] Found saved room!")
                    print("   - Path: \(savedURL.path)")
                    print("   - File name: \(savedURL.lastPathComponent)")
                    print("   - File size: \(fileSize) bytes (\(fileSize / 1024 / 1024) MB)")
                    print("   - Readable: \(FileManager.default.isReadableFile(atPath: savedURL.path))")
                    
                    // Try to list all files in SavedRooms to verify
                    if let files = try? FileManager.default.contentsOfDirectory(atPath: savedRoomsDir.path) {
                        print("   - All files in SavedRooms: \(files)")
                    }
                    
                    return savedURL
                } catch {
                    print("❌ [USDZModel.temporaryURL] Error getting file attributes: \(error)")
                    return savedURL
                }
            } else {
                print("❌ [USDZModel.temporaryURL] Saved room file NOT FOUND!")
                print("   - Expected path: \(savedURL.path)")
                
                // List all files in SavedRooms directory for debugging
                do {
                    let files = try FileManager.default.contentsOfDirectory(atPath: savedRoomsDir.path)
                    print("   - Files in SavedRooms directory: \(files)")
                    print("   - Expected file: \(fileName).usdz")
                } catch {
                    print("   - Could not list SavedRooms directory: \(error)")
                }
                
                return nil
            }
        } else {
            // For bundle rooms, use existing logic
            print("🔍 [USDZModel.temporaryURL] Processing BUNDLE room")
            guard let data = dataAsset?.data else {
                print("❌ [USDZModel.temporaryURL] No dataAsset data for bundle room: \(fileName)")
                return nil
            }
            
            print("🔍 [USDZModel.temporaryURL] Bundle data size: \(data.count) bytes")
            
            let tempDirectory = FileManager.default.temporaryDirectory
            let tempURL = tempDirectory.appendingPathComponent("\(fileName)_\(id.uuidString).usdz")
            
            print("🔍 [USDZModel.temporaryURL] Writing to temp path: \(tempURL.path)")
            
            do {
                try data.write(to: tempURL)
                print("✅ [USDZModel.temporaryURL] Successfully wrote temporary file")
                print("   - Path: \(tempURL.path)")
                print("   - Size: \(data.count) bytes")
                return tempURL
            } catch {
                print("❌ [USDZModel.temporaryURL] Failed to write temporary file: \(error)")
                return nil
            }
        }
    }
}
