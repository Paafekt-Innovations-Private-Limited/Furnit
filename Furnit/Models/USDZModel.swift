import Foundation
import UIKit

struct USDZModel: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let fileName: String
    let dataAsset: NSDataAsset?
    let isSavedRoom: Bool  // Track if this is a saved room
    let fileType: ModelFileType  // File type: .usdz or .ply
    let fileSize: UInt64?  // File size in bytes

    // Standard initializer for USDZ models (backward compatible)
    init(name: String, fileName: String, isSavedRoom: Bool = false) {
        self.name = name
        self.fileName = fileName
        self.isSavedRoom = isSavedRoom
        self.fileType = .usdz
        self.fileSize = nil
        // Only load NSDataAsset for bundle rooms
        self.dataAsset = isSavedRoom ? nil : NSDataAsset(name: fileName)
    }

    // Full initializer with file type and size
    init(name: String, fileName: String, isSavedRoom: Bool, fileType: ModelFileType, fileSize: UInt64?) {
        self.name = name
        self.fileName = fileName
        self.isSavedRoom = isSavedRoom
        self.fileType = fileType
        self.fileSize = fileSize
        // Only load NSDataAsset for bundle rooms with USDZ type
        self.dataAsset = (isSavedRoom || fileType == .ply) ? nil : NSDataAsset(name: fileName)
    }
    
    var displayName: String {
        return name.replacingOccurrences(of: "_", with: " ").capitalized
    }

    /// Formatted file size string (e.g., "45.2 MB")
    var fileSizeFormatted: String {
        guard let size = fileSize else { return "Unknown size" }
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(size))
    }

    /// Subtitle text based on file type
    var subtitleText: String {
        return fileType.displayName
    }
    
    // ✅ UPDATED: Handle both bundle and saved rooms with debug-mode logging
    var temporaryURL: URL? {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            logDebug("🔍 [USDZModel.temporaryURL] ========================================")
            logDebug("🔍 [USDZModel.temporaryURL] Called for: \(fileName)")
            logDebug("🔍 [USDZModel.temporaryURL] isSavedRoom: \(isSavedRoom)")
            logDebug("🔍 [USDZModel.temporaryURL] displayName: \(displayName)")
        }
        
        if isSavedRoom {
            // For saved rooms, return the actual file path
            let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            if debugMode {
                logDebug("🔍 [USDZModel.temporaryURL] Documents directory: \(documentsDirectory.path)")
            }
            
            let savedRoomsDir = documentsDirectory.appendingPathComponent("SavedRooms", isDirectory: true)
            if debugMode {
                logDebug("🔍 [USDZModel.temporaryURL] SavedRooms directory: \(savedRoomsDir.path)")
            }
            
            // Use appropriate file extension based on file type
            let fileExtension = fileType == .ply ? "ply" : "usdz"
            let savedURL = savedRoomsDir.appendingPathComponent("\(fileName).\(fileExtension)")
            if debugMode {
                logDebug("🔍 [USDZModel.temporaryURL] Looking for file at: \(savedURL.path)")
            }
            
            let fileExists = FileManager.default.fileExists(atPath: savedURL.path)
            if debugMode {
                logDebug("🔍 [USDZModel.temporaryURL] File exists: \(fileExists)")
            }
            
            if fileExists {
                do {
                    let attributes = try FileManager.default.attributesOfItem(atPath: savedURL.path)
                    let fileSize = attributes[.size] as? UInt64 ?? 0
                    if debugMode {
                        logDebug("✅ [USDZModel.temporaryURL] Found saved room!")
                        logDebug("   - Path: \(savedURL.path)")
                        logDebug("   - File name: \(savedURL.lastPathComponent)")
                        logDebug("   - File size: \(fileSize) bytes (\(fileSize / 1024 / 1024) MB)")
                        logDebug("   - Readable: \(FileManager.default.isReadableFile(atPath: savedURL.path))")
                        
                        // Try to list all files in SavedRooms to verify
                        if let files = try? FileManager.default.contentsOfDirectory(atPath: savedRoomsDir.path) {
                            logDebug("   - All files in SavedRooms: \(files)")
                        }
                    }
                    
                    return savedURL
                } catch {
                    if debugMode {
                        logDebug("❌ [USDZModel.temporaryURL] Error getting file attributes: \(error)")
                    }
                    return savedURL
                }
            } else {
                if debugMode {
                    logDebug("❌ [USDZModel.temporaryURL] Saved room file NOT FOUND!")
                    logDebug("   - Expected path: \(savedURL.path)")
                    
                    // List all files in SavedRooms directory for debugging
                    do {
                        let files = try FileManager.default.contentsOfDirectory(atPath: savedRoomsDir.path)
                        logDebug("   - Files in SavedRooms directory: \(files)")
                        logDebug("   - Expected file: \(fileName).\(fileExtension)")
                    } catch {
                        logDebug("   - Could not list SavedRooms directory: \(error)")
                    }
                }
                
                return nil
            }
        } else {
            // For bundle rooms, use existing logic
            if debugMode {
                logDebug("🔍 [USDZModel.temporaryURL] Processing BUNDLE room")
            }
            
            guard let data = dataAsset?.data else {
                if debugMode {
                    logDebug("❌ [USDZModel.temporaryURL] No dataAsset data for bundle room: \(fileName)")
                }
                return nil
            }
            
            if debugMode {
                logDebug("🔍 [USDZModel.temporaryURL] Bundle data size: \(data.count) bytes")
            }
            
            let tempDirectory = FileManager.default.temporaryDirectory
            let tempURL = tempDirectory.appendingPathComponent("\(fileName)_\(id.uuidString).usdz")
            
            if debugMode {
                logDebug("🔍 [USDZModel.temporaryURL] Writing to temp path: \(tempURL.path)")
            }
            
            do {
                try data.write(to: tempURL)
                if debugMode {
                    logDebug("✅ [USDZModel.temporaryURL] Successfully wrote temporary file")
                    logDebug("   - Path: \(tempURL.path)")
                    logDebug("   - Size: \(data.count) bytes")
                }
                return tempURL
            } catch {
                if debugMode {
                    logDebug("❌ [USDZModel.temporaryURL] Failed to write temporary file: \(error)")
                }
                return nil
            }
        }
    }
}
