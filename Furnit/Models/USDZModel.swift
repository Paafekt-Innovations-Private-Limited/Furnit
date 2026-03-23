import Foundation
import UIKit

struct USDZModel: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let fileName: String
    let dataAsset: NSDataAsset?
    let isSavedRoom: Bool  // Track if this is a saved room
    let fileType: ModelFileType  // File type: .usdz, .ply, or .meshroom
    let fileSize: UInt64?  // File size in bytes
    let photoOrientation: PhotoOrientation  // Source photo orientation (for PLY/meshroom)
    let roomWidth: Float?  // Room width in meters
    let roomHeight: Float?  // Room height in meters
    let roomDepth: Float?  // Room depth in meters (for meshroom)

    // MARK: - YOLO ratio calibration (optional; from *.meta JSON)
    /// Wall bbox height / reference image height, or 1.0 when using full-frame proxy.
    let yoloWallHeightFrac: Float?
    /// Canonical furniture label → height / reference image height.
    let yoloFurnitureHeightFracByClass: [String: Float]?
    /// Reference image height in pixels at calibration time.
    let yoloRefImageHeightPx: Int?
    /// SHARP / metadata room height (m) when ratios were captured.
    let sharpRoomHeightAtYoloCapture: Float?

    // Standard initializer for USDZ models (backward compatible)
    init(name: String, fileName: String, isSavedRoom: Bool = false) {
        self.name = name
        self.fileName = fileName
        self.isSavedRoom = isSavedRoom
        self.fileType = .usdz
        self.fileSize = nil
        self.photoOrientation = .portrait  // Default for USDZ
        self.roomWidth = nil
        self.roomHeight = nil
        self.roomDepth = nil
        self.yoloWallHeightFrac = nil
        self.yoloFurnitureHeightFracByClass = nil
        self.yoloRefImageHeightPx = nil
        self.sharpRoomHeightAtYoloCapture = nil
        // Only load NSDataAsset for bundle rooms
        self.dataAsset = isSavedRoom ? nil : NSDataAsset(name: fileName)
    }

    // Full initializer with file type and size
    init(
        name: String,
        fileName: String,
        isSavedRoom: Bool,
        fileType: ModelFileType,
        fileSize: UInt64?,
        photoOrientation: PhotoOrientation = .portrait,
        roomWidth: Float? = nil,
        roomHeight: Float? = nil,
        roomDepth: Float? = nil,
        yoloWallHeightFrac: Float? = nil,
        yoloFurnitureHeightFracByClass: [String: Float]? = nil,
        yoloRefImageHeightPx: Int? = nil,
        sharpRoomHeightAtYoloCapture: Float? = nil
    ) {
        self.name = name
        self.fileName = fileName
        self.isSavedRoom = isSavedRoom
        self.fileType = fileType
        self.fileSize = fileSize
        self.photoOrientation = photoOrientation
        self.roomWidth = roomWidth
        self.roomHeight = roomHeight
        self.roomDepth = roomDepth
        self.yoloWallHeightFrac = yoloWallHeightFrac
        self.yoloFurnitureHeightFracByClass = yoloFurnitureHeightFracByClass
        self.yoloRefImageHeightPx = yoloRefImageHeightPx
        self.sharpRoomHeightAtYoloCapture = sharpRoomHeightAtYoloCapture
        // Only load NSDataAsset for bundle rooms with USDZ type
        self.dataAsset = (isSavedRoom || fileType == .ply || fileType == .meshroom) ? nil : NSDataAsset(name: fileName)
    }
    
    var displayName: String {
        var cleanName = name
        // Remove timestamp suffix for PLY files (format: "RoomName_1767917336")
        if fileType == .ply {
            // Check if name ends with underscore followed by digits (timestamp)
            if let range = cleanName.range(of: "_\\d+$", options: .regularExpression) {
                cleanName = String(cleanName[..<range.lowerBound])
            }
        }
        return cleanName.replacingOccurrences(of: "_", with: " ").capitalized
    }

    /// Formatted file size string (e.g., "45.2 MB")
    var fileSizeFormatted: String {
        // Use fileSize if available, otherwise calculate from dataAsset
        let size: UInt64
        if let fs = fileSize {
            size = fs
        } else if let data = dataAsset?.data {
            size = UInt64(data.count)
        } else {
            return "Unknown size"
        }
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(size))
    }

    /// Check if file size is available (from fileSize or dataAsset)
    var hasFileSize: Bool {
        return fileSize != nil || dataAsset?.data != nil
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
            let fileExtension: String
            switch fileType {
            case .ply:
                fileExtension = "ply"
            case .meshroom:
                fileExtension = "meshroom"
            case .glb:
                fileExtension = "glb"
            case .usdz:
                fileExtension = "usdz"
            }
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
                    logDebug("   - File type: \(fileType.rawValue)")

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

            // Use stable filename (no UUID) to avoid re-writing on every render
            let tempDirectory = FileManager.default.temporaryDirectory
            let tempURL = tempDirectory.appendingPathComponent("\(fileName)_bundle.usdz")

            // Check if file already exists with correct size - skip writing if so
            if FileManager.default.fileExists(atPath: tempURL.path) {
                if let existingSize = try? FileManager.default.attributesOfItem(atPath: tempURL.path)[.size] as? Int,
                   existingSize == data.count {
                    if debugMode {
                        logDebug("✅ [USDZModel.temporaryURL] Using cached temp file: \(tempURL.lastPathComponent)")
                    }
                    return tempURL
                }
            }

            if debugMode {
                logDebug("🔍 [USDZModel.temporaryURL] Bundle data size: \(data.count) bytes")
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
