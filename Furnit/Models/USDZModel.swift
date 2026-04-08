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

    /// Optional splat depth-raycast dimensions in **scene units** (PLY space) for ratio fitment vs furniture.
    let roomSceneWidth: Float?
    let roomSceneHeight: Float?
    let roomSceneDepth: Float?

    // MARK: - YOLO ratio calibration (optional; from *.meta JSON)
    /// Wall bbox height / reference image height, or 1.0 when using full-frame proxy.
    let yoloWallHeightFrac: Float?
    /// Canonical furniture label → height / reference image height.
    let yoloFurnitureHeightFracByClass: [String: Float]?
    /// Reference image height in pixels at calibration time.
    let yoloRefImageHeightPx: Int?
    /// SHARP / metadata room height (m) when ratios were captured.
    let sharpRoomHeightAtYoloCapture: Float?
    /// When set (from `*.meta` `displayName`), shown in the list instead of deriving from `name`.
    let customDisplayName: String?
    /// Saved PLY should be treated like SHARP classic orientation/rendering even without `_classic` filename suffix.
    let isClassicPly: Bool
    /// Cached on-disk URL for saved rooms so SwiftUI body reevaluation does not restat the filesystem repeatedly.
    let cachedResolvedURL: URL?

    // Standard initializer for USDZ models (backward compatible)
    init(name: String, fileName: String, isSavedRoom: Bool = false, customDisplayName: String? = nil) {
        self.name = name
        self.fileName = fileName
        self.isSavedRoom = isSavedRoom
        self.fileType = .usdz
        self.fileSize = nil
        self.photoOrientation = .portrait  // Default for USDZ
        self.roomWidth = nil
        self.roomHeight = nil
        self.roomDepth = nil
        self.roomSceneWidth = nil
        self.roomSceneHeight = nil
        self.roomSceneDepth = nil
        self.yoloWallHeightFrac = nil
        self.yoloFurnitureHeightFracByClass = nil
        self.yoloRefImageHeightPx = nil
        self.sharpRoomHeightAtYoloCapture = nil
        self.customDisplayName = customDisplayName
        self.isClassicPly = false
        self.cachedResolvedURL = nil
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
        roomSceneWidth: Float? = nil,
        roomSceneHeight: Float? = nil,
        roomSceneDepth: Float? = nil,
        yoloWallHeightFrac: Float? = nil,
        yoloFurnitureHeightFracByClass: [String: Float]? = nil,
        yoloRefImageHeightPx: Int? = nil,
        sharpRoomHeightAtYoloCapture: Float? = nil,
        customDisplayName: String? = nil,
        isClassicPly: Bool = false,
        cachedResolvedURL: URL? = nil
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
        self.roomSceneWidth = roomSceneWidth
        self.roomSceneHeight = roomSceneHeight
        self.roomSceneDepth = roomSceneDepth
        self.yoloWallHeightFrac = yoloWallHeightFrac
        self.yoloFurnitureHeightFracByClass = yoloFurnitureHeightFracByClass
        self.yoloRefImageHeightPx = yoloRefImageHeightPx
        self.sharpRoomHeightAtYoloCapture = sharpRoomHeightAtYoloCapture
        self.customDisplayName = customDisplayName
        self.isClassicPly = isClassicPly
        self.cachedResolvedURL = cachedResolvedURL
        // Only load NSDataAsset for bundle rooms with USDZ type
        self.dataAsset = (isSavedRoom || fileType == .ply || fileType == .meshroom) ? nil : NSDataAsset(name: fileName)
    }
    
    var displayName: String {
        if let custom = customDisplayName?.trimmingCharacters(in: .whitespacesAndNewlines), !custom.isEmpty {
            return custom
        }
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

    /// Saved meshroom / GLB rooms use manual (default) dimension workflow, not SHARP-measured PLY.
    var isManualSetupRoom: Bool {
        fileType == .meshroom || fileType == .glb
    }

    /// Home list: one line when saved dimensions exist (or W×H + scene depth can reconstruct D).
    /// Manual-setup rooms append “(Default Values)”; SHARP PLY (AI) rooms append “(Near accurate values)”.
    var roomDimensionsListLine: String? {
        let baseLine: String?
        if let w = roomWidth, let h = roomHeight, let d = roomDepth,
           w > 0.05, h > 0.05, d > 0.05, w.isFinite, h.isFinite, d.isFinite {
            baseLine = String(format: "%.2f × %.2f × %.2f m", w, h, d)
        } else if let w = roomWidth, let h = roomHeight,
                  w > 0.05, h > 0.05, w.isFinite, h.isFinite,
                  let sh = roomSceneHeight, sh > 1e-4,
                  let sd = roomSceneDepth, sd > 1e-4 {
            let d = sd * (h / sh)
            guard d > 0.05, d.isFinite else { return nil }
            baseLine = String(format: "%.2f × %.2f × %.2f m", w, h, d)
        } else {
            return nil
        }
        guard let line = baseLine else { return nil }
        if isManualSetupRoom {
            return String(format: "%@ (%@)", line, L10n.RoomViewer.roomDimensionsDefaultValues)
        }
        if fileType == .ply {
            return String(format: "%@ (%@)", line, L10n.RoomViewer.roomDimensionsNearAccurateValues)
        }
        return line
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
            if debugMode, let cachedResolvedURL {
                logDebug("✅ [USDZModel.temporaryURL] Using cached saved-room URL: \(cachedResolvedURL.lastPathComponent)")
            }
            return cachedResolvedURL
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
