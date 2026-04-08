import Foundation
import UIKit
import SceneKit

// MARK: - Saved room .meta JSON (room dims + optional YOLO ratios)

private struct SavedRoomDiskMetadata {
    let orientation: PhotoOrientation
    let width: Float?
    let height: Float?
    let depth: Float?
    /// Depth-raycast / PLY scene units (optional; ratio fitment).
    let sceneWidth: Float?
    let sceneHeight: Float?
    let sceneDepth: Float?
    let yoloWallHeightFrac: Float?
    let yoloFurnitureHeightFracByClass: [String: Float]?
    let yoloRefImageHeightPx: Int?
    let sharpRoomHeightAtYoloCapture: Float?
    /// Optional display name stored in `*.meta` sidecar (list / UI only; file name unchanged).
    let displayName: String?
    let isClassicPly: Bool

    static let empty = SavedRoomDiskMetadata(
        orientation: .portrait,
        width: nil,
        height: nil,
        depth: nil,
        sceneWidth: nil,
        sceneHeight: nil,
        sceneDepth: nil,
        yoloWallHeightFrac: nil,
        yoloFurnitureHeightFracByClass: nil,
        yoloRefImageHeightPx: nil,
        sharpRoomHeightAtYoloCapture: nil,
        displayName: nil,
        isClassicPly: false
    )

    static func parse(dictionary metadata: [String: String]) -> SavedRoomDiskMetadata {
        let orientation = metadata["photoOrientation"].flatMap { PhotoOrientation(rawValue: $0) } ?? .portrait
        let width = metadata["roomWidth"].flatMap { Float($0) }
        let height = metadata["roomHeight"].flatMap { Float($0) }
        let depth = metadata["roomDepth"].flatMap { Float($0) }
        let sceneW = metadata["roomSceneWidth"].flatMap { Float($0) }
        let sceneH = metadata["roomSceneHeight"].flatMap { Float($0) }
        let sceneD = metadata["roomSceneDepth"].flatMap { Float($0) }
        let yoloWall = metadata["yoloWallHeightFrac"].flatMap { Float($0) }
        let yoloRefH = metadata["yoloRefImageHeightPx"].flatMap { Int($0) }
        let sharpCap = metadata["sharpRoomHeightAtYoloCapture"].flatMap { Float($0) }
        let isClassicPly = metadata["isClassicPly"].flatMap { Bool($0) } ?? false
        var furnMap: [String: Float]?
        if let json = metadata["yoloFurnitureHeightFracByClass"],
           let data = json.data(using: .utf8),
           let decoded = try? JSONDecoder().decode([String: Float].self, from: data) {
            furnMap = decoded
        }
        let rawName = metadata["displayName"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let displayName = (rawName?.isEmpty == false) ? rawName : nil
        return SavedRoomDiskMetadata(
            orientation: orientation,
            width: width,
            height: height,
            depth: depth,
            sceneWidth: sceneW,
            sceneHeight: sceneH,
            sceneDepth: sceneD,
            yoloWallHeightFrac: yoloWall,
            yoloFurnitureHeightFracByClass: furnMap,
            yoloRefImageHeightPx: yoloRefH,
            sharpRoomHeightAtYoloCapture: sharpCap,
            displayName: displayName,
            isClassicPly: isClassicPly
        )
    }

    func replacingOrientation(_ newOrientation: PhotoOrientation) -> SavedRoomDiskMetadata {
        SavedRoomDiskMetadata(
            orientation: newOrientation,
            width: width,
            height: height,
            depth: depth,
            sceneWidth: sceneWidth,
            sceneHeight: sceneHeight,
            sceneDepth: sceneDepth,
            yoloWallHeightFrac: yoloWallHeightFrac,
            yoloFurnitureHeightFracByClass: yoloFurnitureHeightFracByClass,
            yoloRefImageHeightPx: yoloRefImageHeightPx,
            sharpRoomHeightAtYoloCapture: sharpRoomHeightAtYoloCapture,
            displayName: displayName,
            isClassicPly: isClassicPly
        )
    }

    func replacingClassicPly(_ newValue: Bool) -> SavedRoomDiskMetadata {
        SavedRoomDiskMetadata(
            orientation: orientation,
            width: width,
            height: height,
            depth: depth,
            sceneWidth: sceneWidth,
            sceneHeight: sceneHeight,
            sceneDepth: sceneDepth,
            yoloWallHeightFrac: yoloWallHeightFrac,
            yoloFurnitureHeightFracByClass: yoloFurnitureHeightFracByClass,
            yoloRefImageHeightPx: yoloRefImageHeightPx,
            sharpRoomHeightAtYoloCapture: sharpRoomHeightAtYoloCapture,
            displayName: displayName,
            isClassicPly: newValue
        )
    }
}

private struct BinaryPlyVertexLayout {
    let headerByteCount: Int
    let vertexCount: Int
    let vertexStride: Int
    let xOffset: Int
    let yOffset: Int
    let zOffset: Int
}

private struct MeasuredPlyRoomDimensions {
    let legacyWidth: Float
    let legacyHeight: Float
    let legacyDepth: Float
    let width: Float
    let height: Float
    let depth: Float
    let sceneWidth: Float
    let sceneHeight: Float
    let sceneDepth: Float
    let zMode: Float
    let zMedian: Float
    let zMean: Float
    let band: Float
    let count: Int
    let floorDiagonal: Float
    let trimmedXSpan: Float
    let trimmedYSpan: Float
    let trimmedZSpan: Float
    let rawWidth: Float
    let rawHeight: Float
}

class USDZModelManager: ObservableObject {
    @Published var models: [USDZModel] = []
    private let supportedSavedRoomExtensions = Set(["usdz", "ply", "meshroom", "glb"])
    
    // Directory for saved rooms
    private let documentsDirectory: URL = {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }()
    
    private var modelsDirectory: URL {
        documentsDirectory.appendingPathComponent("SavedRooms", isDirectory: true)
    }

    /// Exposed for reference image lookup in the room folder (e.g. room thumbnail / meshroom assets).
    var savedRoomsDirectoryURL: URL { modelsDirectory }
    
    init() {
        if AppStateManager.shared.qualitySettings.debugMode {
            logDebug("📦 [USDZModelManager] Initializing...")
        }
        createDirectoriesIfNeeded()
        migrateLegacyClassicSavedRoomsIfNeeded()
        cleanupOrphanSavedRoomArtifactsIfNeeded()
        loadModels()
        if AppStateManager.shared.qualitySettings.debugMode {
            logDebug("📦 [USDZModelManager] Initialization complete. Loaded \(models.count) models")
        }
    }
    
    private func createDirectoriesIfNeeded() {
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: modelsDirectory.path) {
            do {
                try fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
                if AppStateManager.shared.qualitySettings.debugMode {
                    logDebug("✅ [USDZModelManager] Created SavedRooms directory")
                }
            } catch {
                if AppStateManager.shared.qualitySettings.debugMode {
                    logDebug("❌ [USDZModelManager] Failed to create directory: \(error)")
                }
            }
        }
    }

    /// Migrates legacy duplicate `Name_classic.ply` saved-room sidecars into the new single-file `Name.ply` format.
    private func migrateLegacyClassicSavedRoomsIfNeeded() {
        let fileManager = FileManager.default
        guard let directoryContents = try? fileManager.contentsOfDirectory(
            at: modelsDirectory,
            includingPropertiesForKeys: nil
        ) else {
            return
        }

        let legacyClassicURLs = directoryContents.filter { fileURL in
            fileURL.pathExtension.lowercased() == "ply" &&
            fileURL.deletingPathExtension().lastPathComponent.hasSuffix("_classic")
        }
        guard !legacyClassicURLs.isEmpty else { return }

        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        for legacyClassicURL in legacyClassicURLs {
            let legacyClassicStem = legacyClassicURL.deletingPathExtension().lastPathComponent
            let canonicalStem = String(legacyClassicStem.dropLast("_classic".count))
            let canonicalPLYURL = modelsDirectory.appendingPathComponent("\(canonicalStem).ply")

            do {
                if !fileManager.fileExists(atPath: canonicalPLYURL.path) {
                    try fileManager.moveItem(at: legacyClassicURL, to: canonicalPLYURL)
                    if debugMode {
                        logDebug("♻️ [USDZModelManager] Migrated legacy classic-only room \(legacyClassicStem) → \(canonicalStem).ply")
                    }
                } else {
                    try fileManager.removeItem(at: legacyClassicURL)
                    if debugMode {
                        logDebug("♻️ [USDZModelManager] Removed duplicate legacy classic sidecar: \(legacyClassicURL.lastPathComponent)")
                    }
                }
                try mergeClassicPlyFlagIntoSavedRoomMetadata(
                    fileName: canonicalStem,
                    modelFileExtension: "ply",
                    isClassicPly: true
                )
            } catch {
                if debugMode {
                    logDebug("⚠️ [USDZModelManager] Legacy classic migration skipped for \(legacyClassicURL.lastPathComponent): \(error.localizedDescription)")
                }
            }
        }
    }

    private func primarySavedRoomStem(for fileURL: URL) -> String? {
        let ext = fileURL.pathExtension.lowercased()
        guard supportedSavedRoomExtensions.contains(ext) else { return nil }
        return fileURL.deletingPathExtension().lastPathComponent
    }

    private func canonicalPlyStem(for fileName: String) -> String {
        if fileName.hasSuffix("_classic") {
            return String(fileName.dropLast("_classic".count))
        }
        if fileName.hasSuffix("_3dgs") {
            return String(fileName.dropLast("_3dgs".count))
        }
        return fileName
    }

    private func displayNameForSavedRoom(fileName: String, fileType: ModelFileType, metadataDisplayName: String?) -> String? {
        guard fileType == .ply else { return metadataDisplayName }
        let canonicalStem = canonicalPlyStem(for: fileName)
        let baseName = metadataDisplayName ?? canonicalStem.replacingOccurrences(of: "_", with: " ").capitalized
        if fileName.hasSuffix("_classic") {
            return "\(baseName) (Classic PLY)"
        }
        if fileName.hasSuffix("_3dgs") {
            return "\(baseName) (3DGS PLY)"
        }
        return "\(baseName) (PLY)"
    }

    private func orphanArtifactStem(for fileURL: URL) -> String? {
        let fileName = fileURL.lastPathComponent
        if fileName.hasSuffix(".room_metadata.json") {
            return String(fileName.dropLast(".room_metadata.json".count))
        }
        if fileName.hasSuffix(".meta") {
            let withoutMeta = fileURL.deletingPathExtension().lastPathComponent
            return URL(fileURLWithPath: withoutMeta).deletingPathExtension().lastPathComponent
        }
        if fileName.hasSuffix("_thumbnail.jpg") {
            return String(fileName.dropLast("_thumbnail.jpg".count))
        }
        if fileName.hasSuffix("_thumbnail.png") {
            return String(fileName.dropLast("_thumbnail.png".count))
        }
        if fileName.hasSuffix("_sharp_camera.json") {
            return String(fileName.dropLast("_sharp_camera.json".count))
        }
        if fileName.hasSuffix("_classic.ply") {
            return String(fileName.dropLast("_classic.ply".count))
        }
        if fileName.hasSuffix("_3dgs.ply") {
            return String(fileName.dropLast("_3dgs.ply".count))
        }
        return nil
    }

    private func metadataURL(forSavedPlyURL url: URL) -> URL {
        url.deletingLastPathComponent()
            .appendingPathComponent("\(url.deletingPathExtension().lastPathComponent).ply.meta")
    }

    private func binaryPlyLayout(for data: Data) -> BinaryPlyVertexLayout? {
        let endHeaderLF = Data("end_header\n".utf8)
        let endHeaderCRLF = Data("end_header\r\n".utf8)
        let headerEnd: Int
        if let range = data.range(of: endHeaderCRLF) {
            headerEnd = range.upperBound
        } else if let range = data.range(of: endHeaderLF) {
            headerEnd = range.upperBound
        } else {
            return nil
        }

        let headerData = data.prefix(headerEnd)
        let headerString = String(decoding: headerData, as: UTF8.self)
        let lines = headerString.components(separatedBy: .newlines)

        var inVertexElement = false
        var vertexCount: Int?
        var properties: [(name: String, byteWidth: Int)] = []

        func byteWidth(for rawType: String) -> Int? {
            switch rawType {
            case "char", "uchar", "int8", "uint8":
                return 1
            case "short", "ushort", "int16", "uint16":
                return 2
            case "int", "uint", "float", "int32", "uint32", "float32":
                return 4
            case "double", "float64":
                return 8
            default:
                return nil
            }
        }

        for rawLine in lines {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            let parts = line.split(separator: " ").map(String.init)
            guard let keyword = parts.first else { continue }
            if keyword == "element", parts.count >= 3 {
                inVertexElement = (parts[1] == "vertex")
                if inVertexElement {
                    vertexCount = Int(parts[2])
                    properties.removeAll(keepingCapacity: false)
                }
                continue
            }
            guard inVertexElement, keyword == "property", parts.count >= 3 else { continue }
            if parts[1] == "list" {
                return nil
            }
            guard let width = byteWidth(for: parts[1]) else { return nil }
            properties.append((name: parts[2], byteWidth: width))
        }

        guard let count = vertexCount, count > 0 else { return nil }
        var runningOffset = 0
        var xOffset: Int?
        var yOffset: Int?
        var zOffset: Int?
        for property in properties {
            switch property.name {
            case "x": xOffset = runningOffset
            case "y": yOffset = runningOffset
            case "z": zOffset = runningOffset
            default: break
            }
            runningOffset += property.byteWidth
        }
        guard let xOffset, let yOffset, let zOffset, runningOffset > 0 else { return nil }
        return BinaryPlyVertexLayout(
            headerByteCount: headerEnd,
            vertexCount: count,
            vertexStride: runningOffset,
            xOffset: xOffset,
            yOffset: yOffset,
            zOffset: zOffset
        )
    }

    private func readFloat32(from data: Data, offset: Int) -> Float? {
        guard offset >= 0, offset + 4 <= data.count else { return nil }
        var bits: UInt32 = 0
        withUnsafeMutableBytes(of: &bits) { rawBuffer in
            data.copyBytes(to: rawBuffer, from: offset..<(offset + 4))
        }
        return Float(bitPattern: UInt32(littleEndian: bits))
    }

    private func measureRoomDimensions(forPly url: URL, treatAsClassicPly: Bool? = nil) -> MeasuredPlyRoomDimensions? {
        guard let data = try? Data(contentsOf: url),
              let layout = binaryPlyLayout(for: data) else {
            return nil
        }

        let isClassicVariant = treatAsClassicPly ?? url.deletingPathExtension().lastPathComponent.hasSuffix("_classic")
        var normalizedXs: [Float] = []
        var normalizedYs: [Float] = []
        var normalizedDepths: [Float] = []
        normalizedXs.reserveCapacity(layout.vertexCount)
        normalizedYs.reserveCapacity(layout.vertexCount)
        normalizedDepths.reserveCapacity(layout.vertexCount)

        var minX = Float.greatestFiniteMagnitude
        var maxX = -Float.greatestFiniteMagnitude
        var minY = Float.greatestFiniteMagnitude
        var maxY = -Float.greatestFiniteMagnitude
        var minZ = Float.greatestFiniteMagnitude
        var maxZ = -Float.greatestFiniteMagnitude

        for index in 0..<layout.vertexCount {
            let vertexOffset = layout.headerByteCount + index * layout.vertexStride
            guard let storedX = readFloat32(from: data, offset: vertexOffset + layout.xOffset),
                  let storedY = readFloat32(from: data, offset: vertexOffset + layout.yOffset),
                  let storedZ = readFloat32(from: data, offset: vertexOffset + layout.zOffset),
                  storedX.isFinite, storedY.isFinite, storedZ.isFinite else {
                continue
            }

            minX = min(minX, storedX)
            maxX = max(maxX, storedX)
            minY = min(minY, storedY)
            maxY = max(maxY, storedY)
            minZ = min(minZ, storedZ)
            maxZ = max(maxZ, storedZ)

            let normalizedX = storedX
            let normalizedY = isClassicVariant ? -storedY : storedY
            let normalizedZ = isClassicVariant ? -storedZ : storedZ
            let depth = -normalizedZ
            guard depth.isFinite, depth > 0.01 else { continue }

            normalizedXs.append(normalizedX)
            normalizedYs.append(normalizedY)
            normalizedDepths.append(depth)
        }

        guard !normalizedXs.isEmpty,
              minX.isFinite, maxX.isFinite, minY.isFinite, maxY.isFinite, minZ.isFinite, maxZ.isFinite else {
            return nil
        }

        func percentileIndex(_ p: Float, count: Int) -> Int {
            guard count > 1 else { return 0 }
            let raw = Int((Float(count - 1) * p).rounded(.down))
            return min(max(raw, 0), count - 1)
        }

        let sortedAllXs = normalizedXs.sorted()
        let sortedAllYs = normalizedYs.sorted()
        let depths = normalizedDepths.sorted()
        guard sortedAllXs.count >= 64, sortedAllYs.count >= 64, depths.count >= 64 else { return nil }

        let xP3 = sortedAllXs[percentileIndex(0.03, count: sortedAllXs.count)]
        let xP97 = sortedAllXs[percentileIndex(0.97, count: sortedAllXs.count)]
        let yP3 = sortedAllYs[percentileIndex(0.03, count: sortedAllYs.count)]
        let yP97 = sortedAllYs[percentileIndex(0.97, count: sortedAllYs.count)]
        let zP3 = depths[percentileIndex(0.03, count: depths.count)]
        let zP97 = depths[percentileIndex(0.97, count: depths.count)]

        let trimmedXSpan = xP97 - xP3
        let trimmedYSpan = yP97 - yP3
        let trimmedZSpan = zP97 - zP3
        guard trimmedXSpan.isFinite, trimmedYSpan.isFinite, trimmedZSpan.isFinite,
              trimmedXSpan > 0.01, trimmedZSpan > 0.01 else { return nil }

        let floorDiagonal = sqrt(max(0, trimmedXSpan * trimmedXSpan + trimmedZSpan * trimmedZSpan))
        let trimmedDepths = depths.filter { $0 >= zP3 && $0 <= zP97 }
        guard trimmedDepths.count >= 64 else { return nil }

        let binCount = 200
        let binWidth = max(trimmedZSpan / Float(binCount), 1e-4)
        var histogram = [Int](repeating: 0, count: binCount)
        for depth in trimmedDepths {
            var bucket = Int(((depth - zP3) / binWidth).rounded(.down))
            bucket = min(max(bucket, 0), binCount - 1)
            histogram[bucket] += 1
        }
        guard let peakIndex = histogram.enumerated().max(by: { $0.element < $1.element })?.offset else {
            return nil
        }

        let zMode = zP3 + (Float(peakIndex) + 0.5) * binWidth
        let band = max(0.10 * zMode, 1e-4)
        var backWallXs: [Float] = []
        var backWallYs: [Float] = []
        var backWallDepths: [Float] = []
        backWallXs.reserveCapacity(trimmedDepths.count / 2)
        backWallYs.reserveCapacity(trimmedDepths.count / 2)
        backWallDepths.reserveCapacity(trimmedDepths.count / 2)

        for index in 0..<normalizedDepths.count {
            let depth = normalizedDepths[index]
            guard abs(depth - zMode) < band else { continue }
            backWallXs.append(normalizedXs[index])
            backWallYs.append(normalizedYs[index])
            backWallDepths.append(depth)
        }

        guard backWallXs.count >= 64 else { return nil }

        let sortedXs = backWallXs.sorted()
        let sortedYs = backWallYs.sorted()
        let sortedZs = backWallDepths.sorted()
        let idx5 = percentileIndex(0.05, count: sortedXs.count)
        let idx95 = percentileIndex(0.95, count: sortedXs.count)
        let yIdx5 = percentileIndex(0.05, count: sortedYs.count)
        let yIdx95 = percentileIndex(0.95, count: sortedYs.count)
        let rawWidth = sortedXs[idx95] - sortedXs[idx5]
        let rawHeight = sortedYs[yIdx95] - sortedYs[yIdx5]
        let zMedian = sortedZs[sortedZs.count / 2]
        let zMean = sortedZs.reduce(0, +) / Float(sortedZs.count)
        let legacyWidth = rawWidth / 1.2
        let legacyHeight = rawHeight / 1.2
        let legacyDepth = zMedian * 1.08
        let finalDepth = floorDiagonal
        let finalWidth: Float
        if rawHeight > rawWidth {
            finalWidth = sqrt(max(0, rawHeight * rawHeight - rawWidth * rawWidth))
        } else {
            finalWidth = rawWidth
        }
        let finalHeight: Float
        if trimmedYSpan > trimmedXSpan {
            finalHeight = sqrt(max(0, trimmedYSpan * trimmedYSpan - trimmedXSpan * trimmedXSpan))
        } else {
            finalHeight = trimmedYSpan
        }

        return MeasuredPlyRoomDimensions(
            legacyWidth: legacyWidth,
            legacyHeight: legacyHeight,
            legacyDepth: legacyDepth,
            width: finalWidth,
            height: finalHeight,
            depth: finalDepth,
            sceneWidth: maxX - minX,
            sceneHeight: maxY - minY,
            sceneDepth: maxZ - minZ,
            zMode: zMode,
            zMedian: zMedian,
            zMean: zMean,
            band: band,
            count: backWallXs.count,
            floorDiagonal: floorDiagonal,
            trimmedXSpan: trimmedXSpan,
            trimmedYSpan: trimmedYSpan,
            trimmedZSpan: trimmedZSpan,
            rawWidth: rawWidth,
            rawHeight: rawHeight
        )
    }

    private func cleanupOrphanSavedRoomArtifactsIfNeeded() {
        let fileManager = FileManager.default
        guard let directoryContents = try? fileManager.contentsOfDirectory(
            at: modelsDirectory,
            includingPropertiesForKeys: nil
        ) else {
            return
        }

        let livePrimaryStems = Set(directoryContents.compactMap(primarySavedRoomStem(for:)))
        let debugMode = AppStateManager.shared.qualitySettings.debugMode

        for fileURL in directoryContents {
            guard let orphanStem = orphanArtifactStem(for: fileURL),
                  !livePrimaryStems.contains(orphanStem) else {
                continue
            }

            do {
                try fileManager.removeItem(at: fileURL)
                if debugMode {
                    logDebug("🧹 [USDZModelManager] Removed orphan saved-room artifact: \(fileURL.lastPathComponent)")
                }
            } catch {
                if debugMode {
                    logDebug("⚠️ [USDZModelManager] Failed to remove orphan artifact \(fileURL.lastPathComponent): \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func loadModels() {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        cleanupOrphanSavedRoomArtifactsIfNeeded()
        
        if debugMode {
            logDebug("📦 [USDZModelManager] Starting to load models...")
        }
        
        // Bundle models with their fixed orientations
        let bundleModelConfigs: [(name: String, orientation: PhotoOrientation)] = [
            ("vintage_living_room", .landscape),
            ("cozy_living_room_baked", .portrait)
        ]

        if debugMode {
            logDebug("📦 [USDZModelManager] Loading bundle models: \(bundleModelConfigs.map { $0.name })")
        }

        // Load bundle models with fixed orientations
        let bundleModels = bundleModelConfigs.compactMap { config -> USDZModel? in
            let model = USDZModel(
                name: config.name,
                fileName: config.name,
                isSavedRoom: false,
                fileType: .usdz,
                fileSize: nil,
                photoOrientation: config.orientation
            )
            if model.dataAsset != nil {
                if debugMode {
                    logDebug("   ✅ Bundle model loaded: \(config.name) (orientation: \(config.orientation.rawValue))")
                }
                return model
            } else {
                if debugMode {
                    logDebug("   ❌ Bundle model failed: \(config.name)")
                }
                return nil
            }
        }
        
        // Load saved rooms (both USDZ and PLY) and sort by date
        var savedRoomModels: [USDZModel] = []

        // Supported file extensions
        let supportedExtensions = ["usdz", "ply", "meshroom", "glb"]

        do {
            let files = try FileManager.default.contentsOfDirectory(at: modelsDirectory,
                                                                    includingPropertiesForKeys: [.creationDateKey, .fileSizeKey])
            let allModelFiles = files.filter { supportedExtensions.contains($0.pathExtension.lowercased()) }
            let modelFiles = allModelFiles

            if debugMode {
                logDebug("📦 [USDZModelManager] Found \(modelFiles.count) model files in SavedRooms")
            }

            // Get files with dates and sizes
            var filesWithDates: [(url: URL, date: Date, size: UInt64)] = []
            for fileURL in modelFiles {
                let attrs = try fileURL.resourceValues(forKeys: [.creationDateKey, .fileSizeKey])
                let date = attrs.creationDate ?? Date.distantPast
                let size = UInt64(attrs.fileSize ?? 0)
                filesWithDates.append((url: fileURL, date: date, size: size))

                if debugMode {
                    let fileName = fileURL.deletingPathExtension().lastPathComponent
                    let ext = fileURL.pathExtension.lowercased()
                    logDebug("   - \(fileName).\(ext) (created: \(date), size: \(size / 1024) KB)")
                }
            }

            // Sort by date - NEWEST FIRST
            filesWithDates.sort { $0.date > $1.date }

            if debugMode {
                logDebug("📦 [USDZModelManager] Sorted by date (newest first):")
            }

            // Create models in sorted order with proper file type
            for (fileURL, _, size) in filesWithDates {
                let fileName = fileURL.deletingPathExtension().lastPathComponent
                let ext = fileURL.pathExtension.lowercased()

                // Determine file type
                let fileType: ModelFileType
                switch ext {
                case "ply":
                    fileType = .ply
                case "meshroom":
                    fileType = .meshroom
                case "glb":
                    fileType = .glb
                default:
                    fileType = .usdz
                }

                // Load metadata based on file type
                let metadata: SavedRoomDiskMetadata
                switch fileType {
                case .ply:
                    metadata = loadPLYMetadata(for: fileName)
                case .meshroom:
                    metadata = loadMeshRoomMetadata(for: fileName)
                case .glb:
                    metadata = loadGLBMetadata(for: fileName)
                default:
                    metadata = loadUSDZMetadata(for: fileName)
                }

                if debugMode {
                    logDebug("   - \(fileName) (\(fileType.rawValue), orientation: \(metadata.orientation.rawValue), w: \(metadata.width ?? 0), h: \(metadata.height ?? 0), d: \(metadata.depth ?? 0))")
                }

                let model = USDZModel(
                    name: fileName,
                    fileName: fileName,
                    isSavedRoom: true,
                    fileType: fileType,
                    fileSize: size,
                    photoOrientation: metadata.orientation,
                    roomWidth: metadata.width,
                    roomHeight: metadata.height,
                    roomDepth: metadata.depth,
                    roomSceneWidth: metadata.sceneWidth,
                    roomSceneHeight: metadata.sceneHeight,
                    roomSceneDepth: metadata.sceneDepth,
                    yoloWallHeightFrac: metadata.yoloWallHeightFrac,
                    yoloFurnitureHeightFracByClass: metadata.yoloFurnitureHeightFracByClass,
                    yoloRefImageHeightPx: metadata.yoloRefImageHeightPx,
                    sharpRoomHeightAtYoloCapture: metadata.sharpRoomHeightAtYoloCapture,
                    customDisplayName: displayNameForSavedRoom(
                        fileName: fileName,
                        fileType: fileType,
                        metadataDisplayName: metadata.displayName
                    ),
                    isClassicPly: metadata.isClassicPly,
                    cachedResolvedURL: fileURL
                )
                savedRoomModels.append(model)
            }

        } catch {
            if debugMode {
                logDebug("⚠️ [USDZModelManager] Error reading saved rooms: \(error)")
            }
        }
        
        // Combine: saved rooms first (newest at top), then bundle models
        models = savedRoomModels + bundleModels
        
        if debugMode {
            logDebug("📦 [USDZModelManager] Loading complete:")
            logDebug("   - Total models: \(models.count)")
            logDebug("   - Saved rooms: \(savedRoomModels.count)")
            logDebug("   - Bundle models: \(bundleModels.count)")
            if !models.isEmpty {
                logDebug("   - First 3: \(models.prefix(3).map { $0.fileName }.joined(separator: ", "))")
            }
        }
    }
    
    func getModel(by fileName: String) -> USDZModel? {
        return models.first { $0.fileName == fileName }
    }
    
    /// Writes `photoOrientation` into `\(fileName).\(modelFileExtension).meta`, merging with existing keys.
    /// Used when thumbnail EXIF / aspect corrects stale meta so AR roll and reopen use the same orientation.
    func mergePhotoOrientationIntoSavedRoomMetadata(
        fileName: String,
        modelFileExtension: String,
        photoOrientation: PhotoOrientation
    ) throws {
        let metadataURL = modelsDirectory.appendingPathComponent("\(fileName).\(modelFileExtension).meta")
        var dict: [String: String] = [:]
        if FileManager.default.fileExists(atPath: metadataURL.path),
           let data = try? Data(contentsOf: metadataURL),
           let existing = try? JSONDecoder().decode([String: String].self, from: data) {
            dict = existing
        }
        dict["photoOrientation"] = photoOrientation.rawValue
        let out = try JSONEncoder().encode(dict)
        try out.write(to: metadataURL, options: [.atomic])
    }

    /// Writes `isClassicPly` into `\(fileName).\(modelFileExtension).meta`, merging with existing keys.
    private func mergeClassicPlyFlagIntoSavedRoomMetadata(
        fileName: String,
        modelFileExtension: String,
        isClassicPly: Bool
    ) throws {
        let metadataURL = modelsDirectory.appendingPathComponent("\(fileName).\(modelFileExtension).meta")
        var dict: [String: String] = [:]
        if FileManager.default.fileExists(atPath: metadataURL.path),
           let data = try? Data(contentsOf: metadataURL),
           let existing = try? JSONDecoder().decode([String: String].self, from: data) {
            dict = existing
        }
        dict["isClassicPly"] = isClassicPly ? "true" : "false"
        let out = try JSONEncoder().encode(dict)
        try out.write(to: metadataURL, options: [.atomic])
    }

    /// Writes calibrated room dimensions into `\(fileName).\(modelFileExtension).meta`, merging with existing keys.
    func mergeRoomDimensionsIntoSavedRoomMetadata(
        fileName: String,
        modelFileExtension: String,
        roomWidth: Float,
        roomHeight: Float,
        roomDepth: Float
    ) throws {
        let metadataURL = modelsDirectory.appendingPathComponent("\(fileName).\(modelFileExtension).meta")
        var dict: [String: String] = [:]
        if FileManager.default.fileExists(atPath: metadataURL.path),
           let data = try? Data(contentsOf: metadataURL),
           let existing = try? JSONDecoder().decode([String: String].self, from: data) {
            dict = existing
        }
        dict["roomWidth"] = String(format: "%.2f", roomWidth)
        dict["roomHeight"] = String(format: "%.2f", roomHeight)
        dict["roomDepth"] = String(format: "%.2f", roomDepth)
        let out = try JSONEncoder().encode(dict)
        try out.write(to: metadataURL, options: [.atomic])
    }

    /// Merges YOLO ratio calibration keys into an existing `\(fileName).\(modelFileExtension).meta` file without removing other entries.
    func mergeYoloCalibrationMetadata(
        fileName: String,
        modelFileExtension: String,
        wallHeightFrac: Float,
        furnitureFractionsByClass: [String: Float],
        referenceImageHeightPx: Int,
        sharpRoomHeightAtCapture: Float?
    ) throws {
        let metadataURL = modelsDirectory.appendingPathComponent("\(fileName).\(modelFileExtension).meta")
        var dict: [String: String] = [:]
        if FileManager.default.fileExists(atPath: metadataURL.path),
           let data = try? Data(contentsOf: metadataURL),
           let existing = try? JSONDecoder().decode([String: String].self, from: data) {
            dict = existing
        }

        dict["yoloWallHeightFrac"] = String(format: "%.6f", wallHeightFrac)
        dict["yoloRefImageHeightPx"] = String(referenceImageHeightPx)
        if let h = sharpRoomHeightAtCapture {
            dict["sharpRoomHeightAtYoloCapture"] = String(format: "%.4f", h)
        }
        if let jsonData = try? JSONEncoder().encode(furnitureFractionsByClass),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            dict["yoloFurnitureHeightFracByClass"] = jsonString
        }

        let out = try JSONEncoder().encode(dict)
        try out.write(to: metadataURL, options: [.atomic])
    }

    func refreshModels() {
        if AppStateManager.shared.qualitySettings.debugMode {
            logDebug("🔄 [USDZModelManager] Refreshing models...")
        }
        models.removeAll()
        loadModels()
        if AppStateManager.shared.qualitySettings.debugMode {
            logDebug("✅ [USDZModelManager] Refresh complete. Now have \(models.count) models")
        }
    }

    /// Writes `displayName` into the per-room `*.meta` sidecar (file name on disk unchanged).
    func updateDisplayName(for model: USDZModel, newName: String) throws {
        guard model.isSavedRoom else { return }
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw NSError(domain: "USDZModelManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "Empty name"])
        }
        let fileExtension: String
        switch model.fileType {
        case .ply: fileExtension = "ply"
        case .meshroom: fileExtension = "meshroom"
        case .glb: fileExtension = "glb"
        case .usdz: fileExtension = "usdz"
        }
        let metadataURL = modelsDirectory.appendingPathComponent("\(model.fileName).\(fileExtension).meta")
        var dict: [String: String] = [:]
        if FileManager.default.fileExists(atPath: metadataURL.path),
           let data = try? Data(contentsOf: metadataURL),
           let existing = try? JSONDecoder().decode([String: String].self, from: data) {
            dict = existing
        }
        dict["displayName"] = trimmed
        let out = try JSONEncoder().encode(dict)
        try out.write(to: metadataURL, options: [.atomic])
        DispatchQueue.main.async { self.refreshModels() }
    }
    
    // ✅ NEW: Delete model functionality
    func deleteModel(id: UUID) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            logDebug("🗑️ [USDZModelManager] Starting delete for model ID: \(id)")
        }
        
        guard let modelIndex = models.firstIndex(where: { $0.id == id }) else {
            if debugMode {
                logDebug("❌ [USDZModelManager] Model not found with ID: \(id)")
            }
            return
        }
        
        let model = models[modelIndex]
        if debugMode {
            logDebug("🗑️ [USDZModelManager] Found model: \(model.displayName) (isSavedRoom: \(model.isSavedRoom))")
        }
        
        // Only delete file if it's a saved room (not bundle model)
        if model.isSavedRoom {
            // Use appropriate file extension based on file type
            let fileExtension: String
            switch model.fileType {
            case .ply:
                fileExtension = "ply"
            case .meshroom:
                fileExtension = "meshroom"
            case .glb:
                fileExtension = "glb"
            case .usdz:
                fileExtension = "usdz"
            }
            let canonicalFileName = (model.fileType == .ply) ? canonicalPlyStem(for: model.fileName) : model.fileName
            let fileURL = modelsDirectory.appendingPathComponent("\(canonicalFileName).\(fileExtension)")
            let classicSidecarURL = modelsDirectory.appendingPathComponent("\(canonicalFileName)_classic.ply")
            let threeDGSSidecarURL = modelsDirectory.appendingPathComponent("\(canonicalFileName)_3dgs.ply")
            let metadataURL = modelsDirectory.appendingPathComponent("\(canonicalFileName).\(fileExtension).meta")
            let enhancedMetadataSidecarURL = enhancedMetadataURL(forRoomURL: fileURL)
            let thumbnailJPGURL = modelsDirectory.appendingPathComponent("\(canonicalFileName)_thumbnail.jpg")
            let thumbnailPNGURL = modelsDirectory.appendingPathComponent("\(canonicalFileName)_thumbnail.png")
            
            do {
                if FileManager.default.fileExists(atPath: fileURL.path) {
                    try FileManager.default.removeItem(at: fileURL)
                    if debugMode {
                        logDebug("✅ [USDZModelManager] File deleted: \(fileURL.lastPathComponent)")
                    }
                } else {
                    if debugMode {
                        logDebug("⚠️ [USDZModelManager] File not found: \(fileURL.path)")
                    }
                }

                if FileManager.default.fileExists(atPath: metadataURL.path) {
                    try FileManager.default.removeItem(at: metadataURL)
                    if debugMode {
                        logDebug("✅ [USDZModelManager] Metadata file deleted: \(metadataURL.lastPathComponent)")
                    }
                }
                if FileManager.default.fileExists(atPath: classicSidecarURL.path) {
                    try FileManager.default.removeItem(at: classicSidecarURL)
                    if debugMode {
                        logDebug("✅ [USDZModelManager] Classic sidecar deleted: \(classicSidecarURL.lastPathComponent)")
                    }
                }
                if FileManager.default.fileExists(atPath: threeDGSSidecarURL.path) {
                    try FileManager.default.removeItem(at: threeDGSSidecarURL)
                    if debugMode {
                        logDebug("✅ [USDZModelManager] 3DGS sidecar deleted: \(threeDGSSidecarURL.lastPathComponent)")
                    }
                }
                if FileManager.default.fileExists(atPath: enhancedMetadataSidecarURL.path) {
                    try FileManager.default.removeItem(at: enhancedMetadataSidecarURL)
                    if debugMode {
                        logDebug("✅ [USDZModelManager] Enhanced metadata deleted: \(enhancedMetadataSidecarURL.lastPathComponent)")
                    }
                }
                if FileManager.default.fileExists(atPath: thumbnailJPGURL.path) {
                    try FileManager.default.removeItem(at: thumbnailJPGURL)
                    if debugMode {
                        logDebug("✅ [USDZModelManager] Thumbnail deleted: \(thumbnailJPGURL.lastPathComponent)")
                    }
                }
                if FileManager.default.fileExists(atPath: thumbnailPNGURL.path) {
                    try FileManager.default.removeItem(at: thumbnailPNGURL)
                    if debugMode {
                        logDebug("✅ [USDZModelManager] Thumbnail deleted: \(thumbnailPNGURL.lastPathComponent)")
                    }
                }
            } catch {
                if debugMode {
                    logDebug("❌ [USDZModelManager] Failed to delete file: \(error)")
                }
                CrashReporter.shared.report(error, context: "Deleting Room")
            }
            cleanupOrphanSavedRoomArtifactsIfNeeded()
        } else {
            if debugMode {
                logDebug("⚠️ [USDZModelManager] Skipping file deletion - this is a bundle model")
            }
        }
        
        // Rebuild from disk so sidecars / stragglers cannot remain visible in UI.
        refreshModels()
        if debugMode {
            logDebug("✅ [USDZModelManager] Delete finished; models reloaded from disk")
        }
    }
    
    func saveRoom(scene: SCNScene, name: String, completion: @escaping (Bool, String?) -> Void) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            logDebug("💾 [USDZModelManager] Starting to save room: \(name)")
        }
        
        let fileName = sanitizeFileName(name)
        let fileURL = modelsDirectory.appendingPathComponent("\(fileName).usdz")
        
        // Add lighting to the scene before export
        addLightingToScene(scene)
        
        // Export scene to USDZ
        scene.write(to: fileURL, options: nil, delegate: nil) { progress, error, _ in
            if let error = error {
                if debugMode {
                    logDebug("❌ [USDZModelManager] Export failed: \(error)")
                }
                DispatchQueue.main.async {
                    completion(false, error.localizedDescription)
                }
                return
            }
            
            if progress >= 1.0 {
                if debugMode {
                    logDebug("✅ [USDZModelManager] Export complete: \(fileURL.lastPathComponent)")
                }
                
                // Refresh models list after save
                DispatchQueue.main.async {
                    self.refreshModels()
                    completion(true, nil)
                }
            }
        }
    }
    
    private func addLightingToScene(_ scene: SCNScene) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            logDebug("💡 [USDZModelManager] Adding lighting to scene...")
        }
        
        // Remove any existing lights to avoid conflicts
        scene.rootNode.childNodes.filter { $0.light != nil }.forEach { $0.removeFromParentNode() }
        
        // Add ambient light (soft overall illumination)
        let ambientLight = SCNLight()
        ambientLight.type = .ambient
        ambientLight.color = UIColor.white
        ambientLight.intensity = 300
        let ambientNode = SCNNode()
        ambientNode.light = ambientLight
        scene.rootNode.addChildNode(ambientNode)
        if debugMode {
            logDebug("   ✅ Added ambient light (intensity: 300)")
        }
        
        // Add directional light from above (like sun)
        let directionalLight = SCNLight()
        directionalLight.type = .directional
        directionalLight.color = UIColor.white
        directionalLight.intensity = 800
        directionalLight.castsShadow = true
        directionalLight.shadowMode = .deferred
        let directionalNode = SCNNode()
        directionalNode.light = directionalLight
        directionalNode.position = SCNVector3(0, 5, 0)
        directionalNode.eulerAngles = SCNVector3(-Float.pi / 3, 0, 0) // Angle downward
        scene.rootNode.addChildNode(directionalNode)
        if debugMode {
            logDebug("   ✅ Added directional light (intensity: 800)")
        }
        
        // Add fill light from front (helps with shadows)
        let fillLight = SCNLight()
        fillLight.type = .omni
        fillLight.color = UIColor(white: 0.9, alpha: 1.0)
        fillLight.intensity = 400
        let fillNode = SCNNode()
        fillNode.light = fillLight
        fillNode.position = SCNVector3(0, 1.5, 3)
        scene.rootNode.addChildNode(fillNode)
        if debugMode {
            logDebug("   ✅ Added fill light (intensity: 400)")
        }
        
        if debugMode {
            logDebug("💡 [USDZModelManager] Lighting setup complete")
        }
    }
    
    private func sanitizeFileName(_ name: String) -> String {
        let invalidChars = CharacterSet(charactersIn: ":/\\?%*|\"<>")
        let sanitized = name.components(separatedBy: invalidChars).joined()
        return sanitized.replacingOccurrences(of: " ", with: "_")
    }

    private func canonicalRoomStem(for fileURL: URL) -> String {
        let stem = fileURL.deletingPathExtension().lastPathComponent
        if stem.hasSuffix("_classic") {
            return String(stem.dropLast("_classic".count))
        }
        return stem
    }

    private func enhancedMetadataURL(forRoomURL roomURL: URL) -> URL {
        let stem = canonicalRoomStem(for: roomURL)
        return roomURL.deletingLastPathComponent().appendingPathComponent("\(stem).room_metadata.json")
    }

    func loadEnhancedMetadata(forRoomURL roomURL: URL) -> EnhancedRoomMetadata? {
        let url = enhancedMetadataURL(forRoomURL: roomURL)
        if let data = try? Data(contentsOf: url) {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            do {
                return try decoder.decode(EnhancedRoomMetadata.self, from: data)
            } catch {
                logDebug("❌ [USDZModelManager] Failed to decode enhanced metadata at \(url.lastPathComponent): \(error)")
            }
        }

        let legacyURL = roomURL.deletingLastPathComponent()
            .appendingPathComponent("\(canonicalRoomStem(for: roomURL)).\(roomURL.pathExtension).meta")
        guard let legacyData = try? Data(contentsOf: legacyURL),
              let legacy = try? JSONDecoder().decode([String: String].self, from: legacyData) else {
            return nil
        }
        return EnhancedRoomMetadata.fromLegacyFlatMeta(legacy)
    }

    func loadEnhancedMetadata(forSavedRoomNamed fileName: String, fileType: ModelFileType = .ply) -> EnhancedRoomMetadata? {
        let extensionName: String
        switch fileType {
        case .ply: extensionName = "ply"
        case .meshroom: extensionName = "meshroom"
        case .glb: extensionName = "glb"
        case .usdz: extensionName = "usdz"
        }
        let roomURL = modelsDirectory.appendingPathComponent("\(fileName).\(extensionName)")
        return loadEnhancedMetadata(forRoomURL: roomURL)
    }

    func saveEnhancedMetadata(_ metadata: EnhancedRoomMetadata, nextTo roomURL: URL) throws {
        let url = enhancedMetadataURL(forRoomURL: roomURL)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(metadata)
        try data.write(to: url, options: [.atomic])
        logDebug("✅ [USDZModelManager] Enhanced metadata saved: \(url.lastPathComponent)")
    }

    func saveEnhancedMetadata(
        _ metadata: EnhancedRoomMetadata,
        forSavedRoomNamed fileName: String,
        fileType: ModelFileType = .ply
    ) throws {
        let sanitizedName = sanitizeFileName(fileName)
        let extensionName: String
        switch fileType {
        case .ply: extensionName = "ply"
        case .meshroom: extensionName = "meshroom"
        case .glb: extensionName = "glb"
        case .usdz: extensionName = "usdz"
        }
        let roomURL = modelsDirectory.appendingPathComponent("\(sanitizedName).\(extensionName)")
        try saveEnhancedMetadata(metadata, nextTo: roomURL)
    }

    /// Save a PLY file to SavedRooms directory
    func savePLY(
        from sourceURL: URL,
        name: String,
        photoOrientation: PhotoOrientation = .portrait,
        roomWidth: Float? = nil,
        roomHeight: Float? = nil,
        roomDepth: Float? = nil,
        roomSceneWidth: Float? = nil,
        roomSceneHeight: Float? = nil,
        roomSceneDepth: Float? = nil,
        completion: @escaping (Bool, String?) -> Void
    ) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode

        if debugMode {
            logDebug("💾 [USDZModelManager] Starting to save PLY: \(name) (orientation: \(photoOrientation.rawValue), width: \(roomWidth ?? 0), height: \(roomHeight ?? 0), depth: \(roomDepth ?? 0))")
        }

        let fileName = sanitizeFileName(name)
        let destinationURL = modelsDirectory.appendingPathComponent("\(fileName).ply")
        let classicSidecarURL = modelsDirectory.appendingPathComponent("\(fileName)_classic.ply")
        let threeDGSSidecarURL = modelsDirectory.appendingPathComponent("\(fileName)_3dgs.ply")

        // Check if source exists
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            if debugMode {
                logDebug("❌ [USDZModelManager] Source PLY not found: \(sourceURL.path)")
            }
            completion(false, "Source file not found")
            return
        }

        do {
            // Remove existing file if it exists
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            if FileManager.default.fileExists(atPath: classicSidecarURL.path) {
                try FileManager.default.removeItem(at: classicSidecarURL)
            }
            if FileManager.default.fileExists(atPath: threeDGSSidecarURL.path) {
                try FileManager.default.removeItem(at: threeDGSSidecarURL)
            }
            let baseMetadataURL = metadataURL(forSavedPlyURL: destinationURL)
            let classicMetadataURL = metadataURL(forSavedPlyURL: classicSidecarURL)
            let threeDGSMetadataURL = metadataURL(forSavedPlyURL: threeDGSSidecarURL)
            for metadataURL in [baseMetadataURL, classicMetadataURL, threeDGSMetadataURL] {
                if FileManager.default.fileExists(atPath: metadataURL.path) {
                    try FileManager.default.removeItem(at: metadataURL)
                }
            }

            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)

            let sourceEnhancedMetadataURL = enhancedMetadataURL(forRoomURL: sourceURL)
            let destinationEnhancedMetadataURL = enhancedMetadataURL(forRoomURL: destinationURL)
            if FileManager.default.fileExists(atPath: sourceEnhancedMetadataURL.path) {
                if FileManager.default.fileExists(atPath: destinationEnhancedMetadataURL.path) {
                    try FileManager.default.removeItem(at: destinationEnhancedMetadataURL)
                }
                try FileManager.default.copyItem(at: sourceEnhancedMetadataURL, to: destinationEnhancedMetadataURL)
                if debugMode {
                    logDebug("💾 [USDZModelManager] Copied enhanced metadata: \(destinationEnhancedMetadataURL.lastPathComponent)")
                }
            }

            CameraExifSidecar.copySidecarIfPresent(fromRoomURL: sourceURL, toSavedRoomURL: destinationURL)
            SharpCameraSidecar.copySidecarIfPresent(fromRoomURL: sourceURL, toSavedRoomURL: destinationURL)

            let variantDestinations: [(label: String, url: URL, isClassic: Bool)] = [
                ("base_ply", destinationURL, true),
            ].filter { FileManager.default.fileExists(atPath: $0.url.path) }

            for variant in variantDestinations {
                let measured = measureRoomDimensions(forPly: variant.url, treatAsClassicPly: variant.isClassic)
                let finalRoomWidth = measured?.width ?? roomWidth
                let finalRoomHeight = measured?.height ?? roomHeight
                let finalRoomDepth = measured?.depth ?? roomDepth
                let finalSceneWidth = measured?.sceneWidth ?? roomSceneWidth
                let finalSceneHeight = measured?.sceneHeight ?? roomSceneHeight
                let finalSceneDepth = measured?.sceneDepth ?? roomSceneDepth

                let sourceTag: String
                let approachTag: String
                if measured != nil {
                    sourceTag = "measured"
                    approachTag = "pythagoras_diagonal"
                } else if finalRoomWidth != nil || finalRoomHeight != nil || finalRoomDepth != nil {
                    sourceTag = "fallback"
                    approachTag = "preview_active_room_dimensions"
                } else {
                    sourceTag = "unavailable"
                    approachTag = "none"
                }

                if let measured, debugMode {
                    logDebug(
                        "[GREEN][ROOM_DIMS][\(variant.label)] FILE=\(variant.url.lastPathComponent) " +
                        "SOURCE=\(sourceTag.uppercased()) APPROACH=LEGACY_BACK_WALL_PERCENTILE " +
                        "BACK_WALL_Z=\(String(format: "%.4f", measured.zMode)) " +
                        "Z_MEDIAN=\(String(format: "%.4f", measured.zMedian)) " +
                        "BAND=\(String(format: "%.4f", measured.band)) COUNT=\(measured.count) " +
                        "RAW_W=\(String(format: "%.4f", measured.rawWidth)) " +
                        "RAW_H=\(String(format: "%.4f", measured.rawHeight)) " +
                        "W=\(String(format: "%.4f", measured.legacyWidth)) " +
                        "H=\(String(format: "%.4f", measured.legacyHeight)) " +
                        "D=\(String(format: "%.4f", measured.legacyDepth))"
                    )
                    logDebug(
                        "[GREEN][ROOM_DIMS_APP][\(variant.label)] FILE=\(variant.url.lastPathComponent) " +
                        "SOURCE=\(sourceTag.uppercased()) APPROACH=\(approachTag.uppercased()) " +
                        "FLOOR_DIAG=\(String(format: "%.4f", measured.floorDiagonal)) " +
                        "TRIMMED_X=\(String(format: "%.4f", measured.trimmedXSpan)) " +
                        "TRIMMED_Y=\(String(format: "%.4f", measured.trimmedYSpan)) " +
                        "TRIMMED_Z=\(String(format: "%.4f", measured.trimmedZSpan)) " +
                        "BACK_WALL_Z=\(String(format: "%.4f", measured.zMode)) " +
                        "BACK_WALL_Z_MEAN=\(String(format: "%.4f", measured.zMean)) " +
                        "BAND=\(String(format: "%.4f", measured.band)) COUNT=\(measured.count) " +
                        "RAW_W=\(String(format: "%.4f", measured.rawWidth)) " +
                        "RAW_H=\(String(format: "%.4f", measured.rawHeight)) " +
                        "W=\(String(format: "%.4f", measured.width)) " +
                        "H=\(String(format: "%.4f", measured.height)) " +
                        "D=\(String(format: "%.4f", measured.depth)) " +
                        "SCENE_WHD=(\(String(format: "%.4f", measured.sceneWidth))," +
                        "\(String(format: "%.4f", measured.sceneHeight))," +
                        "\(String(format: "%.4f", measured.sceneDepth)))"
                    )
                } else if debugMode, let finalRoomWidth, let finalRoomHeight, let finalRoomDepth {
                    let sceneWidthString = finalSceneWidth.map { String(format: "%.4f", $0) } ?? "nil"
                    let sceneHeightString = finalSceneHeight.map { String(format: "%.4f", $0) } ?? "nil"
                    let sceneDepthString = finalSceneDepth.map { String(format: "%.4f", $0) } ?? "nil"
                    logDebug(
                        "[YELLOW][ROOM_DIMS_APP][\(variant.label)] FILE=\(variant.url.lastPathComponent) " +
                        "SOURCE=\(sourceTag.uppercased()) APPROACH=\(approachTag.uppercased()) " +
                        "W=\(String(format: "%.4f", finalRoomWidth)) " +
                        "H=\(String(format: "%.4f", finalRoomHeight)) " +
                        "D=\(String(format: "%.4f", finalRoomDepth)) " +
                        "SCENE_WHD=(\(sceneWidthString),\(sceneHeightString),\(sceneDepthString))"
                    )
                } else if debugMode {
                    logDebug(
                        "[RED][ROOM_DIMS_APP][\(variant.label)] FILE=\(variant.url.lastPathComponent) " +
                        "SOURCE=\(sourceTag.uppercased()) APPROACH=\(approachTag.uppercased()) unavailable"
                    )
                }

                var metadata: [String: String] = ["photoOrientation": photoOrientation.rawValue]

                if let width = finalRoomWidth {
                    metadata["roomWidth"] = String(format: "%.2f", width)
                }
                if let height = finalRoomHeight {
                    metadata["roomHeight"] = String(format: "%.2f", height)
                }
                if let depth = finalRoomDepth {
                    metadata["roomDepth"] = String(format: "%.2f", depth)
                }
                if let sw = finalSceneWidth {
                    metadata["roomSceneWidth"] = String(format: "%.4f", sw)
                }
                if let sh = finalSceneHeight {
                    metadata["roomSceneHeight"] = String(format: "%.4f", sh)
                }
                if let sd = finalSceneDepth {
                    metadata["roomSceneDepth"] = String(format: "%.4f", sd)
                }
                if let measured {
                    metadata["roomDimsApproach"] = "pythagoras_diagonal"
                    metadata["roomFloorDiagonal"] = String(format: "%.4f", measured.floorDiagonal)
                    metadata["roomTrimmedXSpan"] = String(format: "%.4f", measured.trimmedXSpan)
                    metadata["roomTrimmedYSpan"] = String(format: "%.4f", measured.trimmedYSpan)
                    metadata["roomTrimmedZSpan"] = String(format: "%.4f", measured.trimmedZSpan)
                    metadata["roomBackWallZMode"] = String(format: "%.4f", measured.zMode)
                    metadata["roomBackWallZMean"] = String(format: "%.4f", measured.zMean)
                    metadata["roomBackWallBand"] = String(format: "%.4f", measured.band)
                    metadata["roomBackWallCount"] = "\(measured.count)"
                    metadata["roomBackWallRawWidth"] = String(format: "%.4f", measured.rawWidth)
                    metadata["roomBackWallRawHeight"] = String(format: "%.4f", measured.rawHeight)
                } else if sourceTag == "fallback" {
                    metadata["roomDimsApproach"] = "preview_active_room_dimensions"
                }
                metadata["isClassicPly"] = variant.isClassic ? "true" : "false"

                let metadataURL = metadataURL(forSavedPlyURL: variant.url)
                let metadataData = try JSONEncoder().encode(metadata)
                try metadataData.write(to: metadataURL, options: [.atomic])

                if debugMode {
                    logDebug("✅ [USDZModelManager] Metadata saved to: \(metadataURL.path)")
                }
            }

            // Copy SHARP wall-measurement thumbnail next to saved PLY so list reload can infer capture orientation for AR/UI.
            copyThumbnailFromSHARPSessionIfPresent(sourceURL: sourceURL, savedStem: fileName)

            if debugMode {
                logDebug("✅ [USDZModelManager] PLY saved to: \(destinationURL.path)")
            }

            // Reload models to include the new one
            DispatchQueue.main.async {
                self.loadModels()
                completion(true, nil)
            }
        } catch {
            if debugMode {
                logDebug("❌ [USDZModelManager] Failed to save PLY: \(error)")
            }
            completion(false, error.localizedDescription)
        }
    }

    /// Load all metadata from PLY metadata file (orientation, dimensions, optional YOLO ratios)
    private func loadPLYMetadata(for fileName: String) -> SavedRoomDiskMetadata {
        let metadataURL = modelsDirectory.appendingPathComponent("\(fileName).ply.meta")
        let canonicalStem = canonicalPlyStem(for: fileName)
        let legacyClassicSidecarURL = modelsDirectory.appendingPathComponent("\(canonicalStem)_classic.ply")
        let exactIsClassicVariant = fileName.hasSuffix("_classic")

        let base: SavedRoomDiskMetadata
        if FileManager.default.fileExists(atPath: metadataURL.path),
           let data = try? Data(contentsOf: metadataURL),
           let dict = try? JSONDecoder().decode([String: String].self, from: data) {
            base = SavedRoomDiskMetadata.parse(dictionary: dict)
        } else {
            base = .empty
        }
        let normalizedBase = base.replacingClassicPly(
            exactIsClassicVariant ||
            base.isClassicPly ||
            (!fileName.hasSuffix("_classic") && !fileName.hasSuffix("_3dgs") && FileManager.default.fileExists(atPath: legacyClassicSidecarURL.path))
        )

        // Opening from list: AR roll uses `photoOrientation`. Prefer EXIF from saved thumbnail (same source as SHARP)
        // when present so orientation matches creation even if `.meta` defaulted to portrait or was stale.
        if let thumbURL = savedRoomThumbnailURL(stem: fileName) ?? (canonicalStem != fileName ? savedRoomThumbnailURL(stem: canonicalStem) : nil),
           let image = UIImage(contentsOfFile: thumbURL.path) {
            let fromThumb = PhotoOrientation.detectFromStoredRoomThumbnail(image)
            if fromThumb != normalizedBase.orientation {
                if AppStateManager.shared.qualitySettings.debugMode {
                    logDebug("📐 [USDZModelManager] PLY list orientation: meta=\(normalizedBase.orientation.rawValue) → thumbnail=\(fromThumb.rawValue) (\(thumbURL.lastPathComponent)); persisting to .meta")
                }
                try? mergePhotoOrientationIntoSavedRoomMetadata(
                    fileName: fileName,
                    modelFileExtension: "ply",
                    photoOrientation: fromThumb
                )
                DispatchQueue.main.async { [weak self] in
                    self?.refreshModels()
                }
            } else if AppStateManager.shared.qualitySettings.debugMode {
                logDebug("📐 [USDZModelManager] PLY list orientation: meta=\(normalizedBase.orientation.rawValue) matches thumbnail (\(thumbURL.lastPathComponent))")
            }
            return normalizedBase.replacingOrientation(fromThumb)
        }

        return normalizedBase
    }

    private func savedRoomThumbnailURL(stem: String) -> URL? {
        let jpg = modelsDirectory.appendingPathComponent("\(stem)_thumbnail.jpg")
        if FileManager.default.fileExists(atPath: jpg.path) { return jpg }
        let png = modelsDirectory.appendingPathComponent("\(stem)_thumbnail.png")
        if FileManager.default.fileExists(atPath: png.path) { return png }
        return nil
    }

    private func siblingPlyURLs(for sourceURL: URL) -> (original: URL?, classic: URL?, threeDGS: URL?) {
        let directory = sourceURL.deletingLastPathComponent()
        var stem = sourceURL.deletingPathExtension().lastPathComponent
        if stem.hasSuffix("_classic") {
            stem = String(stem.dropLast("_classic".count))
        } else if stem.hasSuffix("_3dgs") {
            stem = String(stem.dropLast("_3dgs".count))
        }
        let original = directory.appendingPathComponent("\(stem).ply")
        let classic = directory.appendingPathComponent("\(stem)_classic.ply")
        let threeDGS = directory.appendingPathComponent("\(stem)_3dgs.ply")
        let fm = FileManager.default
        return (
            original: fm.fileExists(atPath: original.path) ? original : nil,
            classic: fm.fileExists(atPath: classic.path) ? classic : nil,
            threeDGS: fm.fileExists(atPath: threeDGS.path) ? threeDGS : nil
        )
    }

    private func siblingPlyURLs(forBaseSavedRoomURL baseURL: URL) -> (original: URL, classic: URL, threeDGS: URL) {
        let directory = baseURL.deletingLastPathComponent()
        let stem = baseURL.deletingPathExtension().lastPathComponent
        return (
            original: baseURL,
            classic: directory.appendingPathComponent("\(stem)_classic.ply"),
            threeDGS: directory.appendingPathComponent("\(stem)_3dgs.ply")
        )
    }

    /// Copies `RoomStamp_thumbnail.jpg` (or `.png`) from the SHARP session folder into SavedRooms as `savedStem_thumbnail.*`.
    private func copyThumbnailFromSHARPSessionIfPresent(sourceURL: URL, savedStem: String) {
        let sourceDir = sourceURL.deletingLastPathComponent()
        var sessionStem = sourceURL.deletingPathExtension().lastPathComponent
        if sessionStem.hasSuffix("_classic") {
            sessionStem = String(sessionStem.dropLast("_classic".count))
        }
        let fm = FileManager.default
        for ext in ["jpg", "png"] {
            let src = sourceDir.appendingPathComponent("\(sessionStem)_thumbnail.\(ext)")
            guard fm.fileExists(atPath: src.path) else { continue }
            let dest = modelsDirectory.appendingPathComponent("\(savedStem)_thumbnail.\(ext)")
            try? fm.removeItem(at: dest)
            do {
                try fm.copyItem(at: src, to: dest)
                logDebug("💾 [USDZModelManager] Copied thumbnail \(src.lastPathComponent) → \(dest.lastPathComponent)")
            } catch {
                logDebug("❌ [USDZModelManager] Thumbnail copy failed: \(error.localizedDescription)")
            }
            return
        }
    }

    /// Save a mesh room (image + metadata) to SavedRooms directory
    func saveMeshRoom(image: UIImage, name: String, photoOrientation: PhotoOrientation, roomWidth: Float, roomHeight: Float, roomDepth: Float, completion: @escaping (Bool, String?) -> Void) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode

        if debugMode {
            logDebug("💾 [USDZModelManager] Starting to save mesh room: \(name)")
        }

        let fileName = sanitizeFileName(name)
        let imageURL = modelsDirectory.appendingPathComponent("\(fileName).meshroom")
        let metadataURL = modelsDirectory.appendingPathComponent("\(fileName).meshroom.meta")

        do {
            // Remove existing files if they exist
            if FileManager.default.fileExists(atPath: imageURL.path) {
                try FileManager.default.removeItem(at: imageURL)
            }
            if FileManager.default.fileExists(atPath: metadataURL.path) {
                try FileManager.default.removeItem(at: metadataURL)
            }

            // Save image as JPEG
            guard let imageData = image.jpegData(compressionQuality: 0.85) else {
                completion(false, "Failed to encode image")
                return
            }
            try imageData.write(to: imageURL)

            // Save metadata
            let metadata: [String: String] = [
                "photoOrientation": photoOrientation.rawValue,
                "roomWidth": String(format: "%.2f", roomWidth),
                "roomHeight": String(format: "%.2f", roomHeight),
                "roomDepth": String(format: "%.2f", roomDepth)
            ]
            let metadataData = try JSONEncoder().encode(metadata)
            try metadataData.write(to: metadataURL)

            if debugMode {
                logDebug("✅ [USDZModelManager] Mesh room saved to: \(imageURL.path)")
            }

            // Reload models
            DispatchQueue.main.async {
                self.loadModels()
                completion(true, nil)
            }
        } catch {
            if debugMode {
                logDebug("❌ [USDZModelManager] Failed to save mesh room: \(error)")
            }
            completion(false, error.localizedDescription)
        }
    }

    /// Load metadata from mesh room metadata file
    private func loadMeshRoomMetadata(for fileName: String) -> SavedRoomDiskMetadata {
        let metadataURL = modelsDirectory.appendingPathComponent("\(fileName).meshroom.meta")

        guard FileManager.default.fileExists(atPath: metadataURL.path),
              let data = try? Data(contentsOf: metadataURL),
              let metadata = try? JSONDecoder().decode([String: String].self, from: data) else {
            return .empty
        }

        return SavedRoomDiskMetadata.parse(dictionary: metadata)
    }

    /// Save a GLB (GLTF binary) room to SavedRooms directory
    func saveGLBRoom(glbData: Data, name: String, photoOrientation: PhotoOrientation, roomWidth: Float, roomHeight: Float, roomDepth: Float, completion: @escaping (Bool, String?) -> Void) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode

        if debugMode {
            logDebug("💾 [USDZModelManager] Starting to save GLB room: \(name) (\(glbData.count) bytes)")
        }

        let fileName = sanitizeFileName(name)
        let glbURL = modelsDirectory.appendingPathComponent("\(fileName).glb")
        let metadataURL = modelsDirectory.appendingPathComponent("\(fileName).glb.meta")

        do {
            // Remove existing files if they exist
            if FileManager.default.fileExists(atPath: glbURL.path) {
                try FileManager.default.removeItem(at: glbURL)
            }
            if FileManager.default.fileExists(atPath: metadataURL.path) {
                try FileManager.default.removeItem(at: metadataURL)
            }

            // Save GLB file
            try glbData.write(to: glbURL)

            // Save metadata
            let metadata: [String: String] = [
                "photoOrientation": photoOrientation.rawValue,
                "roomWidth": String(format: "%.2f", roomWidth),
                "roomHeight": String(format: "%.2f", roomHeight),
                "roomDepth": String(format: "%.2f", roomDepth)
            ]
            let metadataData = try JSONEncoder().encode(metadata)
            try metadataData.write(to: metadataURL)

            if debugMode {
                logDebug("✅ [USDZModelManager] GLB room saved to: \(glbURL.path)")
            }

            // Reload models
            DispatchQueue.main.async {
                self.loadModels()
                completion(true, nil)
            }
        } catch {
            if debugMode {
                logDebug("❌ [USDZModelManager] Failed to save GLB room: \(error)")
            }
            completion(false, error.localizedDescription)
        }
    }

    /// Load metadata from GLB metadata file
    private func loadGLBMetadata(for fileName: String) -> SavedRoomDiskMetadata {
        let metadataURL = modelsDirectory.appendingPathComponent("\(fileName).glb.meta")

        guard FileManager.default.fileExists(atPath: metadataURL.path),
              let data = try? Data(contentsOf: metadataURL),
              let metadata = try? JSONDecoder().decode([String: String].self, from: data) else {
            return .empty
        }

        return SavedRoomDiskMetadata.parse(dictionary: metadata)
    }

    /// Optional metadata for saved USDZ rooms (`*.usdz.meta`).
    private func loadUSDZMetadata(for fileName: String) -> SavedRoomDiskMetadata {
        let metadataURL = modelsDirectory.appendingPathComponent("\(fileName).usdz.meta")
        guard FileManager.default.fileExists(atPath: metadataURL.path),
              let data = try? Data(contentsOf: metadataURL),
              let metadata = try? JSONDecoder().decode([String: String].self, from: data) else {
            return .empty
        }
        return SavedRoomDiskMetadata.parse(dictionary: metadata)
    }
}
