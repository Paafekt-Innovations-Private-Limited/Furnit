import Foundation
import SceneKit
import UIKit
import ModelIO

// GLB Model Loader utility for handling glTF binary format
// Provides utilities for loading, validating, and optimizing GLB models for SceneKit
class GLBModelLoader {
    
    // GLB file format constants
    private static let glbMagicNumber: UInt32 = 0x46546C67 // "glTF" in little-endian
    private static let glbVersion2: UInt32 = 2
    
    // GLB chunk types
    private enum ChunkType: UInt32 {
        case json = 0x4E4F534A  // "JSON" in little-endian
        case binary = 0x004E4942 // "BIN\0" in little-endian
    }
    
    // GLB loading errors
    enum GLBError: Error, LocalizedError {
        case invalidMagicNumber
        case unsupportedVersion
        case invalidFileStructure
        case missingJSONChunk
        case corruptedData
        case sceneMorKitLoadError(String)
        
        var errorDescription: String? {
            switch self {
            case .invalidMagicNumber:
                return "Invalid GLB file format"
            case .unsupportedVersion:
                return "Unsupported GLB version"
            case .invalidFileStructure:
                return "Invalid GLB file structure"
            case .missingJSONChunk:
                return "Missing JSON chunk in GLB file"
            case .corruptedData:
                return "Corrupted GLB data"
            case .sceneMorKitLoadError(let message):
                return "SceneKit loading error: \(message)"
            }
        }
    }
    
    // GLB file information
    struct GLBInfo {
        let version: UInt32
        let totalLength: UInt32
        let jsonChunkLength: UInt32
        let binaryChunkLength: UInt32?
        let hasValidStructure: Bool
    }
    
    // Validate GLB file format and structure
    static func validateGLB(data: Data) throws -> GLBInfo {
        guard data.count >= 12 else {
            throw GLBError.invalidFileStructure
        }
        
        let _ = data.withUnsafeBytes { $0.bindMemory(to: UInt8.self) }
        
        // Read GLB header (12 bytes) using proper memory access
        let magic = data.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: 0, as: UInt32.self)
        }
        let version = data.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: 4, as: UInt32.self)
        }
        let totalLength = data.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: 8, as: UInt32.self)
        }
        
        print("🔍 GLB Validation:")
        print("   Magic: 0x\(String(magic, radix: 16))")
        print("   Version: \(version)")
        print("   Total Length: \(totalLength)")
        print("   Actual Data Size: \(data.count)")
        
        // Validate magic number
        guard magic == glbMagicNumber else {
            throw GLBError.invalidMagicNumber
        }
        
        // Validate version (only support version 2)
        guard version == glbVersion2 else {
            throw GLBError.unsupportedVersion
        }
        
        // Validate total length
        guard totalLength <= data.count else {
            throw GLBError.invalidFileStructure
        }
        
        // Read first chunk header (JSON chunk)
        guard data.count >= 20 else {
            throw GLBError.invalidFileStructure
        }
        
        let jsonChunkLength = data.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: 12, as: UInt32.self)
        }
        let jsonChunkType = data.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: 16, as: UInt32.self)
        }
        
        print("   JSON Chunk Length: \(jsonChunkLength)")
        print("   JSON Chunk Type: 0x\(String(jsonChunkType, radix: 16))")
        
        // Validate JSON chunk type
        guard jsonChunkType == ChunkType.json.rawValue else {
            throw GLBError.missingJSONChunk
        }
        
        // Calculate binary chunk info if present
        let jsonChunkEndOffset = 20 + Int(jsonChunkLength)
        var binaryChunkLength: UInt32? = nil
        
        if jsonChunkEndOffset + 8 <= data.count {
            let binaryChunkLen = data.withUnsafeBytes { bytes in
                bytes.load(fromByteOffset: jsonChunkEndOffset, as: UInt32.self)
            }
            let binaryChunkType = data.withUnsafeBytes { bytes in
                bytes.load(fromByteOffset: jsonChunkEndOffset + 4, as: UInt32.self)
            }
            
            if binaryChunkType == ChunkType.binary.rawValue {
                binaryChunkLength = binaryChunkLen
                print("   Binary Chunk Length: \(binaryChunkLen)")
            }
        }
        
        let hasValidStructure = jsonChunkLength > 0 && 
                               Int(totalLength) <= data.count
        
        print("   Valid Structure: \(hasValidStructure)")
        
        return GLBInfo(
            version: version,
            totalLength: totalLength,
            jsonChunkLength: jsonChunkLength,
            binaryChunkLength: binaryChunkLength,
            hasValidStructure: hasValidStructure
        )
    }
    
    // Load GLB data as SceneKit scene using native SceneKit loading (recommended approach)
    static func loadSceneWithModelIO(from data: Data) throws -> SCNScene {
        // Validate GLB format first
        let glbInfo = try validateGLB(data: data)
        
        guard glbInfo.hasValidStructure else {
            throw GLBError.invalidFileStructure
        }
        
        print("📦 Loading GLB as SceneKit scene using native SceneKit...")
        
        // Create temporary file for SceneKit
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("temp_scenekit_\(UUID().uuidString).glb")
        
        do {
            // Write GLB data to temporary file
            try data.write(to: tempURL, options: [.atomicWrite])
            
            print("💾 GLB temporary file: \(tempURL)")
            
            // Default loading options optimized for AR
            let loadingOptions: [SCNSceneSource.LoadingOption: Any] = [
                .checkConsistency: true,
                .convertUnitsToMeters: true,
                .convertToYUp: true,
                .createNormalsIfAbsent: true,
                .flattenScene: false,
                .preserveOriginalTopology: true
            ]
            
            // Load scene from GLB file using native SceneKit
            let scene = try SCNScene(url: tempURL, options: loadingOptions)
            
            print("✅ GLB scene loaded successfully using native SceneKit")
            
            // Log scene information
            logSceneInfo(scene)
            
            // Clean up temporary file
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                try? FileManager.default.removeItem(at: tempURL)
            }
            
            return scene
            
        } catch {
            // Clean up temporary file on error
            try? FileManager.default.removeItem(at: tempURL)
            throw GLBError.sceneMorKitLoadError(error.localizedDescription)
        }
    }
    
    // Load GLB data as SceneKit scene (legacy method using SCNScene directly)
    static func loadScene(from data: Data, options: [SCNSceneSource.LoadingOption: Any] = [:]) throws -> SCNScene {
        // Validate GLB format first
        let glbInfo = try validateGLB(data: data)
        
        guard glbInfo.hasValidStructure else {
            throw GLBError.invalidFileStructure
        }
        
        print("📦 Loading GLB as SceneKit scene...")
        
        do {
            // Create temporary file for SceneKit loading
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("temp_model_\(UUID().uuidString).glb")
            
            // Write GLB data to temporary file
            try data.write(to: tempURL)
            
            // Default loading options optimized for AR
            var loadingOptions = options
            if loadingOptions.isEmpty {
                loadingOptions = [
                    .checkConsistency: true,
                    .convertUnitsToMeters: true,
                    .convertToYUp: true,
                    .createNormalsIfAbsent: true,
                    .flattenScene: false,
                    .preserveOriginalTopology: true
                ]
            }
            
            // Load scene from GLB file
            let scene = try SCNScene(url: tempURL, options: loadingOptions)
            
            print("✅ GLB scene loaded successfully")
            
            // Log scene information
            logSceneInfo(scene)
            
            // Clean up temporary file
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                try? FileManager.default.removeItem(at: tempURL)
            }
            
            return scene
            
        } catch {
            throw GLBError.sceneMorKitLoadError(error.localizedDescription)
        }
    }
    
    // Load GLB data as root node using ModelIO (recommended)
    static func loadRootNodeWithModelIO(from data: Data) throws -> SCNNode {
        let scene = try loadSceneWithModelIO(from: data)
        return scene.rootNode
    }
    
    // Load GLB data as root node (legacy method)
    static func loadRootNode(from data: Data, options: [SCNSceneSource.LoadingOption: Any] = [:]) throws -> SCNNode {
        let scene = try loadScene(from: data, options: options)
        return scene.rootNode
    }
    
    // Optimize GLB model for AR display
    static func optimizeForAR(_ node: SCNNode) -> SCNNode {
        print("⚡ Optimizing GLB model for AR display")
        
        let optimizedNode = node.clone()
        
        // Calculate and apply appropriate scaling
        optimizedNode.scale = calculateOptimalScale(for: optimizedNode)
        
        // Optimize materials for AR lighting
        optimizeMaterials(in: optimizedNode)
        
        // Optimize geometry if needed
        optimizeGeometry(in: optimizedNode)
        
        // Center the model
        centerModel(optimizedNode)
        
        print("✅ GLB model optimized for AR")
        
        return optimizedNode
    }
    
    // Calculate optimal scale for AR display
    private static func calculateOptimalScale(for node: SCNNode) -> SCNVector3 {
        let (minBound, maxBound) = node.boundingBox
        let size = SCNVector3(
            maxBound.x - minBound.x,
            maxBound.y - minBound.y,
            maxBound.z - minBound.z
        )
        
        // Target size for furniture objects (1-2 meters max dimension)
        let targetMaxSize: Float = 1.5
        let currentMaxSize = max(size.x, max(size.y, size.z))
        
        if currentMaxSize > 0 {
            let scaleFactor = targetMaxSize / currentMaxSize
            print("   Calculated scale factor: \(scaleFactor)")
            return SCNVector3(scaleFactor, scaleFactor, scaleFactor)
        }
        
        return SCNVector3(1, 1, 1)
    }
    
    // Optimize materials for AR environment
    private static func optimizeMaterials(in node: SCNNode) {
        var materialCount = 0
        
        node.enumerateChildNodes { childNode, _ in
            guard let geometry = childNode.geometry else { return }
            
            for material in geometry.materials {
                materialCount += 1
                
                // Set appropriate lighting model
                if material.lightingModel == .constant {
                    material.lightingModel = .blinn
                }
                
                // Ensure proper transparency handling
                if material.transparency < 1.0 {
                    material.blendMode = .alpha
                    material.writesToDepthBuffer = false
                }
                
                // Add ambient lighting if missing
                if material.ambient.contents == nil {
                    material.ambient.contents = UIColor(white: 0.1, alpha: 1.0)
                }
                
                // Enhance specular properties for better AR appearance
                if material.specular.contents == nil {
                    material.specular.contents = UIColor(white: 0.2, alpha: 1.0)
                }
                
                // Set reasonable shininess
                if material.shininess < 1.0 {
                    material.shininess = 10.0
                }
            }
        }
        
        print("   Optimized \(materialCount) materials")
    }
    
    // Optimize geometry for performance
    private static func optimizeGeometry(in node: SCNNode) {
        var geometryCount = 0
        
        node.enumerateChildNodes { childNode, _ in
            guard childNode.geometry != nil else { return }
            geometryCount += 1
            
            // Enable automatic normal generation if needed
            // Note: SceneKit automatically generates normals when missing during rendering
        }
        
        print("   Processed \(geometryCount) geometry objects")
    }
    
    // Center the model at origin
    private static func centerModel(_ node: SCNNode) {
        let (minBound, maxBound) = node.boundingBox
        let center = SCNVector3(
            (minBound.x + maxBound.x) / 2,
            (minBound.y + maxBound.y) / 2,
            (minBound.z + maxBound.z) / 2
        )
        
        // Apply centering offset
        let currentScale = node.scale
        node.position = SCNVector3(
            -center.x * currentScale.x,
            -center.y * currentScale.y,
            -center.z * currentScale.z
        )
        
        print("   Model centered at origin")
    }
    
    // Log scene information for debugging
    private static func logSceneInfo(_ scene: SCNScene) {
        let rootNode = scene.rootNode
        let (minBound, maxBound) = rootNode.boundingBox
        
        print("📊 Scene Information:")
        print("   Root node children: \(rootNode.childNodes.count)")
        print("   Bounding box: min(\(minBound)), max(\(maxBound))")
        
        // Count total geometry and materials
        var totalGeometry = 0
        var totalMaterials = 0
        
        rootNode.enumerateChildNodes { node, _ in
            if let geometry = node.geometry {
                totalGeometry += 1
                totalMaterials += geometry.materials.count
            }
        }
        
        print("   Total geometry objects: \(totalGeometry)")
        print("   Total materials: \(totalMaterials)")
        
        // Check for animations
        if !scene.rootNode.animationKeys.isEmpty {
            print("   Animations: \(scene.rootNode.animationKeys.count)")
        }
    }
    
    // Validate if data appears to be GLB format
    static func isGLBData(_ data: Data) -> Bool {
        guard data.count >= 4 else { return false }
        
        let magic = data.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: 0, as: UInt32.self)
        }
        
        return magic == glbMagicNumber
    }
    
    // Get GLB file information without full validation
    static func getGLBInfo(from data: Data) -> GLBInfo? {
        do {
            return try validateGLB(data: data)
        } catch {
            print("⚠️ Failed to get GLB info: \(error)")
            return nil
        }
    }
}