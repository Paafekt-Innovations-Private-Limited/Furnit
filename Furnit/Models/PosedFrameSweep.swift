import ARKit
import CoreVideo
import Foundation
import simd
import UIKit

/// A persisted ARKit keyframe with RGB, optional metric depth, confidence, and camera pose.
struct PosedFrameRecord: Codable, Identifiable, Sendable {
    let id: String
    let imageRelativePath: String
    let depthRelativePath: String?
    let confidenceRelativePath: String?
    /// ARKit camera transform: world-from-camera, right-handed, camera looks down -Z.
    let cameraTransformWorldFromCamera: [Float]
    /// Intrinsics for `rgbResolution`; scale these before unprojecting depth-grid pixels.
    let intrinsicsRGB: [Float]
    let rgbResolution: PixelResolution
    let depthResolution: PixelResolution?
    let trackingState: String
    let timestamp: TimeInterval
}

struct PosedFrameSweepManifest: Codable, Sendable {
    let version: Int
    let createdAt: Date
    let sessionID: String
    let coordinateConvention: String
    let frames: [PosedFrameRecord]
}

struct PixelResolution: Codable, Sendable {
    let width: Int
    let height: Int
}

/// Owns the on-disk folder layout for a posed-frame room sweep.
final class PosedFrameSweepStore {
    let sessionID: String
    let rootURL: URL
    private let framesURL: URL
    private let depthURL: URL
    private let confidenceURL: URL
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private var records: [PosedFrameRecord] = []

    init(sessionID: String = UUID().uuidString) throws {
        self.sessionID = sessionID
        let baseURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("PosedFrameSweeps", isDirectory: true)
            .appendingPathComponent(sessionID, isDirectory: true)
        self.rootURL = baseURL
        self.framesURL = baseURL.appendingPathComponent("frames", isDirectory: true)
        self.depthURL = baseURL.appendingPathComponent("depth", isDirectory: true)
        self.confidenceURL = baseURL.appendingPathComponent("confidence", isDirectory: true)

        try FileManager.default.createDirectory(at: framesURL, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: depthURL, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: confidenceURL, withIntermediateDirectories: true)
    }

    var frameCount: Int { records.count }

    func append(frame: ARFrame) throws -> PosedFrameRecord {
        let index = records.count
        let stem = String(format: "frame_%04d", index)
        let imageURL = framesURL.appendingPathComponent("\(stem).jpg")
        try writeJPEG(from: frame.capturedImage, to: imageURL)

        let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth
        let writtenDepthURL: URL?
        let writtenConfidenceURL: URL?
        let depthResolution: PixelResolution?
        if let depthData {
            let depthMap = depthData.depthMap
            let depthFileURL = depthURL.appendingPathComponent("\(stem).bin")
            try writeFloat32DepthMap(depthMap, to: depthFileURL)
            writtenDepthURL = depthFileURL
            depthResolution = PixelResolution(
                width: CVPixelBufferGetWidth(depthMap),
                height: CVPixelBufferGetHeight(depthMap)
            )

            if let confidenceMap = depthData.confidenceMap {
                let confidenceFileURL = confidenceURL.appendingPathComponent("\(stem).bin")
                try writeUInt8Map(confidenceMap, to: confidenceFileURL)
                writtenConfidenceURL = confidenceFileURL
            } else {
                writtenConfidenceURL = nil
            }
        } else {
            writtenDepthURL = nil
            writtenConfidenceURL = nil
            depthResolution = nil
        }

        let camera = frame.camera
        let record = PosedFrameRecord(
            id: stem,
            imageRelativePath: relativePath(for: imageURL),
            depthRelativePath: writtenDepthURL.map(relativePath(for:)),
            confidenceRelativePath: writtenConfidenceURL.map(relativePath(for:)),
            cameraTransformWorldFromCamera: Self.flat(camera.transform),
            intrinsicsRGB: Self.flat(camera.intrinsics),
            rgbResolution: PixelResolution(
                width: Int(camera.imageResolution.width.rounded()),
                height: Int(camera.imageResolution.height.rounded())
            ),
            depthResolution: depthResolution,
            trackingState: Self.trackingStateDescription(camera.trackingState),
            timestamp: frame.timestamp
        )
        records.append(record)
        try writeManifest()
        return record
    }

    @discardableResult
    func writeManifest() throws -> URL {
        let manifest = PosedFrameSweepManifest(
            version: 1,
            createdAt: Date(),
            sessionID: sessionID,
            coordinateConvention: "ARKit world-from-camera transform; camera looks down -Z; intrinsicsRGB are for rgbResolution and must be scaled for depthResolution.",
            frames: records
        )
        let data = try JSONEncoder.posedFramePretty.encode(manifest)
        let url = rootURL.appendingPathComponent("poses.json")
        try data.write(to: url, options: [.atomic])
        return url
    }

    private func writeJPEG(from pixelBuffer: CVPixelBuffer, to url: URL) throws {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            throw PosedFrameSweepError.imageConversionFailed
        }
        // Keep the JPEG in ARFrame/camera image space. UI rotation would make the
        // saved RGB no longer match `camera.imageResolution` and `intrinsicsRGB`.
        let image = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
        guard let jpeg = image.jpegData(compressionQuality: 0.9) else {
            throw PosedFrameSweepError.imageEncodingFailed
        }
        try jpeg.write(to: url, options: [.atomic])
    }

    private func writeFloat32DepthMap(_ depthMap: CVPixelBuffer, to url: URL) throws {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else {
            throw PosedFrameSweepError.missingPixelBufferBaseAddress
        }
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)
        var data = Data(capacity: width * height * MemoryLayout<Float>.size)
        for y in 0..<height {
            let row = baseAddress.advanced(by: y * rowBytes)
            data.append(row.assumingMemoryBound(to: UInt8.self), count: width * MemoryLayout<Float>.size)
        }
        try data.write(to: url, options: [.atomic])
    }

    private func writeUInt8Map(_ confidenceMap: CVPixelBuffer, to url: URL) throws {
        CVPixelBufferLockBaseAddress(confidenceMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(confidenceMap) else {
            throw PosedFrameSweepError.missingPixelBufferBaseAddress
        }
        let width = CVPixelBufferGetWidth(confidenceMap)
        let height = CVPixelBufferGetHeight(confidenceMap)
        let rowBytes = CVPixelBufferGetBytesPerRow(confidenceMap)
        var data = Data(capacity: width * height)
        for y in 0..<height {
            data.append(baseAddress.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self), count: width)
        }
        try data.write(to: url, options: [.atomic])
    }

    private func relativePath(for url: URL) -> String {
        let rootPath = rootURL.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(rootPath) else { return url.lastPathComponent }
        return String(path.dropFirst(rootPath.count + 1))
    }

    private static func flat(_ matrix: simd_float4x4) -> [Float] {
        [
            matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z, matrix.columns.0.w,
            matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z, matrix.columns.1.w,
            matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z, matrix.columns.2.w,
            matrix.columns.3.x, matrix.columns.3.y, matrix.columns.3.z, matrix.columns.3.w,
        ]
    }

    private static func flat(_ matrix: simd_float3x3) -> [Float] {
        [
            matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z,
            matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z,
            matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z,
        ]
    }

    private static func trackingStateDescription(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal:
            return "normal"
        case .notAvailable:
            return "notAvailable"
        case .limited(let reason):
            return "limited(\(reason))"
        }
    }
}

enum PosedFrameSweepError: LocalizedError {
    case imageConversionFailed
    case imageEncodingFailed
    case missingPixelBufferBaseAddress

    var errorDescription: String? {
        switch self {
        case .imageConversionFailed:
            return "Could not convert ARFrame image to RGB."
        case .imageEncodingFailed:
            return "Could not encode ARFrame image as JPEG."
        case .missingPixelBufferBaseAddress:
            return "Pixel buffer has no readable base address."
        }
    }
}

private extension JSONEncoder {
    static var posedFramePretty: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}
