import Foundation
import simd
import UIKit

struct PosedFrameSweepValidationResult: Sendable {
    let singleFramePLYURL: URL
    let overlayPLYURL: URL?
    let firstFramePointCount: Int
    let secondFramePointCount: Int
    let matchedPointCount: Int
    let medianAlignmentOffsetMeters: Float?
    let p90AlignmentOffsetMeters: Float?
    let firstFrameDepthMedianMeters: Float?

    var summary: String {
        let medianCM = medianAlignmentOffsetMeters.map { String(format: "%.1fcm", $0 * 100) } ?? "n/a"
        let p90CM = p90AlignmentOffsetMeters.map { String(format: "%.1fcm", $0 * 100) } ?? "n/a"
        let depthM = firstFrameDepthMedianMeters.map { String(format: "%.2fm", $0) } ?? "n/a"
        return "points=(\(firstFramePointCount),\(secondFramePointCount)) matched=\(matchedPointCount) medianOffset=\(medianCM) p90=\(p90CM) firstDepthMedian=\(depthM)"
    }
}

/// Debug gate for posed-frame capture: proves depth units, intrinsics scaling, and ARKit pose convention.
enum PosedFrameSweepValidator {
    static func validate(
        sessionURL: URL,
        stride: Int = 4,
        minimumConfidenceRawValue: UInt8 = 1,
        maxAlignmentDistanceMeters: Float = 0.25
    ) throws -> PosedFrameSweepValidationResult {
        let manifest = try loadManifest(sessionURL: sessionURL)
        guard let first = manifest.frames.first else {
            throw PosedFrameSweepValidationError.noFrames
        }

        let firstPoints = try unprojectFrame(
            first,
            sessionURL: sessionURL,
            stride: stride,
            minimumConfidenceRawValue: minimumConfidenceRawValue,
            debugColorOverride: nil
        )
        let singlePLY = sessionURL.appendingPathComponent("debug_single_frame_world.ply")
        try writePLY(points: firstPoints, to: singlePLY)

        var overlayPLY: URL?
        var secondCount = 0
        var matchedCount = 0
        var medianOffset: Float?
        var p90Offset: Float?

        if manifest.frames.count >= 2 {
            let second = manifest.frames[1]
            let secondPoints = try unprojectFrame(
                second,
                sessionURL: sessionURL,
                stride: stride,
                minimumConfidenceRawValue: minimumConfidenceRawValue,
                debugColorOverride: DebugRGB(r: 0, g: 220, b: 255)
            )
            let firstOverlayPoints = firstPoints.map {
                DebugPoint(position: $0.position, color: DebugRGB(r: 255, g: 64, b: 64), depthMeters: $0.depthMeters)
            }
            overlayPLY = sessionURL.appendingPathComponent("debug_two_frame_overlay_world.ply")
            try writePLY(points: firstOverlayPoints + secondPoints, to: overlayPLY!)
            secondCount = secondPoints.count

            let stats = alignmentStats(
                source: firstPoints,
                target: secondPoints,
                maxDistanceMeters: maxAlignmentDistanceMeters
            )
            matchedCount = stats.count
            medianOffset = stats.median
            p90Offset = stats.p90
        }

        return PosedFrameSweepValidationResult(
            singleFramePLYURL: singlePLY,
            overlayPLYURL: overlayPLY,
            firstFramePointCount: firstPoints.count,
            secondFramePointCount: secondCount,
            matchedPointCount: matchedCount,
            medianAlignmentOffsetMeters: medianOffset,
            p90AlignmentOffsetMeters: p90Offset,
            firstFrameDepthMedianMeters: median(firstPoints.map(\.depthMeters))
        )
    }

    private static func loadManifest(sessionURL: URL) throws -> PosedFrameSweepManifest {
        let url = sessionURL.appendingPathComponent("poses.json")
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(PosedFrameSweepManifest.self, from: data)
    }

    private static func unprojectFrame(
        _ frame: PosedFrameRecord,
        sessionURL: URL,
        stride: Int,
        minimumConfidenceRawValue: UInt8,
        debugColorOverride: DebugRGB?
    ) throws -> [DebugPoint] {
        guard frame.trackingState == "normal" else {
            throw PosedFrameSweepValidationError.nonNormalTracking(frame.trackingState)
        }
        guard let depthRelativePath = frame.depthRelativePath,
              let depthResolution = frame.depthResolution else {
            throw PosedFrameSweepValidationError.missingDepth
        }

        let depthURL = sessionURL.appendingPathComponent(depthRelativePath)
        let depth = try readFloat32Grid(url: depthURL, resolution: depthResolution)
        let confidence: [UInt8]? = try frame.confidenceRelativePath.map {
            try readUInt8Grid(url: sessionURL.appendingPathComponent($0), resolution: depthResolution)
        }
        let rgbSampler = RGBSampler(url: sessionURL.appendingPathComponent(frame.imageRelativePath))

        let intrinsics = try Intrinsics(record: frame, depthResolution: depthResolution)
        let transform = try matrix4x4(fromColumnMajor: frame.cameraTransformWorldFromCamera)
        let step = max(1, stride)
        var points: [DebugPoint] = []
        points.reserveCapacity((depthResolution.width / step) * (depthResolution.height / step))

        for y in Swift.stride(from: 0, to: depthResolution.height, by: step) {
            for x in Swift.stride(from: 0, to: depthResolution.width, by: step) {
                let i = y * depthResolution.width + x
                if let confidence, confidence[i] < minimumConfidenceRawValue {
                    continue
                }
                let depthMeters = depth[i]
                guard depthMeters.isFinite, depthMeters > 0.1, depthMeters < 20 else { continue }

                let cameraX = (Float(x) - intrinsics.cxDepth) * depthMeters / intrinsics.fxDepth
                let cameraY = -(Float(y) - intrinsics.cyDepth) * depthMeters / intrinsics.fyDepth
                let cameraZ = -depthMeters
                let world = transform * SIMD4<Float>(cameraX, cameraY, cameraZ, 1)

                let color = debugColorOverride ?? rgbSampler?.color(
                    depthX: x,
                    depthY: y,
                    depthResolution: depthResolution
                ) ?? DebugRGB(r: 180, g: 180, b: 180)
                points.append(
                    DebugPoint(
                        position: SIMD3<Float>(world.x, world.y, world.z),
                        color: color,
                        depthMeters: depthMeters
                    )
                )
            }
        }
        return points
    }

    private static func alignmentStats(
        source: [DebugPoint],
        target: [DebugPoint],
        maxDistanceMeters: Float
    ) -> (count: Int, median: Float?, p90: Float?) {
        guard !source.isEmpty, !target.isEmpty else { return (0, nil, nil) }
        let cellSize = maxDistanceMeters
        var grid: [VoxelKey: [SIMD3<Float>]] = [:]
        grid.reserveCapacity(target.count)
        for point in target {
            grid[VoxelKey(point.position, cellSize: cellSize), default: []].append(point.position)
        }

        var distances: [Float] = []
        distances.reserveCapacity(source.count)
        for point in source {
            let key = VoxelKey(point.position, cellSize: cellSize)
            var best = Float.greatestFiniteMagnitude
            for dz in -1...1 {
                for dy in -1...1 {
                    for dx in -1...1 {
                        let neighbor = VoxelKey(x: key.x + dx, y: key.y + dy, z: key.z + dz)
                        guard let candidates = grid[neighbor] else { continue }
                        for candidate in candidates {
                            best = min(best, simd_distance(point.position, candidate))
                        }
                    }
                }
            }
            if best <= maxDistanceMeters {
                distances.append(best)
            }
        }

        distances.sort()
        return (
            distances.count,
            percentile(sorted: distances, fraction: 0.5),
            percentile(sorted: distances, fraction: 0.9)
        )
    }

    private static func readFloat32Grid(url: URL, resolution: PixelResolution) throws -> [Float] {
        let data = try Data(contentsOf: url)
        let expected = resolution.width * resolution.height * MemoryLayout<Float>.size
        guard data.count >= expected else {
            throw PosedFrameSweepValidationError.invalidGridSize(url.lastPathComponent)
        }
        return data.withUnsafeBytes { rawBuffer in
            let pointer = rawBuffer.bindMemory(to: Float.self)
            return Array(pointer.prefix(resolution.width * resolution.height))
        }
    }

    private static func readUInt8Grid(url: URL, resolution: PixelResolution) throws -> [UInt8] {
        let data = try Data(contentsOf: url)
        let expected = resolution.width * resolution.height
        guard data.count >= expected else {
            throw PosedFrameSweepValidationError.invalidGridSize(url.lastPathComponent)
        }
        return Array(data.prefix(expected))
    }

    private static func writePLY(points: [DebugPoint], to url: URL) throws {
        var output = Data()
        output.append(
            """
            ply
            format ascii 1.0
            element vertex \(points.count)
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header

            """.data(using: .utf8)!
        )
        for point in points {
            output.append(
                String(
                    format: "%.6f %.6f %.6f %d %d %d\n",
                    point.position.x,
                    point.position.y,
                    point.position.z,
                    point.color.r,
                    point.color.g,
                    point.color.b
                ).data(using: .utf8)!
            )
        }
        try output.write(to: url, options: [.atomic])
    }

    private static func matrix4x4(fromColumnMajor values: [Float]) throws -> simd_float4x4 {
        guard values.count == 16 else { throw PosedFrameSweepValidationError.invalidMatrix }
        return simd_float4x4(
            SIMD4(values[0], values[1], values[2], values[3]),
            SIMD4(values[4], values[5], values[6], values[7]),
            SIMD4(values[8], values[9], values[10], values[11]),
            SIMD4(values[12], values[13], values[14], values[15])
        )
    }

    private static func median(_ values: [Float]) -> Float? {
        percentile(sorted: values.sorted(), fraction: 0.5)
    }

    private static func percentile(sorted values: [Float], fraction: Float) -> Float? {
        guard !values.isEmpty else { return nil }
        let clampedFraction = min(Float(1), max(Float(0), fraction))
        let index = Int((Float(values.count - 1) * clampedFraction).rounded())
        return values[index]
    }
}

private struct Intrinsics {
    let fxDepth: Float
    let fyDepth: Float
    let cxDepth: Float
    let cyDepth: Float

    init(record: PosedFrameRecord, depthResolution: PixelResolution) throws {
        guard record.intrinsicsRGB.count == 9 else {
            throw PosedFrameSweepValidationError.invalidMatrix
        }
        let rgbWidth = max(Float(record.rgbResolution.width), 1)
        let rgbHeight = max(Float(record.rgbResolution.height), 1)
        let scaleX = Float(depthResolution.width) / rgbWidth
        let scaleY = Float(depthResolution.height) / rgbHeight

        // Stored column-major ARKit K: columns [(fx,0,0), (0,fy,0), (cx,cy,1)].
        fxDepth = record.intrinsicsRGB[0] * scaleX
        fyDepth = record.intrinsicsRGB[4] * scaleY
        cxDepth = record.intrinsicsRGB[6] * scaleX
        cyDepth = record.intrinsicsRGB[7] * scaleY
    }
}

private struct DebugPoint {
    let position: SIMD3<Float>
    let color: DebugRGB
    let depthMeters: Float
}

private struct DebugRGB {
    let r: UInt8
    let g: UInt8
    let b: UInt8
}

private struct VoxelKey: Hashable {
    let x: Int
    let y: Int
    let z: Int

    init(x: Int, y: Int, z: Int) {
        self.x = x
        self.y = y
        self.z = z
    }

    init(_ point: SIMD3<Float>, cellSize: Float) {
        self.x = Int(floor(point.x / cellSize))
        self.y = Int(floor(point.y / cellSize))
        self.z = Int(floor(point.z / cellSize))
    }
}

private final class RGBSampler {
    private let width: Int
    private let height: Int
    private let pixels: [UInt8]

    init?(url: URL) {
        guard let data = try? Data(contentsOf: url),
              let image = UIImage(data: data)?.cgImage else {
            return nil
        }
        width = image.width
        height = image.height
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        pixels = rgba
    }

    func color(depthX: Int, depthY: Int, depthResolution: PixelResolution) -> DebugRGB {
        let u = Float(depthX) / Float(max(depthResolution.width - 1, 1))
        let v = Float(depthY) / Float(max(depthResolution.height - 1, 1))
        let x = min(width - 1, max(0, Int((u * Float(width - 1)).rounded())))
        let y = min(height - 1, max(0, Int((v * Float(height - 1)).rounded())))
        let offset = (y * width + x) * 4
        return DebugRGB(r: pixels[offset], g: pixels[offset + 1], b: pixels[offset + 2])
    }
}

enum PosedFrameSweepValidationError: LocalizedError {
    case noFrames
    case missingDepth
    case nonNormalTracking(String)
    case invalidGridSize(String)
    case invalidMatrix

    var errorDescription: String? {
        switch self {
        case .noFrames:
            return "Sweep has no posed frames."
        case .missingDepth:
            return "Posed frame is missing LiDAR depth."
        case .nonNormalTracking(let state):
            return "Posed frame was captured with non-normal tracking: \(state)."
        case .invalidGridSize(let name):
            return "Grid file has an invalid byte size: \(name)."
        case .invalidMatrix:
            return "Pose or intrinsics matrix has invalid shape."
        }
    }
}
