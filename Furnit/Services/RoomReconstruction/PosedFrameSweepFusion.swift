import Foundation
import simd
import UIKit

struct PosedFrameSweepFusionResult: Sendable {
    let plyURL: URL
    let frameCount: Int
    let splatCount: Int
    let roomWidthMeters: Float
    let roomHeightMeters: Float
    let roomDepthMeters: Float
    let validationSummary: String?

    var summary: String {
        let dims = String(
            format: "%.2f x %.2f x %.2f m",
            roomWidthMeters,
            roomHeightMeters,
            roomDepthMeters
        )
        let validation = validationSummary.map { " validation={\($0)}" } ?? ""
        return "frames=\(frameCount) splats=\(splatCount) dims=\(dims)\(validation)"
    }
}

enum PosedFrameSweepFusion {
    static func fuse(
        sessionURL: URL,
        outputFileName: String = "lidar_room_fused.ply",
        stride: Int = 2,
        minimumConfidenceRawValue: UInt8 = 1,
        voxelSizeMeters: Float = 0.02,
        minimumDepthMeters: Float = 0.15,
        maximumDepthMeters: Float = 8.0,
        validationResult: PosedFrameSweepValidationResult? = nil
    ) throws -> PosedFrameSweepFusionResult {
        let manifest = try loadManifest(sessionURL: sessionURL)
        guard !manifest.frames.isEmpty else {
            throw PosedFrameSweepFusionError.noFrames
        }

        var voxels: [FusionVoxelKey: FusionVoxelAccumulator] = [:]
        voxels.reserveCapacity(manifest.frames.count * 20_000)
        let step = max(1, stride)
        let cellSize = max(voxelSizeMeters, 0.005)

        for frame in manifest.frames where frame.trackingState == "normal" {
            guard let depthRelativePath = frame.depthRelativePath,
                  let depthResolution = frame.depthResolution else {
                continue
            }

            let depth = try readFloat32Grid(
                url: sessionURL.appendingPathComponent(depthRelativePath),
                resolution: depthResolution
            )
            let confidence: [UInt8]? = try frame.confidenceRelativePath.map {
                try readUInt8Grid(url: sessionURL.appendingPathComponent($0), resolution: depthResolution)
            }
            let intrinsics = try FusionIntrinsics(record: frame, depthResolution: depthResolution)
            let transform = try matrix4x4(fromColumnMajor: frame.cameraTransformWorldFromCamera)
            let rgbSampler = SweepFusionRGBSampler(url: sessionURL.appendingPathComponent(frame.imageRelativePath))

            for y in Swift.stride(from: 0, to: depthResolution.height, by: step) {
                for x in Swift.stride(from: 0, to: depthResolution.width, by: step) {
                    let i = y * depthResolution.width + x
                    if let confidence, confidence[i] < minimumConfidenceRawValue {
                        continue
                    }
                    let depthMeters = depth[i]
                    guard depthMeters.isFinite,
                          depthMeters >= minimumDepthMeters,
                          depthMeters <= maximumDepthMeters else {
                        continue
                    }

                    let cameraX = (Float(x) - intrinsics.cxDepth) * depthMeters / intrinsics.fxDepth
                    let cameraY = -(Float(y) - intrinsics.cyDepth) * depthMeters / intrinsics.fyDepth
                    let cameraZ = -depthMeters
                    let world = transform * SIMD4<Float>(cameraX, cameraY, cameraZ, 1)
                    let position = SIMD3<Float>(world.x, world.y, world.z)
                    let color = rgbSampler?.color(
                        depthX: x,
                        depthY: y,
                        depthResolution: depthResolution
                    ) ?? FusionRGB(r: 180, g: 180, b: 180)

                    let key = FusionVoxelKey(position, cellSize: cellSize)
                    var accumulator = voxels[key] ?? FusionVoxelAccumulator()
                    accumulator.add(position: position, color: color, depthMeters: depthMeters)
                    voxels[key] = accumulator
                }
            }
        }

        var splats = voxels.values.map { $0.splat }
        splats.sort {
            if $0.position.z == $1.position.z {
                if $0.position.y == $1.position.y { return $0.position.x < $1.position.x }
                return $0.position.y < $1.position.y
            }
            return $0.position.z < $1.position.z
        }
        guard !splats.isEmpty else {
            throw PosedFrameSweepFusionError.noUsableDepthPoints
        }

        let bounds = FusionBounds(points: splats.map(\.position))
        let outputURL = sessionURL.appendingPathComponent(outputFileName)
        try writeGaussianPLY(splats: splats, voxelSizeMeters: cellSize, to: outputURL)

        return PosedFrameSweepFusionResult(
            plyURL: outputURL,
            frameCount: manifest.frames.count,
            splatCount: splats.count,
            roomWidthMeters: bounds.width,
            roomHeightMeters: bounds.height,
            roomDepthMeters: bounds.depth,
            validationSummary: validationResult?.summary
        )
    }

    private static func loadManifest(sessionURL: URL) throws -> PosedFrameSweepManifest {
        let url = sessionURL.appendingPathComponent("poses.json")
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(PosedFrameSweepManifest.self, from: data)
    }

    private static func readFloat32Grid(url: URL, resolution: PixelResolution) throws -> [Float] {
        let data = try Data(contentsOf: url)
        let expected = resolution.width * resolution.height * MemoryLayout<Float>.size
        guard data.count >= expected else {
            throw PosedFrameSweepFusionError.invalidGridSize(url.lastPathComponent)
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
            throw PosedFrameSweepFusionError.invalidGridSize(url.lastPathComponent)
        }
        return Array(data.prefix(expected))
    }

    private static func matrix4x4(fromColumnMajor values: [Float]) throws -> simd_float4x4 {
        guard values.count == 16 else { throw PosedFrameSweepFusionError.invalidMatrix }
        return simd_float4x4(
            SIMD4(values[0], values[1], values[2], values[3]),
            SIMD4(values[4], values[5], values[6], values[7]),
            SIMD4(values[8], values[9], values[10], values[11]),
            SIMD4(values[12], values[13], values[14], values[15])
        )
    }

    private static func writeGaussianPLY(splats: [FusionSplat], voxelSizeMeters: Float, to url: URL) throws {
        let header = """
        ply
        format binary_little_endian 1.0
        element vertex \(splats.count)
        property float x
        property float y
        property float z
        property float scale_0
        property float scale_1
        property float scale_2
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        property float opacity
        property uchar red
        property uchar green
        property uchar blue
        end_header

        """
        let vertexStride = 47
        var data = Data()
        data.reserveCapacity(header.utf8.count + splats.count * vertexStride)
        data.append(contentsOf: header.utf8)

        let baseScale = max(voxelSizeMeters * 0.9, 0.012)
        let opacity = logit(0.88)
        let rotW: Float = 1
        let rotX: Float = 0
        let rotY: Float = 0
        let rotZ: Float = 0

        for splat in splats {
            data.appendFloat32LE(splat.position.x)
            data.appendFloat32LE(splat.position.y)
            data.appendFloat32LE(splat.position.z)

            let depthScale = min(max(splat.meanDepthMeters * 0.006, baseScale), 0.04)
            let logScale = log(depthScale)
            data.appendFloat32LE(logScale)
            data.appendFloat32LE(logScale)
            data.appendFloat32LE(logScale)

            data.appendFloat32LE(rotW)
            data.appendFloat32LE(rotX)
            data.appendFloat32LE(rotY)
            data.appendFloat32LE(rotZ)
            data.appendFloat32LE(opacity)
            data.append(splat.color.r)
            data.append(splat.color.g)
            data.append(splat.color.b)
        }
        try data.write(to: url, options: [.atomic])
    }

    private static func logit(_ value: Float) -> Float {
        let clamped = min(max(value, 1e-4), 1 - 1e-4)
        return log(clamped / (1 - clamped))
    }
}

private struct FusionIntrinsics {
    let fxDepth: Float
    let fyDepth: Float
    let cxDepth: Float
    let cyDepth: Float

    init(record: PosedFrameRecord, depthResolution: PixelResolution) throws {
        guard record.intrinsicsRGB.count == 9 else {
            throw PosedFrameSweepFusionError.invalidMatrix
        }
        let rgbWidth = max(Float(record.rgbResolution.width), 1)
        let rgbHeight = max(Float(record.rgbResolution.height), 1)
        let scaleX = Float(depthResolution.width) / rgbWidth
        let scaleY = Float(depthResolution.height) / rgbHeight
        fxDepth = record.intrinsicsRGB[0] * scaleX
        fyDepth = record.intrinsicsRGB[4] * scaleY
        cxDepth = record.intrinsicsRGB[6] * scaleX
        cyDepth = record.intrinsicsRGB[7] * scaleY
    }
}

private struct FusionVoxelKey: Hashable {
    let x: Int
    let y: Int
    let z: Int

    init(_ position: SIMD3<Float>, cellSize: Float) {
        x = Int(floor(position.x / cellSize))
        y = Int(floor(position.y / cellSize))
        z = Int(floor(position.z / cellSize))
    }
}

private struct FusionVoxelAccumulator {
    private var positionSum = SIMD3<Float>.zero
    private var colorSum = SIMD3<Float>.zero
    private var depthSum: Float = 0
    private var count: Int = 0

    mutating func add(position: SIMD3<Float>, color: FusionRGB, depthMeters: Float) {
        positionSum += position
        colorSum += SIMD3<Float>(Float(color.r), Float(color.g), Float(color.b))
        depthSum += depthMeters
        count += 1
    }

    var splat: FusionSplat {
        let divisor = Float(max(count, 1))
        let meanColor = colorSum / divisor
        return FusionSplat(
            position: positionSum / divisor,
            color: FusionRGB(
                r: UInt8(min(max(Int(meanColor.x.rounded()), 0), 255)),
                g: UInt8(min(max(Int(meanColor.y.rounded()), 0), 255)),
                b: UInt8(min(max(Int(meanColor.z.rounded()), 0), 255))
            ),
            meanDepthMeters: depthSum / divisor
        )
    }
}

private struct FusionSplat {
    let position: SIMD3<Float>
    let color: FusionRGB
    let meanDepthMeters: Float
}

private struct FusionRGB {
    let r: UInt8
    let g: UInt8
    let b: UInt8
}

private struct FusionBounds {
    let min: SIMD3<Float>
    let max: SIMD3<Float>

    init(points: [SIMD3<Float>]) {
        var minPoint = points[0]
        var maxPoint = points[0]
        for point in points.dropFirst() {
            minPoint = simd_min(minPoint, point)
            maxPoint = simd_max(maxPoint, point)
        }
        min = minPoint
        max = maxPoint
    }

    var width: Float { max.x - min.x }
    var height: Float { max.y - min.y }
    var depth: Float { max.z - min.z }
}

private final class SweepFusionRGBSampler {
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

    func color(depthX: Int, depthY: Int, depthResolution: PixelResolution) -> FusionRGB {
        let u = Float(depthX) / Float(max(depthResolution.width - 1, 1))
        let v = Float(depthY) / Float(max(depthResolution.height - 1, 1))
        let x = min(width - 1, max(0, Int((u * Float(width - 1)).rounded())))
        let y = min(height - 1, max(0, Int((v * Float(height - 1)).rounded())))
        let offset = (y * width + x) * 4
        return FusionRGB(r: pixels[offset], g: pixels[offset + 1], b: pixels[offset + 2])
    }
}

enum PosedFrameSweepFusionError: LocalizedError, Equatable {
    case noFrames
    case noUsableDepthPoints
    case invalidGridSize(String)
    case invalidMatrix

    var errorDescription: String? {
        switch self {
        case .noFrames:
            return "Sweep has no posed frames."
        case .noUsableDepthPoints:
            return "Sweep fusion found no usable LiDAR depth points."
        case .invalidGridSize(let name):
            return "Grid file has an invalid byte size: \(name)."
        case .invalidMatrix:
            return "Pose or intrinsics matrix has invalid shape."
        }
    }
}

private extension Data {
    mutating func appendFloat32LE(_ value: Float) {
        var bits = value.bitPattern.littleEndian
        Swift.withUnsafeBytes(of: &bits) { append(contentsOf: $0) }
    }
}
