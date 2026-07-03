import Foundation
import simd
import UIKit

struct PosedFrameSweepValidationResult: Sendable {
    let singleFramePLYURL: URL
    let overlayPLYURL: URL?
    let firstFrameIndex: Int
    let secondFrameIndex: Int?
    let firstFrameDiagnostics: PosedFrameSweepFrameDiagnostics
    let secondFrameDiagnostics: PosedFrameSweepFrameDiagnostics?
    let firstFramePointCount: Int
    let secondFramePointCount: Int
    let matchedPointCount: Int
    let medianAlignmentOffsetMeters: Float?
    let p90AlignmentOffsetMeters: Float?
    let medianPlaneResidualMeters: Float?
    let p90PlaneResidualMeters: Float?
    let planeNormalAngleDegrees: Float?
    let planeOffsetMeters: Float?
    let firstFrameDepthMedianMeters: Float?

    var summary: String {
        let medianCM = medianAlignmentOffsetMeters.map { String(format: "%.1fcm", $0 * 100) } ?? "n/a"
        let p90CM = p90AlignmentOffsetMeters.map { String(format: "%.1fcm", $0 * 100) } ?? "n/a"
        let planeMedianCM = medianPlaneResidualMeters.map { String(format: "%.1fcm", $0 * 100) } ?? "n/a"
        let planeP90CM = p90PlaneResidualMeters.map { String(format: "%.1fcm", $0 * 100) } ?? "n/a"
        let planeAngle = planeNormalAngleDegrees.map { String(format: "%.1fdeg", $0) } ?? "n/a"
        let planeOffsetCM = planeOffsetMeters.map { String(format: "%.1fcm", $0 * 100) } ?? "n/a"
        let depthM = firstFrameDepthMedianMeters.map { String(format: "%.2fm", $0) } ?? "n/a"
        let secondIndex = secondFrameIndex.map(String.init) ?? "n/a"
        return "frames=(\(firstFrameIndex),\(secondIndex)) points=(\(firstFramePointCount),\(secondFramePointCount)) matched=\(matchedPointCount) nnMedian=\(medianCM) nnP90=\(p90CM) planeMedian=\(planeMedianCM) planeP90=\(planeP90CM) planeAngle=\(planeAngle) planeOffset=\(planeOffsetCM) firstDepthMedian=\(depthM) first={\(firstFrameDiagnostics.summary)} second={\(secondFrameDiagnostics?.summary ?? "n/a")}"
    }
}

struct PosedFrameSweepFrameDiagnostics: Sendable {
    let frameID: String
    let trackingState: String
    let rgbResolution: PixelResolution
    let depthResolution: PixelResolution
    let fxDepth: Float
    let fyDepth: Float
    let cxDepth: Float
    let cyDepth: Float

    var summary: String {
        "id=\(frameID) tracking=\(trackingState) rgb=\(rgbResolution.width)x\(rgbResolution.height) depth=\(depthResolution.width)x\(depthResolution.height) Kd=(fx:\(String(format: "%.2f", fxDepth)), fy:\(String(format: "%.2f", fyDepth)), cx:\(String(format: "%.2f", cxDepth)), cy:\(String(format: "%.2f", cyDepth)))"
    }
}

/// Debug gate for posed-frame capture.
///
/// Contract:
/// - Nearest-neighbor offsets catch gross overlap failures.
/// - Plane residual, plane angle, and plane offset catch double-wall drift, normal-depth offsets,
///   and plane orientation mismatches.
/// - The overlay PLY is the visual truth gate. Pure in-plane tangential slide can look clean to
///   both nearest-neighbor and plane metrics, so the overlay must be inspected before trusting
///   the capture convention.
enum PosedFrameSweepValidator {
    static func validate(
        sessionURL: URL,
        firstFrameIndex: Int = 0,
        secondFrameIndex requestedSecondFrameIndex: Int? = nil,
        stride: Int = 4,
        minimumConfidenceRawValue: UInt8 = 1,
        maxAlignmentDistanceMeters: Float = 0.25
    ) throws -> PosedFrameSweepValidationResult {
        let manifest = try loadManifest(sessionURL: sessionURL)
        guard !manifest.frames.isEmpty else {
            throw PosedFrameSweepValidationError.noFrames
        }
        let first = try frame(at: firstFrameIndex, in: manifest.frames)
        let resolvedSecondFrameIndex = requestedSecondFrameIndex ?? defaultSecondFrameIndex(
            firstFrameIndex: firstFrameIndex,
            frameCount: manifest.frames.count
        )
        if resolvedSecondFrameIndex == firstFrameIndex {
            throw PosedFrameSweepValidationError.duplicateFrameIndex(firstFrameIndex)
        }

        let firstUnprojection = try unprojectFrame(
            first,
            sessionURL: sessionURL,
            stride: stride,
            minimumConfidenceRawValue: minimumConfidenceRawValue,
            debugColorOverride: nil
        )
        let firstPoints = firstUnprojection.points
        logDebug("[PosedFrameSweepValidator] first frame diagnostics: \(firstUnprojection.diagnostics.summary)")
        let singlePLY = sessionURL.appendingPathComponent("debug_single_frame_world.ply")
        try writePLY(points: firstPoints, to: singlePLY)

        var overlayPLY: URL?
        var secondCount = 0
        var matchedCount = 0
        var medianOffset: Float?
        var p90Offset: Float?
        var medianPlaneResidual: Float?
        var p90PlaneResidual: Float?
        var planeNormalAngle: Float?
        var planeOffset: Float?
        var secondDiagnostics: PosedFrameSweepFrameDiagnostics?

        if let resolvedSecondFrameIndex {
            let second = try frame(at: resolvedSecondFrameIndex, in: manifest.frames)
            let secondUnprojection = try unprojectFrame(
                second,
                sessionURL: sessionURL,
                stride: stride,
                minimumConfidenceRawValue: minimumConfidenceRawValue,
                debugColorOverride: DebugRGB(r: 0, g: 220, b: 255)
            )
            let secondPoints = secondUnprojection.points
            secondDiagnostics = secondUnprojection.diagnostics
            logDebug("[PosedFrameSweepValidator] second frame diagnostics: \(secondUnprojection.diagnostics.summary)")
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

            let planeStats = planeResidualStats(
                first: firstPoints,
                second: secondPoints
            )
            medianPlaneResidual = planeStats.medianResidual
            p90PlaneResidual = planeStats.p90Residual
            planeNormalAngle = planeStats.normalAngleDegrees
            planeOffset = planeStats.planeOffsetMeters
        }

        let result = PosedFrameSweepValidationResult(
            singleFramePLYURL: singlePLY,
            overlayPLYURL: overlayPLY,
            firstFrameIndex: firstFrameIndex,
            secondFrameIndex: resolvedSecondFrameIndex,
            firstFrameDiagnostics: firstUnprojection.diagnostics,
            secondFrameDiagnostics: secondDiagnostics,
            firstFramePointCount: firstPoints.count,
            secondFramePointCount: secondCount,
            matchedPointCount: matchedCount,
            medianAlignmentOffsetMeters: medianOffset,
            p90AlignmentOffsetMeters: p90Offset,
            medianPlaneResidualMeters: medianPlaneResidual,
            p90PlaneResidualMeters: p90PlaneResidual,
            planeNormalAngleDegrees: planeNormalAngle,
            planeOffsetMeters: planeOffset,
            firstFrameDepthMedianMeters: median(firstPoints.map(\.depthMeters))
        )
        logDebug("[PosedFrameSweepValidator] validation summary: \(result.summary)")
        return result
    }

    private static func loadManifest(sessionURL: URL) throws -> PosedFrameSweepManifest {
        let url = sessionURL.appendingPathComponent("poses.json")
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(PosedFrameSweepManifest.self, from: data)
    }

    private static func frame(at index: Int, in frames: [PosedFrameRecord]) throws -> PosedFrameRecord {
        guard frames.indices.contains(index) else {
            throw PosedFrameSweepValidationError.invalidFrameIndex(index: index, frameCount: frames.count)
        }
        return frames[index]
    }

    private static func defaultSecondFrameIndex(firstFrameIndex: Int, frameCount: Int) -> Int? {
        guard frameCount >= 2 else { return nil }
        if firstFrameIndex + 1 < frameCount {
            return firstFrameIndex + 1
        }
        if firstFrameIndex > 0 {
            return firstFrameIndex - 1
        }
        return nil
    }

    private static func unprojectFrame(
        _ frame: PosedFrameRecord,
        sessionURL: URL,
        stride: Int,
        minimumConfidenceRawValue: UInt8,
        debugColorOverride: DebugRGB?
    ) throws -> UnprojectedFrame {
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
        return UnprojectedFrame(
            points: points,
            diagnostics: PosedFrameSweepFrameDiagnostics(
                frameID: frame.id,
                trackingState: frame.trackingState,
                rgbResolution: frame.rgbResolution,
                depthResolution: depthResolution,
                fxDepth: intrinsics.fxDepth,
                fyDepth: intrinsics.fyDepth,
                cxDepth: intrinsics.cxDepth,
                cyDepth: intrinsics.cyDepth
            )
        )
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

    private static func planeResidualStats(
        first: [DebugPoint],
        second: [DebugPoint],
        inlierThresholdMeters: Float = 0.04
    ) -> PlaneResidualStats {
        guard let firstPlane = dominantPlane(points: first.map(\.position), inlierThresholdMeters: inlierThresholdMeters),
              let secondPlane = dominantPlane(points: second.map(\.position), inlierThresholdMeters: inlierThresholdMeters) else {
            return PlaneResidualStats.empty
        }

        var alignedSecondPlane = secondPlane
        if dot(firstPlane.normal, alignedSecondPlane.normal) < 0 {
            alignedSecondPlane = alignedSecondPlane.flipped()
        }

        let normalDot = min(Float(1), max(Float(-1), dot(firstPlane.normal, alignedSecondPlane.normal)))
        let normalAngleDegrees = acos(normalDot) * 180 / .pi
        let planeOffsetMeters = abs(dot(firstPlane.normal, alignedSecondPlane.point - firstPlane.point))

        var residuals: [Float] = []
        residuals.reserveCapacity(firstPlane.inlierCount + alignedSecondPlane.inlierCount)
        for point in first.map(\.position) where abs(firstPlane.signedDistance(to: point)) <= inlierThresholdMeters {
            residuals.append(abs(alignedSecondPlane.signedDistance(to: point)))
        }
        for point in second.map(\.position) where abs(alignedSecondPlane.signedDistance(to: point)) <= inlierThresholdMeters {
            residuals.append(abs(firstPlane.signedDistance(to: point)))
        }

        residuals.sort()
        return PlaneResidualStats(
            medianResidual: percentile(sorted: residuals, fraction: 0.5),
            p90Residual: percentile(sorted: residuals, fraction: 0.9),
            normalAngleDegrees: normalAngleDegrees,
            planeOffsetMeters: planeOffsetMeters
        )
    }

    private static func dominantPlane(
        points: [SIMD3<Float>],
        inlierThresholdMeters: Float
    ) -> DebugPlane? {
        let finitePoints = points.filter { point in
            point.x.isFinite && point.y.isFinite && point.z.isFinite
        }
        guard finitePoints.count >= 12 else { return nil }

        let sample = deterministicSample(finitePoints, maxCount: 6_000)
        guard sample.count >= 12 else { return nil }

        var rng = DeterministicRNG(seed: UInt64(sample.count) &* 1_103_515_245 &+ 12_345)
        let iterations = min(180, max(48, sample.count / 30))
        var bestPlane: DebugPlane?
        var bestInlierCount = 0
        var bestMeanResidual = Float.greatestFiniteMagnitude

        for _ in 0..<iterations {
            let i0 = rng.nextIndex(upperBound: sample.count)
            var i1 = rng.nextIndex(upperBound: sample.count)
            var i2 = rng.nextIndex(upperBound: sample.count)
            if i1 == i0 { i1 = (i1 + 1) % sample.count }
            if i2 == i0 || i2 == i1 { i2 = (i2 + 2) % sample.count }

            let a = sample[i0]
            let b = sample[i1]
            let c = sample[i2]
            let rawNormal = cross(b - a, c - a)
            let normalLength = simd_length(rawNormal)
            guard normalLength > 1e-4 else { continue }

            let normal = rawNormal / normalLength
            let plane = DebugPlane(normal: normal, point: a, inlierCount: 0)
            var inlierCount = 0
            var residualSum: Float = 0

            for point in sample {
                let residual = abs(plane.signedDistance(to: point))
                if residual <= inlierThresholdMeters {
                    inlierCount += 1
                    residualSum += residual
                }
            }

            guard inlierCount > 0 else { continue }
            let meanResidual = residualSum / Float(inlierCount)
            if inlierCount > bestInlierCount || (inlierCount == bestInlierCount && meanResidual < bestMeanResidual) {
                bestPlane = DebugPlane(normal: normal, point: a, inlierCount: inlierCount)
                bestInlierCount = inlierCount
                bestMeanResidual = meanResidual
            }
        }

        guard let bestPlane, bestInlierCount >= max(12, sample.count / 50) else {
            return nil
        }
        return bestPlane
    }

    private static func deterministicSample(_ points: [SIMD3<Float>], maxCount: Int) -> [SIMD3<Float>] {
        guard points.count > maxCount else { return points }
        let step = Float(points.count - 1) / Float(max(maxCount - 1, 1))
        var sample: [SIMD3<Float>] = []
        sample.reserveCapacity(maxCount)
        for index in 0..<maxCount {
            let sourceIndex = min(points.count - 1, Int((Float(index) * step).rounded()))
            sample.append(points[sourceIndex])
        }
        return sample
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

private struct UnprojectedFrame {
    let points: [DebugPoint]
    let diagnostics: PosedFrameSweepFrameDiagnostics
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

private struct DebugPlane {
    let normal: SIMD3<Float>
    let point: SIMD3<Float>
    let inlierCount: Int

    func signedDistance(to point: SIMD3<Float>) -> Float {
        dot(normal, point - self.point)
    }

    func flipped() -> DebugPlane {
        DebugPlane(normal: -normal, point: point, inlierCount: inlierCount)
    }
}

private struct PlaneResidualStats {
    let medianResidual: Float?
    let p90Residual: Float?
    let normalAngleDegrees: Float?
    let planeOffsetMeters: Float?

    static let empty = PlaneResidualStats(
        medianResidual: nil,
        p90Residual: nil,
        normalAngleDegrees: nil,
        planeOffsetMeters: nil
    )
}

private struct DeterministicRNG {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
    }

    mutating func nextIndex(upperBound: Int) -> Int {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int(state % UInt64(upperBound))
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

enum PosedFrameSweepValidationError: LocalizedError, Equatable {
    case noFrames
    case missingDepth
    case nonNormalTracking(String)
    case invalidGridSize(String)
    case invalidMatrix
    case invalidFrameIndex(index: Int, frameCount: Int)
    case duplicateFrameIndex(Int)

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
        case .invalidFrameIndex(let index, let frameCount):
            return "Frame index \(index) is outside the sweep frame range 0..<\(frameCount)."
        case .duplicateFrameIndex(let index):
            return "Validator needs two different frames; frame \(index) was selected twice."
        }
    }
}
