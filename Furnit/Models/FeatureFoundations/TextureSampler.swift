import CoreGraphics
import Foundation
import simd

public protocol SplatColorReadable: AnyObject {
    var supportsColorReadback: Bool { get }
    func colorAt(screenPoint: CGPoint) -> SIMD3<Float>?
}

public extension SplatColorReadable {
    var supportsColorReadback: Bool { false }
}

public struct SurfaceSamplePoint: Sendable {
    public enum SurfaceType: String, Sendable {
        case floor
        case wall
        case ceiling
        case unknown
    }

    public let screenPoint: CGPoint
    public let worldPosition: SIMD3<Float>
    public let surface: SurfaceType

    public init(screenPoint: CGPoint, worldPosition: SIMD3<Float>, surface: SurfaceType) {
        self.screenPoint = screenPoint
        self.worldPosition = worldPosition
        self.surface = surface
    }
}

private func kMeansColors(_ colors: [SIMD3<Float>], k: Int, iterations: Int = 16) -> [SIMD3<Float>] {
    guard !colors.isEmpty, k > 0 else { return [] }
    if colors.count <= k { return colors }

    var centroids = Array(colors.shuffled().prefix(k))
    for _ in 0..<iterations {
        var clusters = Array(repeating: [SIMD3<Float>](), count: centroids.count)
        for color in colors {
            let nearest = centroids.enumerated().min { lhs, rhs in
                length_squared(lhs.element - color) < length_squared(rhs.element - color)
            }?.offset ?? 0
            clusters[nearest].append(color)
        }

        centroids = zip(centroids, clusters).map { centroid, cluster in
            guard !cluster.isEmpty else { return centroid }
            return cluster.reduce(.zero, +) / Float(cluster.count)
        }
    }

    return centroids
}

private func classifyMaterial(_ color: SIMD3<Float>) -> SurfacePalette.MaterialHint {
    let r = color.x
    let g = color.y
    let b = color.z
    let maxChannel = max(r, max(g, b))
    let minChannel = min(r, min(g, b))
    let saturation = maxChannel > 0 ? (maxChannel - minChannel) / maxChannel : 0
    let brightness = maxChannel

    if brightness > 0.86, saturation < 0.10 { return .plaster }
    if brightness < 0.18 { return .fabric }
    if r > g, g > b, saturation > 0.12 { return .wood }
    if brightness > 0.60, saturation < 0.15 { return .concrete }
    if brightness > 0.45, saturation > 0.05 { return .tile }
    return .unknown
}

public struct TextureSampler {
    private let depthQuery: SplatDepthQueryable
    private let colorReader: SplatColorReadable
    private let roomModel: RoomModel

    public init(
        depthQuery: SplatDepthQueryable,
        colorReader: SplatColorReadable,
        roomModel: RoomModel
    ) {
        self.depthQuery = depthQuery
        self.colorReader = colorReader
        self.roomModel = roomModel
    }

    public func samplePalette(gridResolution: Int = 24) -> SurfacePalette {
        let samples = collectSamples(gridResolution: max(8, gridResolution))
        return buildPalette(from: samples)
    }

    private func collectSamples(gridResolution: Int) -> [SurfaceSamplePoint] {
        let viewport = depthQuery.viewportSize
        guard viewport.width > 0, viewport.height > 0 else { return [] }

        let rows = gridResolution
        let cols = gridResolution
        var samples: [SurfaceSamplePoint] = []
        samples.reserveCapacity(rows * cols)

        for row in 0..<rows {
            for col in 0..<cols {
                let point = CGPoint(
                    x: (CGFloat(col) + 0.5) / CGFloat(cols) * viewport.width,
                    y: (CGFloat(row) + 0.5) / CGFloat(rows) * viewport.height
                )
                guard let depth = depthQuery.depthAt(screenPoint: point),
                      let world = depthQuery.unproject(screenPoint: point, depth: depth) else { continue }
                samples.append(SurfaceSamplePoint(screenPoint: point, worldPosition: world, surface: classifySurface(world)))
            }
        }

        logDebug("🎨 [TextureSampler] collectedSamples=\(samples.count) grid=\(rows)x\(cols)")
        return samples
    }

    private func classifySurface(_ point: SIMD3<Float>) -> SurfaceSamplePoint.SurfaceType {
        if let floor = roomModel.floorPlane, abs(floor.distance(to: point)) < 0.10 {
            return .floor
        }
        if let ceiling = roomModel.ceilingPlane, abs(ceiling.distance(to: point)) < 0.10 {
            return .ceiling
        }
        if roomModel.wallPlanes.contains(where: { abs($0.distance(to: point)) < 0.10 }) {
            return .wall
        }
        return .unknown
    }

    private func buildPalette(from samples: [SurfaceSamplePoint]) -> SurfacePalette {
        func colors(for surface: SurfaceSamplePoint.SurfaceType) -> SurfacePalette.SurfaceColors? {
            let screenPoints = samples.filter { $0.surface == surface }.map(\.screenPoint)
            let colors = screenPoints.compactMap { colorReader.colorAt(screenPoint: $0) }
            guard colors.count >= 3 else { return nil }
            let clustered = kMeansColors(colors, k: min(5, colors.count))
            return SurfacePalette.SurfaceColors(
                dominantColors: clustered,
                materialHints: clustered.map(classifyMaterial)
            )
        }

        return SurfacePalette(
            floor: colors(for: .floor),
            walls: colors(for: .wall),
            ceiling: colors(for: .ceiling)
        )
    }
}
