import CoreGraphics
import Foundation
import simd

/// Abstracts depth access from the active Gaussian splat renderer.
/// Callers should not touch Metal textures directly.
public protocol SplatDepthQueryable: AnyObject {
    /// Camera-relative depth in scene units at the provided screen point.
    func depthAt(screenPoint: CGPoint) -> Float?

    /// Reconstructs a scene-space point from a screen point and a camera-relative depth value.
    func unproject(screenPoint: CGPoint, depth: Float) -> SIMD3<Float>?

    /// Bulk depth sampling for geometry extraction. Conformers should override for efficiency.
    func sampleDepthGrid(rows: Int, cols: Int) -> [[Float?]]

    /// Bulk point-cloud reconstruction for geometry extraction. Conformers should override for efficiency.
    func buildPointCloud(rows: Int, cols: Int, maxDistance: Float) -> [SIMD3<Float>]

    /// Viewport size in points.
    var viewportSize: CGSize { get }
}

public extension SplatDepthQueryable {
    func sampleDepthGrid(rows: Int, cols: Int) -> [[Float?]] {
        guard rows > 0, cols > 0, viewportSize.width > 0, viewportSize.height > 0 else { return [] }

        let width = viewportSize.width
        let height = viewportSize.height
        var grid = [[Float?]](repeating: [Float?](repeating: nil, count: cols), count: rows)

        for row in 0..<rows {
            for col in 0..<cols {
                let x = (CGFloat(col) + 0.5) / CGFloat(cols) * width
                let y = (CGFloat(row) + 0.5) / CGFloat(rows) * height
                grid[row][col] = depthAt(screenPoint: CGPoint(x: x, y: y))
            }
        }
        return grid
    }

    func buildPointCloud(
        rows: Int = 64,
        cols: Int = 64,
        maxDistance: Float = 10.0
    ) -> [SIMD3<Float>] {
        guard rows > 0, cols > 0, viewportSize.width > 0, viewportSize.height > 0 else { return [] }

        let width = viewportSize.width
        let height = viewportSize.height
        let grid = sampleDepthGrid(rows: rows, cols: cols)
        var points: [SIMD3<Float>] = []
        points.reserveCapacity(rows * cols)

        for row in 0..<grid.count {
            for col in 0..<grid[row].count {
                guard let depth = grid[row][col], depth.isFinite, depth > 0, depth < maxDistance else { continue }
                let x = (CGFloat(col) + 0.5) / CGFloat(cols) * width
                let y = (CGFloat(row) + 0.5) / CGFloat(rows) * height
                if let point = unproject(screenPoint: CGPoint(x: x, y: y), depth: depth) {
                    points.append(point)
                }
            }
        }

        return points
    }
}
