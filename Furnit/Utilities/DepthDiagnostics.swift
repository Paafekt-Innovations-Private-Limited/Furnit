import CoreGraphics
import Foundation

enum DepthDiagnostics {
    static func logGridSummary(label: String, grid: [[Float?]]) {
        let rows = grid.count
        let cols = grid.first?.count ?? 0
        let valid = grid.flatMap { $0 }.compactMap { $0 }
        let minDepth = valid.min() ?? 0
        let maxDepth = valid.max() ?? 0
        logDebug(
            "🩺 [DepthDiagnostics] \(label) rows=\(rows) cols=\(cols) valid=\(valid.count) " +
            "depthRange=\(String(format: "%.3f", minDepth))...\(String(format: "%.3f", maxDepth))"
        )
    }

    static func logViewportSample(label: String, point: CGPoint, depth: Float?) {
        let depthText = depth.map { String(format: "%.4f", $0) } ?? "nil"
        logDebug(
            "🩺 [DepthDiagnostics] \(label) point=(\(String(format: "%.1f", point.x)),\(String(format: "%.1f", point.y))) depth=\(depthText)"
        )
    }
}
