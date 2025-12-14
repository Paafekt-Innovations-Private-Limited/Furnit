// SmartyPantsGeometry.swift
// Alpha Shape (Concave Hull) using Delaunay Triangulation

import Foundation

// MARK: - Point2D
struct Point2D: Hashable {
    let x: Float
    let y: Float

    static func == (lhs: Point2D, rhs: Point2D) -> Bool {
        return abs(lhs.x - rhs.x) < 1e-6 && abs(lhs.y - rhs.y) < 1e-6
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(Int(x * 1000))
        hasher.combine(Int(y * 1000))
    }
}

// MARK: - Edge2D
struct Edge2D: Hashable {
    let p1: Point2D
    let p2: Point2D

    // Normalized edge (order-independent for hashing)
    var normalized: Edge2D {
        if p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y) {
            return Edge2D(p1: p1, p2: p2)
        } else {
            return Edge2D(p1: p2, p2: p1)
        }
    }

    static func == (lhs: Edge2D, rhs: Edge2D) -> Bool {
        let l = lhs.normalized
        let r = rhs.normalized
        return l.p1 == r.p1 && l.p2 == r.p2
    }

    func hash(into hasher: inout Hasher) {
        let n = normalized
        hasher.combine(n.p1)
        hasher.combine(n.p2)
    }
}

// MARK: - Triangle2D
struct Triangle2D {
    let p1: Point2D
    let p2: Point2D
    let p3: Point2D

    var edges: [Edge2D] {
        return [
            Edge2D(p1: p1, p2: p2),
            Edge2D(p1: p2, p2: p3),
            Edge2D(p1: p3, p2: p1)
        ]
    }

    // Circumcenter and circumradius
    func circumcircle() -> (center: Point2D, radius: Float) {
        let ax = p1.x, ay = p1.y
        let bx = p2.x, by = p2.y
        let cx = p3.x, cy = p3.y

        let d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        guard abs(d) > 1e-10 else {
            // Degenerate triangle
            return (Point2D(x: ax, y: ay), Float.infinity)
        }

        let ax2 = ax * ax + ay * ay
        let bx2 = bx * bx + by * by
        let cx2 = cx * cx + cy * cy

        let ux = (ax2 * (by - cy) + bx2 * (cy - ay) + cx2 * (ay - by)) / d
        let uy = (ax2 * (cx - bx) + bx2 * (ax - cx) + cx2 * (bx - ax)) / d

        let center = Point2D(x: ux, y: uy)
        let radius = sqrtf((ax - ux) * (ax - ux) + (ay - uy) * (ay - uy))

        return (center, radius)
    }

    // Check if point is inside circumcircle
    func circumcircleContains(_ p: Point2D) -> Bool {
        let (center, radius) = circumcircle()
        let dx = p.x - center.x
        let dy = p.y - center.y
        return sqrtf(dx * dx + dy * dy) <= radius
    }
}

// MARK: - Alpha Shape
final class AlphaShape {

    /// Maximum points for Delaunay (O(n²) complexity)
    private static let maxPointsForDelaunay = 300

    /// Compute alpha shape from points using Delaunay triangulation
    /// - Parameters:
    ///   - points: Array of (x, y) tuples
    ///   - alpha: Alpha parameter (higher = more concave, 0 = convex hull)
    /// - Returns: Array of boundary edges forming the concave hull
    static func compute(points: [(x: Int, y: Int)], alpha: Float = 0.5) -> [(start: (x: Int, y: Int), end: (x: Int, y: Int))] {
        guard points.count >= 3 else {
            return []
        }

        // Downsample points if too many (Delaunay is O(n²))
        let sampledPoints: [(x: Int, y: Int)]
        if points.count > maxPointsForDelaunay {
            sampledPoints = downsamplePoints(points, targetCount: maxPointsForDelaunay)
        } else {
            sampledPoints = points
        }

        // Convert to Point2D
        let pts = sampledPoints.map { Point2D(x: Float($0.x), y: Float($0.y)) }

        // Remove duplicates
        let uniquePts = Array(Set(pts))
        guard uniquePts.count >= 3 else {
            return []
        }

        // Step 1: Delaunay triangulation using Bowyer-Watson
        let triangles = delaunayTriangulation(points: uniquePts)

        guard !triangles.isEmpty else {
            return []
        }

        // Step 2: Calculate circumradius for each triangle and filter by alpha
        let alphaThreshold = 1.0 / max(alpha, 1e-6)
        var keptTriangles: [Triangle2D] = []

        for tri in triangles {
            let (_, radius) = tri.circumcircle()
            if radius < alphaThreshold {
                keptTriangles.append(tri)
            }
        }

        guard !keptTriangles.isEmpty else {
            // Fall back to convex hull if no triangles kept
            return convexHullEdges(points: points)
        }

        // Step 3: Find boundary edges (edges that appear in exactly one triangle)
        var edgeCount: [Edge2D: Int] = [:]
        for tri in keptTriangles {
            for edge in tri.edges {
                let normalized = edge.normalized
                edgeCount[normalized, default: 0] += 1
            }
        }

        // Boundary edges appear exactly once
        let boundaryEdges = edgeCount.filter { $0.value == 1 }.map { $0.key }

        guard !boundaryEdges.isEmpty else {
            return []
        }

        // Step 4: Order edges to form polygon(s)
        let orderedEdges = orderEdges(boundaryEdges)

        // Convert back to integer tuples
        return orderedEdges.map { edge in
            (start: (x: Int(edge.p1.x), y: Int(edge.p1.y)),
             end: (x: Int(edge.p2.x), y: Int(edge.p2.y)))
        }
    }

    /// Bowyer-Watson algorithm for Delaunay triangulation
    private static func delaunayTriangulation(points: [Point2D]) -> [Triangle2D] {
        guard points.count >= 3 else { return [] }

        // Find bounding box
        var minX = Float.infinity, maxX = -Float.infinity
        var minY = Float.infinity, maxY = -Float.infinity
        for p in points {
            minX = min(minX, p.x)
            maxX = max(maxX, p.x)
            minY = min(minY, p.y)
            maxY = max(maxY, p.y)
        }

        let dx = maxX - minX
        let dy = maxY - minY
        let deltaMax = max(dx, dy) * 2

        // Create super-triangle that contains all points
        let superP1 = Point2D(x: minX - deltaMax, y: minY - deltaMax)
        let superP2 = Point2D(x: minX + deltaMax * 2, y: minY - deltaMax)
        let superP3 = Point2D(x: minX + dx / 2, y: maxY + deltaMax)
        let superTriangle = Triangle2D(p1: superP1, p2: superP2, p3: superP3)

        var triangles: [Triangle2D] = [superTriangle]

        // Add points one at a time
        for point in points {
            var badTriangles: [Triangle2D] = []

            // Find all triangles whose circumcircle contains the point
            for tri in triangles {
                if tri.circumcircleContains(point) {
                    badTriangles.append(tri)
                }
            }

            // Find boundary of polygonal hole
            var polygon: [Edge2D] = []
            for tri in badTriangles {
                for edge in tri.edges {
                    // Edge is on boundary if it's not shared by another bad triangle
                    let isShared = badTriangles.contains { other in
                        if other.p1 == tri.p1 && other.p2 == tri.p2 && other.p3 == tri.p3 {
                            return false // Same triangle
                        }
                        return other.edges.contains { $0 == edge }
                    }
                    if !isShared {
                        polygon.append(edge)
                    }
                }
            }

            // Remove bad triangles
            triangles.removeAll { tri in
                badTriangles.contains { bad in
                    bad.p1 == tri.p1 && bad.p2 == tri.p2 && bad.p3 == tri.p3
                }
            }

            // Re-triangulate the hole
            for edge in polygon {
                let newTri = Triangle2D(p1: edge.p1, p2: edge.p2, p3: point)
                triangles.append(newTri)
            }
        }

        // Remove triangles that share vertices with super-triangle
        triangles.removeAll { tri in
            let superVertices = [superP1, superP2, superP3]
            return superVertices.contains { sv in
                tri.p1 == sv || tri.p2 == sv || tri.p3 == sv
            }
        }

        return triangles
    }

    /// Order edges to form a continuous polygon
    private static func orderEdges(_ edges: [Edge2D]) -> [Edge2D] {
        guard !edges.isEmpty else { return [] }

        var remaining = edges
        var ordered: [Edge2D] = []

        // Start with first edge
        var current = remaining.removeFirst()
        ordered.append(current)
        var currentEnd = current.p2

        while !remaining.isEmpty {
            // Find edge that connects to current end
            var found = false
            for (i, edge) in remaining.enumerated() {
                if edge.p1 == currentEnd {
                    ordered.append(edge)
                    currentEnd = edge.p2
                    remaining.remove(at: i)
                    found = true
                    break
                } else if edge.p2 == currentEnd {
                    // Reverse the edge
                    let reversed = Edge2D(p1: edge.p2, p2: edge.p1)
                    ordered.append(reversed)
                    currentEnd = reversed.p2
                    remaining.remove(at: i)
                    found = true
                    break
                }
            }

            if !found {
                // Disconnected polygon - start new chain
                if !remaining.isEmpty {
                    current = remaining.removeFirst()
                    ordered.append(current)
                    currentEnd = current.p2
                }
            }
        }

        return ordered
    }

    /// Compute convex hull edges as fallback
    private static func convexHullEdges(points: [(x: Int, y: Int)]) -> [(start: (x: Int, y: Int), end: (x: Int, y: Int))] {
        guard points.count >= 3 else { return [] }

        // Graham scan for convex hull
        let sorted = points.sorted { a, b in
            if a.y != b.y { return a.y < b.y }
            return a.x < b.x
        }

        let start = sorted[0]
        let rest = sorted.dropFirst().sorted { a, b in
            let angle1 = atan2(Float(a.y - start.y), Float(a.x - start.x))
            let angle2 = atan2(Float(b.y - start.y), Float(b.x - start.x))
            return angle1 < angle2
        }

        var hull = [start]
        for p in rest {
            while hull.count > 1 && cross(hull[hull.count - 2], hull[hull.count - 1], p) <= 0 {
                hull.removeLast()
            }
            hull.append(p)
        }

        // Create edges from hull
        var edges: [(start: (x: Int, y: Int), end: (x: Int, y: Int))] = []
        for i in 0..<hull.count {
            let next = (i + 1) % hull.count
            edges.append((start: hull[i], end: hull[next]))
        }

        return edges
    }

    private static func cross(_ o: (x: Int, y: Int), _ a: (x: Int, y: Int), _ b: (x: Int, y: Int)) -> Int {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }

    /// Grid-based downsampling to reduce point count
    /// Keeps one representative point per grid cell
    private static func downsamplePoints(_ points: [(x: Int, y: Int)], targetCount: Int) -> [(x: Int, y: Int)] {
        guard points.count > targetCount else { return points }

        // Find bounding box
        var minX = Int.max, maxX = Int.min
        var minY = Int.max, maxY = Int.min
        for p in points {
            minX = min(minX, p.x)
            maxX = max(maxX, p.x)
            minY = min(minY, p.y)
            maxY = max(maxY, p.y)
        }

        let width = maxX - minX + 1
        let height = maxY - minY + 1

        // Calculate grid cell size to achieve target count
        // gridCells ≈ targetCount, so cellSize = sqrt(area / targetCount)
        let area = Float(width * height)
        let cellSize = max(1, Int(sqrtf(area / Float(targetCount))))

        // Grid-based sampling: keep one point per cell
        var grid: [Int: (x: Int, y: Int)] = [:]  // cellKey -> point

        for p in points {
            let cellX = (p.x - minX) / cellSize
            let cellY = (p.y - minY) / cellSize
            let key = cellY * 10000 + cellX  // Unique cell key

            // Keep first point in each cell (or could average)
            if grid[key] == nil {
                grid[key] = p
            }
        }

        return Array(grid.values)
    }
}
