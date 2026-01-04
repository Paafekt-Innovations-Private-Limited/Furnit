import Foundation
import simd
import Metal

// MARK: - Splat Structure (internal representation from SHARP)
/// Representation of a Gaussian splat decoded from SHARP.
/// NOT directly sent to Metal - use SplatVertex for rendering.
struct Splat {
    var position: SIMD3<Float>   // World space position
    var scale: SIMD3<Float>      // Standard deviations in x,y,z
    var quat: SIMD4<Float>       // Rotation quaternion
    var color: SIMD3<Float>      // RGB in [0,1]
    var opacity: Float           // Alpha in [0,1]
}

// MARK: - SplatVertex (Metal-compatible vertex format)
/// Using SIMD4 to avoid SIMD3 padding issues (Swift SIMD3 = 16 bytes, Metal float3 = 12)
/// Total: 32 bytes per vertex, no padding surprises
struct SplatVertex {
    var positionRadius: SIMD4<Float>  // xyz = position, w = radius (16 bytes)
    var colorOpacity: SIMD4<Float>    // rgb = color, a = opacity (16 bytes)
    // Total: 32 bytes exactly
}

// NOTE: SplatUniforms is defined in SplatRenderer.swift to match Metal shader

// MARK: - Conversion from GaussianSplat
extension Splat {
    /// Convert from SinglePhotoRoomReconstructor.GaussianSplat to renderable Splat
    init(from gaussianSplat: SinglePhotoRoomReconstructor.GaussianSplat) {
        self.position = gaussianSplat.position
        self.scale = gaussianSplat.scale
        self.quat = gaussianSplat.quaternion
        self.color = gaussianSplat.color
        self.opacity = gaussianSplat.opacity
    }
}

// MARK: - Heap Item for Top-K Selection
/// Used by fromSHARP for memory-efficient top-K opacity selection
private struct SplatHeapItem {
    var opacity: Float
    var splat: SinglePhotoRoomReconstructor.GaussianSplat
}

// MARK: - Splat Array Utilities
extension Array where Element == Splat {
    /// Maximum splats to render after diversity selection
    static let maxSplatCount = 50_000

    /// DEBUG: Set to true to bypass diversity and use raw semantic splats
    static let debugBypassDiversity = true

    /// DEBUG: Smaller target size so furniture looks object-like, not spread out
    static let debugTargetSize: Float = 6.0

    /// DEBUG: Max splats to show (fewer = tighter blob, less galaxy)
    static let debugMaxSplats = 20_000

    /// DEBUG: Max radius to include (cull giant blurry blobs)
    static let debugMaxRadius: Float = 0.3

    /// Create splat array from SHARP output with semantic filtering and importance scoring.
    static func fromSHARP(
        _ gaussianSplats: [SinglePhotoRoomReconstructor.GaussianSplat],
        targetSize: Float = 4.0  // Room size in meters
    ) -> [Splat] {
        // Use bigger target size in debug mode
        let effectiveTargetSize = debugBypassDiversity ? debugTargetSize : targetSize

        print("========== fromSHARP START ==========")
        print("Input: \(gaussianSplats.count) raw splats, targetSize=\(effectiveTargetSize) (debug=\(debugBypassDiversity))")

        guard !gaussianSplats.isEmpty else {
            print("ERROR: No input splats!")
            return []
        }

        // --------------------------------------------------------
        // 1. Find position bounds first (needed for semantic filtering)
        // --------------------------------------------------------
        var minPos = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxPos = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        for gs in gaussianSplats {
            minPos = simd_min(minPos, gs.position)
            maxPos = simd_max(maxPos, gs.position)
        }

        let extent = maxPos - minPos + SIMD3<Float>(repeating: 1e-6)

        // Compute median Z for depth threshold
        let allZ = gaussianSplats.map { ($0.position.z - minPos.z) / extent.z }
        let sortedZ = allZ.sorted()
        let medianZ = sortedZ[sortedZ.count / 2]

        print("SHARP bounds: min=\(minPos), max=\(maxPos)")
        print("SHARP extent: \(extent), medianZ=\(medianZ)")

        // --------------------------------------------------------
        // 2. Semantic filtering: opacity, Y-band, depth threshold
        // --------------------------------------------------------
        let minOpacity: Float = 0.25           // Drop hazy splats
        let furnitureBandMinY: Float = 0.0     // Floor level (normalized)
        let furnitureBandMaxY: Float = 0.45    // ~first 1.2m above floor (furniture height)
        let depthThreshold = medianZ + 0.08    // Keep splats near/in front of median

        var skippedLowOpacity = 0
        var skippedHighY = 0
        var skippedFarDepth = 0

        var semanticFiltered: [(gs: SinglePhotoRoomReconstructor.GaussianSplat, normY: Float, normZ: Float)] = []
        semanticFiltered.reserveCapacity(gaussianSplats.count / 3)

        for gs in gaussianSplats {
            // Normalize position to 0-1 range
            let normY = (gs.position.y - minPos.y) / extent.y
            let normZ = (gs.position.z - minPos.z) / extent.z

            // 1) Opacity gate
            if gs.opacity < minOpacity {
                skippedLowOpacity += 1
                continue
            }

            // 2) Furniture Y-band (skip ceiling/upper walls)
            if normY < furnitureBandMinY || normY > furnitureBandMaxY {
                skippedHighY += 1
                continue
            }

            // 3) Depth gate (skip far walls/background)
            if normZ > depthThreshold {
                skippedFarDepth += 1
                continue
            }

            semanticFiltered.append((gs, normY, normZ))
        }

        print("Semantic filter: kept \(semanticFiltered.count), dropped \(skippedLowOpacity) low opacity, \(skippedHighY) out-of-band Y, \(skippedFarDepth) far depth")

        // Fallback if too aggressive filtering removed everything
        if semanticFiltered.isEmpty {
            print("Warning: All splats filtered out! Relaxing filters...")
            // Fallback: just use opacity filter with lower threshold
            for gs in gaussianSplats where gs.opacity >= 0.15 {
                let normY = (gs.position.y - minPos.y) / extent.y
                let normZ = (gs.position.z - minPos.z) / extent.z
                semanticFiltered.append((gs, normY, normZ))
            }
        }

        guard !semanticFiltered.isEmpty else {
            print("ERROR: No splats remaining after fallback filter")
            return []
        }

        // ========================================================
        // DEBUG BYPASS: Skip diversity, use semantic splats directly
        // ========================================================
        if debugBypassDiversity {
            print("!!!!! DEBUG MODE: BYPASSING DIVERSITY !!!!!")
            print("Semantic splats available: \(semanticFiltered.count)")

            // Pre-compute scale for radius filtering
            let center = (minPos + maxPos) * 0.5
            let maxExtent = Swift.max(extent.x, Swift.max(extent.y, extent.z))
            let scale: Float = maxExtent > 0 ? (effectiveTargetSize / maxExtent) : 1.0

            print("DEBUG: center=\(center), maxExtent=\(maxExtent), scale=\(scale)")

            // Filter out giant blurry blobs BEFORE sorting
            var culledBigRadius = 0
            let radiusFiltered = semanticFiltered.filter { item in
                let baseRadius = Swift.max(item.gs.scale.x, item.gs.scale.y) * scale * 0.5
                if baseRadius > debugMaxRadius {
                    culledBigRadius += 1
                    return false
                }
                return true
            }

            print("DEBUG: Culled \(culledBigRadius) giant radius splats (>\(debugMaxRadius))")
            print("DEBUG: After radius filter: \(radiusFiltered.count) splats")

            // Sort by opacity (highest first) and take top N
            let maxDebugSplats = Swift.min(debugMaxSplats, radiusFiltered.count)
            let sorted = radiusFiltered.sorted { $0.gs.opacity > $1.gs.opacity }
            let topSplats = sorted.prefix(maxDebugSplats)

            print("DEBUG: Taking top \(topSplats.count) splats by opacity")

            var result: [Splat] = []
            result.reserveCapacity(topSplats.count)

            for (i, item) in topSplats.enumerated() {
                let gs = item.gs
                let worldPos = (gs.position - center) * scale

                // Moderate radius - visible but not bloated
                let baseRadius = Swift.max(gs.scale.x, gs.scale.y) * scale * 0.5
                let debugRadius = Swift.max(baseRadius, 0.03)  // Minimum 0.03, no 2x bloat

                result.append(
                    Splat(
                        position: worldPos,
                        scale: SIMD3<Float>(debugRadius, debugRadius, debugRadius),
                        quat: gs.quaternion,
                        color: gs.color,
                        opacity: Swift.min(Swift.max(gs.opacity, 0.0), 1.0)
                    )
                )

                // Log first 10 splats
                if i < 10 {
                    print("DEBUG splat[\(i)]: pos=\(worldPos), radius=\(debugRadius), opacity=\(gs.opacity), color=\(gs.color)")
                }
            }

            // Log bounds of final splats
            let finalXs = result.map { $0.position.x }
            let finalYs = result.map { $0.position.y }
            let finalZs = result.map { $0.position.z }
            let finalRadii = result.map { $0.scale.x }

            print("""
            DEBUG FINAL STATS:
              count = \(result.count)
              X: [\(finalXs.min() ?? 0), \(finalXs.max() ?? 0)]
              Y: [\(finalYs.min() ?? 0), \(finalYs.max() ?? 0)]
              Z: [\(finalZs.min() ?? 0), \(finalZs.max() ?? 0)]
              radius: [\(finalRadii.min() ?? 0), \(finalRadii.max() ?? 0)]
            """)
            print("========== fromSHARP END (DEBUG) ==========")

            return result
        }

        // --------------------------------------------------------
        // 3. Importance scoring with tent-shaped size preference
        //    (NORMAL PATH - only runs if debugBypassDiversity = false)
        // --------------------------------------------------------
        print("NORMAL PATH: Running diversity selection...")

        // Tent function: 0 at edges, 1 at middle
        func tent(_ v: Float, _ minV: Float, _ midV: Float, _ maxV: Float) -> Float {
            if v <= minV || v >= maxV { return 0 }
            if v == midV { return 1 }
            return v < midV
                ? (v - minV) / (midV - minV)
                : (maxV - v) / (maxV - midV)
        }

        // Size thresholds in SHARP scale units
        let minScale: Float = 5.0      // Too tiny = noise
        let midScale: Float = 18.0     // Furniture-sized blobs
        let maxScale: Float = 30.0     // Too large = background

        // Score each splat (store splat + score for now)
        let scored: [(splat: SinglePhotoRoomReconstructor.GaussianSplat, score: Float)] = semanticFiltered.map { item in
            let gs = item.gs
            let normZ = item.normZ

            // Opacity score (higher = better)
            let opacityScore = gs.opacity

            // Tent-shaped size score: prefer medium-sized splats
            let scaleXY = Swift.max(gs.scale.x, gs.scale.y)
            let sizeScore = Swift.max(0, Swift.min(1, tent(scaleXY, minScale, midScale, maxScale)))

            // Depth score: 1.0 at medianZ, falls off toward back
            let depthDelta = (normZ - medianZ + 0.05) / 0.15
            let depthScore = Swift.max(0, Swift.min(1, 1.0 - depthDelta))

            let score = opacityScore * (0.5 * depthScore + 0.5 * sizeScore)
            return (gs, score)
        }

        // --------------------------------------------------------
        // 4. Spatial diversity: grid-based selection
        //    IMPORTANT: Normalize X/Y across the SCORED set's actual bounds
        // --------------------------------------------------------
        let gridSize = 32
        let targetCount = 50_000  // Target splat count after diversity

        // Find actual X/Y bounds of scored splats
        let scoredXs = scored.map { $0.splat.position.x }
        let scoredYs = scored.map { $0.splat.position.y }

        guard let sMinX = scoredXs.min(), let sMaxX = scoredXs.max(),
              let sMinY = scoredYs.min(), let sMaxY = scoredYs.max() else {
            print("Error: Could not compute scored bounds")
            return []
        }

        let invRangeX: Float = sMaxX > sMinX ? 1.0 / (sMaxX - sMinX) : 1.0
        let invRangeY: Float = sMaxY > sMinY ? 1.0 / (sMaxY - sMinY) : 1.0

        print("Diversity grid bounds: X[\(sMinX), \(sMaxX)], Y[\(sMinY), \(sMaxY)]")

        // Type alias for clarity
        typealias CellItem = (splat: SinglePhotoRoomReconstructor.GaussianSplat, score: Float)

        // Build grid cells using properly normalized coordinates
        var cells: [[[CellItem]]] = (0..<gridSize).map { _ in
            (0..<gridSize).map { _ in [CellItem]() }
        }

        for item in scored {
            let p = item.splat.position
            let nx = Swift.max(0, Swift.min(1, (p.x - sMinX) * invRangeX))
            let ny = Swift.max(0, Swift.min(1, (p.y - sMinY) * invRangeY))

            let ix = Swift.max(0, Swift.min(gridSize - 1, Int(nx * Float(gridSize))))
            let iy = Swift.max(0, Swift.min(gridSize - 1, Int(ny * Float(gridSize))))
            cells[ix][iy].append((item.splat, item.score))
        }

        // Count used cells first to compute dynamic per-cell cap
        var usedCellCount = 0
        for x in 0..<gridSize {
            for y in 0..<gridSize {
                if !cells[x][y].isEmpty { usedCellCount += 1 }
            }
        }

        // Dynamic per-cell cap: targetCount / usedCells, minimum 10
        let basePerCell = Swift.max(10, targetCount / Swift.max(1, usedCellCount))

        print("Diversity: \(usedCellCount) cells used, basePerCell = \(basePerCell)")

        // Take top basePerCell from each cell
        var diversified: [SinglePhotoRoomReconstructor.GaussianSplat] = []
        diversified.reserveCapacity(Swift.min(targetCount, scored.count))

        for x in 0..<gridSize {
            for y in 0..<gridSize {
                let cell = cells[x][y]
                if cell.isEmpty { continue }
                let bestInCell = cell
                    .sorted { $0.score > $1.score }
                    .prefix(basePerCell)
                    .map { $0.splat }
                diversified.append(contentsOf: bestInCell)
            }
        }

        print("Diversity grid: collected \(diversified.count) splats")

        // Final cap if still too many
        let finalSplats: [SinglePhotoRoomReconstructor.GaussianSplat]
        if diversified.count > maxSplatCount {
            // Re-score and take top
            let reScored = diversified.map { gs -> (SinglePhotoRoomReconstructor.GaussianSplat, Float) in
                let scaleXY = Swift.max(gs.scale.x, gs.scale.y)
                let sizeScore = Swift.max(0, Swift.min(1, tent(scaleXY, minScale, midScale, maxScale)))
                return (gs, gs.opacity * sizeScore)
            }
            finalSplats = reScored
                .sorted { $0.1 > $1.1 }
                .prefix(maxSplatCount)
                .map { $0.0 }
            print("Final cap: \(diversified.count) -> \(finalSplats.count)")
        } else {
            finalSplats = diversified
        }

        // Log diversity stats
        let finalXs = finalSplats.map { $0.position.x }
        let finalYs = finalSplats.map { $0.position.y }
        let finalZs = finalSplats.map { $0.position.z }
        let finalScales = finalSplats.map { Swift.max($0.scale.x, $0.scale.y) }

        print("""
        DIVERSIFIED STATS:
          count = \(finalSplats.count)
          X: [\(finalXs.min() ?? 0), \(finalXs.max() ?? 0)]
          Y: [\(finalYs.min() ?? 0), \(finalYs.max() ?? 0)]
          Z: [\(finalZs.min() ?? 0), \(finalZs.max() ?? 0)]
          scale: [\(finalScales.min() ?? 0), \(finalScales.max() ?? 0)]
        """)

        // --------------------------------------------------------
        // 5. Build world-space Splat array
        // --------------------------------------------------------
        let center = (minPos + maxPos) * 0.5
        let maxExtent = Swift.max(extent.x, Swift.max(extent.y, extent.z))
        let scale: Float = maxExtent > 0 ? (targetSize / maxExtent) : 1.0

        var result: [Splat] = []
        result.reserveCapacity(finalSplats.count)

        for gs in finalSplats {
            let worldPos = (gs.position - center) * scale
            let worldScale = gs.scale * scale * 0.5  // stddev -> radius

            result.append(
                Splat(
                    position: worldPos,
                    scale: worldScale,
                    quat: gs.quaternion,
                    color: gs.color,
                    opacity: Swift.min(Swift.max(gs.opacity, 0.0), 1.0)
                )
            )
        }

        return result
    }

    /// Sort splats by depth for correct alpha blending (back to front)
    mutating func sortByDepth(cameraPosition: SIMD3<Float>, cameraForward: SIMD3<Float>) {
        // Compute depth as distance along camera forward direction
        self.sort { a, b in
            let depthA = simd_dot(a.position - cameraPosition, cameraForward)
            let depthB = simd_dot(b.position - cameraPosition, cameraForward)
            return depthA < depthB  // Back to front
        }
    }

    /// Filter splats by opacity and radius thresholds
    func filtered(minOpacity: Float = 0.15, maxRadius: Float = 30.0) -> [Splat] {
        var kept = 0
        var droppedOpacity = 0
        var droppedRadius = 0

        let result = self.filter { s in
            // Drop low opacity (hazy/noise)
            if s.opacity < minOpacity {
                droppedOpacity += 1
                return false
            }

            // Drop gigantic blobs (usually noise/background)
            let radius = Swift.max(s.scale.x, s.scale.y)
            if radius > maxRadius {
                droppedRadius += 1
                return false
            }

            kept += 1
            return true
        }

        print("Splat filter: kept \(kept), dropped \(droppedOpacity) low opacity, \(droppedRadius) large radius")
        return result
    }

    /// Create Metal vertex buffer with properly aligned SplatVertex data.
    /// Maps SHARP positions into room space coordinates.
    /// Returns buffer, count, and cloud bounds for camera auto-framing.
    func makeVertexBuffer(device: MTLDevice, maxCount: Int = 120_000) -> (buffer: MTLBuffer, count: Int, bounds: CloudBounds)? {
        print("========== makeVertexBuffer START ==========")
        print("Input splats: \(self.count), maxCount: \(maxCount)")

        let count = Swift.min(self.count, maxCount)
        guard count > 0 else {
            print("ERROR: No splats to make buffer from!")
            return nil
        }

        var vertices: [SplatVertex] = []
        vertices.reserveCapacity(count)

        // Room dimensions (should match settings)
        let roomWidth: Float = 4.0
        let roomDepth: Float = 4.5
        let roomHeight: Float = 2.8

        print("Room dimensions: \(roomWidth) x \(roomDepth) x \(roomHeight)")

        // Find bounds and compute average position (center of mass)
        var minPos = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxPos = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var positionSum = SIMD3<Float>(repeating: 0)
        for s in self.prefix(count) {
            minPos = simd_min(minPos, s.position)
            maxPos = simd_max(maxPos, s.position)
            positionSum += s.position
        }
        let avgPos = positionSum / Float(count)  // Simple average (center of mass)

        let range = maxPos - minPos + SIMD3<Float>(repeating: 1e-6)

        print("Input position range: min=\(minPos), max=\(maxPos)")
        print("Range: \(range)")

        // Debug: print first few splats
        print("First 5 splats before vertex conversion:")
        for (i, s) in self.prefix(5).enumerated() {
            print("[\(i)] pos=\(s.position), scale=\(s.scale), color=\(s.color), opacity=\(s.opacity)")
        }

        // DEBUG: Use splat positions directly (they're already centered around origin from fromSHARP)
        // Bigger radius for visibility
        let minR: Float = 0.15
        let maxR: Float = 0.5
        let radiusScale: Float = 0.02

        print("DEBUG: Using direct positions (centered at origin), no squash/slab")
        print("Radius params: minR=\(minR), maxR=\(maxR), radiusScale=\(radiusScale)")

        // Pack into SIMD4 format - positions already in world space from fromSHARP
        for s in self.prefix(count) {
            // Use position directly (already centered around origin)
            let worldX = s.position.x
            let worldY = s.position.y
            let worldZ = s.position.z

            // Radius: clamp raw scale and map to visible range
            let rawScale = Swift.max(s.scale.x, Swift.max(s.scale.y, s.scale.z))
            let clamped = Swift.max(0.0, Swift.min(rawScale, 40.0))
            let t = clamped * radiusScale
            let radius = minR + (maxR - minR) * Swift.min(t, 1.0)

            let posR = SIMD4<Float>(worldX, worldY, worldZ, radius)

            // Pack color (rgb) + opacity (a) into SIMD4, clamped to [0,1]
            let colA = SIMD4<Float>(
                Swift.max(0, Swift.min(1, s.color.x)),
                Swift.max(0, Swift.min(1, s.color.y)),
                Swift.max(0, Swift.min(1, s.color.z)),
                Swift.max(0, Swift.min(1, s.opacity))
            )

            vertices.append(SplatVertex(positionRadius: posR, colorOpacity: colA))
        }

        // Debug: print first few packed vertices
        print("First 10 packed vertices:")
        for i in 0..<Swift.min(10, vertices.count) {
            let v = vertices[i]
            print("[\(i)] pos=(\(v.positionRadius.x), \(v.positionRadius.y), \(v.positionRadius.z)) radius=\(v.positionRadius.w) color=(\(v.colorOpacity.x), \(v.colorOpacity.y), \(v.colorOpacity.z)) opacity=\(v.colorOpacity.w)")
        }

        // 🧪 BIG RED DEBUG SPOT at center of mass (avgPos, not bounding-box center)
        let debugCenterSplat = SplatVertex(
            positionRadius: SIMD4<Float>(avgPos.x, avgPos.y, avgPos.z, 0.6),
            colorOpacity: SIMD4<Float>(1.0, 0.0, 0.0, 1.0)     // bright red, full opacity
        )
        vertices.append(debugCenterSplat)
        print("DEBUG: Added BIG RED center splat at avgPos=\(avgPos) with radius 0.6")
        print("DEBUG: (bounding-box center was \((minPos + maxPos) * 0.5))")

        // Log final vertex stats
        let allX = vertices.map { $0.positionRadius.x }
        let allY = vertices.map { $0.positionRadius.y }
        let allZ = vertices.map { $0.positionRadius.z }
        let allR = vertices.map { $0.positionRadius.w }
        let allOpacity = vertices.map { $0.colorOpacity.w }

        print("""
        VERTEX BUFFER STATS:
          count = \(vertices.count)
          X: [\(allX.min() ?? 0), \(allX.max() ?? 0)]
          Y: [\(allY.min() ?? 0), \(allY.max() ?? 0)]
          Z: [\(allZ.min() ?? 0), \(allZ.max() ?? 0)]
          radius: [\(allR.min() ?? 0), \(allR.max() ?? 0)]
          opacity: [\(allOpacity.min() ?? 0), \(allOpacity.max() ?? 0)]
        """)

        let stride = MemoryLayout<SplatVertex>.stride
        let length = vertices.count * stride
        print("SplatVertex stride: \(stride) bytes (should be 32)")
        print("Creating buffer: \(vertices.count) vertices, \(length) bytes")

        guard let buffer = device.makeBuffer(bytes: vertices, length: length, options: [.storageModeShared]) else {
            print("ERROR: Failed to create Metal buffer!")
            return nil
        }

        // Create CloudBounds from the actual vertex positions (not input splats)
        let finalBounds = CloudBounds(min: minPos, max: maxPos, averageCenter: avgPos)
        print("Cloud bounds for camera: averageCenter=\(finalBounds.averageCenter), boxCenter=\(finalBounds.boxCenter), radius=\(finalBounds.radius)")

        print("========== makeVertexBuffer END ==========")
        return (buffer, vertices.count, finalBounds)
    }
}
