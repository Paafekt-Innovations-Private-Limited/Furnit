// RoomGeometryEngine.swift
// Drop-in replacement. Requires LinearAlgebra.swift in the same target.
//
// EXTERNAL DEPENDENCIES (defined elsewhere in the project — do NOT import or redefine):
//   SplatDepthQueryable, SplatColorReadable, TextureSampler
//   SourceCameraInfo, DetectedPlane, RoomModel, RoomCorner
//   FreeFloorRegion, FloorUVBounds, AABB3, SurfacePalette
//   logDebug(_:)
//   normalizeOrZero(_:)                         ← LinearAlgebra.swift
//   smallestEigenvector(ofSymmetric3x3:)         ← LinearAlgebra.swift
//
// CHANGES IN THIS REVISION:
//   1. RoomGeometryConfig: added `useBoundsBasedRoomSize` toggle (default true)
//   2. RoomGeometryConfig: added six magic-number fields
//   3. Hardcoded magic numbers replaced throughout with config references
//   4. estimateSceneToMeters(floor:ceiling:bounds:plyHeightSU:) branched by toggle
//   5. computeFreespace(points:floor:sceneToMeters:) branched by toggle
//   6. Everything else preserved exactly from the user's current file

import Foundation
import simd
import Darwin

// MARK: - Color readback fallback

private final class NullSplatColorReader: SplatColorReadable {
    var supportsColorReadback: Bool { false }
    func colorAt(screenPoint _: CGPoint) -> SIMD3<Float>? { nil }
}

// MARK: - Errors

public enum GeometryExtractionError: Error {
    case insufficientPoints(Int)
    case noFloorDetected
    case noCeilingDetected
    case degenerateGeometry(String)
}

// MARK: - Configuration

/// Centralised tunables for the geometry pipeline.
public struct RoomGeometryConfig {
    public init() {}

    // RANSAC
    public var ransacIterations: Int          = 200
    public var ransacInlierThreshold: Float   = 0.04   // scene units
    public var ransacMinInlierFraction: Float = 0.15

    // Plane deduplication
    public var deduplicationNormalDotMin: Float       = 0.60
    public var deduplicationDistanceTolerance: Float  = 0.05

    // Scene-to-metres estimation
    public var standardCeilingHeightM: Float  = 2.44

    // ── Change 1 ──────────────────────────────────────────────────────────────
    /// When `true`, use legacy AABB-based room sizing and simple floor-extent freespace.
    /// When `false`, use plane-distance scaling and occupancy-grid freespace.
    public var useBoundsBasedRoomSize: Bool = true

    // ── Change 2 ──────────────────────────────────────────────────────────────
    /// Minimum **|dot(n, ref)|** for a plane parallel to `ref` (floor/ceiling vs `ref = floor.normal` or world +Y when `ref` unset).
    /// Relaxed from 0.72 → 0.50 to better handle high-angle, camera-looking-down shots where the
    /// reconstructed floor normal tilts toward Z and has a smaller Y component.
    public var horizontalNormalYMin: Float = 0.50

    /// Maximum **|dot(n, ref)|** for a plane perpendicular to `ref` (walls vs `ref = floor.normal`, else world +Y).
    public var verticalNormalYMax: Float = 0.32

    /// Maximum **|dot(n, ref)|** used in `refinePlaneFit` for wall refinement (slightly relaxed vs RANSAC).
    public var verticalNormalYMaxRefined: Float = 0.35

    /// Dot-product upper bound below which two wall normals are considered orthogonal.
    public var cornerOrthogonalityMax: Float = 0.75

    /// Side length of one occupancy-grid cell in metres (~15 cm).
    public var freespaceGridCellMeters: Float = 0.15

    /// Points above this fraction of ceiling height are treated as ceiling artefacts,
    /// not furniture/obstacles.
    public var obstacleCeilingHeightFraction: Float = 0.7
}

// MARK: - Room-space transform (floor-aligned coordinates)

/// Orthonormal basis aligned to the detected floor. Room space is defined so that:
///   - Y axis (`up`) is the floor normal
///   - origin is on the floor plane (`floorOrigin`)
///   - X (`right`) and Z (`forward`) span the floor, chosen to be roughly world aligned.
struct RoomSpaceTransform {

    let floorOrigin: SIMD3<Float>
    let right: SIMD3<Float>
    let up: SIMD3<Float>
    let forward: SIMD3<Float>

    init(floor: DetectedPlane) {
        let up = normalizeOrZero(floor.normal)
        let worldZ = SIMD3<Float>(0, 0, 1)
        let worldX = SIMD3<Float>(1, 0, 0)
        let candidate: SIMD3<Float> = abs(simd_dot(up, worldZ)) < 0.9 ? worldZ : worldX
        let right = normalizeOrZero(simd_cross(candidate, up))
        let forward = simd_cross(up, right)

        self.floorOrigin = floor.pointOnPlane
        self.right = right
        self.up = up
        self.forward = forward
    }

    @inline(__always)
    func toRoomSpace(_ point: SIMD3<Float>) -> SIMD3<Float> {
        let d = point - floorOrigin
        return SIMD3<Float>(
            simd_dot(d, right),
            simd_dot(d, up),
            simd_dot(d, forward)
        )
    }

    @inline(__always)
    func toSceneSpace(_ point: SIMD3<Float>) -> SIMD3<Float> {
        floorOrigin
            + right * point.x
            + up * point.y
            + forward * point.z
    }

    func transformPoints(_ points: [SIMD3<Float>]) -> [SIMD3<Float>] {
        points.map { toRoomSpace($0) }
    }

    func transformPlane(_ plane: DetectedPlane, toRoomSpace: Bool) -> DetectedPlane {
        if toRoomSpace {
            let n = SIMD3<Float>(
                simd_dot(plane.normal, right),
                simd_dot(plane.normal, up),
                simd_dot(plane.normal, forward)
            )
            let p = self.toRoomSpace(plane.pointOnPlane)
            return DetectedPlane(
                type: plane.type,
                normal: normalizeOrZero(n),
                pointOnPlane: p
            )
        } else {
            let nScene = right * plane.normal.x
                + up * plane.normal.y
                + forward * plane.normal.z
            let pScene = self.toSceneSpace(plane.pointOnPlane)
            return DetectedPlane(
                type: plane.type,
                normal: normalizeOrZero(nScene),
                pointOnPlane: pScene
            )
        }
    }
}

// MARK: - Engine

public final class RoomGeometryEngine {

    // MARK: Stored properties

    public var config: RoomGeometryConfig

    /// User-confirmed or measured ceiling height in metres.
    public var referenceCeilingHeightMeters: Float?

    /// Camera/subject metadata used in the subjectDistance fallback.
    public var cameraInfo: SourceCameraInfo

    private let depthQuery: SplatDepthQueryable
    private let colorRead: SplatColorReadable

    // MARK: Init

    public init(
        depthQuery: SplatDepthQueryable,
        colorReader: SplatColorReadable?,
        cameraInfo: SourceCameraInfo,
        referenceCeilingHeightMeters: Float? = nil,
        config: RoomGeometryConfig = RoomGeometryConfig()
    ) {
        self.depthQuery = depthQuery
        self.colorRead = colorReader ?? NullSplatColorReader()
        self.cameraInfo = cameraInfo
        self.config = config
        self.referenceCeilingHeightMeters = referenceCeilingHeightMeters
    }

    // MARK: - Public entry point

    /// Builds a complete `RoomModel` from a splat depth point cloud.
    public func extractRoomModel(points: [SIMD3<Float>]) throws -> RoomModel {
        guard points.count >= 3 else {
            throw GeometryExtractionError.insufficientPoints(points.count)
        }

        // Make RANSAC plane fits deterministic for a given point cloud by seeding the
        // C RNG that backs `drand48()` used in `randomPlaneSample(from:)`.
        srand48(42)

        logDebug("📐 [RoomGeometryEngine] extractRoomModel — \(points.count) points (session / caller grid)")

        // Scene-space bounds and floor.
        let sceneBounds = computeAABB(points: points)
        let sceneFloor = try extractFloorPlane(points: points, bounds: sceneBounds)

        // Room-space transform (floor → up axis).
        let roomTransform = RoomSpaceTransform(floor: sceneFloor)
        let roomPoints = roomTransform.transformPoints(points)
        let roomBounds = computeAABB(points: roomPoints)

        let roomFloor = DetectedPlane(
            type: .floor,
            normal: SIMD3<Float>(0, 1, 0),
            pointOnPlane: .zero
        )
        let floorExtentsStr = String(format: "%.3f×%.3f×%.3f", roomBounds.size.x, roomBounds.size.y, roomBounds.size.z)
        logDebug("📐 [RoomGeometryEngine] floor_extents_su=\(floorExtentsStr)")

        // PLY subsample (full reconstruction) — prefer this vertical span for scale, not depth-grid AABB/ceiling.
        let plyExtentsResult: (width: Float, height: Float, depth: Float)? = {
            guard let splatPositions = depthQuery.trimmedSplatPositions, splatPositions.count >= 1000 else {
                return nil
            }
            return measurePLYExtents(
                splatPositions: splatPositions,
                floorNormal: sceneFloor.normal,
                floorOrigin: sceneFloor.pointOnPlane
            )
        }()
        let plyHeightSUForScale: Float? = {
            guard let e = plyExtentsResult else { return nil }
            let h = e.height
            return (h > 0.001 && h.isFinite) ? h : nil
        }()

        // Ceiling in room space: RANSAC on upper band, else AABB-top synthetic (Step 2).
        let roomCeiling = extractCeilingPlane(points: roomPoints, bounds: roomBounds, floor: roomFloor)
            ?? syntheticCeiling(floor: roomFloor, bounds: roomBounds)
        let scale = estimateSceneToMeters(
            floor: roomFloor,
            ceiling: roomCeiling,
            bounds: roomBounds,
            plyHeightSU: plyHeightSUForScale
        )

        let planeHSUFromCeiling = abs(roomCeiling.distance(to: roomFloor.pointOnPlane))
        let planeHSUForLog = plyHeightSUForScale ?? planeHSUFromCeiling
        let ceilingM = referenceCeilingHeightMeters ?? config.standardCeilingHeightM
        let sceneToMetersStr = String(format: "%.4f", scale)
        let cmStr = String(format: "%.3f", ceilingM)
        let aabbYspanStr = String(format: "%.4f", roomBounds.size.y)
        logDebug("📐 [RoomGeometryEngine] sceneToMeters=\(sceneToMetersStr) plane_h_su=\(String(format: "%.4f", planeHSUForLog)) (ceiling_plane_h_su=\(String(format: "%.4f", planeHSUFromCeiling))) aabb_y_span=\(aabbYspanStr) ref_ceiling_m=\(cmStr)")

        if let plyExtents = plyExtentsResult,
           let splatPositions = depthQuery.trimmedSplatPositions,
           splatPositions.count >= 1000 {
            logDebug("📐 [RoomGeometryEngine] Floor-aligned PLY extents (P3–P97, projected onto floor normal)")
            let plySU = String(format: "%.3f×%.3f×%.3f", plyExtents.width, plyExtents.height, plyExtents.depth)
            let plyM = String(
                format: "%.3f×%.3f×%.3f",
                plyExtents.width * scale,
                plyExtents.height * scale,
                plyExtents.depth * scale
            )
            let depthSU = String(format: "%.3f×%.3f×%.3f", roomBounds.size.x, roomBounds.size.y, roomBounds.size.z)
            let scaleNote = plyHeightSUForScale != nil ? " feeds sceneToMeters when set" : " (no PLY height for scale)"
            logDebug("📐 [RoomGeometryEngine] ply_extents_su=\(plySU) (P3–P97 floor-aligned, \(splatPositions.count) splats)\(scaleNote)")
            logDebug("📐 [RoomGeometryEngine] ply_extents_m=\(plyM)")
            logDebug("📐 [RoomGeometryEngine] depth_grid_extents_su=\(depthSU) (viewport grid, room space)")
        }

        let sceneFloorN = sceneFloor.normal
        let sceneFloorP = sceneFloor.pointOnPlane
        let plyP397Str: String
        if let pe = plyExtentsResult {
            plyP397Str = String(format: "%.6f,%.6f,%.6f", pe.width, pe.height, pe.depth)
        } else {
            plyP397Str = "n/a"
        }
        let depthGridStr = String(
            format: "%.6f,%.6f,%.6f",
            roomBounds.size.x, roomBounds.size.y, roomBounds.size.z
        )
        let plyHForScaleStr = plyHeightSUForScale.map { String(format: "%.6f", $0) } ?? "n/a"
        logDebug(
            "[ROOM_CREATE_COMPARE] phase=room_geometry " +
            "floor_n_scene=(\(String(format: "%.6f", sceneFloorN.x)),\(String(format: "%.6f", sceneFloorN.y)),\(String(format: "%.6f", sceneFloorN.z))) " +
            "floor_p_scene=(\(String(format: "%.6f", sceneFloorP.x)),\(String(format: "%.6f", sceneFloorP.y)),\(String(format: "%.6f", sceneFloorP.z))) " +
            "depth_grid_size_su_xyz=\(depthGridStr) ply_P397_floor_aligned_su_whd=\(plyP397Str) ply_h_for_scale_su=\(plyHForScaleStr) " +
            "sceneToMeters=\(sceneToMetersStr) plane_h_su=\(String(format: "%.6f", planeHSUForLog)) " +
            "ceiling_plane_h_su=\(String(format: "%.6f", planeHSUFromCeiling)) ref_ceiling_m=\(cmStr) " +
            "aabb_y_span_su=\(aabbYspanStr)"
        )

        // Walls / corners in room space.
        let roomWalls = try extractWallPlanes(points: roomPoints, floor: roomFloor, ceiling: roomCeiling)
        let roomCorners = findCorners(walls: roomWalls, floor: roomFloor, sceneToMeters: scale)

        // Compute free-floor regions in scene space (keep UVs tied to real floor plane).
        let sceneCeiling = roomTransform.transformPlane(roomCeiling, toRoomSpace: false)
        let freeRegions = computeFreespace(points: points, floor: sceneFloor, ceiling: sceneCeiling, sceneToMeters: scale)

        // Transform planes and corners back to scene space for RoomModel consumers.
        let sceneWalls = roomWalls.map { roomTransform.transformPlane($0, toRoomSpace: false) }
        let sceneCorners = roomCorners.map { corner -> RoomCorner in
            let posScene = roomTransform.toSceneSpace(corner.position)
            return RoomCorner(position: posScene, uv: corner.uv)
        }

        let stub = RoomModel(
            aabb: sceneBounds,
            floor: sceneFloor,
            ceiling: sceneCeiling,
            walls: sceneWalls,
            corners: sceneCorners,
            freeFloorRegions: freeRegions,
            surfacePalette: .empty,
            cameraInfo: nil,
            sceneToMeters: scale
        )

        logOpposingWallPairDiagnostics(
            walls: sceneWalls,
            sceneToMeters: scale,
            referenceWidthMeters: stub.widthMeters,
            referenceDepthMeters: stub.depthMeters
        )

        let palette: SurfacePalette
        if colorRead.supportsColorReadback {
            let sampler = TextureSampler(depthQuery: depthQuery, colorReader: colorRead, roomModel: stub)
            palette = sampler.samplePalette()
        } else {
            palette = .empty
        }

        return RoomModel(
            aabb: sceneBounds,
            floor: sceneFloor,
            ceiling: sceneCeiling,
            walls: sceneWalls,
            corners: sceneCorners,
            freeFloorRegions: freeRegions,
            surfacePalette: palette,
            cameraInfo: cameraInfo,
            sceneToMeters: scale
        )
    }

    /// PLY / splat extent helper for **logged** ``ply_extents_*`` lines (see ``extractRoomModel(points:)``).
    ///
    /// - Projects each position onto **floor-relative** axes (`right`, floor normal as “up”, `forward`).
    /// - On each axis, takes **P3–P97** (default ``trimLow`` / ``trimHigh`` = 0.03 / 0.97): sort that axis’s
    ///   coordinates, use values at the low/high percentile indices, return **hi − lo** as the span.
    /// - Returns **(width, height, depth)** in scene units along those axes (not the raw world AABB).
    public func measurePLYExtents(
        splatPositions: [SIMD3<Float>],
        floorNormal: SIMD3<Float>,
        floorOrigin: SIMD3<Float>,
        trimLow: Float = 0.03,
        trimHigh: Float = 0.97
    ) -> (width: Float, height: Float, depth: Float) {
        guard !splatPositions.isEmpty else { return (0, 0, 0) }

        let up = normalizeOrZero(floorNormal)
        let worldZ = SIMD3<Float>(0, 0, 1)
        let worldX = SIMD3<Float>(1, 0, 0)
        let candidate: SIMD3<Float> = abs(simd_dot(up, worldZ)) < 0.9 ? worldZ : worldX
        let right = normalizeOrZero(simd_cross(candidate, up))
        let forward = simd_cross(up, right)

        var xs: [Float] = []
        var ys: [Float] = []
        var zs: [Float] = []
        xs.reserveCapacity(splatPositions.count)
        ys.reserveCapacity(splatPositions.count)
        zs.reserveCapacity(splatPositions.count)

        for p in splatPositions {
            let d = p - floorOrigin
            xs.append(simd_dot(d, right))
            ys.append(simd_dot(d, up))
            zs.append(simd_dot(d, forward))
        }

        func trimmedSpan(_ values: [Float]) -> Float {
            let sorted = values.sorted()
            let n = sorted.count
            let lo = sorted[max(0, Int(Float(n) * trimLow))]
            let hi = sorted[min(n - 1, Int(Float(n) * trimHigh))]
            return max(0, hi - lo)
        }

        return (
            width: trimmedSpan(xs),
            height: trimmedSpan(ys),
            depth: trimmedSpan(zs)
        )
    }

    // MARK: - estimateSceneToMeters  (Change 4)

    /// Estimates the scene-unit → metre conversion factor from PLY/ceiling depth span and reference ceiling height
    /// (or subject distance as a last resort). Wall-YOLO / pinhole tape-height scaling has been removed.
    ///
    /// - Parameter plyHeightSU: P3–P97 vertical span (scene units) from full PLY via ``measurePLYExtents``.
    ///   When set, used before sparse depth-grid ceiling distance for height denominators.
    public func estimateSceneToMeters(
        floor:       DetectedPlane?,
        ceiling:     DetectedPlane?,
        bounds:      AABB3,
        plyHeightSU: Float? = nil
    ) -> Float {

        let ceilingM = referenceCeilingHeightMeters ?? config.standardCeilingHeightM

        let heightSUFromCeilingPlane: Float? = {
            guard let floor, let ceiling else { return nil }
            let h = abs(ceiling.distance(to: floor.pointOnPlane))
            return h > 0.001 ? h : nil
        }()
        let planeHeightSU: Float? = {
            if let ply = plyHeightSU, ply > 0.001, ply.isFinite { return ply }
            return heightSUFromCeilingPlane
        }()
        let verticalFallbackSU = max((plyHeightSU ?? bounds.size.y), 0.001)

        if config.useBoundsBasedRoomSize {
            logDebug("📐 [RoomGeometryEngine] sceneToMeters: BOUNDS-BASED mode active")
            if let ph = planeHeightSU {
                let scale = ceilingM / ph
                logDebug(
                    "📐 [RoomGeometryEngine] sceneToMeters BOUNDS-BASED ceiling_m / plane height: " +
                    "ceiling_m=\(String(format: "%.3f", ceilingM)) plane_h_su=\(String(format: "%.4f", ph)) " +
                    "aabb_y_su=\(String(format: "%.4f", bounds.size.y)) scale=\(String(format: "%.4f", scale))"
                )
                return scale
            }
            let fallback = ceilingM / verticalFallbackSU
            logDebug(
                "📐 [RoomGeometryEngine] sceneToMeters BOUNDS-BASED fallback ceiling_m/vertical SU=\(String(format: "%.4f", fallback))"
            )
            return fallback
        }

        // ── PLANE-BASED ─────────────────────────────────────────────────────
        if let ph = planeHeightSU, ph > 0.001 {
            let scale = ceilingM / ph
            logDebug(
                "📐 [RoomGeometryEngine] sceneToMeters OK (reference/standard ceiling + height SU): " +
                "ceiling_m=\(String(format: "%.3f", ceilingM)) sceneH_su=\(String(format: "%.4f", ph)) " +
                "scale=\(String(format: "%.4f", scale))"
            )
            return scale
        }

        if let subjectDistance = cameraInfo.subjectDistanceMeters,
           subjectDistance > 0.1,
           bounds.size.z > 0.05 {
            let scale = subjectDistance / max(bounds.size.z, 0.001)
            logDebug("📐 [RoomGeometryEngine] sceneToMeters via subjectDistance=\(String(format: "%.4f", scale))")
            return scale
        }

        let fallback = ceilingM / verticalFallbackSU
        logDebug("📐 [RoomGeometryEngine] sceneToMeters fallback ceiling_m/vertical SU=\(String(format: "%.4f", fallback))")
        return fallback
    }

    // MARK: - Floor Plane Extraction

    public func extractFloorPlane(
        points: [SIMD3<Float>],
        bounds: AABB3
    ) throws -> DetectedPlane {
        // Candidate points in the lower 20% of the Y range
        let threshold = bounds.min.y + (bounds.size.y * 0.20)
        let candidates = points.filter { $0.y <= threshold }
        guard !candidates.isEmpty else {
            throw GeometryExtractionError.noFloorDetected
        }

        guard let hypothesis = ransacFitPlane(points: candidates, isHorizontal: true) else {
            throw GeometryExtractionError.noFloorDetected
        }
        var n = hypothesis.normal
        let p = hypothesis.pointOnPlane
        if n.y < 0 { n = -n }
        let plane = DetectedPlane(type: .floor, normal: normalizeOrZero(n), pointOnPlane: p)
        logDebug("📐 [RoomGeometryEngine] floor plane normal=\(plane.normal) point=\(plane.pointOnPlane)")
        return plane
    }

    /// Detects a horizontal ceiling in **room space** (floor normal = +Y). Upper ~20% of the vertical AABB.
    public func extractCeilingPlane(
        points: [SIMD3<Float>],
        bounds: AABB3,
        floor: DetectedPlane
    ) -> DetectedPlane? {
        guard bounds.size.y > 1e-3 else { return nil }
        let threshold = bounds.max.y - bounds.size.y * 0.20
        let candidates = points.filter { $0.y >= threshold }
        guard candidates.count >= 10 else { return nil }

        guard let hypothesis = ransacFitPlane(
            points: candidates,
            isHorizontal: true,
            referenceHorizontalNormal: SIMD3<Float>(0, 1, 0)
        ) else { return nil }

        var n = hypothesis.normal
        if n.y > 0 { n = -n }
        let plane = DetectedPlane(
            type: .ceiling,
            normal: normalizeOrZero(n),
            pointOnPlane: hypothesis.pointOnPlane
        )
        guard plane.pointOnPlane.y > floor.pointOnPlane.y + 0.08 else { return nil }
        logDebug("📐 [RoomGeometryEngine] ceiling RANSAC normal=\(plane.normal) point=\(plane.pointOnPlane)")
        return plane
    }

    // MARK: - Synthetic Ceiling

    /// Horizontal ceiling at `bounds.max.y` in the **same coordinate space as `bounds`** (AABB top).
    /// ``extractRoomModel`` uses reference ceiling height ÷ ``sceneToMeters`` instead; kept for callers/tests.
    public func syntheticCeiling(
        floor:  DetectedPlane,
        bounds: AABB3
    ) -> DetectedPlane {
        let normal = SIMD3<Float>(0, -1, 0)
        let point = SIMD3<Float>(bounds.center.x, bounds.max.y, bounds.center.z)
        logDebug("📐 [RoomGeometryEngine] syntheticCeiling point=\(point)")
        return DetectedPlane(type: .ceiling, normal: normal, pointOnPlane: point)
    }

    // MARK: - Opposing Wall Pair Diagnostics

    /// Logs pairs of roughly opposing walls and their estimated separation in metres.
    public func logOpposingWallPairDiagnostics(
        walls:                  [DetectedPlane],
        sceneToMeters:          Float,
        referenceWidthMeters:  Float? = nil,
        referenceDepthMeters:  Float? = nil
    ) {
        guard walls.count >= 2 else { return }

        for i in 0 ..< walls.count {
            for j in (i + 1) ..< walls.count {
                let w1 = walls[i]
                let w2 = walls[j]
                let dotN = simd_dot(w1.normal, w2.normal)

                let label: String
                if dotN < -0.5 {
                    label = "OPPOSING"
                } else if abs(dotN) < 0.3 {
                    label = "PERPENDICULAR"
                } else {
                    label = "SIMILAR"
                }

                let dotStr = String(format: "%.3f", dotN)
                logDebug("📐 [DIAG] wall[\(i)]↔wall[\(j)] dot=\(dotStr) \(label)")

                if dotN < -0.5 {
                    let sepSU = abs(w2.distance(to: w1.pointOnPlane))
                    let sepM = sepSU * sceneToMeters
                    let sepStr = String(format: "%.3f", sepM)

                    let ax = abs(w1.normal.x)
                    let az = abs(w1.normal.z)
                    let axisLabel = ax >= az ? "width" : "depth"
                    let ref = ax >= az ? referenceWidthMeters : referenceDepthMeters
                    let gapStr: String
                    if let ref, ref.isFinite, ref > 0.05 {
                        gapStr = String(format: "%.3f", abs(sepM - ref))
                    } else {
                        gapStr = "n/a"
                    }

                    logDebug("📐 [DIAG] wall[\(i)]↔wall[\(j)] gap=\(sepStr)m axis=\(axisLabel) metricGap_m=\(gapStr)")
                }
            }
        }
    }

    // MARK: - Wall Plane Extraction  (Change 3)

    public func extractWallPlanes(
        points:  [SIMD3<Float>],
        floor:   DetectedPlane,
        ceiling: DetectedPlane
    ) throws -> [DetectedPlane] {

        // Wall height band: 10–90% span between floor and ceiling planes (room or scene space).
        let floorY = floor.pointOnPlane.y
        let ceilingY = ceiling.pointOnPlane.y
        let roomH = abs(ceilingY - floorY)
        var bandMin = min(floorY, ceilingY) + roomH * 0.10
        var bandMax = max(floorY, ceilingY) - roomH * 0.10

        var candidates = wallCandidateHeightBand(points: points, bandMin: bandMin, bandMax: bandMax)

        // Depth grids often cluster in room-Y; a plane-based band can exclude most samples (e.g. 296/2245).
        // If too few pass, use the middle bulk of observed Y (percentiles) so RANSAC still sees walls.
        let targetMin = max(180, Int(ceil(Float(points.count) * 0.18)))
        if candidates.count < targetMin, points.count >= 24 {
            let ys = points.map(\.y).sorted()
            let n = ys.count
            let lo = ys[min(n - 1, max(0, Int(Float(n) * 0.06)))]
            let hi = ys[min(n - 1, max(0, Int(Float(n) * 0.94)))]
            let span = hi - lo
            if span > 1e-3 {
                bandMin = lo + span * 0.03
                bandMax = hi - span * 0.03
                if bandMax > bandMin {
                    let alt = wallCandidateHeightBand(points: points, bandMin: bandMin, bandMax: bandMax)
                    if alt.count > candidates.count {
                        candidates = alt
                        logDebug("📐 [RoomGeometryEngine] wall band widened (Y percentiles) → \(candidates.count) pts (thin plane-band)")
                    }
                }
            }
        }

        guard !candidates.isEmpty else {
            throw GeometryExtractionError.degenerateGeometry("no wall-band candidates")
        }

        var remaining = candidates
        var walls     = [DetectedPlane]()
        let minCount  = max(3, Int(Float(candidates.count) * config.ransacMinInlierFraction))

        while remaining.count >= 3 {
            // Walls: perpendicular to floor normal (not world Y — SHARP rooms can be tilted in YZ).
            guard let plane = ransacFitPlane(
                points:                     remaining,
                isHorizontal:               false,
                referenceHorizontalNormal:  floor.normal
            ) else { break }

            let inliers  = remaining.filter { abs(plane.distance(to: $0)) < config.ransacInlierThreshold }
            let outliers = remaining.filter { abs(plane.distance(to: $0)) >= config.ransacInlierThreshold }

            guard inliers.count >= minCount else { break }

            let candidate: DetectedPlane
            if let refined = refinePlaneFit(points: inliers, referenceHorizontalNormal: floor.normal) {
                candidate = refined
            } else {
                candidate = DetectedPlane(type: .wall, normal: plane.normal, pointOnPlane: plane.pointOnPlane)
            }

            // Diversity gate: skip walls whose normals are too similar to an existing wall,
            // but still drop their inliers so future iterations can discover new orientations.
            let tooSimilar = walls.contains { existing in
                abs(simd_dot(existing.normal, candidate.normal)) > config.deduplicationNormalDotMin
            }
            if tooSimilar {
                remaining = outliers
                continue
            }

            walls.append(candidate)
            remaining = outliers
        }

        let deduped = deduplicatedWalls(walls)
        logDebug("📐 [RoomGeometryEngine] extractWallPlanes — \(deduped.count) walls from \(candidates.count) pts")
        let fn = normalizeOrZero(floor.normal)
        for (idx, w) in deduped.enumerated() {
            let parallelToFloor = abs(simd_dot(w.normal, fn))
            logDebug(
                "📐 [WALL] wall[\(idx)] normal=\(w.normal) |dot(floorN)|=\(String(format: "%.3f", parallelToFloor)) " +
                    "(≈0 ⇒ vertical/room wall vs tilted floor)"
            )
        }
        return deduped
    }

    // MARK: - Corner Detection  (Change 3d)

    /// Detects room corners from pairs of orthogonal wall planes.
    public func findCorners(
        walls:         [DetectedPlane],
        floor:         DetectedPlane,
        sceneToMeters: Float
    ) -> [RoomCorner] {

        var corners = [RoomCorner]()
        let basis = makePlaneBasis(normal: floor.normal)

        for i in 0 ..< walls.count {
            for j in (i + 1) ..< walls.count {
                let w1 = walls[i]
                let w2 = walls[j]

                let orthogonality = abs(simd_dot(w1.normal, w2.normal))
                // ── Change 3d: was < 0.75
                guard orthogonality < config.cornerOrthogonalityMax else { continue }

                guard let xzPoint = solveWallIntersectionXZ(wall1: w1, wall2: w2) else { continue }

                let position = SIMD3<Float>(xzPoint.x, floor.pointOnPlane.y, xzPoint.z)
                let uv = projectToPlaneUV(position, origin: floor.pointOnPlane, u: basis.u, v: basis.v)
                corners.append(RoomCorner(position: position, uv: uv))
            }
        }

        let deduped = deduplicatedCorners(corners, sceneToMeters: sceneToMeters)
        logDebug("📐 [RoomGeometryEngine] findCorners — \(deduped.count) corners from \(walls.count) walls")
        return deduped
    }

    // MARK: - RANSAC Plane Fit  (Change 3a / 3b)

    /// Fits a plane to `points` using RANSAC.
    ///
    /// - Parameters:
    ///   - isHorizontal: `true` → normal nearly parallel to ``referenceHorizontalNormal`` (floor/ceiling vs tilted scenes).
    ///                   `false` → normal nearly perpendicular to that reference (walls).
    ///   - referenceHorizontalNormal: Up direction of the *floor* (unit-ish). `nil` → world +Y (legacy axis-aligned rooms).
    @discardableResult
    public func ransacFitPlane(
        points:                    [SIMD3<Float>],
        isHorizontal:              Bool,
        inlierThreshold:           Float? = nil,
        minInlierCount:            Int?   = nil,
        referenceHorizontalNormal: SIMD3<Float>? = nil
    ) -> DetectedPlane? {

        guard points.count >= 3 else { return nil }

        let thresh    = inlierThreshold ?? config.ransacInlierThreshold
        let minInlier = minInlierCount  ?? max(3, Int(Float(points.count) * config.ransacMinInlierFraction))

        let refUp: SIMD3<Float>
        if let raw = referenceHorizontalNormal {
            let n = normalizeOrZero(raw)
            refUp = simd_length_squared(n) > 1e-12 ? n : SIMD3<Float>(0, 1, 0)
        } else {
            refUp = SIMD3<Float>(0, 1, 0)
        }

        var best:           DetectedPlane?
        var bestInlierCount = 0

        for _ in 0 ..< config.ransacIterations {
            guard let hypothesis = randomPlaneSample(from: points) else { continue }

            let align = abs(simd_dot(hypothesis.normal, refUp))
            if isHorizontal {
                guard align > config.horizontalNormalYMin else { continue }
            } else {
                guard align < config.verticalNormalYMax   else { continue }
            }

            let count = points.filter { abs(hypothesis.distance(to: $0)) < thresh }.count
            if count > bestInlierCount {
                bestInlierCount = count
                best = hypothesis
            }
        }

        guard bestInlierCount >= minInlier, let candidate = best else { return nil }
        return candidate
    }

    // MARK: - Floor-projected room extents (no wall RANSAC)

    public func measureRoomExtentsFromFloor(
        points: [SIMD3<Float>],
        floor: DetectedPlane
    ) -> (width: Float, height: Float, depth: Float) {
        // Reuse RoomSpaceTransform so that extents are measured in the same
        // basis as walls / corners when running in room space.
        let xform = RoomSpaceTransform(floor: floor)
        let roomPoints = xform.transformPoints(points)
        let aabb = computeAABB(points: roomPoints)
        return (
            width:  aabb.size.x,
            height: aabb.size.y,
            depth:  aabb.size.z
        )
    }

    // MARK: - Plane Refinement  (Change 3c)

    /// Refines a plane through PCA on `points`.
    /// Returns nil if the result is degenerate or too parallel to ``referenceHorizontalNormal`` (near-horizontal slab in wall usage).
    public func refinePlaneFit(
        points:                    [SIMD3<Float>],
        referenceHorizontalNormal: SIMD3<Float>? = nil
    ) -> DetectedPlane? {
        guard points.count >= 3 else { return nil }

        let centroid = points.reduce(.zero, +) / Float(points.count)
        let cov      = covarianceMatrix(points: points, centroid: centroid)
        guard let rawNormal = smallestEigenvector(ofSymmetric3x3: cov) else { return nil }
        let normal = normalizeOrZero(rawNormal)

        let refUp: SIMD3<Float>
        if let raw = referenceHorizontalNormal {
            let n = normalizeOrZero(raw)
            refUp = simd_length_squared(n) > 1e-12 ? n : SIMD3<Float>(0, 1, 0)
        } else {
            refUp = SIMD3<Float>(0, 1, 0)
        }
        guard abs(simd_dot(normal, refUp)) < config.verticalNormalYMaxRefined else { return nil }

        return DetectedPlane(type: .wall, normal: normal, pointOnPlane: centroid)
    }

    // MARK: - Random Plane Sample

    private func randomPlaneSample(from points: [SIMD3<Float>]) -> DetectedPlane? {
        guard points.count >= 3 else { return nil }
        var idxSet = Set<Int>()
        while idxSet.count < 3 {
            let r = Int(drand48() * Double(points.count))
            if r >= 0 && r < points.count {
                idxSet.insert(r)
            }
        }
        let pts = idxSet.map { points[$0] }
        let normal = normalizeOrZero(simd_cross(pts[1] - pts[0], pts[2] - pts[0]))
        guard simd_length_squared(normal) > 1e-8 else { return nil }
        return DetectedPlane(type: .wall, normal: normal, pointOnPlane: pts[0])
    }

    // MARK: - Deduplication

    private func deduplicatedWalls(_ walls: [DetectedPlane]) -> [DetectedPlane] {
        var result = [DetectedPlane]()
        for wall in walls {
            let isDupe = result.contains { ex in
                abs(simd_dot(wall.normal, ex.normal)) >= config.deduplicationNormalDotMin &&
                abs(ex.distance(to: wall.pointOnPlane)) < config.deduplicationDistanceTolerance
            }
            if !isDupe { result.append(wall) }
        }
        return result
    }

    private func deduplicatedCorners(
        _ corners:     [RoomCorner],
        sceneToMeters: Float
    ) -> [RoomCorner] {
        let mergeRadius = 0.30 / max(sceneToMeters, 0.0001)  // 30 cm in scene units
        var result = [RoomCorner]()
        for corner in corners {
            let isDupe = result.contains { ex in
                let dx = ex.position.x - corner.position.x
                let dz = ex.position.z - corner.position.z
                return (dx * dx + dz * dz).squareRoot() < mergeRadius
            }
            if !isDupe { result.append(corner) }
        }
        return result
    }

    // MARK: - Wall Candidate Height Band

    /// Returns points whose Y coordinate falls within the expected wall height band.
    private func wallCandidateHeightBand(
        points:  [SIMD3<Float>],
        bandMin: Float,
        bandMax: Float
    ) -> [SIMD3<Float>] {
        guard bandMax >= bandMin else { return [] }
        return points.filter { $0.y >= bandMin && $0.y <= bandMax }
    }

    // MARK: - Wall Intersection  (unchanged rename)

    /// Projects two wall planes onto XZ and returns their 2-D intersection.
    /// Returns nil when the walls are parallel in XZ.
    private func solveWallIntersectionXZ(
        wall1: DetectedPlane,
        wall2: DetectedPlane
    ) -> SIMD3<Float>? {
        let n1 = SIMD2<Float>(wall1.normal.x, wall1.normal.z)
        let n2 = SIMD2<Float>(wall2.normal.x, wall2.normal.z)
        let d1 = simd_dot(n1, SIMD2<Float>(wall1.pointOnPlane.x, wall1.pointOnPlane.z))
        let d2 = simd_dot(n2, SIMD2<Float>(wall2.pointOnPlane.x, wall2.pointOnPlane.z))
        let det = n1.x * n2.y - n1.y * n2.x
        guard abs(det) > 1e-6 else { return nil }
        let x = (d1 * n2.y - d2 * n1.y) / det
        let z = (n1.x * d2 - n2.x * d1) / det
        return SIMD3<Float>(x, 0, z)
    }

    // MARK: - computeFreespace  (Change 5)

    /// Computes free floor region(s).
    ///
    /// **Bounds-based mode** (`config.useBoundsBasedRoomSize == true`):
    ///   Classic AABB-only path — returns one `FreeFloorRegion` with `occupancyRatio: nil`.
    ///
    /// **Plane-based mode** (`config.useBoundsBasedRoomSize == false`):
    ///   Occupancy-grid subtraction: marks cells that contain obstacle point projections
    ///   as occupied, returns free area adjusted for furniture.
    ///
    /// - Note: TODO — connected-component labelling for multiple distinct free regions.
    public func computeFreespace(
        points:        [SIMD3<Float>],
        floor:         DetectedPlane,
        ceiling:       DetectedPlane,
        sceneToMeters: Float
    ) -> [FreeFloorRegion] {

        let basis       = makePlaneBasis(normal: floor.normal)
        let floorPoints = points
            .filter { abs(floor.distance(to: $0)) < config.ransacInlierThreshold * 1.5 }
            .map { projectToPlaneUV($0, origin: floor.pointOnPlane, u: basis.u, v: basis.v) }

        guard !floorPoints.isEmpty else { return [] }

        var minUV = floorPoints[0]
        var maxUV = floorPoints[0]
        for pt in floorPoints.dropFirst() {
            minUV = simd_min(minUV, pt)
            maxUV = simd_max(maxUV, pt)
        }

        let polygon = [
            SIMD2<Float>(minUV.x, minUV.y),
            SIMD2<Float>(maxUV.x, minUV.y),
            SIMD2<Float>(maxUV.x, maxUV.y),
            SIMD2<Float>(minUV.x, maxUV.y)
        ]
        let uvBounds = FloorUVBounds(min: minUV, max: maxUV)

        // ── BOUNDS-BASED path ─────────────────────────────────────────────
        if config.useBoundsBasedRoomSize {
            let sceneArea = max(0, (maxUV.x - minUV.x) * (maxUV.y - minUV.y))
            let areaSqM   = sceneArea * sceneToMeters * sceneToMeters
            logDebug(
                "📐 [RoomGeometryEngine] freeRegion BOUNDS-BASED " +
                "sceneArea=\(String(format: "%.3f", sceneArea)) areaSqM=\(String(format: "%.3f", areaSqM))"
            )
            return [
                FreeFloorRegion(
                    polygon: polygon,
                    areaSqM: areaSqM,
                    uvBounds: uvBounds,
                    occupancyRatio: nil
                ),
            ]
        }

        // ── OCCUPANCY-GRID path ───────────────────────────────────────────
        let cellSizeSU = config.freespaceGridCellMeters / max(sceneToMeters, 0.0001)
        let rangeU     = maxUV.x - minUV.x
        let rangeV     = maxUV.y - minUV.y
        let gridW      = max(1, Int(rangeU / cellSizeSU))
        let gridH      = max(1, Int(rangeV / cellSizeSU))
        let totalCells = gridW * gridH

        // Obstacle points: above floor by at least 3× inlier threshold,
        // below obstacleCeilingHeightFraction of detected ceiling height
        let ceilingHeightSU = max(abs(ceiling.distance(to: floor.pointOnPlane)), 0.05)
        let maxObstacleH = ceilingHeightSU * config.obstacleCeilingHeightFraction
        let obstacleUVs = points.filter {
            let h = floor.distance(to: $0)   // signed distance above floor
            return h > config.ransacInlierThreshold * 3 && h < maxObstacleH
        }.map { projectToPlaneUV($0, origin: floor.pointOnPlane, u: basis.u, v: basis.v) }

        var occupied = [Bool](repeating: false, count: totalCells)
        for uv in obstacleUVs {
            let gx = Int((uv.x - minUV.x) / cellSizeSU)
            let gy = Int((uv.y - minUV.y) / cellSizeSU)
            guard gx >= 0, gx < gridW, gy >= 0, gy < gridH else { continue }
            occupied[gy * gridW + gx] = true
        }

        let occupiedCount  = occupied.filter { $0 }.count
        let freeCount      = totalCells - occupiedCount
        let occupancyRatio = Float(occupiedCount) / Float(max(totalCells, 1))
        let cellAreaSqM    = cellSizeSU * cellSizeSU * sceneToMeters * sceneToMeters
        let freeAreaSqM    = Float(freeCount) * cellAreaSqM
        let totalAreaSqM   = rangeU * rangeV * sceneToMeters * sceneToMeters

        logDebug(
            "📐 [RoomGeometryEngine] freeRegion GRID \(gridW)×\(gridH) " +
            "cellSU=\(String(format: "%.4f", cellSizeSU)) " +
            "totalArea=\(String(format: "%.2f", totalAreaSqM))m² " +
            "freeArea=\(String(format: "%.2f", freeAreaSqM))m² " +
            "occupied=\(String(format: "%.0f", occupancyRatio * 100))% " +
            "obstaclePoints=\(obstacleUVs.count)"
        )

        // TODO: connected-component labelling for multiple distinct free regions
        return [FreeFloorRegion(
            polygon:        polygon,
            areaSqM:        freeAreaSqM,
            uvBounds:       uvBounds,
            occupancyRatio: occupancyRatio
        )]
    }

    // MARK: - AABB

    public func computeAABB(points: [SIMD3<Float>]) -> AABB3 {
        guard !points.isEmpty else { return AABB3(min: .zero, max: .zero) }
        var lo = points[0], hi = points[0]
        for p in points.dropFirst() {
            lo = simd_min(lo, p)
            hi = simd_max(hi, p)
        }
        return AABB3(min: lo, max: hi)
    }

    /// **`roomSpaceHeightSUForScale`:** In room space, floor is **y = 0**. Ceiling-height proxy for scale math
    /// uses the **P97 of Y** over ``roomPoints`` as a **trimmed top** — not always ``roomBounds.max.y`` / raw
    /// AABB high (rare upper outliers are capped; tiny clouds fall back to ``roomBounds.max.y``). Prefer this
    /// over **full AABB Y span** when you need a stable vertical numerator without ceiling RANSAC.
    ///
    /// - Note: Optional helper; pipeline may use detected ceiling plane distance instead when present.
    private func roomSpaceHeightSUForScale(roomPoints: [SIMD3<Float>], roomBounds: AABB3) -> Float {
        let maxY = roomBounds.max.y
        guard maxY > 1e-4 else { return 1e-4 }
        guard roomPoints.count >= 6 else { return maxY }
        let ys = roomPoints.map(\.y).sorted()
        let n = ys.count
        let iHi = min(n - 1, max(0, Int(Float(n - 1) * 0.97)))
        let p97Top = ys[iHi]
        return max(min(p97Top, maxY), 1e-4)
    }

    // MARK: - UV Projection Helpers

    public struct PlaneBasis {
        public let u: SIMD3<Float>
        public let v: SIMD3<Float>
    }

    /// Returns an orthonormal UV basis tangent to a plane with the given normal.
    public func makePlaneBasis(normal: SIMD3<Float>) -> PlaneBasis {
        let ref: SIMD3<Float> = abs(normal.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
        let u = normalizeOrZero(simd_cross(ref, normal))
        let v = simd_cross(normal, u)
        return PlaneBasis(u: u, v: v)
    }

    /// Projects a 3-D point onto the UV plane defined by `origin`, `u`, and `v`.
    public func projectToPlaneUV(
        _ point:  SIMD3<Float>,
        origin:   SIMD3<Float>,
        u:        SIMD3<Float>,
        v:        SIMD3<Float>
    ) -> SIMD2<Float> {
        let d = point - origin
        return SIMD2<Float>(simd_dot(d, u), simd_dot(d, v))
    }

    // MARK: - Covariance Matrix  (outer-product form)

    /// Computes the 3×3 covariance matrix via outer-product accumulation.
    public func covarianceMatrix(
        points:   [SIMD3<Float>],
        centroid: SIMD3<Float>
    ) -> simd_float3x3 {
        let n = Float(max(points.count, 1))
        return points.reduce(simd_float3x3(0)) { acc, p in
            let d = p - centroid
            // Column-major outer product d ⊗ d
            return acc + simd_float3x3(
                SIMD3<Float>(d.x * d.x, d.y * d.x, d.z * d.x),
                SIMD3<Float>(d.x * d.y, d.y * d.y, d.z * d.y),
                SIMD3<Float>(d.x * d.z, d.y * d.z, d.z * d.z)
            )
        } * (1.0 / n)
    }

    // MARK: - RMS Plane Error

    public func rmsPlaneError(points: [SIMD3<Float>], plane: DetectedPlane) -> Float {
        guard !points.isEmpty else { return 0 }
        let sumSq = points.reduce(Float(0)) { acc, p in
            let d = plane.distance(to: p)
            return acc + d * d
        }
        return (sumSq / Float(points.count)).squareRoot()
    }
}

// MARK: - AABB3 extension

extension AABB3 {
    /// Returns a copy of the box expanded uniformly by `amount` on every face.
    public func expanded(by amount: Float) -> AABB3 {
        AABB3(
            min: min - SIMD3<Float>(repeating: amount),
            max: max + SIMD3<Float>(repeating: amount)
        )
    }
}
