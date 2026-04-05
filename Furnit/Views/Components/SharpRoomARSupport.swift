import Foundation
import ARKit
import UIKit
import simd

struct ARPlaneRoomMeasurement: Equatable, Sendable {
    let widthMeters: Float
    let heightMeters: Float
    let depthMeters: Float
    let floorAnchorCount: Int
    let wallAnchorCount: Int
    let usedDefaultCeilingHeight: Bool
    /// True when no horizontal floor was found; W×D use wall patch + default depth (ARKit never saw the floor).
    let usedWallOnlyFallback: Bool

    var sourceLabel: String {
        if usedWallOnlyFallback {
            return "ARKit_plane_wall_only_m_W×H×D"
        }
        if usedDefaultCeilingHeight {
            return "ARKit_plane_floor_only_m_W×H×D"
        }
        return "ARKit_plane_detection_m_W×H×D"
    }
}

struct SharpRoomFurnitureItem: Identifiable, Equatable {
    let id = UUID()
    let category: String
    let dimensions: SIMD3<Float>
    let tint: UIColor
}

struct SharpRoomPlacedFurniture: Identifiable {
    let id: UUID
    let item: SharpRoomFurnitureItem
    var position: SIMD3<Float>
    var rotationY: Float
    var fits: Bool
    var clearanceMeters: Float
}

enum SharpRoomFurnitureCatalog {
    static let standardItems: [SharpRoomFurnitureItem] = [
        SharpRoomFurnitureItem(category: "Sofa", dimensions: SIMD3<Float>(1.85, 0.85, 0.90), tint: .systemGreen),
        SharpRoomFurnitureItem(category: "Bed", dimensions: SIMD3<Float>(2.00, 0.60, 1.50), tint: .systemBlue),
        SharpRoomFurnitureItem(category: "Table", dimensions: SIMD3<Float>(1.50, 0.75, 0.90), tint: .systemOrange),
        SharpRoomFurnitureItem(category: "Chair", dimensions: SIMD3<Float>(0.50, 0.85, 0.50), tint: .systemTeal),
        SharpRoomFurnitureItem(category: "Wardrobe", dimensions: SIMD3<Float>(1.50, 2.00, 0.60), tint: .systemPurple),
        SharpRoomFurnitureItem(category: "Desk", dimensions: SIMD3<Float>(1.20, 0.75, 0.60), tint: .systemPink),
    ]
}

final class ARMotionTracker: NSObject, ARSessionDelegate {
    private var session: ARSession?
    private let sessionDelegateQueue = DispatchQueue(label: "com.furnit.sharproom.ar-motion", qos: .userInitiated)
    var initialTransform: simd_float4x4?
    var onRelativePoseUpdate: ((simd_float4x4) -> Void)?
    var onTrackingStatus: ((String) -> Void)?
    var onScanGuidanceUpdate: ((String) -> Void)?
    var onPlaneRoomMeasurementUpdate: ((ARPlaneRoomMeasurement?) -> Void)?
    /// Debug: log at most once per second while waiting for `.normal` before first reference.
    private var lastLimitedInitialLogTime: CFAbsoluteTime = 0
    private var planeAnchorsByID: [UUID: ARPlaneAnchor] = [:]
    private var lastPublishedPlaneMeasurement: ARPlaneRoomMeasurement?
    private var lastTrackingState: ARCamera.TrackingState?
    private var lastPublishedScanGuidance: String?

    /// Sharp Live Room: 6DoF pose plus ARKit plane detection.
    static func makeLiveRoomConfiguration() -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.planeDetection = [.horizontal, .vertical]
        config.isLightEstimationEnabled = false
        return config
    }

    private func ensureSession() -> ARSession {
        if let session {
            return session
        }
        let newSession = ARSession()
        newSession.delegate = self
        newSession.delegateQueue = sessionDelegateQueue
        session = newSession
        return newSession
    }

    func start() {
        guard ARWorldTrackingConfiguration.isSupported else {
            onTrackingStatus?("AR motion tracking unavailable on this device")
            logDebug("❌ [ARMotionTracker] world tracking unsupported")
            return
        }
        let config = Self.makeLiveRoomConfiguration()
        let session = ensureSession()
        initialTransform = nil
        lastLimitedInitialLogTime = 0
        planeAnchorsByID.removeAll()
        lastTrackingState = nil
        lastPublishedScanGuidance = nil
        publishPlaneRoomMeasurement(nil)
        CameraOwnershipDiagnostics.log(owner: "ARMotionTracker", event: "ar_run", details: "reason=start")
        logDebug("🚀 [ARMotionTracker] starting Live Room AR session worldAlignment=gravity planeDetection=horizontal+vertical frameSemantics=none")
        logMemorySnapshot("ARMotionTracker.start", details: "phase=before_session_run")
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        logMemorySnapshot("ARMotionTracker.start", details: "phase=after_session_run")
    }

    func stop() {
        CameraOwnershipDiagnostics.log(owner: "ARMotionTracker", event: "ar_pause", details: "reason=stop")
        logDebug("🛑 [ARMotionTracker] stopping tracking-only AR session")
        session?.pause()
        initialTransform = nil
        planeAnchorsByID.removeAll()
        lastTrackingState = nil
        lastPublishedScanGuidance = nil
        publishPlaneRoomMeasurement(nil)
    }

    /// Pauses frame delivery so ARKit stops flooding the main queue (SwiftUI alerts / `TextField` stay responsive).
    func pauseForModal() {
        CameraOwnershipDiagnostics.log(owner: "ARMotionTracker", event: "ar_pause", details: "reason=modal_or_camera_handoff")
        session?.pause()
        logDebug("⏸️ [ARMotionTracker] session paused for modal UI")
    }

    /// Resumes after ``pauseForModal`` without resetting the world map (pair with Sharp Room modal dismiss).
    func resumeAfterModal() {
        guard ARWorldTrackingConfiguration.isSupported else { return }
        let config = Self.makeLiveRoomConfiguration()
        let session = ensureSession()
        CameraOwnershipDiagnostics.log(owner: "ARMotionTracker", event: "ar_run", details: "reason=resume")
        session.run(config, options: [])
        logDebug("▶️ [ARMotionTracker] session resumed after modal UI")
    }

    deinit {
        CameraOwnershipDiagnostics.log(owner: "ARMotionTracker", event: "deinit")
    }

    /// Clears the stored reference pose so the next `didUpdate` uses the current device pose as identity (Sharp Room recenter).
    func resetReferencePose() {
        initialTransform = nil
        // Do **not** set `lastLimitedInitialLogTime = 0`: `now - 0` is huge, so the defer log throttle
        // would pass on every frame while waiting for `.normal`. Refresh the window from "now" instead.
        lastLimitedInitialLogTime = CFAbsoluteTimeGetCurrent()
        logDebug("📍 [ARMotionTracker] reference pose cleared — next frame becomes new origin (after .normal)")
    }

    private func isFloorPlane(_ plane: ARPlaneAnchor) -> Bool {
        switch plane.classification {
        case .floor:
            return true
        case .none:
            return plane.alignment == .horizontal
        default:
            return false
        }
    }

    private func isWallPlane(_ plane: ARPlaneAnchor) -> Bool {
        switch plane.classification {
        case .wall:
            return true
        case .none:
            return plane.alignment == .vertical
        default:
            return false
        }
    }

    private func publishPlaneRoomMeasurement(_ measurement: ARPlaneRoomMeasurement?) {
        guard lastPublishedPlaneMeasurement != measurement else { return }
        lastPublishedPlaneMeasurement = measurement
        DispatchQueue.main.async { [weak self] in
            self?.onPlaneRoomMeasurementUpdate?(measurement)
        }
    }

    private func publishScanGuidance(_ guidance: String) {
        guard lastPublishedScanGuidance != guidance else { return }
        lastPublishedScanGuidance = guidance
        DispatchQueue.main.async { [weak self] in
            self?.onScanGuidanceUpdate?(guidance)
        }
        logDebug("🟣 [AR_GUIDE] \(guidance)")
    }

    private func refreshScanGuidance() {
        let floorCount = planeAnchorsByID.values.filter(isFloorPlane).count
        let wallCount = planeAnchorsByID.values.filter(isWallPlane).count

        if let trackingState = lastTrackingState {
            switch trackingState {
            case .notAvailable:
                publishScanGuidance("Move phone slowly to start AR")
                return
            case .limited(let reason):
                switch reason {
                case .initializing:
                    publishScanGuidance("Move phone slowly while AR initializes")
                    return
                case .insufficientFeatures:
                    publishScanGuidance("Aim at floor-wall edges with more texture and light")
                    return
                case .excessiveMotion:
                    publishScanGuidance("Hold steadier for a moment")
                    return
                case .relocalizing:
                    publishScanGuidance("Return to the same view for AR relocalization")
                    return
                @unknown default:
                    publishScanGuidance("Move phone slowly to help AR stabilize")
                    return
                }
            case .normal:
                break
            }
        }

        if floorCount == 0 && wallCount == 0 {
            publishScanGuidance("Tilt down and step back so floor and wall are both visible")
        } else if floorCount == 0 && wallCount > 0 {
            publishScanGuidance("Good wall found. Tilt down until floor enters the frame")
        } else if floorCount > 0 && wallCount == 0 {
            publishScanGuidance("Good floor found. Lift up slightly to include a wall")
        } else {
            publishScanGuidance("Good coverage. Sweep left and right slowly, then hold still")
        }
    }

    private static let defaultCeilingHeightMeters: Float = 2.44
    /// When only a wall patch is visible (no floor), depth has no AR ground truth — conservative default room depth.
    private static let defaultRoomDepthWallOnlyMeters: Float = 3.0

    /// For vertical planes, ARKit often stores the tall dimension in `extent.x` or `extent.z`, not `.y`.
    /// Project each local axis onto gravity to recover vertical and horizontal spans in metres.
    private func gravityAlignedWallSpansMeters(_ plane: ARPlaneAnchor) -> (vertical: Float, horizontal: Float) {
        let worldUp = SIMD3<Float>(0, 1, 0)
        let t = plane.transform
        let e = plane.extent
        let axisVectorsAndLengths: [(SIMD3<Float>, Float)] = [
            (SIMD3<Float>(t.columns.0.x, t.columns.0.y, t.columns.0.z), e.x),
            (SIMD3<Float>(t.columns.1.x, t.columns.1.y, t.columns.1.z), e.y),
            (SIMD3<Float>(t.columns.2.x, t.columns.2.y, t.columns.2.z), e.z)
        ]
        var maxVertical: Float = 0
        var maxHorizontal: Float = 0
        for (axisVec, length) in axisVectorsAndLengths {
            guard length.isFinite, length > 1e-5 else { continue }
            let lenAxis = simd_length(axisVec)
            guard lenAxis > 1e-5 else { continue }
            let d = axisVec / lenAxis
            let vertical = abs(simd_dot(d, worldUp)) * length
            let cosUp = simd_dot(d, worldUp)
            let sinThetaSq = max(0, 1 - cosUp * cosUp)
            let horizontal = sqrt(sinThetaSq) * length
            maxVertical = max(maxVertical, vertical)
            maxHorizontal = max(maxHorizontal, horizontal)
        }
        if maxVertical < 0.2 || maxHorizontal < 0.2 {
            let sorted = [plane.extent.x, plane.extent.y, plane.extent.z].sorted()
            let thin = sorted[0], mid = sorted[1], thick = sorted[2]
            _ = thin
            return (thick, mid)
        }
        return (maxVertical, maxHorizontal)
    }

    private func bestWallHeightFromPlanes(_ wallPlanes: [ARPlaneAnchor]) -> Float {
        wallPlanes.map { gravityAlignedWallSpansMeters($0).vertical }.max() ?? 0
    }

    private func wallPatchAreaScore(_ plane: ARPlaneAnchor) -> Float {
        let spans = gravityAlignedWallSpansMeters(plane)
        return spans.vertical * spans.horizontal
    }

    private func recomputePlaneRoomMeasurement() {
        let floorPlanes = planeAnchorsByID.values.filter(isFloorPlane)
        let wallPlanes = planeAnchorsByID.values.filter(isWallPlane)
        refreshScanGuidance()

        let bestFloor = floorPlanes.max(by: {
            ($0.extent.x * $0.extent.z) < ($1.extent.x * $1.extent.z)
        })

        let wallVerticalSpans = wallPlanes.map { gravityAlignedWallSpansMeters($0).vertical }
        let bestWallHeight = wallVerticalSpans.max()

        if let bestFloor {
            let floorArea = bestFloor.extent.x * bestFloor.extent.z
            let hasQualifyingWall = (bestWallHeight ?? 0) > 0.3
            let usedDefaultCeiling = !hasQualifyingWall
            let heightMeters = hasQualifyingWall ? bestWallHeight! : Self.defaultCeilingHeightMeters

            if !hasQualifyingWall {
                logDebug(
                    "🟡 [AR_MEASURE] floor found but no qualifying wall (best_wall_vert_gravity=\(bestWallHeight.map { String(format: "%.2f", $0) } ?? "none"), " +
                        "threshold=0.30m) — using default ceiling \(String(format: "%.2f", Self.defaultCeilingHeightMeters))m " +
                        "(floors=\(floorPlanes.count) walls=\(wallPlanes.count) floor_area=\(String(format: "%.2f", floorArea))m² " +
                        "wall_vert_spans=\(wallVerticalSpans.map { String(format: "%.2f", $0) }))"
                )
            }

            let measurement = ARPlaneRoomMeasurement(
                widthMeters: bestFloor.extent.x,
                heightMeters: heightMeters,
                depthMeters: bestFloor.extent.z,
                floorAnchorCount: floorPlanes.count,
                wallAnchorCount: hasQualifyingWall ? wallPlanes.count : 0,
                usedDefaultCeilingHeight: usedDefaultCeiling,
                usedWallOnlyFallback: false
            )
            logDebug(
                "🟢 [AR_MEASURE] room from planes W×H×D=" +
                    "\(String(format: "%.3f", measurement.widthMeters))×" +
                    "\(String(format: "%.3f", measurement.heightMeters))×" +
                    "\(String(format: "%.3f", measurement.depthMeters))m " +
                    "floors=\(measurement.floorAnchorCount) walls=\(measurement.wallAnchorCount) " +
                    "default_ceiling=\(usedDefaultCeiling) wall_only=false"
            )
            publishPlaneRoomMeasurement(measurement)
            return
        }

        guard let bestWall = wallPlanes.max(by: { wallPatchAreaScore($0) < wallPatchAreaScore($1) }) else {
            logDebug(
                "🟡 [AR_MEASURE] gate: no floor and no wall planes " +
                    "(total_anchors=\(planeAnchorsByID.count) floors=\(floorPlanes.count) walls=\(wallPlanes.count))"
            )
            publishPlaneRoomMeasurement(nil)
            return
        }

        let spans = gravityAlignedWallSpansMeters(bestWall)
        let wallVert = spans.vertical
        let wallHoriz = spans.horizontal
        let hasQualifyingWallHeight = wallVert > 0.3
        let heightMeters = hasQualifyingWallHeight ? wallVert : Self.defaultCeilingHeightMeters
        let widthMeters = max(wallHoriz, 1.0)
        let depthMeters = Self.defaultRoomDepthWallOnlyMeters

        logDebug(
            "🟡 [AR_MEASURE] no floor plane — wall-only fallback " +
                "(walls=\(wallPlanes.count) patch_gravity_v×h=\(String(format: "%.2f", wallVert))×\(String(format: "%.2f", wallHoriz))m " +
                "extent_xyz=\(String(format: "%.2f", bestWall.extent.x))×\(String(format: "%.2f", bestWall.extent.y))×\(String(format: "%.2f", bestWall.extent.z)) " +
                "default_depth=\(String(format: "%.2f", depthMeters))m tip=tilt_down_to_find_floor)"
        )

        let measurement = ARPlaneRoomMeasurement(
            widthMeters: widthMeters,
            heightMeters: heightMeters,
            depthMeters: depthMeters,
            floorAnchorCount: 0,
            wallAnchorCount: wallPlanes.count,
            usedDefaultCeilingHeight: !hasQualifyingWallHeight,
            usedWallOnlyFallback: true
        )
        logDebug(
            "🟢 [AR_MEASURE] room wall-only W×H×D=" +
                "\(String(format: "%.3f", measurement.widthMeters))×" +
                "\(String(format: "%.3f", measurement.heightMeters))×" +
                "\(String(format: "%.3f", measurement.depthMeters))m " +
                "walls=\(measurement.wallAnchorCount) default_ceiling=\(!hasQualifyingWallHeight) wall_only=true"
        )
        publishPlaneRoomMeasurement(measurement)
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Copy pose state immediately; never retain `ARFrame` past this method (no async captures, no stored frame).
        let currentTransform = frame.camera.transform
        let trackingState = frame.camera.trackingState
        if initialTransform == nil {
            // Sharp Live Room uses pose as a motion hint over the splat; IMU-backed poses are still useful
            // when visual tracking is `.limited` (e.g. phone aimed at the on-screen room). Only `.notAvailable`
            // has no usable 6DoF.
            if case .notAvailable = trackingState {
                let now = CFAbsoluteTimeGetCurrent()
                if now - lastLimitedInitialLogTime > 1.0 {
                    lastLimitedInitialLogTime = now
                    logDebug(
                        "📍 [ARMotionTracker] waiting for pose — trackingState=\(trackingState)"
                    )
                }
                return
            }
            initialTransform = currentTransform
            logDebug("📍 [ARMotionTracker] captured initial camera transform (trackingState=\(trackingState))")
        }
        guard let initialTransform else { return }
        // Full device rotation (tilt to look at furniture). Position: floor height locked to reference Y,
        // and horizontal motion projected onto opening **forward** (XZ) so walking reads as pinch-style
        // dolly into/out of the scene — not lateral “one-finger pan” strafe from sideways steps.
        let p0 = SIMD3<Float>(
            initialTransform.columns.3.x,
            initialTransform.columns.3.y,
            initialTransform.columns.3.z
        )
        let p1 = SIMD3<Float>(
            currentTransform.columns.3.x,
            currentTransform.columns.3.y,
            currentTransform.columns.3.z
        )
        let deltaH = SIMD3<Float>(p1.x - p0.x, 0, p1.z - p0.z)
        var forwardXZ = SIMD3<Float>(
            -initialTransform.columns.2.x,
            0,
            -initialTransform.columns.2.z
        )
        let forwardLen = simd_length(forwardXZ)
        if forwardLen < 1e-4 {
            forwardXZ = SIMD3<Float>(0, 0, -1)
        } else {
            forwardXZ /= forwardLen
        }
        let alongForward = simd_dot(deltaH, forwardXZ)
        let dollyH = forwardXZ * alongForward
        let pSynth = SIMD3<Float>(p0.x + dollyH.x, p0.y, p0.z + dollyH.z)

        var floorPlaneCamera = currentTransform
        floorPlaneCamera.columns.3 = SIMD4<Float>(pSynth.x, pSynth.y, pSynth.z, 1)
        let relativeTransform = simd_mul(simd_inverse(initialTransform), floorPlaneCamera)
        // Always hop to main: never run consumer work synchronously in `didUpdate` — a slow callback
        // retains many `ARFrame`s and triggers "delegate is retaining N ARFrames" from ARKit.
        DispatchQueue.main.async { [weak self] in
            self?.onRelativePoseUpdate?(relativeTransform)
        }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        lastTrackingState = camera.trackingState
        let status: String
        let detailReason: String
        switch camera.trackingState {
        case .normal:
            status = "AR tracking normal"
            detailReason = "normal"
        case .notAvailable:
            status = "AR tracking unavailable"
            detailReason = "notAvailable"
        case .limited(let reason):
            status = "AR tracking limited"
            switch reason {
            case .initializing:
                detailReason = "limited(initializing)"
            case .excessiveMotion:
                detailReason = "limited(excessiveMotion)"
            case .insufficientFeatures:
                detailReason = "limited(insufficientFeatures)"
            case .relocalizing:
                detailReason = "limited(relocalizing)"
            @unknown default:
                detailReason = "limited(unknown)"
            }
        }
        let planeCount = planeAnchorsByID.count
        let floorCount = planeAnchorsByID.values.filter(isFloorPlane).count
        let wallCount = planeAnchorsByID.values.filter(isWallPlane).count
        onTrackingStatus?(status)
        logDebug(
            "📷 [ARMotionTracker] \(detailReason) planes=\(planeCount)(f:\(floorCount) w:\(wallCount))"
        )
        refreshScanGuidance()
    }

    private func planeLabel(_ plane: ARPlaneAnchor) -> String {
        let kind: String
        switch plane.classification {
        case .floor:   kind = "floor"
        case .wall:    kind = "wall"
        case .ceiling: kind = "ceiling"
        case .table:   kind = "table"
        case .seat:    kind = "seat"
        case .door:    kind = "door"
        case .window:  kind = "window"
        case .none:    kind = plane.alignment == .horizontal ? "horiz" : "vert"
        @unknown default: kind = "unknown"
        }
        let ext = String(format: "%.2f×%.2f×%.2f", plane.extent.x, plane.extent.y, plane.extent.z)
        if isWallPlane(plane) {
            let g = gravityAlignedWallSpansMeters(plane)
            let grav = String(format: "grav_v×h=%.2f×%.2f", g.vertical, g.horizontal)
            return "\(kind) ext=\(ext) \(grav)"
        }
        return "\(kind) \(ext)"
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        var changed = false
        for anchor in anchors {
            guard let plane = anchor as? ARPlaneAnchor else { continue }
            planeAnchorsByID[plane.identifier] = plane
            changed = true
            logDebug("🔵 [AR_PLANE] +added \(planeLabel(plane)) id=\(plane.identifier.uuidString.prefix(8)) total=\(planeAnchorsByID.count)")
        }
        if changed {
            recomputePlaneRoomMeasurement()
        }
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        var changed = false
        for anchor in anchors {
            guard let plane = anchor as? ARPlaneAnchor else { continue }
            planeAnchorsByID[plane.identifier] = plane
            changed = true
        }
        if changed {
            let floorCount = planeAnchorsByID.values.filter(isFloorPlane).count
            let wallCount = planeAnchorsByID.values.filter(isWallPlane).count
            logDebug("🔵 [AR_PLANE] ~updated total=\(planeAnchorsByID.count) floors=\(floorCount) walls=\(wallCount)")
            recomputePlaneRoomMeasurement()
        }
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        var changed = false
        for anchor in anchors {
            guard anchor is ARPlaneAnchor else { continue }
            planeAnchorsByID.removeValue(forKey: anchor.identifier)
            changed = true
            logDebug("🔴 [AR_PLANE] -removed id=\(anchor.identifier.uuidString.prefix(8)) total=\(planeAnchorsByID.count)")
        }
        if changed {
            recomputePlaneRoomMeasurement()
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        let message = "AR tracking failed: \(error.localizedDescription)"
        onTrackingStatus?(message)
        logDebug("🔴🔴🔴 [ARMotionTracker] ❌ SESSION FAILED: \(message)")
    }
}
