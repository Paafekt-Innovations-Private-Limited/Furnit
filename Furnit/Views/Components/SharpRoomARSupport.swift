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
    let ceilingAnchorCount: Int
    let totalPlaneCount: Int
    /// True when no floor was found (W×D come from wall/other plane corners only).
    let usedWallOnlyFallback: Bool

    var sourceLabel: String {
        if ceilingAnchorCount > 0 && floorAnchorCount > 0 {
            return "ARKit_floor+ceiling+walls_m_W×H×D"
        }
        if floorAnchorCount > 0 && wallAnchorCount > 0 {
            return "ARKit_floor+walls_m_W×H×D"
        }
        if floorAnchorCount > 0 {
            return "ARKit_floor_only_m_W×H×D"
        }
        if wallAnchorCount > 0 {
            return "ARKit_walls_only_m_W×H×D"
        }
        return "ARKit_planes_m_W×H×D"
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

        let ceilingCount = planeAnchorsByID.values.filter({ $0.classification == .ceiling }).count
        let totalDetected = floorCount + wallCount + ceilingCount

        if let trackingState = lastTrackingState {
            switch trackingState {
            case .notAvailable:
                publishScanGuidance("Point phone at the REAL room around you — move slowly side-to-side")
                return
            case .limited(let reason):
                switch reason {
                case .initializing:
                    publishScanGuidance("Point at real floor/walls (not the screen). Slowly move side-to-side")
                    return
                case .insufficientFeatures:
                    publishScanGuidance("Need more texture — aim at floor edges, furniture, corners. Move phone side-to-side slowly")
                    return
                case .excessiveMotion:
                    publishScanGuidance("Too fast — hold steadier for a moment, then resume slow sweep")
                    return
                case .relocalizing:
                    publishScanGuidance("Return to the same view for AR relocalization")
                    return
                @unknown default:
                    publishScanGuidance("Move phone slowly side-to-side to help AR stabilize")
                    return
                }
            case .normal:
                break
            }
        }

        if totalDetected == 0 {
            publishScanGuidance("Aim camera at floor-wall junction. Sweep left-right slowly")
        } else if floorCount == 0 && wallCount > 0 {
            publishScanGuidance("Wall found (\(wallCount)). Tilt down to show the floor")
        } else if floorCount > 0 && wallCount == 0 {
            publishScanGuidance("Floor found (\(floorCount)). Lift up to show a wall")
        } else if ceilingCount > 0 {
            publishScanGuidance("Floor+wall+ceiling detected (\(totalDetected) planes). Hold or keep sweeping for accuracy")
        } else {
            publishScanGuidance("Floor+wall detected (\(totalDetected) planes). Tilt up for ceiling, or keep sweeping")
        }
    }

    // MARK: - World-space plane corner projection

    /// Returns the four world-space corners of an `ARPlaneAnchor`'s estimated rectangle.
    private func worldCorners(of plane: ARPlaneAnchor) -> [SIMD3<Float>] {
        let t = plane.transform
        let halfW = plane.extent.x * 0.5
        let halfD = plane.extent.z * 0.5
        let localCorners: [SIMD4<Float>] = [
            SIMD4<Float>(-halfW, 0, -halfD, 1),
            SIMD4<Float>( halfW, 0, -halfD, 1),
            SIMD4<Float>(-halfW, 0,  halfD, 1),
            SIMD4<Float>( halfW, 0,  halfD, 1)
        ]
        return localCorners.map { local in
            let w = t * local
            return SIMD3<Float>(w.x, w.y, w.z)
        }
    }

    /// Returns the world-space center of an `ARPlaneAnchor`.
    private func worldCenter(of plane: ARPlaneAnchor) -> SIMD3<Float> {
        let c = plane.transform * SIMD4<Float>(plane.center.x, plane.center.y, plane.center.z, 1)
        return SIMD3<Float>(c.x, c.y, c.z)
    }

    /// Gravity-aligned vertical span of a wall anchor (projects local axes onto world up).
    private func wallVerticalSpan(_ plane: ARPlaneAnchor) -> Float {
        let worldUp = SIMD3<Float>(0, 1, 0)
        let t = plane.transform
        let e = plane.extent
        let axes: [(SIMD3<Float>, Float)] = [
            (SIMD3<Float>(t.columns.0.x, t.columns.0.y, t.columns.0.z), e.x),
            (SIMD3<Float>(t.columns.1.x, t.columns.1.y, t.columns.1.z), e.y),
            (SIMD3<Float>(t.columns.2.x, t.columns.2.y, t.columns.2.z), e.z)
        ]
        var maxV: Float = 0
        for (axisVec, length) in axes {
            guard length.isFinite, length > 1e-5 else { continue }
            let len = simd_length(axisVec)
            guard len > 1e-5 else { continue }
            let v = abs(simd_dot(axisVec / len, worldUp)) * length
            maxV = max(maxV, v)
        }
        if maxV < 0.01 {
            return max(plane.extent.x, max(plane.extent.y, plane.extent.z))
        }
        return maxV
    }

    /// Approximate wall normal in world XZ (horizontal direction the wall faces).
    private func wallWorldNormal(_ plane: ARPlaneAnchor) -> SIMD3<Float> {
        let t = plane.transform
        let localNormal = SIMD4<Float>(0, 1, 0, 0)
        let wn = t * localNormal
        var n = SIMD3<Float>(wn.x, 0, wn.z)
        let len = simd_length(n)
        if len > 1e-5 { n /= len } else { n = SIMD3<Float>(0, 0, 1) }
        return n
    }

    /// Human-readable cardinal direction from a wall's world normal.
    private func wallDirection(_ plane: ARPlaneAnchor) -> String {
        let n = wallWorldNormal(plane)
        let absX = abs(n.x)
        let absZ = abs(n.z)
        if absX > absZ {
            return n.x > 0 ? "facing+X(right)" : "facing-X(left)"
        } else {
            return n.z > 0 ? "facing+Z(back)" : "facing-Z(front)"
        }
    }

    // MARK: - Collated room measurement from ALL planes

    private func recomputePlaneRoomMeasurement() {
        let allPlanes = Array(planeAnchorsByID.values)
        let floorPlanes = allPlanes.filter(isFloorPlane)
        let wallPlanes = allPlanes.filter(isWallPlane)
        let ceilingPlanes = allPlanes.filter { $0.classification == .ceiling }
        refreshScanGuidance()

        guard !allPlanes.isEmpty else {
            logDebug(
                "🟡 [AR_MEASURE] no planes at all (total_anchors=\(planeAnchorsByID.count))"
            )
            publishPlaneRoomMeasurement(nil)
            return
        }

        var worldMinX: Float =  .greatestFiniteMagnitude
        var worldMaxX: Float = -.greatestFiniteMagnitude
        var worldMinY: Float =  .greatestFiniteMagnitude
        var worldMaxY: Float = -.greatestFiniteMagnitude
        var worldMinZ: Float =  .greatestFiniteMagnitude
        var worldMaxZ: Float = -.greatestFiniteMagnitude

        for plane in allPlanes {
            for corner in worldCorners(of: plane) {
                worldMinX = min(worldMinX, corner.x)
                worldMaxX = max(worldMaxX, corner.x)
                worldMinY = min(worldMinY, corner.y)
                worldMaxY = max(worldMaxY, corner.y)
                worldMinZ = min(worldMinZ, corner.z)
                worldMaxZ = max(worldMaxZ, corner.z)
            }
        }

        let hasAnyExtent = worldMinX < worldMaxX || worldMinY < worldMaxY || worldMinZ < worldMaxZ

        guard hasAnyExtent else {
            logDebug(
                "🟡 [AR_MEASURE] planes found but no usable world extent " +
                    "(floors=\(floorPlanes.count) walls=\(wallPlanes.count) ceilings=\(ceilingPlanes.count) total=\(allPlanes.count))"
            )
            publishPlaneRoomMeasurement(nil)
            return
        }

        let spanX = worldMaxX - worldMinX
        let spanY = worldMaxY - worldMinY
        let spanZ = worldMaxZ - worldMinZ

        let wallHeights = wallPlanes.map { wallVerticalSpan($0) }
        let bestWallVert = wallHeights.max() ?? 0

        let widthMeters  = max(spanX, spanZ)
        let depthMeters  = min(spanX, spanZ)
        let heightMeters = max(spanY, bestWallVert)

        let planeDetails = allPlanes.map { plane -> String in
            let kind: String
            switch plane.classification {
            case .floor:   kind = "floor"
            case .wall:    kind = "wall"
            case .ceiling: kind = "ceiling"
            case .table:   kind = "table"
            case .seat:    kind = "seat"
            case .door:    kind = "door"
            case .window:  kind = "window"
            case .none:    kind = plane.alignment == .horizontal ? "horiz?" : "vert?"
            @unknown default: kind = "unk"
            }
            let center = worldCenter(of: plane)
            let ext = String(format: "%.2f×%.2f", plane.extent.x, plane.extent.z)
            let cStr = String(format: "(%.2f,%.2f,%.2f)", center.x, center.y, center.z)
            if isWallPlane(plane) {
                let dir = wallDirection(plane)
                let vSpan = String(format: "%.2f", wallVerticalSpan(plane))
                return "\(kind)[\(ext) v=\(vSpan)m \(dir) @\(cStr)]"
            }
            return "\(kind)[\(ext) @\(cStr)]"
        }

        logDebug(
            "🟢 [AR_MEASURE] collated room W×H×D=" +
                "\(String(format: "%.3f", widthMeters))×" +
                "\(String(format: "%.3f", heightMeters))×" +
                "\(String(format: "%.3f", depthMeters))m " +
                "bbox X[\(String(format: "%.2f", worldMinX)),\(String(format: "%.2f", worldMaxX))] " +
                "Y[\(String(format: "%.2f", worldMinY)),\(String(format: "%.2f", worldMaxY))] " +
                "Z[\(String(format: "%.2f", worldMinZ)),\(String(format: "%.2f", worldMaxZ))] " +
                "floors=\(floorPlanes.count) walls=\(wallPlanes.count) ceilings=\(ceilingPlanes.count) total=\(allPlanes.count) " +
                "planes=[\(planeDetails.joined(separator: "; "))]"
        )

        let measurement = ARPlaneRoomMeasurement(
            widthMeters: widthMeters,
            heightMeters: heightMeters,
            depthMeters: depthMeters,
            floorAnchorCount: floorPlanes.count,
            wallAnchorCount: wallPlanes.count,
            ceilingAnchorCount: ceilingPlanes.count,
            totalPlaneCount: allPlanes.count,
            usedWallOnlyFallback: floorPlanes.isEmpty && !wallPlanes.isEmpty
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
            let vSpan = wallVerticalSpan(plane)
            let dir = wallDirection(plane)
            return "\(kind) ext=\(ext) vert=\(String(format: "%.2f", vSpan))m \(dir)"
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
