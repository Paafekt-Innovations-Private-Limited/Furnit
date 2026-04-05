import SwiftUI
import UIKit
import simd

/// Bridges ``GaussianSplatView.Coordinator`` so SwiftUI parents (e.g. ``SharpRoomView``) can call ``measureRoom()`` after a frame has been rendered.
final class GaussianSplatMeasurementHost: ObservableObject {
    weak var coordinator: GaussianSplatView.Coordinator?
    @Published var arModeEnabled: Bool = false
    @Published var arStatusText: String = "AR camera off"
    @Published var arPlaneRoomMeasurement: ARPlaneRoomMeasurement?
    @Published var furnitureStatusText: String = "Furniture: none"
    @Published var placedFurnitureCount: Int = 0

    var depthQuery: (any SplatDepthQueryable)? {
        coordinator
    }

    var colorReader: (any SplatColorReadable)? {
        coordinator
    }

    /// Schedules one `setNeedsDisplay` so the next frame fills the depth scratch (call before ``measureRoom()`` if the view may have been idle).
    func requestRedrawForDepthMeasure() {
        coordinator?.requestRedrawForDepthMeasure()
    }

    /// Resets orbit, zoom, scene scale, and AR motion baseline (same as Recenter toolbar / notification).
    func recenterSharpRoomCamera() {
        coordinator?.performSharpRoomRecenter()
    }

    /// Euclidean distance from the **splat virtual camera** to the surface hit along the view ray, in **scene units**.
    /// Point must be in **window** coordinates (same as ``UIView.convert(_:to:)`` with `nil`). Call from the main thread.
    /// Used by Furniture Fit in Live Room so metric depth matches the rendered splat, not the floor-contact + device-pitch heuristic.
    func splatCameraToSurfaceDistanceSceneUnits(atWindowPoint windowPoint: CGPoint) -> Float? {
        guard let mtkView = coordinator?.view else { return nil }
        let pointInSplat = mtkView.convert(windowPoint, from: nil)
        return coordinator?.depthAt(screenPoint: pointInSplat)
    }

    /// Reads the last splat depth buffer and builds a trimmed grid point cloud (see ``GaussianSplatView.Coordinator/measureRoomFromDepthBuffer()``). Call on the main thread after at least one frame.
    func measureRoom() -> RoomRaycastDimensions? {
        coordinator?.measureRoomFromDepthBuffer()
    }

    func sampleDepthGrid(rows: Int, cols: Int) -> [[Float?]] {
        coordinator?.sampleDepthGrid(rows: rows, cols: cols) ?? []
    }

    /// Dense grid of world-space points from the splat depth buffer (for ``RoomGeometryEngine``).
    func buildPointCloudForRoomGeometry(rows: Int = 48, cols: Int = 48, maxDistance: Float = 12) -> [SIMD3<Float>] {
        coordinator?.buildPointCloud(rows: rows, cols: cols, maxDistance: maxDistance) ?? []
    }

    /// Captures the next rendered splat frame via Metal readback (reliable for `MTKView` / `CAMetalLayer`).
    func captureScreenshot(completion: @escaping (UIImage?) -> Void) {
        guard let coordinator else {
            DispatchQueue.main.async {
                completion(nil)
            }
            return
        }
        coordinator.scheduleScreenshotCapture(completion: completion)
    }

    func setARModeEnabled(_ enabled: Bool) {
        arModeEnabled = enabled
        coordinator?.setARModeEnabled(enabled)
    }

    /// Runs a short exclusive AR session while the splat renderer is frozen, then returns the best plane-based room measurement.
    func measureRoomWithARBurst(completion: @escaping (ARPlaneRoomMeasurement?) -> Void) {
        coordinator?.measureRoomWithARBurst(completion: completion) ?? DispatchQueue.main.async {
            completion(nil)
        }
    }

    /// Pauses ARKit while alerts/sheets need the main thread (e.g. save-room name field).
    func setModalHeavyWorkPaused(_ paused: Bool) {
        coordinator?.setModalHeavyWorkPaused(paused)
    }

    func setPendingFurnitureItem(_ item: SharpRoomFurnitureItem?) {
        coordinator?.setPendingFurnitureItem(item)
    }

    func clearPlacedFurniture() {
        coordinator?.clearPlacedFurniture()
    }

    func rotateSelectedFurniture(by radians: Float) {
        coordinator?.rotateSelectedFurniture(by: radians)
    }

    func updateARStatus(_ text: String) {
        arStatusText = text
    }

    func updateARPlaneRoomMeasurement(_ measurement: ARPlaneRoomMeasurement?) {
        arPlaneRoomMeasurement = measurement
    }

    func updateFurnitureStatus(_ text: String, count: Int) {
        furnitureStatusText = text
        placedFurnitureCount = count
    }

    /// After splat room geometry extraction for an **unsaved** Sharp session: logs world W×H×D plus AR virtual-camera–relative room AABB (metres). Always-on `AR_ROOM` / `[AR_ROOM_MEASURE]`.
    func logARRoomMeasureAfterGeometryExtraction(
        roomModel: RoomModel,
        plyURL: URL,
        boundBased: Bool,
        pointCloudCount: Int
    ) {
        let plyName = plyURL.lastPathComponent
        let stm = roomModel.sceneToMeters
        let sharpWidthMeters = roomModel.widthMeters
        let sharpHeightMeters = roomModel.heightMeters
        let sharpDepthMeters = roomModel.depthMeters
        let ext = roomModel.planeAwareSceneExtent
        let sanitizedStatus = arStatusText
            .replacingOccurrences(of: "\"", with: "'")
            .replacingOccurrences(of: "\n", with: " ")
        let authoritativeMeasurement = arPlaneRoomMeasurement
        let roomWidthMeters = authoritativeMeasurement?.widthMeters ?? sharpWidthMeters
        let roomHeightMeters = authoritativeMeasurement?.heightMeters ?? sharpHeightMeters
        let roomDepthMeters = authoritativeMeasurement?.depthMeters ?? sharpDepthMeters
        let authoritativeSource = authoritativeMeasurement?.sourceLabel ?? "SHARP_geometry_world_m_W×H×D"
        let arPlaneMetricFragment: String
        if let authoritativeMeasurement {
            arPlaneMetricFragment =
                " arkit_plane_m_W×H×D=\(arRoomMeasureFmt(authoritativeMeasurement.widthMeters))×" +
                "\(arRoomMeasureFmt(authoritativeMeasurement.heightMeters))×" +
                "\(arRoomMeasureFmt(authoritativeMeasurement.depthMeters)) " +
                "arkit_plane_counts_floor×wall=\(authoritativeMeasurement.floorAnchorCount)×\(authoritativeMeasurement.wallAnchorCount)"
        } else {
            arPlaneMetricFragment = " arkit_plane_note=not_ready"
        }

        guard arModeEnabled else {
            logARRoomMeasure(
                "phase=geometry_done live_room=off ply=\(plyName) bound_based=\(boundBased) pts=\(pointCloudCount) " +
                    "world_m_W×H×D=\(arRoomMeasureFmt(roomWidthMeters))×\(arRoomMeasureFmt(roomHeightMeters))×\(arRoomMeasureFmt(roomDepthMeters)) sceneToMeters=\(arRoomMeasureFmt(stm)) " +
                    "sharp_geometry_m_W×H×D=\(arRoomMeasureFmt(sharpWidthMeters))×\(arRoomMeasureFmt(sharpHeightMeters))×\(arRoomMeasureFmt(sharpDepthMeters)) " +
                    arPlaneMetricFragment +
                    " " +
                    "plane_ext_su_W×H×D=\(arRoomMeasureFmt(ext.width))×\(arRoomMeasureFmt(ext.height))×\(arRoomMeasureFmt(ext.depth)) " +
                    "ar_note=no_live_room_skip_camera_relative room_dims_authoritative_source=\(authoritativeSource)"
            )
            return
        }

        guard let camWorld = coordinator?.lastARVirtualCameraWorldForDiagnostics else {
            logARRoomMeasure(
                "phase=splat_box_in_virtual_cam live_room=on ply=\(plyName) bound_based=\(boundBased) pts=\(pointCloudCount) " +
                    "world_m_W×H×D=\(arRoomMeasureFmt(roomWidthMeters))×\(arRoomMeasureFmt(roomHeightMeters))×\(arRoomMeasureFmt(roomDepthMeters)) sceneToMeters=\(arRoomMeasureFmt(stm)) " +
                    "sharp_geometry_m_W×H×D=\(arRoomMeasureFmt(sharpWidthMeters))×\(arRoomMeasureFmt(sharpHeightMeters))×\(arRoomMeasureFmt(sharpDepthMeters)) " +
                    arPlaneMetricFragment +
                    " " +
                    "plane_ext_su_W×H×D=\(arRoomMeasureFmt(ext.width))×\(arRoomMeasureFmt(ext.height))×\(arRoomMeasureFmt(ext.depth)) " +
                    "ar_status=\(sanitizedStatus) ar_note=no_camera_matrix_yet_frame_pending " +
                    "room_dims_authoritative_source=\(authoritativeSource)"
            )
            return
        }

        let aabb = roomModel.roomBounds
        let corners: [SIMD3<Float>] = [
            SIMD3(aabb.min.x, aabb.min.y, aabb.min.z),
            SIMD3(aabb.max.x, aabb.min.y, aabb.min.z),
            SIMD3(aabb.min.x, aabb.max.y, aabb.min.z),
            SIMD3(aabb.max.x, aabb.max.y, aabb.min.z),
            SIMD3(aabb.min.x, aabb.min.y, aabb.max.z),
            SIMD3(aabb.max.x, aabb.min.y, aabb.max.z),
            SIMD3(aabb.min.x, aabb.max.y, aabb.max.z),
            SIMD3(aabb.max.x, aabb.max.y, aabb.max.z),
        ]
        let invCam = simd_inverse(camWorld)
        var minV = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
        var maxV = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
        for corner in corners {
            let homogeneous = invCam * SIMD4<Float>(corner.x, corner.y, corner.z, 1)
            let p = SIMD3<Float>(homogeneous.x, homogeneous.y, homogeneous.z)
            minV = simd_min(minV, p)
            maxV = simd_max(maxV, p)
        }
        let spanV = maxV - minV
        let rightM = spanV.x * stm
        let upM = spanV.y * stm
        let fwdM = spanV.z * stm

        let camSU = SIMD3<Float>(camWorld.columns.3.x, camWorld.columns.3.y, camWorld.columns.3.z)
        let chordSU = simd_length(roomModel.centroid - camSU)
        let chordM = chordSU * stm

        logARRoomMeasure(
            "phase=splat_box_in_virtual_cam live_room=on ply=\(plyName) bound_based=\(boundBased) pts=\(pointCloudCount) " +
                "world_m_W×H×D=\(arRoomMeasureFmt(roomWidthMeters))×\(arRoomMeasureFmt(roomHeightMeters))×\(arRoomMeasureFmt(roomDepthMeters)) sceneToMeters=\(arRoomMeasureFmt(stm)) " +
                "sharp_geometry_m_W×H×D=\(arRoomMeasureFmt(sharpWidthMeters))×\(arRoomMeasureFmt(sharpHeightMeters))×\(arRoomMeasureFmt(sharpDepthMeters)) " +
                arPlaneMetricFragment +
                " " +
                "plane_ext_su_W×H×D=\(arRoomMeasureFmt(ext.width))×\(arRoomMeasureFmt(ext.height))×\(arRoomMeasureFmt(ext.depth)) " +
                "ar_status=\(sanitizedStatus) " +
                "ar_cam_pos_world_su=(\(arRoomMeasureFmt(camSU.x)),\(arRoomMeasureFmt(camSU.y)),\(arRoomMeasureFmt(camSU.z))) " +
                "ar_cam_to_centroid_m=\(arRoomMeasureFmt(chordM)) " +
                "room_aabb_view_space_span_m_right×up×fwd=\(arRoomMeasureFmt(rightM))×\(arRoomMeasureFmt(upM))×\(arRoomMeasureFmt(fwdM)) " +
                "note=view_space_extents_splat_box_in_AR_virtual_camera_frame " +
                "room_dims_authoritative_source=\(authoritativeSource)"
        )
    }
}

private func arRoomMeasureFmt(_ x: Float) -> String {
    String(format: "%.4f", x)
}
