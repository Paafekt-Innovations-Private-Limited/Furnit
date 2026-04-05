import SwiftUI
import UIKit
import simd

/// Bridges ``GaussianSplatView.Coordinator`` so SwiftUI parents (e.g. ``SharpRoomView``) can call ``measureRoom()`` after a frame has been rendered.
final class GaussianSplatMeasurementHost: ObservableObject {
    weak var coordinator: GaussianSplatView.Coordinator?
    @Published var arModeEnabled: Bool = false
    @Published var arStatusText: String = "AR camera off"
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

    func updateFurnitureStatus(_ text: String, count: Int) {
        furnitureStatusText = text
        placedFurnitureCount = count
    }
}
