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
