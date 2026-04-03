import SwiftUI
import UIKit

/// Bridges ``GaussianSplatView.Coordinator`` so SwiftUI parents (e.g. ``SharpRoomView``) can call ``measureRoom()`` after a frame has been rendered.
final class GaussianSplatMeasurementHost: ObservableObject {
    weak var coordinator: GaussianSplatView.Coordinator?

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

    /// Reads the last splat depth buffer and raycasts five samples. Call from the main thread after the MTKView has rendered at least one frame.
    func measureRoom() -> RoomRaycastDimensions? {
        coordinator?.measureRoomFromDepthBuffer()
    }

    func sampleDepthGrid(rows: Int, cols: Int) -> [[Float?]] {
        coordinator?.sampleDepthGrid(rows: rows, cols: cols) ?? []
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
}
