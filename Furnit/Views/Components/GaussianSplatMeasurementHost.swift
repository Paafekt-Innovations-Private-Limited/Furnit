import SwiftUI

/// Bridges ``GaussianSplatView.Coordinator`` so SwiftUI parents (e.g. ``SharpRoomView``) can call ``measureRoom()`` after a frame has been rendered.
final class GaussianSplatMeasurementHost: ObservableObject {
    weak var coordinator: GaussianSplatView.Coordinator?

    /// Schedules one `setNeedsDisplay` so the next frame fills the depth scratch (call before ``measureRoom()`` if the view may have been idle).
    func requestRedrawForDepthMeasure() {
        coordinator?.requestRedrawForDepthMeasure()
    }

    /// Reads the last splat depth buffer and raycasts five samples. Call from the main thread after the MTKView has rendered at least one frame.
    func measureRoom() -> RoomRaycastDimensions? {
        coordinator?.measureRoomFromDepthBuffer()
    }
}
