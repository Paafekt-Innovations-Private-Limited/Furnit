import SwiftUI

/// Bridges ``GaussianSplatView.Coordinator`` so SwiftUI parents (e.g. ``SharpRoomView``) can call ``measureRoom()`` after a frame has been rendered.
final class GaussianSplatMeasurementHost: ObservableObject {
    weak var coordinator: GaussianSplatView.Coordinator?

    /// Reads the last splat depth buffer and raycasts five samples. Call from the main thread after the MTKView has rendered at least one frame.
    func measureRoom() -> RoomRaycastDimensions? {
        coordinator?.measureRoomFromDepthBuffer()
    }
}
