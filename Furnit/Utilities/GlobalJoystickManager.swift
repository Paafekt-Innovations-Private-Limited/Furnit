import SwiftUI
import Combine

/// Global joystick manager that provides consistent joystick functionality across all views
/// Camera movement managers directly read `joystickOffset` - no callbacks needed
class GlobalJoystickManager: ObservableObject {
    static let shared = GlobalJoystickManager()

    /// Current joystick offset (raw values from VirtualJoystick, max ~30 in each direction)
    @Published var joystickOffset: CGSize = .zero

    private init() {
        logDebug("🕹️ [GlobalJoystickManager] Initialized")
    }

    /// Update joystick position - called by VirtualJoystick
    func updateOffset(_ offset: CGSize) {
        joystickOffset = offset
    }

    /// Reset joystick to center
    func reset() {
        joystickOffset = .zero
    }
}

/// Reusable joystick overlay that can be added to any view
struct GlobalJoystickOverlay: View {
    @ObservedObject private var joystickManager = GlobalJoystickManager.shared
    @State private var localOffset: CGSize = .zero

    var body: some View {
        VStack {
            Spacer()
            HStack {
                VirtualJoystick(joystickOffset: $localOffset)
                    .onChange(of: localOffset) { _, newOffset in
                        joystickManager.updateOffset(newOffset)
                    }
                    .padding(.leading, 30)
                    .padding(.bottom, 40)
                Spacer()
            }
        }
        .allowsHitTesting(true)
    }
}

#Preview {
    ZStack {
        Color.gray.opacity(0.3)
        GlobalJoystickOverlay()
    }
}
