import RealityKit
import ARKit
import SwiftUI
import simd

class RealityKitCameraMovementManager: ObservableObject {
    // Define MovementSpeed enum locally with BALANCED values for smooth movement
    enum MovementSpeed: Float {
        case slow = 0.0016   // 2x original for gentle movement
        case normal = 0.003  // 2x original for comfortable cruising
        case fast = 0.006    // 2x original for quicker navigation
    }

    // Camera movement properties
    weak var arView: ARView?
    private var boundaryManager: RealityKitBoundaryManager?
    private var displayLink: CADisplayLink?
    private var currentJoystickOffset: CGSize = .zero

    // Custom camera control for non-AR mode
    private weak var cameraAnchor: AnchorEntity?

    // Current speed setting
    @Published var currentSpeed: MovementSpeed = .normal

    // Movement configuration - BALANCED FOR SMOOTH MOVEMENT
    private var movementSpeed: Float = 0.04 // Moderate speed for smooth control
    private let joystickSensitivity: Float = 0.001 // Keep original sensitivity
    private let joystickDeadZone: Float = 5.0 // Dead zone threshold

    // Smooth movement
    private var targetVelocity: SIMD2<Float> = .zero
    private var currentVelocity: SIMD2<Float> = .zero
    private let smoothingFactor: Float = 0.18 // Balanced for smooth transitions

    // Callback for camera movement notifications
    var onCameraMove: (() -> Void)?

    init() {
        // ❌ DISABLED: GlobalCameraController now handles all camera movement
        // displayLink = CADisplayLink(target: self, selector: #selector(updateCameraPosition))
        // displayLink?.add(to: .main, forMode: .common)

        logDebug("🎮 Camera movement manager initialized (displayLink DISABLED - using GlobalCameraController)")
    }
    
    deinit {
        // Clean up display link
        displayLink?.invalidate()
    }
    
    // Set the ARView reference for camera manipulation
    func setARView(_ arView: ARView) {
        self.arView = arView
        logDebug("📱 Camera movement manager ARView set")
    }
    
    func setupARView(_ arView: ARView) {
        setARView(arView)
    }
    
    // Set camera anchor reference for direct camera control
    func setCameraAnchor(_ anchor: AnchorEntity) {
        // Only log if anchor is changing
        let isNewAnchor = self.cameraAnchor !== anchor
        self.cameraAnchor = anchor

        if isNewAnchor {
            logDebug("🎮 Camera anchor set")
        }
    }

    // Set boundary manager reference (shared from RealityKitView)
    func setBoundaryManager(_ manager: RealityKitBoundaryManager) {
        self.boundaryManager = manager
        logDebug("🏠 Camera movement manager using shared boundary manager")
    }

    // Update movement speed from settings
    func updateMovementSpeed(_ speed: MovementSpeed) {
        setSpeed(speed)
    }
    
    func setSpeed(_ speed: MovementSpeed) {
        // Map enum to actual speed values with moderate scaling for smooth movement
        movementSpeed = speed.rawValue * 20.0  // Balanced multiplier for smooth control
        currentSpeed = speed
        logDebug("🏃 Speed set to \(speed) (\(movementSpeed))")
    }

}
