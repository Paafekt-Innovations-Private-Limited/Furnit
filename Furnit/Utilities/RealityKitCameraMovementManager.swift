import Foundation
import RealityKit
import ARKit
import Combine

class RealityKitCameraMovementManager: ObservableObject {
    enum MovementSpeed: Float {
        case slow = 0.0008
        case normal = 0.0015
        case fast = 0.003
    }
    
    weak var arView: ARView?
    weak var boundaryManager: RealityKitBoundaryManager?
    @Published var isActive = false
    @Published var currentSpeed: MovementSpeed = .normal
    
    var onCameraMove: (() -> Void)?
    private var cameraAnchor: AnchorEntity!
    private var cameraEntity: Entity!
    private var displayLink: CADisplayLink?
    private var inputState = InputState()
    private var brightLightAnchor: AnchorEntity?
    
    private struct InputState {
        var forward = false
        var backward = false
        var left = false
        var right = false
        var up = false
        var down = false
        var rotationDelta: SIMD2<Float> = .zero
        var pitch: Float = 0
        var yaw: Float = 0
    }
    
    init() {
        print("📷 RealityKit Camera movement manager initialized")
    }
    
    func setupARView(_ arView: ARView) {
        self.arView = arView
        setupCamera()
        print("📷 RealityKit Camera movement manager ARView set")
    }
    
    func setARView(_ arView: ARView) {
        setupARView(arView)
    }
    
    func updateMovementSpeed(_ speed: MovementSpeed) {
        setSpeed(speed)
    }
    
    func updateJoystickInput(_ offset: CGSize) {
        let x = Float(offset.width)
        let y = Float(offset.height)
        
        // Handle joystick movement
        if abs(x) > 0.1 || abs(y) > 0.1 {
            // Stop all previous movement
            stopMoving(direction: "forward")
            stopMoving(direction: "backward")
            stopMoving(direction: "left")
            stopMoving(direction: "right")
            
            // Apply new movement based on joystick
            if abs(x) > 0.1 {
                if x > 0 {
                    startMoving(direction: "right")
                } else {
                    startMoving(direction: "left")
                }
            }
            
            if abs(y) > 0.1 {
                if y > 0 {
                    startMoving(direction: "forward")
                } else {
                    startMoving(direction: "backward")
                }
            }
        } else {
            // Stop all movement when joystick is centered
            stopMoving(direction: "forward")
            stopMoving(direction: "backward")
            stopMoving(direction: "left")
            stopMoving(direction: "right")
        }
    }
    
    private func setupCamera() {
        guard arView != nil else { return }
        
        // Create camera anchor at world origin
        cameraAnchor = AnchorEntity(world: .zero)
        
        // Create camera entity
        cameraEntity = Entity()
        cameraEntity.name = "FirstPersonCamera"
        
        // Note: PerspectiveCameraComponent doesn't have those properties
        // The camera functionality is handled differently in RealityKit
        // We just need the entity for positioning and movement
        
        cameraAnchor.addChild(cameraEntity)
        
        print("📷 Camera entity created for movement control")
    }
    
    func startCameraMode() {
        guard let arView = arView, let boundaryManager = boundaryManager else {
            print("❌ Cannot start camera mode: ARView or BoundaryManager not set")
            return
        }
        
        isActive = true
        
        // ===== LIGHTING FIX SECTION =====
        // Boost environment to maximum brightness
        arView.environment.background = .color(UIColor(white: 0.5, alpha: 1.0))
        arView.environment.lighting.intensityExponent = 5.0
        
        // Add super bright light that follows camera
        let cameraLightEntity = Entity()
        cameraLightEntity.name = "CameraLight"
        cameraLightEntity.components.set(PointLightComponent(
            color: .white,
            intensity: 10000,
            attenuationRadius: 30.0
        ))
        cameraLightEntity.position = [0, 0.5, 1]  // Slightly in front of camera
        
        // Add extra bright directional light
        let directionalEntity = Entity()
        directionalEntity.components.set(DirectionalLightComponent(
            color: .white,
            intensity: 5000,
            isRealWorldProxy: false
        ))
        directionalEntity.orientation = simd_quatf(angle: -.pi/4, axis: [1, 0, 0])
        
        let lightAnchor = AnchorEntity(world: .zero)
        lightAnchor.addChild(directionalEntity)
        arView.scene.addAnchor(lightAnchor)
        
        // Add multiple point lights for maximum visibility
        let lightPositions = [
            SIMD3<Float>(5, 3, 5),
            SIMD3<Float>(-5, 3, -5),
            SIMD3<Float>(5, 3, -5),
            SIMD3<Float>(-5, 3, 5)
        ]
        
        for position in lightPositions {
            let extraLight = Entity()
            extraLight.components.set(PointLightComponent(
                color: .white,
                intensity: 3000,
                attenuationRadius: 15.0
            ))
            extraLight.position = position
            lightAnchor.addChild(extraLight)
        }
        
        print("💡 Camera lighting activated - Maximum brightness")
        // ===== END OF LIGHTING FIX =====
        
        // Get the center of the room for starting position
        let centerPosition = boundaryManager.getRoomCenter()
        let eyeHeight: Float = 1.6 // Average eye height in meters
        
        // Set the starting position
        let startPosition = SIMD3<Float>(centerPosition.x, eyeHeight, centerPosition.z)
        
        // Use safe position from boundary manager
        let safePosition = boundaryManager.getSafeCameraPosition(near: startPosition)
        cameraAnchor.position = safePosition
        
        // Attach camera light to camera entity (this light moves with camera)
        cameraEntity.addChild(cameraLightEntity)
        
        // Reset camera entity's local position and rotation
        cameraEntity.position = .zero
        
        // Look straight ahead initially (0 degree pitch)
        inputState.pitch = 0
        inputState.yaw = 0
        updateCameraRotation()
        
        // Add the camera anchor to the scene
        arView.scene.addAnchor(cameraAnchor)
        
        // Start the update loop for movement
        startUpdateLoop()
        
        // Add a white test cube to verify visibility
        let testMesh = MeshResource.generateBox(size: 0.3)
        let testMaterial = SimpleMaterial(color: .white, roughness: 0.5, isMetallic: false)
        let testCube = ModelEntity(mesh: testMesh, materials: [testMaterial])
        testCube.position = [0, eyeHeight, -2] // 2 meters in front
        
        let testAnchor = AnchorEntity(world: .zero)
        testAnchor.addChild(testCube)
        arView.scene.addAnchor(testAnchor)
        
        print("📷 Custom camera positioned at: \(cameraAnchor.position)")
        print("🟦 White test cube added 2 meters in front")
        print("📷 Camera mode fully activated")
    }
    
    func setCameraAnchor(_ anchor: AnchorEntity) {
        // Store the camera anchor if needed
        // or just ignore it if not used
        print("📷 Camera anchor reference set")
    }
    
    func setBoundaryManager(_ boundaryManager: RealityKitBoundaryManager) {
        self.boundaryManager = boundaryManager
        print("📍 Boundary manager set for camera movement")
    }
    
    private func setupBrightLighting(in arView: ARView) {
        // Remove old bright light if exists
        brightLightAnchor?.removeFromParent()
        
        // Boost environment lighting to MAXIMUM
        arView.environment.background = .color(UIColor(white: 0.5, alpha: 1.0))
        arView.environment.lighting.intensityExponent = 5.0  // Maximum brightness
        
        // Add multiple super bright lights
        let lights = [
            (position: SIMD3<Float>(0, 3, 0), intensity: Float(10000)),    // Very bright overhead
            (position: SIMD3<Float>(5, 3, 5), intensity: Float(5000)),
            (position: SIMD3<Float>(-5, 3, -5), intensity: Float(5000)),
            (position: SIMD3<Float>(5, 3, -5), intensity: Float(5000)),
            (position: SIMD3<Float>(-5, 3, 5), intensity: Float(5000))
        ]
        
        brightLightAnchor = AnchorEntity(world: .zero)
        
        for light in lights {
            let lightEntity = Entity()
            lightEntity.components.set(PointLightComponent(
                color: .white,
                intensity: light.intensity,
                attenuationRadius: 50
            ))
            lightEntity.position = light.position
            
            brightLightAnchor?.addChild(lightEntity)
        }
        
        // Add directional light
        let directionalEntity = Entity()
        directionalEntity.components.set(DirectionalLightComponent(
            color: .white,
            intensity: 5000,
            isRealWorldProxy: false
        ))
        directionalEntity.orientation = simd_quatf(angle: -.pi/4, axis: [1, 0, 0])
        brightLightAnchor?.addChild(directionalEntity)
        
        arView.scene.addAnchor(brightLightAnchor!)
        
        // Also add a light that follows the camera
        let cameraLightEntity = Entity()
        cameraLightEntity.components.set(PointLightComponent(
            color: .white,
            intensity: 3000,
            attenuationRadius: 15
        ))
        cameraLightEntity.position = [0, 0.2, 0.5]  // Slightly in front
        
        cameraEntity.addChild(cameraLightEntity)
        
        print("💡 MAXIMUM BRIGHTNESS activated - Multiple lights added")
    }
    
    private func addTestCube(to arView: ARView) {
        // Add a white test cube to verify visibility
        let mesh = MeshResource.generateBox(size: 0.3)
        let material = SimpleMaterial(color: .white, roughness: 0.5, isMetallic: false)
        
        let testCube = ModelEntity(mesh: mesh, materials: [material])
        testCube.position = [0, 1.5, -2]  // 2 meters in front
        
        let testAnchor = AnchorEntity(world: .zero)
        testAnchor.addChild(testCube)
        arView.scene.addAnchor(testAnchor)
        
        print("🟦 White test cube added at position [0, 1.5, -2]")
        print("   If you can't see this cube, the camera view itself may be the issue")
    }
    
    func stopCameraMode() {
        guard let arView = arView else { return }
        
        isActive = false
        
        // Remove bright lights
        brightLightAnchor?.removeFromParent()
        brightLightAnchor = nil
        
        // Reset environment
        arView.environment.lighting.intensityExponent = 1.0
        
        // Remove camera from scene
        cameraAnchor.removeFromParent()
        
        // Stop update loop
        stopUpdateLoop()
        
        print("✅ Camera mode DEACTIVATED")
    }
    
    func startMoving(direction: String) {
        switch direction.lowercased() {
        case "forward", "w": inputState.forward = true
        case "backward", "s": inputState.backward = true
        case "left", "a": inputState.left = true
        case "right", "d": inputState.right = true
        case "up", "q": inputState.up = true
        case "down", "e": inputState.down = true
        default: break
        }
    }
    
    func stopMoving(direction: String) {
        switch direction.lowercased() {
        case "forward", "w": inputState.forward = false
        case "backward", "s": inputState.backward = false
        case "left", "a": inputState.left = false
        case "right", "d": inputState.right = false
        case "up", "q": inputState.up = false
        case "down", "e": inputState.down = false
        default: break
        }
    }
    
    func rotate(deltaX: Float, deltaY: Float) {
        inputState.rotationDelta = SIMD2<Float>(deltaX, deltaY)
    }
    
    func setSpeed(_ speed: MovementSpeed) {
        currentSpeed = speed
        print("🏃 Speed: \(speed)")
    }
    
    private func startUpdateLoop() {
        stopUpdateLoop()
        displayLink = CADisplayLink(target: self, selector: #selector(updateCamera))
        displayLink?.add(to: .main, forMode: .common)
    }
    
    private func stopUpdateLoop() {
        displayLink?.invalidate()
        displayLink = nil
    }
    
    @objc private func updateCamera() {
        guard isActive else { return }
        
        // Update rotation
        if inputState.rotationDelta.x != 0 || inputState.rotationDelta.y != 0 {
            inputState.yaw -= inputState.rotationDelta.x * 0.5
            inputState.pitch -= inputState.rotationDelta.y * 0.5
            inputState.pitch = max(-80, min(80, inputState.pitch))
            
            updateCameraRotation()
            inputState.rotationDelta = .zero
        }
        
        // Calculate movement
        var movement = SIMD3<Float>.zero
        let speed = currentSpeed.rawValue
        
        let yawRadians = inputState.yaw * .pi / 180
        let forward = SIMD3<Float>(sin(yawRadians), 0, -cos(yawRadians))
        let right = SIMD3<Float>(cos(yawRadians), 0, sin(yawRadians))
        
        if inputState.forward { movement += forward * speed }
        if inputState.backward { movement -= forward * speed }
        if inputState.right { movement += right * speed }
        if inputState.left { movement -= right * speed }
        if inputState.up { movement.y += speed }
        if inputState.down { movement.y -= speed }
        
        // Apply movement
        if movement != .zero {
            cameraAnchor.position += movement
        }
    }
    
    private func updateCameraRotation() {
        let pitchRadians = inputState.pitch * .pi / 180
        let yawRadians = inputState.yaw * .pi / 180
        
        let pitchQuat = simd_quatf(angle: pitchRadians, axis: [1, 0, 0])
        let yawQuat = simd_quatf(angle: yawRadians, axis: [0, 1, 0])
        
        cameraEntity.orientation = yawQuat * pitchQuat
    }
}
