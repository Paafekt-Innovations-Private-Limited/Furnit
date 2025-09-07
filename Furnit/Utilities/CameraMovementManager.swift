import SceneKit
import SwiftUI

class CameraMovementManager: ObservableObject {
    // Camera movement properties
    weak var sceneView: SCNView?
    private var displayLink: CADisplayLink?
    private var currentJoystickOffset: CGSize = .zero
    
    // Movement configuration
    private let movementSpeed: Float = 0.05 // Units per frame
    private let smoothingFactor: Float = 0.8 // Movement smoothing (0.0 = instant, 1.0 = no movement)
    
    init() {
        // Set up display link for smooth continuous movement
        displayLink = CADisplayLink(target: self, selector: #selector(updateCameraPosition))
        displayLink?.add(to: .main, forMode: .common)
    }
    
    deinit {
        // Clean up display link
        displayLink?.invalidate()
    }
    
    // Set the scene view reference for camera manipulation
    func setSceneView(_ sceneView: SCNView) {
        self.sceneView = sceneView
    }
    
    // Update joystick input from the virtual joystick
    func updateJoystickInput(_ offset: CGSize) {
        currentJoystickOffset = offset
    }
    
    // Continuous camera position updates based on joystick input
    @objc private func updateCameraPosition() {
        guard let sceneView = sceneView,
              let cameraNode = sceneView.pointOfView else { return }
        
        // Skip if no joystick input
        guard abs(currentJoystickOffset.width) > 1 || abs(currentJoystickOffset.height) > 1 else { return }
        
        // Convert joystick input to movement vectors
        let forwardBackward = Float(-currentJoystickOffset.height) * movementSpeed // Negative for intuitive forward movement
        let leftRight = Float(currentJoystickOffset.width) * movementSpeed
        
        // Get camera's current transform for directional movement
        let cameraTransform = cameraNode.transform
        
        // Extract camera's forward and right vectors from transform matrix
        let forwardVector = SCNVector3(
            -cameraTransform.m31, // Forward is negative Z in camera space
            0, // Keep movement horizontal
            -cameraTransform.m33
        )
        
        let rightVector = SCNVector3(
            cameraTransform.m11, // Right is positive X in camera space
            0, // Keep movement horizontal
            cameraTransform.m13
        )
        
        // Normalize vectors for consistent movement speed
        let normalizedForward = normalizeVector(forwardVector)
        let normalizedRight = normalizeVector(rightVector)
        
        // Calculate movement delta based on joystick input
        let movementDelta = SCNVector3(
            normalizedForward.x * forwardBackward + normalizedRight.x * leftRight,
            0, // No vertical movement
            normalizedForward.z * forwardBackward + normalizedRight.z * leftRight
        )
        
        // Apply movement to camera position
        let newPosition = SCNVector3(
            cameraNode.position.x + movementDelta.x,
            cameraNode.position.y, // Keep same height
            cameraNode.position.z + movementDelta.z
        )
        
        // Update camera position smoothly
        cameraNode.position = newPosition
    }
    
    // Helper function to normalize a 3D vector
    private func normalizeVector(_ vector: SCNVector3) -> SCNVector3 {
        let length = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)
        if length == 0 { return vector }
        
        return SCNVector3(
            vector.x / length,
            vector.y / length,
            vector.z / length
        )
    }
    
    // Reset camera to default position
    func resetCameraPosition() {
        guard let sceneView = sceneView,
              let cameraNode = sceneView.pointOfView else { return }
        
        SCNTransaction.begin()
        SCNTransaction.animationDuration = 0.5
        
        // Reset to default indoor viewing position
        let defaultPosition = SCNVector3(x: -2, y: 1.6, z: 2)
        let lookAtPoint = SCNVector3(x: 0, y: 1, z: 0)
        
        cameraNode.position = defaultPosition
        cameraNode.look(at: lookAtPoint)
        
        SCNTransaction.commit()
    }
}