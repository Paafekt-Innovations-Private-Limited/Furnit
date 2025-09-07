import SceneKit
import UIKit

class GestureHandlers: NSObject {
    weak var scnView: SCNView?
    private var boundaryManager: BoundaryManager?
    private var lastPanPoint: CGPoint = .zero
    private var initialCameraPosition: SCNVector3 = SCNVector3Zero
    private var initialCameraRotation: SCNVector4 = SCNVector4Zero
    
    init(scnView: SCNView) {
        self.scnView = scnView
        self.boundaryManager = BoundaryManager(scnView: scnView)
        super.init()
        setupGestures()
    }
    
    func setBoundaryManager(_ manager: BoundaryManager) {
        self.boundaryManager = manager
    }
    
    private func setupGestures() {
        guard let scnView = scnView else { return }
        
        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        panGesture.maximumNumberOfTouches = 1
        scnView.addGestureRecognizer(panGesture)
        
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        scnView.addGestureRecognizer(pinchGesture)
        
        let doubleTapGesture = UITapGestureRecognizer(target: self, action: #selector(handleDoubleTap(_:)))
        doubleTapGesture.numberOfTapsRequired = 2
        scnView.addGestureRecognizer(doubleTapGesture)
    }
    
    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        guard let scnView = scnView,
              let cameraNode = scnView.pointOfView else { return }
        
        let translation = gesture.translation(in: scnView)
        
        switch gesture.state {
        case .began:
            lastPanPoint = gesture.location(in: scnView)
            initialCameraPosition = cameraNode.position
            initialCameraRotation = cameraNode.rotation
            
        case .changed:
            let sensitivity: Float = 0.01
            let rotationY = Float(translation.x) * sensitivity
            let rotationX = Float(translation.y) * sensitivity
            
            cameraNode.rotation = SCNVector4(1, 0, 0, initialCameraRotation.w - rotationX)
            cameraNode.rotation = SCNVector4(0, 1, 0, initialCameraRotation.w - rotationY)
            
        default:
            break
        }
    }
    
    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard let scnView = scnView,
              let cameraNode = scnView.pointOfView else { return }
        
        switch gesture.state {
        case .began:
            initialCameraPosition = cameraNode.position
            
        case .changed:
            let scale = Float(gesture.scale)
            let scaleFactor = 1.0 / scale
            let newPosition = SCNVector3(
                initialCameraPosition.x * scaleFactor,
                initialCameraPosition.y * scaleFactor,
                initialCameraPosition.z * scaleFactor
            )
            
            let minDistance: Float = 1.0
            let maxDistance: Float = 20.0
            let distance = sqrt(newPosition.x * newPosition.x + 
                              newPosition.y * newPosition.y + 
                              newPosition.z * newPosition.z)
            
            if distance >= minDistance && distance <= maxDistance {
                let constrainedPosition = boundaryManager?.constrainCameraPosition(newPosition) ?? newPosition
                cameraNode.position = constrainedPosition
            }
            
        default:
            break
        }
    }
    
    @objc private func handleDoubleTap(_ gesture: UITapGestureRecognizer) {
        guard let scnView = scnView,
              let cameraNode = scnView.pointOfView else { return }
        
        // Reset camera to default indoor viewing position
        SCNTransaction.begin()
        SCNTransaction.animationDuration = 0.5
        
        // Position camera inside room at human eye level
        let defaultCameraPosition = SCNVector3(x: -2, y: 1.6, z: 2)
        let defaultLookAtPoint = SCNVector3(x: 0, y: 1, z: 0)
        
        cameraNode.position = defaultCameraPosition
        cameraNode.rotation = SCNVector4(x: 0, y: 0, z: 0, w: 0)
        cameraNode.look(at: defaultLookAtPoint)
        
        SCNTransaction.commit()
    }
}