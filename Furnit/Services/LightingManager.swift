import Foundation
import RealityKit
import ARKit

class LightingManager {
    enum LightingQuality {
        case low
        case standard
        case high
    }
    
    weak var arView: ARView?
    private var currentQuality: LightingQuality = .standard
    var intensityMultiplier: Float = 2.0
    
    private var lightAnchors: [AnchorEntity] = []
    
    init() {
        print("💡 LightingManager initialized")
    }
    
    func setupARView(_ arView: ARView) {
        self.arView = arView
        setupLighting()
        print("💡 Lighting setup complete")
    }
    
    private func setupLighting() {
        guard let arView = arView else { return }
        
        // Clear existing lights
        lightAnchors.forEach { $0.removeFromParent() }
        lightAnchors.removeAll()
        
        // 1. Set bright environment lighting (this is how RealityKit handles ambient light)
        arView.environment.lighting.intensityExponent = 2.0 * intensityMultiplier
        arView.environment.background = .color(UIColor(white: 0.2, alpha: 1.0))
        
        // 2. Add Directional Light Component (correct syntax)
        let directionalLight = DirectionalLightComponent()
        
        let directionalEntity = Entity()
        directionalEntity.components.set(DirectionalLightComponent(
            color: .white,
            intensity: 3000 * intensityMultiplier,
            isRealWorldProxy: false
        ))
        directionalEntity.orientation = simd_quatf(angle: -.pi/4, axis: [1, 0, 0])
        
        let directionalAnchor = AnchorEntity(world: .zero)
        directionalAnchor.addChild(directionalEntity)
        arView.scene.addAnchor(directionalAnchor)
        lightAnchors.append(directionalAnchor)
        
        // 3. Add Point Light Component (correct syntax)
        let pointEntity = Entity()
        pointEntity.components.set(PointLightComponent(
            color: .white,
            intensity: 2500 * intensityMultiplier,
            attenuationRadius: 15.0
        ))
        pointEntity.position = [0, 3, 0]
        
        let pointAnchor = AnchorEntity(world: .zero)
        pointAnchor.addChild(pointEntity)
        arView.scene.addAnchor(pointAnchor)
        lightAnchors.append(pointAnchor)
        
        // 4. Add additional point lights for better coverage
        let positions: [SIMD3<Float>] = [
            [5, 3, 5],
            [-5, 3, -5],
            [5, 3, -5],
            [-5, 3, 5]
        ]
        
        for position in positions {
            let entity = Entity()
            entity.components.set(PointLightComponent(
                color: .white,
                intensity: 1500 * intensityMultiplier,
                attenuationRadius: 10.0
            ))
            entity.position = position
            
            let anchor = AnchorEntity(world: .zero)
            anchor.addChild(entity)
            arView.scene.addAnchor(anchor)
            lightAnchors.append(anchor)
        }
        
        print("💡 Added \(lightAnchors.count) light anchors to scene")
    }
    
    func setQuality(_ quality: LightingQuality) {
        currentQuality = quality
        
        switch quality {
        case .low:
            intensityMultiplier = 1.5
        case .standard:
            intensityMultiplier = 2.0
        case .high:
            intensityMultiplier = 3.0
        }
        
        updateLighting()
    }
    
    func updateLighting() {
        guard let arView = arView else { return }
        
        // Update environment intensity
        arView.environment.lighting.intensityExponent = 2.0 * intensityMultiplier
        
        print("💡 Lighting intensity updated: \(intensityMultiplier)x")
    }
    
    func addObjectSpotlight(for modelEntity: Entity) {
        let spotlightEntity = Entity()
        spotlightEntity.components.set(SpotLightComponent(
            color: .white,
            intensity: 1500 * intensityMultiplier,
            innerAngleInDegrees: 45,
            outerAngleInDegrees: 60,
            attenuationRadius: 5.0
        ))
        
        let bounds = modelEntity.visualBounds(relativeTo: modelEntity)
        let height = bounds.max.y - bounds.min.y
        spotlightEntity.position = [0, height + 0.5, 0]
        
        modelEntity.addChild(spotlightEntity)
        
        print("💡 Added spotlight for object")
    }
    
    // Call this when entering camera mode for maximum brightness
    func forceMaximumBrightness() {
        guard let arView = arView else { return }
        
        // Set environment to maximum brightness
        arView.environment.lighting.intensityExponent = 5.0
        arView.environment.background = .color(UIColor(white: 0.5, alpha: 1.0))
        
        // Add extra bright light for camera mode
        let cameraLightEntity = Entity()
        cameraLightEntity.name = "ExtraCameraLight"
        cameraLightEntity.components.set(PointLightComponent(
            color: .white,
            intensity: 10000,
            attenuationRadius: 30.0
        ))
        cameraLightEntity.position = [0, 2, 0]
        
        let anchor = AnchorEntity(world: .zero)
        anchor.addChild(cameraLightEntity)
        arView.scene.addAnchor(anchor)
        lightAnchors.append(anchor)
        
        print("💡 MAXIMUM BRIGHTNESS ACTIVATED")
    }
}
