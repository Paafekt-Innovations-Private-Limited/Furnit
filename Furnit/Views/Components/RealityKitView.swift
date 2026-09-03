import SwiftUI
import RealityKit
import ARKit

struct RealityKitView: UIViewRepresentable {
    let model: USDZModel
    let cameraMovementManager: CameraMovementManager
    let arObjectPlacementManager: ARObjectPlacementManager
    let isARActive: Bool
    let infiniteZoom: Bool

    // ✅ NEW: Snapshot capability - for capturing clean 3D room
    @Binding var shouldCaptureSnapshot: Bool
    @Binding var capturedSnapshot: UIImage?

    // ✅ NEW: Camera reset trigger - resets camera to optimal position
    @Binding var shouldResetCamera: Bool
    
    // Access quality settings from environment
    @Environment(\.appState) private var appState
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    func makeUIView(context: Context) -> ARView {
        logDebug("🎨 [RealityKitView.makeUIView] ========================================")
        logDebug("🎨 [RealityKitView.makeUIView] Creating ARView for model: \(model.displayName)")
        logDebug("   - File name: \(model.fileName)")
        logDebug("   - Is saved room: \(model.isSavedRoom)")
        
        // Use .nonAR mode for custom camera control that allows rotation without moving position
        let arView = ARView(frame: .zero, cameraMode: .nonAR, automaticallyConfigureSession: false)
        arView.isAccessibilityElement = true
        arView.accessibilityIdentifier = "saved_room_viewport_loading"
        
        // ✅ NEW: Store ARView reference in Coordinator for snapshot
        context.coordinator.arView = arView
        
        // Configure ARView for room viewing in non-AR mode
        arView.renderOptions.insert(.disablePersonOcclusion)
        arView.renderOptions.insert(.disableMotionBlur)
        Self.configureViewerBackground(arView, roomCoordinateFrame: model.roomCoordinateFrame)
        
        // Apply quality settings
        let quality = appState.currentQuality
        configureRenderingQuality(arView: arView, quality: quality)
        
        logDebug("🎨 Applying quality setting: \(quality.displayName)")
        
        // Set up coordinator and custom camera for non-AR mode
        context.coordinator.setupGestures(for: arView, placementManager: arObjectPlacementManager)
        context.coordinator.gestureHandlers?.setInfiniteZoomEnabled(infiniteZoom)
        context.coordinator.setupCustomCamera(for: arView)

        loadModel(into: arView, coordinator: context.coordinator)

        // Set up camera movement manager with the ARView (for other features)
        cameraMovementManager.setupARView(arView)
        if let cameraAnchor = context.coordinator.cameraAnchor {
            cameraMovementManager.setCameraAnchor(cameraAnchor)
        }

        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
        context.coordinator.gestureHandlers?.setInfiniteZoomEnabled(infiniteZoom)

        if let cameraAnchor = context.coordinator.cameraAnchor {
            cameraMovementManager.setCameraAnchor(cameraAnchor)
        }

        // ✅ Check if model changed - reset camera position if so
        if context.coordinator.currentModelID != model.id {
            logDebug("🔄 [RealityKitView.updateUIView] MODEL CHANGED! Resetting camera position...")
            logDebug("   Old model: \(context.coordinator.currentModelID?.uuidString ?? "nil")")
            logDebug("   New model: \(model.id.uuidString) (\(model.displayName))")

            // Reset camera to optimal position using stored boundary manager
            if let cameraAnchor = context.coordinator.cameraAnchor,
               let boundaryManager = context.coordinator.boundaryManager,
               boundaryManager.bounds != nil {
                context.coordinator.cameraLookAtTarget = Self.repositionOptimalCamera(
                    cameraAnchor: cameraAnchor,
                    cameraEntity: context.coordinator.cameraEntity,
                    boundaryManager: boundaryManager,
                    model: model,
                    cameraMetadata: context.coordinator.depthCameraMetadata
                )
                context.coordinator.configureNavigationContract(for: model)

                logDebug("📷 [RealityKitView.updateUIView] Camera RESET to: \(cameraAnchor.transform.translation)")
            }

            context.coordinator.currentModelID = model.id
            context.coordinator.lastViewportSize = .zero
        }

        if let cameraAnchor = context.coordinator.cameraAnchor,
           let boundaryManager = context.coordinator.boundaryManager,
           boundaryManager.bounds != nil {
            let viewportSize = uiView.bounds.size
            if context.coordinator.shouldReframeForViewportChange(viewportSize) {
                if model.roomCoordinateFrame == .depthAnythingImageDepthMeters {
                    context.coordinator.cameraLookAtTarget = Self.repositionOptimalCamera(
                        cameraAnchor: cameraAnchor,
                        cameraEntity: context.coordinator.cameraEntity,
                        boundaryManager: boundaryManager,
                        model: model,
                        cameraMetadata: context.coordinator.depthCameraMetadata
                    )
                    context.coordinator.configureNavigationContract(for: model)
                } else {
                    let optimalPose = boundaryManager.getOptimalCameraPosition(
                        roomCoordinateFrame: model.roomCoordinateFrame,
                        photoOrientation: model.photoOrientation
                    )
                    Self.configureVolumetricRoomCameraFieldOfView(
                        context.coordinator.cameraEntity,
                        boundaryManager: boundaryManager,
                        cameraPosition: optimalPose.position,
                        lookAtPosition: optimalPose.lookAt,
                        photoOrientation: model.photoOrientation,
                        viewportSize: viewportSize
                    )
                }
            }
        }

        Self.configureViewerBackground(uiView, roomCoordinateFrame: model.roomCoordinateFrame)

        // Update rendering quality if settings changed
        let currentQuality = appState.currentQuality
        configureRenderingQuality(arView: uiView, quality: currentQuality)

        // Keep RealityKit camera movement on the stable default speed.
        DispatchQueue.main.async { [weak cameraMovementManager] in
            cameraMovementManager?.setSpeed(.normal)
        }
        
        // ✅ Handle camera reset requests (triggered on view appear)
        if shouldResetCamera {
            let debugMode = AppStateManager.shared.qualitySettings.debugMode
            
            if debugMode {
                logDebug("🔄 [RealityKitView.updateUIView] CAMERA RESET TRIGGERED")
            }
            
            if let cameraAnchor = context.coordinator.cameraAnchor,
               let boundaryManager = context.coordinator.boundaryManager,
               boundaryManager.bounds != nil {
                context.coordinator.cameraLookAtTarget = Self.repositionOptimalCamera(
                    cameraAnchor: cameraAnchor,
                    cameraEntity: context.coordinator.cameraEntity,
                    boundaryManager: boundaryManager,
                    model: model,
                    cameraMetadata: context.coordinator.depthCameraMetadata
                )
                context.coordinator.configureNavigationContract(for: model)

                if debugMode {
                    logDebug("📷 [RealityKitView] Camera RESET to optimal position: \(cameraAnchor.transform.translation)")
                }
            } else {
                if debugMode {
                    logDebug("⚠️ [RealityKitView] Cannot reset camera - missing cameraAnchor or boundaryManager")
                }
            }

            // Clear the flag
            DispatchQueue.main.async {
                self.shouldResetCamera = false
            }
        }

        // ✅ FIXED: Handle snapshot requests
        if shouldCaptureSnapshot {
            logDebug("📸 [RealityKitView] Snapshot requested, capturing ARView...")
            
            // Capture the bindings to mutate them in the async closure
            let capturedSnapshotBinding = $capturedSnapshot
            let shouldCaptureSnapshotBinding = $shouldCaptureSnapshot
            
            // Use ARView's built-in snapshot method to capture ONLY 3D content
            uiView.snapshot(saveToHDR: false) { image in
                DispatchQueue.main.async {
                    capturedSnapshotBinding.wrappedValue = image
                    shouldCaptureSnapshotBinding.wrappedValue = false
                    
                    if let image = image {
                        logDebug("✅ [RealityKitView] Snapshot captured: \(Int(image.size.width))x\(Int(image.size.height))")
                    } else {
                        logDebug("❌ [RealityKitView] Snapshot failed - no image returned")
                    }
                }
            }
        }
    }

    @discardableResult
    private static func applyCameraPose(
        _ cameraAnchor: AnchorEntity,
        position: SIMD3<Float>,
        lookAt: SIMD3<Float>
    ) -> SIMD3<Float> {
        let direction = normalizedDirection(from: position, to: lookAt)
        cameraAnchor.transform.translation = position
        cameraAnchor.transform.rotation = fixedWorldUpLookRotation(forward: direction)
        return direction
    }

    private static func configureDepthAnythingCameraFieldOfView(
        _ cameraEntity: PerspectiveCamera?,
        model: USDZModel,
        authoredVerticalFieldOfView: Float?,
        viewportSize: CGSize?,
        cameraMetadata: [String: Double]?
    ) {
        guard let cameraEntity,
              model.roomCoordinateFrame == .depthAnythingImageDepthMeters else {
            return
        }
        if let authoredVerticalFieldOfView {
            var displayVerticalFieldOfView = authoredVerticalFieldOfView
            if let viewportSize,
               viewportSize.width > 1,
               viewportSize.height > 1 {
                let metadata = cameraMetadata ?? model.temporaryURL.map {
                    CameraExifSidecar.load(roomURL: $0)
                } ?? [:]
                let imageWidth = metadata["depthMeshImageWidthPx"]
                    ?? metadata["imageWidthPx"]
                let imageHeight = metadata["depthMeshImageHeightPx"]
                    ?? metadata["imageHeightPx"]
                if let imageWidth, let imageHeight,
                   imageWidth > 1, imageHeight > 1 {
                    // Match preview's aspect-fill framing. A wide viewport must crop the top and
                    // bottom of a 4:3 photo; showing the whole vertical FOV exposes nonexistent
                    // pixels at the sides and was what made the extension skirt visible.
                    displayVerticalFieldOfView = Float(
                        DepthAnythingFlatPhotoCameraFraming.projectiveDisplayVerticalFieldOfViewDegrees(
                            authoredVerticalFieldOfViewDegrees: authoredVerticalFieldOfView,
                            imageWidth: Int(imageWidth.rounded()),
                            imageHeight: Int(imageHeight.rounded()),
                            photoOrientation: model.photoOrientation,
                            viewportSize: viewportSize
                        )
                    )
                }
            }
            cameraEntity.camera.fieldOfViewOrientation = .vertical
            cameraEntity.camera.fieldOfViewInDegrees = displayVerticalFieldOfView
        } else if model.photoOrientation == .landscape {
            cameraEntity.camera.fieldOfViewOrientation = .horizontal
            cameraEntity.camera.fieldOfViewInDegrees = 60.0
        } else {
            cameraEntity.camera.fieldOfViewOrientation = .vertical
            cameraEntity.camera.fieldOfViewInDegrees = 60.0
        }
    }

    private static func configureVolumetricRoomCameraFieldOfView(
        _ cameraEntity: PerspectiveCamera?,
        boundaryManager: RealityKitBoundaryManager,
        cameraPosition: SIMD3<Float>,
        lookAtPosition: SIMD3<Float>,
        photoOrientation: PhotoOrientation,
        viewportSize: CGSize?
    ) {
        guard let cameraEntity,
              let verticalFieldOfView = boundaryManager.verticalFieldOfViewToFrameFrontWall(
                  cameraPosition: cameraPosition,
                  lookAtPosition: lookAtPosition,
                  photoOrientation: photoOrientation,
                  viewportSize: viewportSize
              ) else {
            return
        }
        cameraEntity.camera.fieldOfViewOrientation = .vertical
        cameraEntity.camera.fieldOfViewInDegrees = verticalFieldOfView
    }

    private static func authoredDepthCaptureVerticalFieldOfView(
        for model: USDZModel,
        cameraMetadata: [String: Double]?
    ) -> Float? {
        guard model.roomCoordinateFrame == .depthAnythingImageDepthMeters else {
            return nil
        }
        let metadata = cameraMetadata ?? model.temporaryURL.map {
            CameraExifSidecar.load(roomURL: $0)
        } ?? [:]
        guard metadata["depthMeshProjectionVersion", default: 0] >= 1,
              metadata["depthMeshIsFlatPhotoPlane", default: 0] < 0.5,
              let fieldOfView = metadata["depthMeshVerticalFovDegrees"].map(Float.init),
              fieldOfView.isFinite,
              (10...140).contains(fieldOfView) else {
            return nil
        }
        return fieldOfView
    }

    /// Correct legacy flat USDZ files created before the flat-mesh exporter preserved photo aspect. The
    /// room-specific camera sidecar survives the save copy and records the original pixel size.
    /// New files already match and therefore receive a scale of approximately 1.
    private static func restoreDepthAnythingPhotoAspectIfNeeded(
        _ entity: ModelEntity,
        modelURL: URL,
        photoOrientation: PhotoOrientation,
        cameraMetadata: [String: Double]?
    ) {
        let metadata = cameraMetadata ?? CameraExifSidecar.load(roomURL: modelURL)
        guard var pixelWidth = metadata["imageWidthPx"].map(Float.init),
              var pixelHeight = metadata["imageHeightPx"].map(Float.init),
              pixelWidth > 1,
              pixelHeight > 1 else {
            return
        }

        switch photoOrientation {
        case .landscape where pixelWidth < pixelHeight,
             .portrait where pixelWidth > pixelHeight:
            swap(&pixelWidth, &pixelHeight)
        default:
            break
        }

        let desiredAspect = pixelWidth / pixelHeight
        let bounds = entity.visualBounds(relativeTo: entity)
        // Current depth-surface USDZs already preserve pixel aspect. Aspect repair only belongs
        // to legacy image planes; applying it to spatial depth would deform the reconstruction.
        guard bounds.extents.z <= 0.2 else { return }
        guard bounds.extents.x > 0.001,
              bounds.extents.y > 0.001,
              desiredAspect.isFinite,
              desiredAspect > 0 else {
            return
        }
        let currentAspect = bounds.extents.x / bounds.extents.y
        let correction = desiredAspect / currentAspect
        guard correction.isFinite,
              correction > 0.5,
              correction < 2.0,
              abs(correction - 1) > 0.002 else {
            return
        }

        entity.scale.x *= correction
        logDebug(
            "📐 [RealityKitView] Restored photo aspect " +
                "current=\(String(format: "%.4f", currentAspect)) " +
                "source=\(String(format: "%.4f", desiredAspect)) " +
                "scaleX=\(String(format: "%.4f", correction))"
        )
    }

    @discardableResult
    private static func repositionOptimalCamera(
        cameraAnchor: AnchorEntity,
        cameraEntity: PerspectiveCamera?,
        boundaryManager: RealityKitBoundaryManager,
        model: USDZModel,
        cameraMetadata: [String: Double]?
    ) -> SIMD3<Float>? {
        guard let bounds = boundaryManager.bounds else { return nil }
        let authoredVerticalFieldOfView = authoredDepthCaptureVerticalFieldOfView(
            for: model,
            cameraMetadata: cameraMetadata
        )
        configureDepthAnythingCameraFieldOfView(
            cameraEntity,
            model: model,
            authoredVerticalFieldOfView: authoredVerticalFieldOfView,
            viewportSize: boundaryManager.arView?.bounds.size,
            cameraMetadata: cameraMetadata
        )
        if authoredVerticalFieldOfView != nil {
            // Perspective depth meshes are authored around the original optical center. Keeping
            // the eye at that origin and restoring the authored focal length makes the first
            // saved-room frame a pixel-exact reprojection of the source photograph.
            let cameraPosition = SIMD3<Float>(0, 0, 0)
            // Android/glTF camera space looks down -Z. Target the farthest authored depth from the
            // capture origin so recenter, gestures and D-pad share its full-room baseline.
            let lookAtPosition = SIMD3<Float>(0, 0, min(bounds.min.z, -0.2))
            _ = applyCameraPose(
                cameraAnchor,
                position: cameraPosition,
                lookAt: lookAtPosition
            )
            return lookAtPosition
        }
        let (cameraPosition, lookAtPosition) = boundaryManager.getOptimalCameraPosition(
            roomCoordinateFrame: model.roomCoordinateFrame,
            photoOrientation: model.photoOrientation,
            inferencePlaneWidthMeters: model.roomCoordinateFrame == .depthAnythingImageDepthMeters
                ? model.roomWidth
                : nil,
            inferencePlaneHeightMeters: model.roomCoordinateFrame == .depthAnythingImageDepthMeters
                ? model.roomHeight
                : nil
        )
        cameraAnchor.transform.translation = cameraPosition
        _ = applyCameraPose(
            cameraAnchor,
            position: cameraPosition,
            lookAt: lookAtPosition
        )
        if model.roomCoordinateFrame != .depthAnythingImageDepthMeters {
            configureVolumetricRoomCameraFieldOfView(
                cameraEntity,
                boundaryManager: boundaryManager,
                cameraPosition: cameraPosition,
                lookAtPosition: lookAtPosition,
                photoOrientation: model.photoOrientation,
                viewportSize: boundaryManager.arView?.bounds.size
            )
        }
        return lookAtPosition
    }

    private static func normalizedDirection(from position: SIMD3<Float>, to target: SIMD3<Float>) -> SIMD3<Float> {
        let delta = target - position
        let lengthSquared = simd_length_squared(delta)
        guard lengthSquared > 1e-8 else {
            return SIMD3<Float>(0, 0, -1)
        }
        return delta / sqrt(lengthSquared)
    }

    private static func fixedWorldUpLookRotation(forward: SIMD3<Float>) -> simd_quatf {
        var up = SIMD3<Float>(0, 1, 0)
        if abs(simd_dot(forward, up)) > 0.98 {
            up = SIMD3<Float>(0, 0, 1)
        }

        let right = simd_normalize(simd_cross(forward, up))
        let correctedUp = simd_normalize(simd_cross(right, forward))
        let localToWorld = float3x3(
            columns: (
                right,
                correctedUp,
                -forward
            )
        )
        return simd_quatf(localToWorld)
    }
    
    class Coordinator {
        var gestureHandlers: RealityKitGestureHandlers?
        var scene: RealityKit.Scene?
        weak var arObjectPlacementManager: ARObjectPlacementManager?

        // Custom camera control for non-AR mode
        var cameraEntity: PerspectiveCamera?
        var cameraAnchor: AnchorEntity?

        // World anchor for object placement (the model anchor)
        var worldAnchor: AnchorEntity?

        // ✅ NEW: Store ARView reference for snapshot
        weak var arView: ARView?

        // ✅ Track current model to detect room changes
        var currentModelID: UUID?
        var boundaryManager: RealityKitBoundaryManager?
        var depthCameraMetadata: [String: Double]?
        /// Point used for exterior orbit framing. The gesture handler independently switches to
        /// turn-in-place navigation whenever the camera lies inside a volumetric room.
        var cameraLookAtTarget: SIMD3<Float>? {
            didSet {
                gestureHandlers?.setOrbitTarget(cameraLookAtTarget)
            }
        }
        var lastViewportSize: CGSize = .zero
        private var cameraMoveNotificationTokens: [NSObjectProtocol] = []
        deinit {
            removeCameraMoveObservers()
        }

        func installCameraMoveObservers() {
            removeCameraMoveObservers()
            let nc = NotificationCenter.default
            let namesAndDeltas: [(NSNotification.Name, SIMD3<Float>)] = [
                (
                    PaafektViewerCameraMoveNotification.left,
                    SIMD3<Float>(-DepthAnythingPhotoCameraInteraction.dPadHorizontalStep, 0, 0)
                ),
                (
                    PaafektViewerCameraMoveNotification.right,
                    SIMD3<Float>(DepthAnythingPhotoCameraInteraction.dPadHorizontalStep, 0, 0)
                ),
                (
                    PaafektViewerCameraMoveNotification.up,
                    SIMD3<Float>(0, DepthAnythingPhotoCameraInteraction.dPadVerticalStep, 0)
                ),
                (
                    PaafektViewerCameraMoveNotification.down,
                    SIMD3<Float>(0, -DepthAnythingPhotoCameraInteraction.dPadVerticalStep, 0)
                ),
            ]
            for (name, delta) in namesAndDeltas {
                let token = nc.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                    self?.nudgeCamera(by: delta)
                }
                cameraMoveNotificationTokens.append(token)
            }
        }

        func removeCameraMoveObservers() {
            let nc = NotificationCenter.default
            for token in cameraMoveNotificationTokens {
                nc.removeObserver(token)
            }
            cameraMoveNotificationTokens.removeAll()
        }

        /// Nudge the custom camera like Splat/GLB D-pad: world X walk, world Y lift, orbit target moves with camera.
        private func nudgeCamera(by worldDelta: SIMD3<Float>) {
            guard let cameraAnchor else { return }

            let rotationNudge = DepthAnythingPhotoCameraInteraction.rotationNudge(
                forWorldDelta: worldDelta
            )
            let didTurn = gestureHandlers?.nudgeCapturedPhotoView(
                yawDelta: rotationNudge.yaw,
                pitchDelta: rotationNudge.pitch
            ) == true
            if didTurn { return }

            let position = cameraAnchor.transform.translation
            let forward = cameraAnchor.transform.rotation.act(SIMD3<Float>(0, 0, -1))
            let lookDistance: Float
            if let storedLookAt = cameraLookAtTarget {
                lookDistance = max(simd_length(storedLookAt - position), 0.5)
            } else {
                lookDistance = 3.0
            }
            let lookAt = position + forward * lookDistance

            var newPosition = position + worldDelta
            if let boundaryManager {
                newPosition = boundaryManager.constrainCameraPosition(newPosition)
            }
            let appliedDelta = newPosition - position
            let newLookAt = lookAt + appliedDelta

            _ = RealityKitView.applyCameraPose(
                cameraAnchor,
                position: newPosition,
                lookAt: newLookAt
            )
            cameraLookAtTarget = newLookAt
            gestureHandlers?.syncRotationState()
        }

        func configureNavigationContract(for model: USDZModel) {
            let metadata = model.temporaryURL.map { CameraExifSidecar.load(roomURL: $0) } ?? [:]
            let isFlatPhotoPlane = model.roomCoordinateFrame == .depthAnythingImageDepthMeters
                && metadata["depthMeshIsFlatPhotoPlane", default: 0] >= 0.5
            let hasCompletedBackground = model.roomCoordinateFrame == .depthAnythingImageDepthMeters
                && metadata["depthMeshHasCompletedBackground", default: 0] >= 0.5
            let hasNavigableDepthSurface = hasCompletedBackground
                || (model.roomCoordinateFrame == .depthAnythingImageDepthMeters
                    && metadata["depthMeshUsesContinuousSurface", default: 0] >= 0.5)
            gestureHandlers?.setFlatPhotoNavigationEnabled(isFlatPhotoPlane)
            let opticalCenter = model.roomCoordinateFrame == .depthAnythingImageDepthMeters
                && !isFlatPhotoPlane
                && !hasNavigableDepthSurface
                ? cameraAnchor?.transform.translation
                : nil
            gestureHandlers?.setCapturedPhotoOpticalCenter(opticalCenter)
            let sourceVerticalFieldOfView = metadata["depthMeshVerticalFovDegrees"].map(Float.init)
            let imageWidth = metadata["depthMeshImageWidthPx"] ?? metadata["imageWidthPx"]
            let imageHeight = metadata["depthMeshImageHeightPx"] ?? metadata["imageHeightPx"]
            let sourceHorizontalFieldOfView: Float? = {
                guard let sourceVerticalFieldOfView,
                      let imageWidth, let imageHeight,
                      imageWidth > 1, imageHeight > 1 else { return nil }
                let verticalHalfFov = sourceVerticalFieldOfView * .pi / 360
                return 2 * atan(tan(verticalHalfFov) * Float(imageWidth / imageHeight)) * 180 / .pi
            }()
            gestureHandlers?.setLayeredPhotoLookLimits(
                maximumYaw: hasNavigableDepthSurface
                    ? metadata["depthMeshMaxYawRadians"].map(Float.init)
                    : nil,
                maximumPitch: hasNavigableDepthSurface
                    ? metadata["depthMeshMaxPitchRadians"].map(Float.init)
                    : nil,
                sourceHorizontalFieldOfView: hasNavigableDepthSurface
                    ? sourceHorizontalFieldOfView
                    : nil,
                sourceVerticalFieldOfView: hasNavigableDepthSurface
                    ? sourceVerticalFieldOfView
                    : nil,
                nearestReliableDepth: hasNavigableDepthSurface
                    ? metadata["depthMeshNearestReliableDepthM"].map(Float.init)
                    : nil
            )
        }

        func shouldReframeForViewportChange(_ size: CGSize) -> Bool {
            guard size.width > 1, size.height > 1 else { return false }
            let deltaWidth = abs(size.width - lastViewportSize.width)
            let deltaHeight = abs(size.height - lastViewportSize.height)
            guard lastViewportSize == .zero || deltaWidth > 8 || deltaHeight > 8 else {
                return false
            }
            lastViewportSize = size
            return true
        }
        
        func setupGestures(for arView: ARView, placementManager: ARObjectPlacementManager) {
            gestureHandlers = RealityKitGestureHandlers(arView: arView)
            self.arObjectPlacementManager = placementManager

            // Connect object placement manager to gesture handlers for manipulation support
            gestureHandlers?.setObjectPlacementManager(placementManager)

            // Add tap gesture for AR object placement
            let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
            arView.addGestureRecognizer(tapGesture)
        }
        
        // Set up custom camera for non-AR mode with controllable rotation
        // NOTE: This only CREATES the camera, does NOT add to scene yet (that happens after model loads)
        func setupCustomCamera(for arView: ARView) {
            // Create perspective camera entity with reasonable field of view
            cameraEntity = PerspectiveCamera()
            cameraEntity?.camera.fieldOfViewOrientation = .vertical
            cameraEntity?.camera.fieldOfViewInDegrees = 60.0

            // Create camera anchor - initial position will be set after model loads and bounds are calculated
            cameraAnchor = AnchorEntity(world: SIMD3<Float>(0, 0, 0)) // Temporary position
            cameraAnchor?.name = "CustomCameraAnchor" // Give it a name for identification

            // Add camera entity to anchor with no offset (prevents orbital rotation)
            if let camera = cameraEntity, let anchor = cameraAnchor {
                camera.position = SIMD3<Float>(0, 0, 0) // No offset from anchor center
                camera.name = "CustomPerspectiveCamera" // Name the camera
                
                // Configure camera for full screen rendering
                camera.camera.near = 0.1
                camera.camera.far = 100.0
                
                anchor.addChild(camera)
                // ❌ DON'T add to scene here - will add AFTER model loads and positioning
                // arView.scene.addAnchor(anchor)

                // Pass camera references to gesture handlers for direct camera control
                gestureHandlers?.setCameraReferences(camera: camera, cameraAnchor: anchor)
                // The look-at target can be resolved before this camera exists; replay it so
                // the first single-finger drag already orbits the room.
                gestureHandlers?.setOrbitTarget(cameraLookAtTarget)
                installCameraMoveObservers()

                logDebug("📷 Custom camera CREATED (position will be set after model loads and bounds calculated)")
            }
        }

        // Add camera to scene - called AFTER model loads to ensure camera takes precedence
        func addCameraToScene(arView: ARView) {
            guard let cameraAnchor = cameraAnchor else {
                logDebug("❌ [addCameraToScene] No camera anchor to add!")
                return
            }

            // First, find and log any existing cameras in the scene
            var existingCameraCount = 0
            for anchor in arView.scene.anchors {
                func findCameras(in entity: Entity) {
                    if entity is PerspectiveCamera {
                        existingCameraCount += 1
                        logDebug("⚠️ Found existing PerspectiveCamera: \(entity.name.isEmpty ? "unnamed" : entity.name)")
                    }
                    for child in entity.children {
                        findCameras(in: child)
                    }
                }
                findCameras(in: anchor)
            }
            logDebug("📷 [addCameraToScene] Found \(existingCameraCount) existing cameras in scene")

            // Add our camera anchor to the scene LAST
            arView.scene.addAnchor(cameraAnchor)
            logDebug("📷 [addCameraToScene] Camera anchor added to scene as LAST anchor")
            logDebug("   Total anchors in scene: \(arView.scene.anchors.count)")
            
            // ✅ Try a more aggressive approach - remove ALL cameras then add ours
            if let cameraEntity = cameraEntity {
                logDebug("🧹 Clearing all existing cameras from scene before adding ours")
                
                // Collect all existing camera entities
                var existingCameras: [PerspectiveCamera] = []
                for anchor in arView.scene.anchors {
                    func collectCameras(in entity: Entity) {
                        if let perspectiveCamera = entity as? PerspectiveCamera,
                           perspectiveCamera !== cameraEntity {
                            existingCameras.append(perspectiveCamera)
                        }
                        for child in entity.children {
                            collectCameras(in: child)
                        }
                    }
                    collectCameras(in: anchor)
                }
                
                // Remove all existing cameras
                for camera in existingCameras {
                    camera.parent?.removeChild(camera)
                    logDebug("🗑️ Removed existing camera: \(camera.name)")
                }
                
                logDebug("✅ [addCameraToScene] Scene cleared. Our camera should now be the only one.")
                logDebug("   Camera Entity: \(cameraEntity)")
                logDebug("   Camera Name: \(cameraEntity.name)")
                logDebug("   Camera FOV: \(cameraEntity.camera.fieldOfViewInDegrees)")
                logDebug("   Camera Position: \(cameraAnchor.transform.translation)")
            }
        }
        
        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            // Handle AR object placement if AR is active
            Task { @MainActor in
                if let arManager = arObjectPlacementManager,
                   arManager.isReadyToPlace {
                    let location = gesture.location(in: gesture.view)
                    let _ = arManager.handleTapToPlace(at: location)
                }
            }
        }
    }
    
    // ✅ FIXED: Handle both bundle rooms and saved rooms
    private func loadModel(into arView: ARView, coordinator: Coordinator) {
        logDebug("🎨 [RealityKitView.loadModel] ========================================")
        logDebug("🎨 [RealityKitView.loadModel] Starting to load model: \(model.displayName)")
        logDebug("   - Is saved room: \(model.isSavedRoom)")
        
        // Get the model URL (works for both bundle and saved rooms)
        guard let modelURL = model.temporaryURL else {
            logDebug("❌ [RealityKitView.loadModel] CRITICAL: No URL for model!")
            logDebug("   - Model name: \(model.name)")
            logDebug("   - File name: \(model.fileName)")
            logDebug("   - Is saved room: \(model.isSavedRoom)")
            return
        }
        coordinator.depthCameraMetadata = model.roomCoordinateFrame == .depthAnythingImageDepthMeters
            ? CameraExifSidecar.load(roomURL: modelURL)
            : nil
        
        logDebug("🎨 [RealityKitView.loadModel] Got model URL: \(modelURL.path)")
        logDebug("   - Last path component: \(modelURL.lastPathComponent)")
        
        // Verify file exists
        let fileExists = FileManager.default.fileExists(atPath: modelURL.path)
        logDebug("   - File exists: \(fileExists)")
        
        if fileExists {
            do {
                let attributes = try FileManager.default.attributesOfItem(atPath: modelURL.path)
                if let fileSize = attributes[.size] as? UInt64 {
                    logDebug("   - File size: \(fileSize) bytes (\(Double(fileSize) / 1024.0 / 1024.0) MB)")
                }
                let isReadable = FileManager.default.isReadableFile(atPath: modelURL.path)
                logDebug("   - Is readable: \(isReadable)")
            } catch {
                logDebug("   - Error getting file attributes: \(error)")
            }
        } else {
            logDebug("❌ [RealityKitView.loadModel] CRITICAL: File does not exist at path!")
            return
        }
        
        // Load USDZ model using RealityKit's Entity loading
        logDebug("🎨 [RealityKitView.loadModel] Starting async entity load...")
        
        Task { @MainActor in
            do {
                // Use the async initializer of ModelEntity to load the model without blocking
                logDebug("🎨 [RealityKitView.loadModel] Loading model asynchronously using ModelEntity(contentsOf:)")
                let modelEntity = try await ModelEntity(contentsOf: modelURL)
                
                logDebug("✅ [RealityKitView.loadModel] Entity loaded successfully!")
                logDebug("   - Entity name: '\(modelEntity.name)'")
                logDebug("   - Entity position: \(modelEntity.position)")
                logDebug("   - Entity scale: \(modelEntity.scale)")
                logDebug("   - Has children: \(modelEntity.children.count)")
                
                if !modelEntity.children.isEmpty {
                    logDebug("   - Child entities:")
                    for (index, child) in modelEntity.children.enumerated().prefix(5) {
                        logDebug("     [\(index)] \(child.name.isEmpty ? "unnamed" : child.name) - position: \(child.position)")
                    }
                    if modelEntity.children.count > 5 {
                        logDebug("     ... and \(modelEntity.children.count - 5) more children")
                    }
                }

                if model.roomCoordinateFrame == .depthAnythingImageDepthMeters {
                    Self.restoreDepthAnythingPhotoAspectIfNeeded(
                        modelEntity,
                        modelURL: modelURL,
                        photoOrientation: model.photoOrientation,
                        cameraMetadata: coordinator.depthCameraMetadata
                    )
                }

                // Ensure model has proper materials for visibility
                ensureModelHasMaterials(modelEntity)
                if model.roomCoordinateFrame == .depthAnythingImageDepthMeters {
                    normalizePhotoRoomMaterials(in: modelEntity)
                }

                // Calculate model bounds for camera positioning
                let bounds = modelEntity.components[ModelComponent.self]?.mesh.bounds
                if let bounds = bounds {
                    logDebug("📦 Model bounds after loading: min(\(bounds.min)), max(\(bounds.max))")
                } else {
                    logDebug("📦 Model bounds after loading: no bounds")
                }
                
                // Check for lights in the scene
                logDebug("🎨 [RealityKitView.loadModel] Checking for lights in scene...")
                #if false
                var lightCount = 0
                
                func countLights(in entity: Entity) {
                    if entity.components[PointLightComponent.self] != nil {
                        lightCount += 1
                        logDebug("     💡 Found PointLight: \(entity.name.isEmpty ? "unnamed" : entity.name)")
                    }
                    if entity.components[DirectionalLightComponent.self] != nil {
                        lightCount += 1
                        logDebug("     💡 Found DirectionalLight: \(entity.name.isEmpty ? "unnamed" : entity.name)")
                    }
                    if entity.components[SpotLightComponent.self] != nil {
                        lightCount += 1
                        logDebug("     💡 Found SpotLight: \(entity.name.isEmpty ? "unnamed" : entity.name)")
                    }
                    
                    for child in entity.children {
                        countLights(in: child)
                    }
                }
                
                countLights(in: modelEntity)
                logDebug("   - Total lights found in model: \(lightCount)")
                #endif
                // Use helper method to count lights in the model entity
                let lightCount = self.countLights(in: modelEntity)
                logDebug("   - Total lights found in model: \(lightCount)")
                
                if lightCount == 0,
                   model.roomCoordinateFrame != .depthAnythingImageDepthMeters {
                    logDebug("⚠️ [RealityKitView.loadModel] WARNING: NO LIGHTS IN SCENE!")
                    logDebug("   - This explains the black screen!")
                    logDebug("   - Adding emergency lighting...")
                    
                    // Add emergency lighting directly to the model entity
                    let pointLight = PointLight()
                    pointLight.light.intensity = 2000
                    pointLight.light.attenuationRadius = 100
                    pointLight.position = [0, 2, 0]
                    modelEntity.addChild(pointLight)
                    logDebug("   - ✅ Added emergency point light at [0, 2, 0]")
                    
                    let ambientLight = PointLight()
                    ambientLight.light.intensity = 1000
                    ambientLight.light.attenuationRadius = 100
                    ambientLight.position = [0, 3, 2]
                    modelEntity.addChild(ambientLight)
                    logDebug("   - ✅ Added emergency ambient light at [0, 3, 2]")
                    
                    let fillLight = PointLight()
                    fillLight.light.intensity = 1500
                    fillLight.light.attenuationRadius = 100
                    fillLight.position = [2, 1, 2]
                    modelEntity.addChild(fillLight)
                    logDebug("   - ✅ Added emergency fill light at [2, 1, 2]")
                } else if lightCount == 0 {
                    // A photo room is authored as an unlit/constant material. Adding fallback
                    // lights makes its base texture combine with emissive data in older saved
                    // USDZ files and visibly brightens the original photograph.
                    logDebug("✅ [RealityKitView.loadModel] Keeping photo room color-faithful without fallback lights")
                }
                
                // Clean up any previous model anchor to avoid state pollution
                logDebug("🎨 [RealityKitView.loadModel] Adding model to scene...")
                if let oldAnchor = coordinator.worldAnchor {
                    arView.scene.removeAnchor(oldAnchor)
                    coordinator.worldAnchor = nil
                    logDebug("🧹 [RealityKitView] Removed previous model anchor from scene")
                }

                // Also remove camera anchor if it exists (will re-add after model)
                if let cameraAnchor = coordinator.cameraAnchor {
                    arView.scene.removeAnchor(cameraAnchor)
                    logDebug("🧹 [RealityKitView] Removed camera anchor (will re-add after model)")
                }

                let modelAnchor = AnchorEntity(world: SIMD3<Float>(0, 0, 0))
                modelAnchor.addChild(modelEntity)
                arView.scene.addAnchor(modelAnchor)
                coordinator.scene = arView.scene
                
                // Store and broadcast the new world/model anchor
                coordinator.worldAnchor = modelAnchor
                // Ensure object placement manager uses the fresh scene and anchor
                arObjectPlacementManager.setSceneReferences(arView: arView, scene: arView.scene)
                arObjectPlacementManager.setWorldAnchor(modelAnchor)
                logDebug("📌 [RealityKitView] World anchor set on placement manager")

                // Set up boundary manager for camera constraints
                let boundaryManager = RealityKitBoundaryManager(arView: arView)
                let navigationMetadata = coordinator.depthCameraMetadata ?? [:]
                let hasCompletedBackground = model.roomCoordinateFrame == .depthAnythingImageDepthMeters
                    && navigationMetadata["depthMeshHasCompletedBackground", default: 0] >= 0.5
                let hasNavigableDepthSurface = hasCompletedBackground
                    || (model.roomCoordinateFrame == .depthAnythingImageDepthMeters
                        && navigationMetadata["depthMeshUsesContinuousSurface", default: 0] >= 0.5)
                boundaryManager.setUsesCapturedPhotoFrustum(
                    model.roomCoordinateFrame == .depthAnythingImageDepthMeters
                        && !hasNavigableDepthSurface
                )
                // Option B: Ensure fresh bounds per model load (avoid inheriting previous room bounds)
                boundaryManager.reset()
                boundaryManager.setCompletedPhotoCameraEnvelope(
                    forwardTranslation: hasNavigableDepthSurface
                        ? navigationMetadata["depthMeshForwardTranslationM"]
                            .map(Float.init)
                            ?? navigationMetadata["depthMeshMaxTranslationM"].map(Float.init)
                        : nil,
                    lateralTranslation: hasNavigableDepthSurface
                        ? navigationMetadata["depthMeshLateralTranslationM"].map(Float.init)
                        : nil,
                    backwardTranslation: hasNavigableDepthSurface
                        ? navigationMetadata["depthMeshBackwardTranslationM"].map(Float.init)
                        : nil
                )
                logDebug("🧹 [RealityKitView] Boundary manager reset before calculating new room bounds")
                boundaryManager.calculateRoomBounds(from: modelEntity)
                coordinator.gestureHandlers?.setBoundaryManager(boundaryManager)

                // ✅ Store boundary manager and model ID in coordinator for camera reset on revisit
                coordinator.boundaryManager = boundaryManager
                coordinator.currentModelID = self.model.id
                logDebug("📝 [RealityKitView] Stored model ID: \(self.model.id) for tracking")

                // Share boundary manager with camera movement manager
                self.cameraMovementManager.setupARView(arView)
                self.cameraMovementManager.setBoundaryManager(boundaryManager)
                
                // Keep RealityKit camera movement on the stable default speed.
                DispatchQueue.main.async { [weak cameraMovementManager = self.cameraMovementManager] in
                    cameraMovementManager?.setSpeed(.normal)
                }
                
                // Position custom camera inside the room bounds
                logDebug("🔍 [RealityKitView] === CAMERA POSITIONING DEBUG ===")
                logDebug("   cameraAnchor exists: \(coordinator.cameraAnchor != nil)")
                logDebug("   boundaryManager.bounds exists: \(boundaryManager.bounds != nil)")

                if let cameraAnchor = coordinator.cameraAnchor, let bounds = boundaryManager.bounds {
                    let placementLabel = model.roomCoordinateFrame == .depthAnythingImageDepthMeters
                        ? "DEPTH ANYTHING front-center"
                        : (model.roomCoordinateFrame.usesFrontFacingRealityKitCamera ? "front-center" : "back-left corner")
                    logDebug("✅ [RealityKitView] BOUNDS AVAILABLE - using \(placementLabel) positioning")
                    logDebug("   Room bounds min: \(bounds.min)")
                    logDebug("   Room bounds max: \(bounds.max)")

                    coordinator.cameraLookAtTarget = Self.repositionOptimalCamera(
                        cameraAnchor: cameraAnchor,
                        cameraEntity: coordinator.cameraEntity,
                        boundaryManager: boundaryManager,
                        model: model,
                        cameraMetadata: coordinator.depthCameraMetadata
                    )
                    coordinator.lastViewportSize = arView.bounds.size
                    let lookAt = coordinator.cameraLookAtTarget ?? SIMD3<Float>(
                        (bounds.min.x + bounds.max.x) * 0.5,
                        (bounds.min.y + bounds.max.y) * 0.5,
                        (bounds.min.z + bounds.max.z) * 0.5
                    )
                    let lookDirection = Self.normalizedDirection(
                        from: cameraAnchor.transform.translation,
                        to: lookAt
                    )
                    logDebug("📷 [RealityKitView] Camera \(placementLabel) positioned:")
                    logDebug("   📍 Final Position: \(cameraAnchor.transform.translation)")
                    logDebug("   👁️ Looking at: \(lookAt)")
                    logDebug("   🧭 Direction: \(lookDirection)")

                    // ✅ Add camera to scene AFTER model and AFTER positioning (ensures camera takes precedence)
                    coordinator.addCameraToScene(arView: arView)

                    // ✅ Sync gesture handler rotation state with camera's new orientation
                    coordinator.configureNavigationContract(for: model)
                    arView.accessibilityIdentifier = "saved_room_viewport"
                    logDebug("✅ [RealityKitView] Camera ready - gestures synced with camera orientation")
                } else if let cameraAnchor = coordinator.cameraAnchor {
                    // Fallback if no bounds - use default position
                    logDebug("⚠️ [RealityKitView] NO BOUNDS - using DEFAULT position")
                    let defaultPosition: SIMD3<Float>
                    let defaultLookAt: SIMD3<Float>
                    if model.roomCoordinateFrame == .depthAnythingImageDepthMeters {
                        defaultPosition = SIMD3(0, 0, -2.5)
                        defaultLookAt = SIMD3(0, 0, 0)
                    } else {
                        defaultPosition = SIMD3(0, 1.5, 3)
                        defaultLookAt = SIMD3(0, 1.4, 0)
                    }
                    cameraAnchor.transform.translation = defaultPosition

                    _ = Self.applyCameraPose(
                        cameraAnchor,
                        position: defaultPosition,
                        lookAt: defaultLookAt
                    )
                    coordinator.cameraLookAtTarget = defaultLookAt

                    logDebug("📷 Custom camera positioned at default: \(defaultPosition) (no bounds available)")

                    // ✅ Add camera to scene AFTER model and AFTER positioning
                    coordinator.addCameraToScene(arView: arView)

                    // ✅ Sync gesture handler rotation state with camera's new orientation
                    coordinator.configureNavigationContract(for: model)
                    logDebug("✅ [RealityKitView] Camera ready - gestures synced with camera orientation")
                } else {
                    logDebug("❌ [RealityKitView] NO CAMERA ANCHOR - cannot position camera!")
                }
                logDebug("🔍 [RealityKitView] === END CAMERA POSITIONING DEBUG ===")
                
                // Set up camera movement manager with custom camera references
                self.cameraMovementManager.setupARView(arView)
                
                // Share camera references with camera movement manager for joystick control
                if let cameraAnchor = coordinator.cameraAnchor {
                    self.cameraMovementManager.setCameraAnchor(cameraAnchor)
                }
                
                // Set up camera movement callback
                self.cameraMovementManager.onCameraMove = {
                    // Camera movement callback - ready for future enhancements
                }
                
                logDebug("✅ [RealityKitView.loadModel] Complete setup finished successfully")
                logDebug("🎨 [RealityKitView.loadModel] ========================================")
                
            } catch {
                logDebug("❌ [RealityKitView.loadModel] FAILED TO LOAD ENTITY!")
                logDebug("   - Error: \(error)")
                logDebug("   - Error description: \(error.localizedDescription)")
                logDebug("   - Model URL: \(modelURL.path)")
                logDebug("🎨 [RealityKitView.loadModel] ========================================")
                CrashReporter.shared.report(error, context: "Loading 3D Model")
            }
        }
    }
    
    private func setupCamera(for arView: ARView, with bounds: BoundingBox?) {
        // Calculate room dimensions from bounds
        let roomSize: SIMD3<Float>
        let roomCenter: SIMD3<Float>
        
        if let bounds = bounds {
            roomSize = bounds.max - bounds.min
            roomCenter = (bounds.min + bounds.max) / 2
        } else {
            // Default room size
            roomSize = SIMD3<Float>(5, 3, 5)
            roomCenter = SIMD3<Float>(0, 1.5, 0)
        }
        
        // Position camera INSIDE the room, slightly above floor level
        let cameraHeight = bounds?.min.y ?? 0.0 + (roomSize.y * 0.4) // 40% up from floor
        let viewingDistance = min(roomSize.x, roomSize.z) * 0.3 // 30% of smaller horizontal dimension
        
        // Position camera inside room, looking toward the center
        let cameraPosition = SIMD3<Float>(
            roomCenter.x - viewingDistance,
            cameraHeight,
            roomCenter.z + viewingDistance * 0.5
        )
        
        // Create camera transform
        var cameraTransform = Transform.identity
        cameraTransform.translation = cameraPosition
        
        // Look at room center
        let lookDirection = normalize(roomCenter - cameraPosition)
        cameraTransform.rotation = simd_quatf(from: SIMD3<Float>(0, 0, -1), to: lookDirection)
        
        logDebug("📷 Camera configured at position: \(cameraPosition)")
    }
    
    private func setupLighting(for arView: ARView) {
        let quality = appState.currentQuality
        let lightingMultiplier = quality.lightingIntensity
        
        // Create ambient light
        let ambientLightComponent = DirectionalLightComponent(
            color: .white,
            intensity: Float(300 * 3.0),
            isRealWorldProxy: false
        )
        
        let ambientLightEntity = Entity()
        ambientLightEntity.components.set(ambientLightComponent)
        
        // Create key light
        let keyLightComponent = DirectionalLightComponent(
            color: .white,
            intensity: Float(800 * lightingMultiplier),
            isRealWorldProxy: false
        )
        
        let keyLightEntity = Entity()
        keyLightEntity.components.set(keyLightComponent)
        keyLightEntity.orientation = simd_quatf(angle: .pi / 4, axis: SIMD3<Float>(1, 0, 0))
        keyLightEntity.position = SIMD3<Float>(5, 10, 5)
        
        // Create additional light specifically for placed 3D objects
        let objectLightComponent = DirectionalLightComponent(
            color: .white,
            intensity: Float(1500 * lightingMultiplier), // Brighter for 3D objects
            isRealWorldProxy: false
        )

        let objectLightEntity = Entity()
        objectLightEntity.components.set(objectLightComponent)
        objectLightEntity.orientation = simd_quatf(angle: .pi / 6, axis: SIMD3<Float>(1, 0, 0)) // 30 degrees down
        objectLightEntity.position = SIMD3<Float>(0, 8, 2) // Above and slightly forward

        // Add lights to scene
        let lightingAnchor = AnchorEntity(.world(transform: matrix_identity_float4x4))
        lightingAnchor.addChild(ambientLightEntity)
        lightingAnchor.addChild(keyLightEntity)
        lightingAnchor.addChild(objectLightEntity)
        arView.scene.addAnchor(lightingAnchor)

        logDebug("💡 Applied lighting intensity: \(lightingMultiplier)x for \(quality.displayName) quality")
        logDebug("💡 Added dedicated lighting for placed 3D objects")
    }
    
    /// Photo rooms carry their own color in an unlit texture. Keep the viewport and IBL neutral so
    /// uncovered depth samples cannot read as a grey haze around or through the saved mesh.
    private static func configureViewerBackground(_ arView: ARView, roomCoordinateFrame: RoomCoordinateFrame) {
        if roomCoordinateFrame == .depthAnythingImageDepthMeters {
            arView.environment.background = .color(.black)
            arView.environment.lighting.resource = nil
            arView.environment.lighting.intensityExponent = 0
        } else {
            arView.environment.background = .color(.init(white: 0, alpha: 1.0))
        }
    }

    // Configure rendering quality based on user settings
    private func configureRenderingQuality(arView: ARView, quality: AssetQuality) {
        if model.roomCoordinateFrame == .depthAnythingImageDepthMeters {
            // The room surface is a photograph, not a relightable 3D material. Preserve source
            // color independent of the global quality profile or environment-light estimate.
            arView.renderOptions.insert(.disableAREnvironmentLighting)
            if #available(iOS 15.0, *) {
                arView.environment.sceneUnderstanding.options = []
            }
            logDebug("🔄 Applied color-faithful photo-room rendering")
            return
        }

        switch quality {
        case .standard:
            arView.renderOptions.remove(.disableAREnvironmentLighting)
            if #available(iOS 15.0, *) {
                arView.environment.sceneUnderstanding.options = []
            }
        case .high:
            arView.renderOptions.insert(.disableAREnvironmentLighting)
            if #available(iOS 15.0, *) {
                arView.environment.sceneUnderstanding.options = .collision
            }
        case .best:
            arView.renderOptions.insert(.disableAREnvironmentLighting)
            if #available(iOS 15.0, *) {
                arView.environment.sceneUnderstanding.options = [.collision, .physics]
            }
        }
        
        logDebug("🔄 Updated rendering quality to: \(quality.displayName)")
    }

    // Ensure loaded model has proper materials for visibility
    private func normalizePhotoRoomMaterials(in entity: Entity) {
        if var modelComponent = entity.components[ModelComponent.self] {
            modelComponent.materials = modelComponent.materials.map { material in
                let sourceColor: UnlitMaterial.BaseColor
                if let unlit = material as? UnlitMaterial {
                    sourceColor = unlit.color
                } else if let pbr = material as? PhysicallyBasedMaterial {
                    sourceColor = pbr.baseColor
                } else if let simple = material as? SimpleMaterial {
                    sourceColor = simple.color
                } else {
                    return material
                }

                var opaqueColor = sourceColor
                opaqueColor.tint = sourceColor.tint.withAlphaComponent(1)

                // The saved room is a captured photograph, not a relightable surface. Converting
                // any importer-created PBR/Simple material also protects older saved USDZ files
                // whose SceneKit `.constant` material was not preserved as unlit by RealityKit.
                var unlit = UnlitMaterial(applyPostProcessToneMap: false)
                unlit.color = opaqueColor
                unlit.blending = .opaque
                unlit.opacityThreshold = nil
                unlit.faceCulling = .none
                unlit.readsDepth = true
                unlit.writesDepth = true
                return unlit
            }
            entity.components.set(modelComponent)
        }
        for child in entity.children {
            normalizePhotoRoomMaterials(in: child)
        }
    }

    private func ensureModelHasMaterials(_ entity: Entity) {
        // Check if entity itself has a model component and materials
        if var modelComponent = entity.components[ModelComponent.self] {
            if modelComponent.materials.isEmpty {
                // Add default material if none exists
                let defaultMaterial = SimpleMaterial(color: .white, roughness: 0.5, isMetallic: false)
                modelComponent.materials = [defaultMaterial]
                entity.components.set(modelComponent)
                logDebug("🎨 Added default white material to model entity")
            } else {
                logDebug("🎨 Model has \(modelComponent.materials.count) existing materials")
            }
        }

        // Recursively check child entities
        for child in entity.children {
            ensureModelHasMaterials(child)
        }
    }

    /// Recursively count the number of light components in an entity hierarchy.
    /// - Parameter entity: The root entity to inspect.
    /// - Returns: The total number of `PointLightComponent`, `DirectionalLightComponent`, or `SpotLightComponent` present in the entity and its descendants.
    ///
    /// This helper is annotated with `@MainActor` because RealityKit entity properties such as
    /// `components` and `children` are actor-isolated. By executing on the main actor, we can
    /// safely traverse the entity tree without triggering concurrency violations.
    @MainActor
    private func countLights(in entity: Entity) -> Int {
        var total = 0
        if entity.components[PointLightComponent.self] != nil {
            total += 1
        }
        if entity.components[DirectionalLightComponent.self] != nil {
            total += 1
        }
        if entity.components[SpotLightComponent.self] != nil {
            total += 1
        }
        for child in entity.children {
            total += countLights(in: child)
        }
        return total
    }
}

// MARK: - Extensions for SIMD math operations are defined in RealityKitObjectPlacementManager.swift
