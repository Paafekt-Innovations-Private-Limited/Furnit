# Engineering Help Request: Single-Photo Room Completion and Inside-Room Navigation

Status: dated investigation record; not the current implementation guide
Date: 2026-08-14
Repository: `Paafekt-Innovations-Private-Limited/Furnit`
Branch: `main`
Last pushed commit: `316aa0bcadc9e5f45cac4bd868a7f7ae0979c43e`
Platforms: iOS and Android

## Current outcome pointer (2026-08-15 push candidate)

- iOS and Android now generate one opaque version-5 continuous depth surface before preview. Neither
  active path emits the investigated completed-background/foreground enclosure.
- Each platform previews its generated final artifact and Save promotes that same USDZ/GLB instead
  of rerunning inference or rebuilding geometry. Android additionally checks byte identity while
  promoting the preview GLB.
- Both platforms include synthetic-fixture UI regression coverage. No private room photograph is
  included in the test targets.
- Neither platform's visible result is documented as resolved. Fresh portrait/landscape manual tests,
  including preview → save → reopen comparison with the chair/fan room, remain required.

The remainder of this document preserves the pre-implementation problem statement and alternatives
for engineering history. For current behavior use `Furnit/docs/README.md` and
`android/docs/ANDROID_ROOM_CREATION.md`.

## Problem statement to send to another engineer

Furnit creates a room-like 3D asset from one room photograph. Metric monocular depth is available,
along with camera focal estimates and RTMDet instance segmentation. The initial preview looks correct,
but saved-room navigation does not feel like a human standing inside a room and turning their head.

We need a technically sound design and implementation for **limited inside-room movement and
turn-in-place navigation from one source photograph**, without the following artifacts:

- furniture and ceiling-fan pixels moving away from their intended positions;
- duplicate chair/fan images left on a far backing plane;
- gray holes or trails exposed behind foreground objects;
- depth-discontinuity triangles stretching object-edge pixels into long wedges;
- camera interaction feeling like orbiting or dragging a hanging picture/plane;
- a saved room behaving differently from its visually correct preview for unexplained reasons.

The desired interaction is:

- one-finger drag right turns the virtual camera/eye right (the rendered scene moves left);
- one-finger drag rotates around the camera position, not around the room or photo plane;
- pinch should provide a believable limited move into/out of the reconstructed room when the asset
  supports translation;
- the camera should be constrained to a conservative valid-view volume;
- walls/floor/ceiling revealed behind segmented foreground objects should use completed background
  texture, not copied chair/fan pixels;
- unseen content must not be represented as measured/observed truth; generated completion may be
  used if explicitly treated as appearance completion;
- preview behavior should stay sharp and stable.

The central question is not merely gesture direction. It is:

> What asset representation and camera-validity contract should we use so a single photo can support
> convincing, bounded head rotation and modest translation, while preserving foreground placement and
> preventing duplicated or stretched source pixels?

## User-observed evidence

The source scene contains a dark office chair in the foreground and a ceiling fan. Reported results:

1. Moving/zooming into the projective depth mesh separated the chair and fan from the background.
2. The far backing photo still contained the original chair/fan pixels, leaving duplicates/trails.
3. Removing or exposing the backing produced gray holes.
4. Keeping the camera at the authored optical center removed the trails, but turning did not feel like
   walking/looking inside a room; it felt like a plane suspended in front of the camera.
5. On iOS, one-finger rotation did not run for current saved rooms because those rooms are tagged as
   flat photo planes and the gesture handler returned through its photo-pan branch.
6. Preview is reported as visually correct; saved-room reopening is the problematic path.

Private local screenshots supplied during debugging showed a close view with heavily stretched
chair/wall pixels, Android gray gaps around chair/fan, iOS distorted/duplicated foreground regions,
repeated chair/fan silhouettes after zoom, and a duplicated/stretched ceiling fan. The files are not
part of the repository.

Earlier Android logs also showed the depth-photo model being loaded correctly; the visible problem was
rendering/navigation, not RTMDet initialization. A separate WebView setup failure occurred during one
experimental camera-bounds change (`Cannot read properties of null (reading 'maxX')`) and was not the
root cause of the visual artifacts.

## Repository state when the investigation began

The working tree initially contained an **unfinished, uncompiled, unverified iOS experiment** in:

- `Furnit/Services/RoomReconstruction/DepthAnythingRoomReconstructor.swift`
- `Furnit/Utilities/RealityKitBoundaryManager.swift`
- `Furnit/Views/Components/RealityKitView.swift`

That experiment attempted foreground masking, deterministic background filling, a two-layer USDZ,
and different camera bounds. Its active save selection was rejected in favor of the stable iOS
version-3 flat plane. Some generalized helper and compatibility code remains compiled but dormant.

The pushed gesture-range change is commit `316aa0bc`. It allows unrestricted yaw and near-vertical
pitch for captured projective rooms while preserving the authored optical center, but it does not
solve missing side/background content.

## End-to-end flow and why preview is not saved-room evidence

### iOS preview path

Entry and renderer:

```text
SinglePhotoRoomViewer
  -> makeDepthAnythingPreviewDestination
  -> DepthAnythingPreviewRoomView
  -> DepthAnythingPreviewSceneView (SceneKit SCNView)
  -> one SCNPlane textured with the full photo
```

The preview deliberately skips all expensive reconstruction:

```swift
private func makeDepthAnythingPreviewDestination(
    image: UIImage,
    cameraMetadata: [String: Double],
    photoOrientation _: PhotoOrientation
) throws -> DepthAnythingPreparedPreview {
    let fixed = image.fixedOrientation()
    let previewImage = downsampleDepthAnythingPreviewImage(fixed, maxDimension: 1600)
    // ...write JPEG...
    let roomWidth: Float = 2.0
    let roomHeight = max(1.8, roomWidth * Float(height) / Float(width))
    let roomDepth: Float = 3.0
    logDebug(
        "[DepthAnythingRoom][PreviewFast] skipping " +
        "depth_anything/geocalib/rtmdet/room_height/usdz_export during creation"
    )
    // ...return preview metadata...
}
```

The visual is one sharp photo plane:

```swift
private func makeScene(context: Context) -> SCNScene {
    let scene = SCNScene()
    let image = UIImage(contentsOfFile: imageURL.path)
    let plane = SCNPlane(
        width: CGFloat(max(roomWidthMeters, 0.05)),
        height: CGFloat(max(roomHeightMeters, 0.05))
    )
    let material = SCNMaterial()
    material.diffuse.contents = image
    material.lightingModel = .constant
    material.isDoubleSided = true
    plane.materials = [material]
    scene.rootNode.addChildNode(SCNNode(geometry: plane))
    // ...fixed camera setup...
    return scene
}
```

Preview gestures pan and zoom the framed photograph; they do not navigate reconstructed geometry:

```swift
@objc private func handlePan(_ recognizer: UIPanGestureRecognizer) {
    // translation becomes cameraOffset X/Y
    cameraOffset.wrappedValue = CGSize(
        width: panStartOffset.width - horizontalUnits,
        height: panStartOffset.height + verticalUnits
    )
}

@objc private func handlePinch(_ recognizer: UIPinchGestureRecognizer) {
    cameraZoom.wrappedValue = min(max(pinchStartZoom * recognizer.scale, 0.55), 4.0)
}
```

Therefore “preview is correct” means the full photograph is framed correctly. It does **not** prove
that the saved USDZ, RealityKit camera, or saved-room gesture contract is correct.

### iOS save and reopen path on pushed `main`

On first Save:

```text
DepthAnythingRoomReconstructor.reconstructWithResult
  -> downsample and orient image
  -> run GeoCalib, Depth Anything, and RTMDet concurrently
  -> calibrate metric depth and W x H x D
  -> buildPhotoPlaneMesh (current pushed format version 3)
  -> export USDZ
  -> reopen through ModelViewerView -> RealityKitView
```

The currently pushed exporter intentionally replaced the former depth-displaced surface with a flat
saved photo plane:

```swift
// Pushed main at 316aa0bc
enrichedCalibrationMetadata["depthMeshProjectionVersion"] = 3
enrichedCalibrationMetadata["depthMeshIsFlatPhotoPlane"] = 1
enrichedCalibrationMetadata["depthMeshFocalLengthXPx"] = Double(meshFocalXPixels)
enrichedCalibrationMetadata["depthMeshFocalLengthYPx"] = Double(meshFocalYPixels)
enrichedCalibrationMetadata["depthMeshVerticalFovDegrees"] = Double(meshVerticalFovDegrees)

let mesh = try buildPhotoPlaneMesh(
    image: workingImage,
    depthMap: calibratedDepthMap,
    widthMeters: meshDisplayDimensions.width,
    heightMeters: meshDisplayDimensions.height
)
let url = try exportUSDZ(mesh: mesh, textureImage: workingImage)
```

That choice prevents depth-edge tearing, but cannot surround the camera. Rotating away from it must
look like turning away from a photograph.

The codebase still contains a calibrated pinhole mesh builder:

```swift
func buildPerspectiveMesh(
    image: UIImage,
    depthMap: [[Float]],
    focalXPixels: Float,
    focalYPixels: Float
) throws -> MDLMesh {
    try buildMesh(
        image: image,
        depthMap: depthMap,
        projection: .pinhole(
            focalXPixels: focalXPixels,
            focalYPixels: focalYPixels
        )
    )
}
```

Pinhole vertices are created as:

```swift
case let .pinhole(focalXPixels, focalYPixels):
    x = -(Float(column) - centerX) * depth / focalXPixels
    y =  (Float(row) - centerY) * depth / focalYPixels
    z = depth
```

Triangles across depth discontinuities are omitted. This avoids long wedges, but creates holes that
need valid background content.

## Shipped iOS gesture and camera behavior

Saved USDZ rooms use `RealityKitView` and `RealityKitGestureHandlers`, not the preview SceneKit
coordinator.

On pushed `main`, navigation classification is:

```swift
func configureNavigationContract(for model: USDZModel) {
    let metadata = model.temporaryURL.map { CameraExifSidecar.load(roomURL: $0) } ?? [:]
    let usesFlatPhotoPlane = model.roomCoordinateFrame == .depthAnythingImageDepthMeters
        && metadata["depthMeshIsFlatPhotoPlane", default: 0] >= 0.5
    gestureHandlers?.setFlatPhotoNavigationEnabled(usesFlatPhotoPlane)
    let opticalCenter = model.roomCoordinateFrame == .depthAnythingImageDepthMeters
        && !usesFlatPhotoPlane
        ? cameraAnchor?.transform.translation
        : nil
    gestureHandlers?.setCapturedPhotoOpticalCenter(opticalCenter)
}
```

This explains the reported one-finger failure for current version-3 saves. `usesFlatPhotoNavigation`
returns before rotation and translates X/Y:

```swift
if usesFlatPhotoNavigation {
    switch gesture.state {
    case .began:
        initialCameraTransform = cameraAnchor.transform
    case .changed:
        var transform = initialCameraTransform
        transform.translation.x += Float(translation.x) * panSensitivity
        transform.translation.y -= Float(translation.y) * panSensitivity
        cameraAnchor.transform = transform
    default:
        break
    }
    return
}
```

The rotation branch itself computes incremental yaw/pitch and can turn in place:

```swift
let deltaYaw = Float(deltaTranslation.x) * rotationSensitivity * 0.5
let deltaPitch = Float(deltaTranslation.y) * rotationSensitivity * 0.5
accumulatedYaw += -deltaYaw
accumulatedPitch += -deltaPitch

let yawRotation = simd_quatf(angle: accumulatedYaw, axis: SIMD3<Float>(0, 1, 0))
let pitchRotation = simd_quatf(angle: accumulatedPitch, axis: SIMD3<Float>(1, 0, 0))
let combinedRotation = yawRotation * pitchRotation

if activeRotationTurnsInPlace {
    newTransform.translation = capturedPhotoOpticalCenter ?? initialCameraTransform.translation
}
newTransform.rotation = combinedRotation
cameraAnchor.transform = newTransform
```

For captured projective rooms, pinch is intentionally field-of-view zoom at a fixed optical center:

```swift
if capturedPhotoOpticalCenter != nil || usesFlatPhotoNavigation {
    let scale = max(Float(gesture.scale), 0.01)
    let initialHalfFov = capturedFrustumPinchStartFieldOfView * .pi / 360
    let zoomedFov = 2 * atan(tan(initialHalfFov) / scale) * 180 / .pi
    cameraEntity.camera.fieldOfViewInDegrees = min(max(zoomedFov, 12), 120)
    return
}
```

For a true navigable volume, pinch moves the camera along its forward vector and then applies room
bounds. That branch was unsafe for the old single-photo projective surface because it revealed
unobserved/duplicated content.

## Android comparison

Android saved AI rooms currently use a projective depth GLB (`photo_room_depth`). The generator
pinhole-unprojects sampled depth pixels:

```kotlin
val x = (column - centerX) * depth / focalXPixels
val y = (centerY - row) * depth / focalYPixels
val z = -depth
```

It omits triangles whose four depths differ by more than `0.15 m`, then adds a far plane textured
with the **entire original photograph**:

```kotlin
val backingDepth = farDepth + 0.02f
// four full-frame backing corners with UV 0...1
indices.addAll(
    listOf(
        backingStart, backingStart + 1, backingStart + 2,
        backingStart, backingStart + 2, backingStart + 3
    )
)
```

That backing explains duplicate foreground pixels: the depth surface carries the chair/fan at their
estimated depths, and the backing carries the same source pixels again at `backingDepth`.

After commit `316aa0bc`, Android one-finger movement rotates at the camera position with full yaw:

```javascript
interiorEuler.setFromQuaternion(camera.quaternion, 'YXZ');
interiorEuler.y -= (event.clientX - previousPoint.x) * 0.005;
interiorEuler.x -= (event.clientY - previousPoint.y) * 0.005;
const pitchLimit = Math.PI / 2 - 0.05;
interiorEuler.x = THREE.MathUtils.clamp(interiorEuler.x, -pitchLimit, pitchLimit);
camera.quaternion.setFromEuler(interiorEuler);
```

But `photoDepth` pinch remains FOV-only and the camera remains at its capture origin to avoid exposing
the bad backing:

```javascript
if (navigationMode === 'photoDepth') {
    depthPhotoZoom = THREE.MathUtils.clamp(depthPhotoZoom * pinchRatio, 0.5, 4.0);
    updatePhotoProjection();
    syncFirstPersonTarget();
    return;
}
```

This is stable reprojection, not human walking.

## Discarded local iOS experiment (historical)

The current working tree tries the following version-4 contract:

1. Run Apple Vision `GenerateForegroundInstanceMaskRequest` concurrently with GeoCalib, Depth
   Anything, and RTMDet.
2. Use the class-agnostic salient foreground mask so an object like a ceiling fan can be included
   even if RTMDet/COCO does not classify it.
3. Dilate the foreground mask to remove antialiased object-edge pixels.
4. Fill masked pixels with a nearest-known-background propagation followed by six local relaxation
   passes.
5. Export the original calibrated depth surface with the original photograph.
6. Export a second far-depth surface textured with the completed background.
7. Store metadata:
   - `depthMeshProjectionVersion = 4`
   - `depthMeshIsFlatPhotoPlane = 0`
   - `depthMeshHasCompletedBackground = 1`
8. Allow RealityKit to treat version 4 as a navigable volume rather than pinning the eye to the
   authored optical center.

Core experimental selection:

```swift
if let foregroundMask,
   let completedImage = Self.completeBackground(
       image: workingImage,
       foregroundMask: foregroundMask,
       imageWidth: imageWidth,
       imageHeight: imageHeight
   ) {
    mesh = try buildPerspectiveMesh(
        image: workingImage,
        depthMap: calibratedDepthMap,
        focalXPixels: meshFocalXPixels,
        focalYPixels: meshFocalYPixels
    )

    let farDepth = max(maximumDepth + 0.05, percentile95Depth + 0.10)
    let backingDepthMap = Array(
        repeating: Array(repeating: farDepth, count: imageWidth),
        count: imageHeight
    )
    completedBackgroundMesh = try buildPerspectiveMesh(
        image: completedImage,
        depthMap: backingDepthMap,
        focalXPixels: meshFocalXPixels,
        focalYPixels: meshFocalYPixels
    )
    enrichedCalibrationMetadata["depthMeshProjectionVersion"] = 4
    enrichedCalibrationMetadata["depthMeshIsFlatPhotoPlane"] = 0
    enrichedCalibrationMetadata["depthMeshHasCompletedBackground"] = 1
} else {
    // Safe fallback to pushed version-3 flat photo plane.
}
```

Experimental navigation classification:

```swift
let isFlatPhotoPlane = model.roomCoordinateFrame == .depthAnythingImageDepthMeters
    && metadata["depthMeshIsFlatPhotoPlane", default: 0] >= 0.5
let hasCompletedBackground = model.roomCoordinateFrame == .depthAnythingImageDepthMeters
    && metadata["depthMeshHasCompletedBackground", default: 0] >= 0.5

let usesPreviewPhotoPan = isFlatPhotoPlane && !model.isSavedRoom
gestureHandlers?.setFlatPhotoNavigationEnabled(usesPreviewPhotoPan)

let opticalCenter = model.roomCoordinateFrame == .depthAnythingImageDepthMeters
    && !usesPreviewPhotoPan
    && !hasCompletedBackground
    ? cameraAnchor?.transform.translation
    : nil
gestureHandlers?.setCapturedPhotoOpticalCenter(opticalCenter)
```

This experiment needs critical review before completion. Known concerns:

- A single far plane is not equivalent to fitted walls/floor/ceiling and may still feel like layered
  cardboard during translation.
- Nearest-neighbor plus relaxation is deterministic and offline, but it is not semantic/generative
  inpainting; large chair regions can become visibly smeared.
- Apple Vision salient-instance segmentation may mask curtains or other large regions that should
  remain structural texture.
- The original depth surface still includes background and foreground in one mesh. It relies on
  depth-discontinuity triangle removal rather than explicit foreground submeshes.
- A robust design may need foreground-specific meshes and a separately reconstructed structural
  room shell rather than “depth mesh plus far plane.”
- Camera bounds derived from mesh min/max do not directly describe a valid-view volume for layered
  depth images.
- `RealityKitBoundaryManager` currently conflates geometric bounds with safe camera bounds.
- USDZ material ordering, depth precision, z-fighting, face winding, and SceneKit export behavior
  need validation.
- The experimental SceneKit export helper should be reformatted/reviewed before compilation.
- Equivalent Android generation has not yet been changed to completed-background textures.

## Why “extend the wall” is plausible but underspecified

The user's intended solution is approximately:

> Take observed wall/floor/ceiling pixels, extend them behind chair/fan masks, and keep the objects at
> their own depth so camera movement reveals clean background instead of dragged object pixels.

That direction is sound for bounded view synthesis, but “wall extension” still requires decisions:

1. **Foreground mask ownership**
   - RTMDet masks known COCO furniture but may miss the fan.
   - Apple Vision salient foreground is class-agnostic but may over-mask.
   - Depth discontinuities can supplement either mask.

2. **Structural geometry**
   - Constant far plane is simplest but least room-like.
   - Manhattan/cuboid planes need wall-floor-ceiling boundary estimation and homographies.
   - A smoothed background depth surface retains non-planar structure but can wobble.

3. **Texture completion**
   - Edge-aware diffusion/Telea/Fast Marching is deterministic and small.
   - PatchMatch is better for repeated wall/floor texture.
   - On-device generative inpainting can hallucinate plausible content but adds a large model,
     latency, memory, licensing, and determinism concerns.

4. **Foreground representation**
   - A textured depth submesh preserves exact observed appearance.
   - A billboard always faces the camera and is wrong for head translation.
   - A layered depth image (multiple depths/colors per source ray) is ideal for limited novel views.
   - Full object completion cannot be recovered from one photograph without generation or a shape
     prior.

5. **Valid camera envelope**
   - It should be based on disocclusion coverage and completed texture confidence, not only room
     dimensions.
   - Translation should probably be a small fraction of estimated depth, with rotation less
     restricted than translation.

## Candidate representations for expert review

### A. Layered depth image (recommended direction to evaluate)

Create at least two layers per source ray:

- layer 0: visible foreground/scene color and calibrated depth;
- layer 1: completed background color and fitted/smoothed structural depth;
- optional extra layers at strong depth boundaries.

Render by splatting or triangulating each layer independently. Never texture the background layer
with unmasked source foreground pixels. Limit viewpoint movement using a confidence/coverage metric.

Pros: directly models disocclusion; best match for this exact failure.
Cons: needs custom representation/export/rendering or multiple meshes and careful seam handling.

### B. Structural room shell plus foreground depth meshes

Estimate floor, ceiling, and visible wall planes from depth/gravity/room boundaries. Project clean
background pixels onto those planes, inpaint occluded regions, and place foreground segmented depth
meshes separately.

Pros: strongest “inside a room” cue; stable walls and perspective.
Cons: plane segmentation/boundary errors are highly visible; irregular rooms need more than a cuboid.

### C. Completed background depth surface plus original foreground surface

The unfinished local experiment approximates this using a constant far depth. Improve it by
estimating structural background depth under each foreground mask from neighboring planes/surfaces.

Pros: minimal change to current USDZ/GLB pipeline.
Cons: still a 2.5D view and may fail for large translation or side views.

### D. Keep fixed optical center

Maintain exact source reprojection and provide only turn/FOV controls.

Pros: no trails and no invented geometry.
Cons: cannot satisfy the requested human walking/inside-room feeling and quickly exposes empty view
outside the source frustum.

## Questions for the reviewing engineer

Please answer these concretely against the code and constraints above:

1. Is a multi-mesh USDZ/GLB layered-depth representation sufficient for the requested limited motion,
   or is a custom renderer required for correct disocclusion/compositing?
2. Should the background be a set of fitted room planes, a completed background depth surface, or a
   layered depth image? What is the simplest representation that will actually feel like a room?
3. How should foreground masks combine RTMDet, Apple Vision salient masks, and depth discontinuities
   without erasing curtains/walls?
4. What on-device inpainting method is appropriate for wall/floor/ceiling completion at up to roughly
   2048 px, with no cloud dependency?
5. How should background depth under an occluder be inferred from surrounding samples and plane fits?
6. How should mesh triangles be partitioned so foreground never bridges to background and no gray
   cracks appear at antialiased boundaries?
7. How should camera motion limits be computed from depth, mask size, and completed-background
   confidence?
8. Can RealityKit and Three.js render the same asset contract consistently, including material order,
   depth testing, face orientation, and transparency?
9. How can version-3 flat saves and older projective saves remain readable while new saves use the new
   representation?
10. What intermediate debug artifacts should be persisted or displayed (foreground mask, completed
    background, structural depth, mesh layers, valid camera volume) so failures can be diagnosed from
    device screenshots?

## Non-negotiable constraints

- Input is one room photograph.
- All core processing is on device; room photos are not uploaded as a requirement.
- iOS uses Core ML/Vision; Android uses LiteRT/ONNX as currently packaged.
- Existing metric W x H x D measurement should not be replaced by mesh bounds.
- Do not claim hidden/generated pixels are observed measurements.
- Preview must remain immediate and visually stable.
- New saved assets need a versioned metadata contract and backward-compatible viewer behavior.
- Manual device visual testing is required before documenting the issue as resolved.
- Do not commit huge model/build artifacts, credentials, or signing material.

## Acceptance criteria for an actual fix

Use the same chair/fan test photograph and verify both platforms:

1. The first saved-room frame matches the source photograph closely enough that chair/fan placement
   does not jump relative to walls.
2. One-finger drag right turns the virtual eye right; it does not translate or orbit the room.
3. Full horizontal turning behaves as eye rotation, with no roll and sensible pitch limits.
4. A conservative pinch/translation produces parallax between chair/fan and walls.
5. Newly revealed pixels behind chair/fan contain plausible completed structure, never duplicate
   chair/fan source pixels.
6. No long stretched wedges appear at depth discontinuities.
7. No gray cracks/trails appear at object boundaries during repeated zoom in/out.
8. Recenter exactly restores the authored capture pose and field of view.
9. Preview remains sharp, correctly oriented, and unaffected.
10. Reopening the saved room produces the same result as immediately after save.
11. Legacy flat and projective rooms still open without crashes.
12. Device memory and save latency remain acceptable and are logged by stage.

## Useful files

### iOS

- `Furnit/Views/Components/SinglePhotoRoomViewer.swift`
  - immediate SceneKit preview and first-save entry point;
- `Furnit/Services/RoomReconstruction/DepthAnythingRoomReconstructor.swift`
  - metric depth, measurement, mesh construction, USDZ export;
- `Furnit/Views/ModelViewerView.swift`
  - saved-room UI host;
- `Furnit/Views/Components/RealityKitView.swift`
  - saved USDZ loading, camera framing, navigation contract;
- `Furnit/Utilities/RealityKitGestureHandlers.swift`
  - one-/two-finger pan and pinch camera behavior;
- `Furnit/Utilities/RealityKitBoundaryManager.swift`
  - model bounds and camera constraints;
- `Furnit/Services/OnDevice/RTMDetImageInference.swift`
  - detections, masks, affinity graph, cached mask generation;
- `Furnit/Utilities/CameraExifSidecar.swift`
  - projection/calibration metadata persisted beside assets.

### Android

- `android/app/src/main/java/com/furnit/android/GLBRoomActivity.kt`
  - saved/preview viewer and Three.js navigation;
- `android/app/src/main/java/com/furnit/android/services/GlbGenerator.kt`
  - flat preview GLB and projective saved-depth GLB;
- `android/app/src/main/java/com/furnit/android/services/DepthAnythingRoomMeasurer.kt`
  - reconstruction entry point;
- `android/app/src/main/java/com/furnit/android/roomreconstruction/DepthAnythingRoomMeasurementPipeline.kt`
  - calibrated depth and measurement result;
- `android/app/src/main/java/com/furnit/android/roomreconstruction/MeasurementObjectDetection.kt`
  - RTMDet object detection used during measurement;
- `android/app/src/main/java/com/furnit/android/services/FurnitureFitManager.kt`
  - LiteRT RTMDet mask/cutout generation.

## Commands for a reviewer

Inspect pushed state versus the unfinished experiment:

```bash
cd /path/to/Furnit
git status --short --branch
git diff --stat
git diff -- Furnit/Services/RoomReconstruction/DepthAnythingRoomReconstructor.swift \
  Furnit/Utilities/RealityKitBoundaryManager.swift \
  Furnit/Views/Components/RealityKitView.swift
git show 316aa0bc:Furnit/Services/RoomReconstruction/DepthAnythingRoomReconstructor.swift
```

Do not build merely to decide the architecture. After code review and implementation, normal builds
are:

```bash
# Android, from android/
./gradlew :app:assembleDebug --no-daemon

# iOS, from repository root
xcodebuild -project "Furnit.xcodeproj" -scheme "Furnit" -configuration Debug \
  -sdk iphoneos -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO COMPILER_INDEX_STORE_ENABLE=NO ENABLE_PREVIEWS=NO \
  -jobs 2 build
```

Automated builds are provisional for this visual/navigation issue. Final validation must be manual on
iOS and Android devices using the chair/fan scene.
