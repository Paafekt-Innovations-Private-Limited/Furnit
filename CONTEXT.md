# Furnit Project Context

Furnit is a cross-platform image-to-3D room generation and furniture-fitment project.

- **iOS app:** `Furnit/` (Swift, SwiftUI/UIKit, Core ML, MetalSplatter).
- **Android app:** `android/` (Kotlin, ONNX Runtime RTMDet, CameraX, WebView/Three.js GLB room viewer).
- **Shared docs:** `docs/`, `Furnit/docs/`, `android/docs/`.
- **Model/export scripts:** `scripts/`, `android/pyfiles/`, `android/scripts/`.

## Current iOS Architecture

### RTMDet / Furniture Fit

The active iOS Furniture Fit "brain" path is RTMDet-Ins-m through Core ML.

Important files:

- `Furnit/Services/OnDevice/RTMDetModelService.swift` — model loading.
- `Furnit/Services/OnDevice/RTMDetImageInference.swift` — image/live inference, raw-head decode, confidence-first NMS, mask-head execution, mask-affinity grouping, cached mask rebuilds.
- `Furnit/Views/FurnitureFit/FurnitureFitView.swift` — live overlay, selection, segmentation, cutout display, pinch/pan gesture ownership.
- `Furnit/Views/SettingsFurnitureFitImageScanView.swift` — Settings still-image diagnostic path.
- `Furnit/Views/FurnitureFit/README.md` — main RTMDet/Furniture Fit pipeline documentation.
- `Furnit/diagrams/rtmdet-swift-flow.svg` — visual flow.

Current behavior:

- Core ML image input is preferred; BGR mean/std normalization is expected inside the model graph.
- Raw outputs are `cls/bbox/kernel` at 80/40/20 plus `mask_feat`.
- NMS is class-aware and confidence-first; `maxDetectionCount` may be `nil` for no artificial cap.
- Object-piece fusion is class-agnostic mask-affinity grouping, not chair-specific logic.
- Settings image scan should mirror the live RTMDet path: uncapped detections, fused instance masks, and pixel-level RGBA union.
- In room viewers, the room layer also owns pinch zoom; when a segmented cutout is visible, Furniture Fit must capture two-finger touches so pinch scales the furniture cluster instead of the room camera.
- **Full video mode** displays cluster-level bounding boxes (union bbox per affinity group) rather than individual detection boxes; tapping a cluster selects all its members. During `segmentSelected`, the live camera preview hides and transparent cutouts composite over the 3D room.
- **Brain default** opens in `segmentPrimary`: auto-segments the highest-confidence detection with no tap. Tap-to-select lives behind the in-room **text.viewfinder** button (`showFullVideoWithIdentifications`).
- **Multi-select placement** (Regime A): when multiple items are selected and segmented, each gets an independent overlay with stable `UUID` identity. Items are frozen at selection, not updated from live detections. Each can be independently panned/pinched.
- **Onboarding hints**: priority-ordered, one-at-a-time transient hints with `@AppStorage` persistence. A "?" button shows all eligible hints on demand. Hints are mode-scoped (browsing, furnitureFit, fullVideo).
- **Room toolbar**: room viewers expose measurement and gesture helpers in the top controls; Android mirrors the Swift visual structure with floating back, center ruler/pinch/tap helpers, recenter/save, and AR controls.

### Thermal & Cadence Management

- **Live cadence**: RTMDet live-identify runs at ~5fps (`rtmdetLiveTargetInterval = 200ms`), creating genuine idle gaps between inference runs.
- **Placement pause**: inference is fully skipped (not just results-discarded) when independent overlay items are active (`inferencePausedForPlacement`). Camera preview stays alive.
- **Background pause**: `UIApplication.willResignActiveNotification` stops the capture session; `didBecomeActiveNotification` resumes it.
- **Thermal backoff**: observes `ProcessInfo.thermalStateDidChangeNotification`. Maps `.nominal/.fair` → 200ms, `.serious` → 400ms, `.critical` → pause inference entirely while keeping last-displayed boxes.
- **Camera ownership**: AR↔AVCapture transitions use a 150ms settle delay after AR pause before AVCapture starts, reducing `-17281` contention errors.

### Room Generation — GeoCalib + Depth Anything + RTMDet Anchor (default)

The active iOS single-photo room-generation backend is **Depth Anything V2 Metric Indoor + GeoCalib
+ RTMDet object-anchor measurement**, split into **two phases**:

1. **Instant preview (no ML)** — `makeDepthAnythingPreviewDestination` in `SinglePhotoRoomViewer.swift`
   downsamples the photo, writes a JPEG sidecar, and opens `DepthAnythingPreviewRoomView` with
   placeholder dimensions (W=2 m, H=aspect×W, D=3 m). No GeoCalib, Depth Anything, RTMDet, or USDZ
   export runs during creation (`[DepthAnythingRoom][PreviewFast]` log).
2. **First save (full ML)** — tapping Save calls `DepthAnythingRoomReconstructor.reconstructWithResult`,
   which runs the parallel inference stack below, exports USDZ, and copies the room into saved rooms
   with measured W×H×D in `.usdz.meta`.

**Alternate manual path:** orange manual setup card → boundary editor → `SinglePhotoRoomReconstructor`
+ `SyntheticDepthEstimator` (CIFilter placeholder depth, not MiDaS) → `MeshRoomView`. No GeoCalib or
Depth Anything on this path.

**Saved PLY splats:** `SplatRoomView` → `GaussianSplatView` renders via **MetalSplatter + SplatIO**
(spz-swift). SparkJS was removed; no WebGL splat bundle ships.

Important files:

- `Furnit/Services/RoomReconstruction/GeoCalibCalibrationService.swift` — on-device GeoCalib CNN + Swift LM optimizer; letterboxed full-frame input; square-pixel focal (`fx = fy`) in the working image grid.
- `Furnit/Services/RoomReconstruction/DepthAnythingRoomReconstructor.swift` — Depth Anything Core ML inference, depth resize to the working grid, point-grid room sizing, object-anchor metric depth scale, textured mesh build, USDZ export.
- `Furnit/Models/DepthAnything/DepthAnythingV2MetricIndoorSmall.mlpackage` — bundled Apache-licensed metric indoor depth model.
- `Furnit/Models/GeoCalib/GeoCalibPinholeCNN.mlpackage` — bundled GeoCalib perspective-field CNN (LM refinement in Swift).
- `Furnit/Services/OnDevice/RTMDetImageInference.swift` — one-shot object-anchor bbox used by the room measurement path.
- `Furnit/Views/Components/SinglePhotoRoomViewer.swift` — preview dispatch (`PreviewFast`), save
  (`reconstructWithResult`), camera sidecars.
- `Furnit/Views/ModelViewerView.swift` / `Furnit/Views/Components/RealityKitView.swift` — RealityKit USDZ preview and camera framing.
- `Furnit/Utilities/RealityKitBoundaryManager.swift` — front-facing camera for `.depthAnythingImageDepthMeters` rooms.
- `scripts/run_geocalib.py`, `scripts/geocalib_write_sidecar.py`, `scripts/export_geocalib_to_coreml.py` — GeoCalib export and offline validation.
- `scripts/depthanything_measure_room.py` — Python reference for metric depth + mesh export.

Pipeline order (one measurement grid, **on first save only**):

1. **Full working frame** (e.g. 1200×1600 after downsample) — same grid for everything.
2. **Parallel inference** — GeoCalib focal/gravity, Depth Anything metric depth, and RTMDet object anchor run from the same image.
3. **Metric calibration** — compare GeoCalib, sidecar/EXIF/ARKit capture metadata, and object-anchor measurements to resolve focal and depth scale. **ARKit capture gravity and camera height override GeoCalib** when present. Vanishing-point gravity refiner (`VanishingPointGravity`) is a stub (`vps=0`, unimplemented).
4. **Unproject together** — a shared point grid feeds room extent, single-view height/width, object masks, and final W×H×D.
5. **Mesh + export** — build a textured mesh from the calibrated depth grid, export USDZ, persist camera sidecars and metadata.

Current behavior:

- **Preview** is a flat photo plane in `DepthAnythingPreviewRoomView` with placeholder dims; brain/segment can load RTMDet separately for Furniture Fit.
- **Save** produces **USDZ** with texture UVs and measured W×H×D.
- Room W×H×D exposed by the room viewer controls and stored in `.usdz.meta` come from the measurement grid, not mesh bounds or wall-rect frustum math.
- Saved rooms use `roomCoordinateFrame=depth_anything_image_depth_meters`.
- The old model-backed splat generation service/model path is removed from the active iOS Swift code.

Important caveats:

- Single-image relief mesh, not full metric reconstruction; chair anchor + EXIF disambiguation improve scale but are not tape-measure grade.
- The experimental LiDAR sweep-fusion stack was removed in July 2026. Current production room creation is single-photo based.

## Current Android Architecture

Android room generation uses an optimized **flat full-photo GLB** preview plus a wired Depth Anything measurement path with RTMDet and optional GeoCalib.
iOS uses **instant preview (no ML)** then **GeoCalib + Depth Anything + RTMDet object anchor → USDZ on first save**.

Current Android room-creation behavior:

- `SinglePhotoRoomActivity` shows the AI/manual method picker immediately after photo selection.
- Photo decode is EXIF-aware, sampled to a bounded max dimension off the UI thread, and handed to generation after the picker UI has rendered.
- `SinglePhotoRoomReconstructor` no longer adds artificial wait time; preview generation is driven by real work only.
- `GlbGenerator.generateFlatPhotoGlb` embeds the full-photo texture as JPEG to reduce GLB size and speed up preview/save.
- Generated room width, height, and depth metadata are passed into `GLBRoomActivity` so the viewer ruler dialog and camera framing use the saved dimensions.

Both platforms share the same inline brain / full-video segmentation UX:

- Brain default: auto-segment highest-confidence primary over the 3D room.
- **Viewfinder** button (Android `ic_text_viewfinder`, Swift `text.viewfinder`): live camera + detection/cluster boxes → multi-select → Segment → transparent cutouts over 3D room.
- Android `GLBRoomActivity` uses Swift-parity floating top controls instead of a full-width top band: back, center ruler/pinch/tap helper capsule, recenter/save, and AR resize.
- Android `FurnitureFitManager` shares a process-wide ONNX Runtime backend (`OrtEnvironment`, `OrtSession`, session options, and single-thread inference executor). The identify-only path requests only cls/bbox outputs and skips mask planes/affinity work; segmentation modes still request kernels plus `mask_feat`.

Important files/docs:

- `android/README.md` — Android overview, model staging, logs.
- `android/docs/TEST_AND_SETTINGS.md` — run/settings checklist.
- `android/app/src/main/java/com/furnit/android/` — app source.

Default agent compile check:

```bash
cd android && ./gradlew :app:assembleDebug --no-daemon
```

## Build And Test Rules

Follow repository rules before running builds/tests.

- `CLAUDE.md` requires compile before concluding code changes.
- `.cursor/rules/model-usage-preference.mdc` says mechanical execution such as `xcodebuild`, `./gradlew`, and git mechanics should use Claude Sonnet when available.
- `.cursor/rules/ios-build-preference.mdc` defines the preferred iOS compile command:

```bash
xcodebuild -project "Furnit.xcodeproj" -scheme "Furnit" -configuration Debug \
  -sdk iphoneos -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO COMPILER_INDEX_STORE_ENABLE=NO ENABLE_PREVIEWS=NO \
  -jobs 2 build
```

- Do not use Simulator destinations unless explicitly requested.
- Do not pass `-derivedDataPath` into the repo unless explicitly requested.
- `.cursor/rules/android-build-preference.mdc` defines the preferred Android compile command shown above.

## Xcode Cache Safety

The user has an external SSD-backed Xcode cache setup. Be conservative.

- See `.cursor/rules/xcode-cache-safety.mdc`.
- Diagnose Xcode/SwiftPM cache issues by reading/listing only.
- Do not create, delete, replace, or repair symlinks/folders under Xcode/SwiftPM cache paths unless the user explicitly approves that repair in the current conversation.
- If `xcodebuild` fails because of DerivedData, SwiftPM, SourcePackages, permissions, symlinks, missing external volume, or disk space, stop and report the exact error.

## Documentation Map

Start here when updating docs:

- `README.md` — top-level documentation index.
- `Furnit/docs/README.md` — iOS docs and diagram index.
- `Furnit/Views/FurnitureFit/README.md` — RTMDet/Furniture Fit pipeline.
- `docs/RTMDET_IOS_SWIFT_SPIKE.md` — RTMDet Core ML loader/export and Swift postprocess status.
- `docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md` — room/furniture sizing and overlay scaling.
- `Furnit/CHANGELOG.md` — recent iOS changes.
- `android/README.md` and `android/docs/` — Android-specific docs.

When code behavior changes, update the nearest detailed doc plus any affected diagram/index.

## Practical Debug Notes

- A screenshot of the live RTMDet overlay is not the same as the original camera frame; re-scanning a screenshot can produce different scores/detections.
- If segmented cutout pinch does not work in the main room flow, first verify whether `FurnitureFitContainerView.handlePinch(_:)` logs appear. If not, debug hit-testing/gesture ownership before changing scale math.
- Python RTMDet probes are useful for export inspection, but Swift/Core ML app tests are the source of truth for current image-input behavior.
- For RTMDet script class labels, `scripts/rtmdet-coco80-classes.json` is standard COCO-80 (`56 == chair`). The apps use their localized runtime class maps.
