# Furnit Project Context

Furnit is a cross-platform image-to-3D room generation and furniture-fitment project.

- **iOS app:** `Furnit/` (Swift, SwiftUI/UIKit, Core ML, MetalSplatter).
- **Android app:** `android/` (Kotlin, ExecuTorch, NCNN/ONNX support, Vulkan-focused SHARP path).
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
- **Full video mode** displays cluster-level bounding boxes (union bbox per affinity group) rather than individual detection boxes; tapping a cluster selects all its members.
- **Multi-select placement** (Regime A): when multiple items are selected and segmented, each gets an independent overlay with stable `UUID` identity. Items are frozen at selection, not updated from live detections. Each can be independently panned/pinched.
- **Onboarding hints**: priority-ordered, one-at-a-time transient hints with `@AppStorage` persistence. A "?" button shows all eligible hints on demand. Hints are mode-scoped (browsing, furnitureFit, fullVideo).
- **Toolbar dimensions**: W×H×D room measurements displayed directly in the navigation bar when available (replaces the old ruler icon and floating chip).

### Thermal & Cadence Management

- **Live cadence**: RTMDet live-identify runs at ~5fps (`rtmdetLiveTargetInterval = 200ms`), creating genuine idle gaps between inference runs.
- **Placement pause**: inference is fully skipped (not just results-discarded) when independent overlay items are active (`inferencePausedForPlacement`). Camera preview stays alive.
- **Background pause**: `UIApplication.willResignActiveNotification` stops the capture session; `didBecomeActiveNotification` resumes it.
- **Thermal backoff**: observes `ProcessInfo.thermalStateDidChangeNotification`. Maps `.nominal/.fair` → 200ms, `.serious` → 400ms, `.critical` → pause inference entirely while keeping last-displayed boxes.
- **Camera ownership**: AR↔AVCapture transitions use a 150ms settle delay after AR pause before AVCapture starts, reducing `-17281` contention errors.

### Room Generation / SHARP Replacement

The active iOS single-photo room-generation backend is now **Depth Anything Metric USDZ**, selected by `RoomGenerationImplementation.defaultImplementation`.
The older SHARP Core ML path is still present for comparison/debugging, but it is not the default shipping path because SHARP model weights are not license-clean for commercial release.

Important files:

- `Furnit/Models/RoomGenerationImplementation.swift` — room-generation enum and default backend (`.depthAnythingMetricUSDZ`).
- `Furnit/Services/RoomReconstruction/DepthAnythingRoomReconstructor.swift` — Depth Anything V2 Metric Indoor Core ML inference, proportional-XY/depth-relief mesh construction, USDZ export.
- `Furnit/Models/DepthAnything/DepthAnythingV2MetricIndoorSmall.mlpackage` — bundled Apache-licensed metric indoor depth model.
- `Furnit/Views/Components/SinglePhotoRoomViewer.swift` — dispatches the selected generation backend, opens Depth Anything USDZ preview, and saves generated USDZ rooms.
- `Furnit/Views/ModelViewerView.swift` / `Furnit/Views/Components/RealityKitView.swift` — RealityKit USDZ preview and camera framing.
- `Furnit/Utilities/RealityKitBoundaryManager.swift` — front-facing camera placement for image-depth meshes.
- `Furnit/Models/USDZModel.swift` / `Furnit/Models/USDZModelManager.swift` — saved room metadata, coordinate-frame persistence, and list loading.
- `Furnit/Services/RoomReconstruction/SwiftSharpMathRoomReconstructor.swift` — no-ML flat-plane prototype/debug backend.
- `Furnit/Services/OnDevice/SHARPService.swift`, `Furnit/Views/SharpRoomView.swift`, `Furnit/Views/Components/GaussianSplatView.swift` — retained SHARP/Core ML Gaussian-splat path and viewer.
- `Furnit/diagrams/sharp-swift-flow.svg` — visual flow.

Current behavior:

- **Depth Anything Metric USDZ is default.** It runs the metric indoor model through Vision/Core ML (`computeUnits = .all`), resizes depth back to the working image, builds a connected grid mesh, skips triangles across >0.4m depth discontinuities, and exports USDZ.
- Depth Anything geometry uses proportional image-space X/Y, not pinhole `pixel * depth / focalLength`; Z is depth relief (`-(depthMax - depth)`) so near geometry comes toward the viewer.
- The generated USDZ uses a texture image plus UVs. After the camera-facing fix, the exporter uses non-inverted V coordinates so newly generated rooms are not upside down in the preview.
- Fresh Depth Anything previews have a bottom-center **Save** button. Save copies the generated USDZ into `Documents/SavedRooms`, writes a `.usdz.meta` sidecar, persists `roomCoordinateFrame=depth_anything_image_depth_meters`, dimensions, photo orientation, and display name, then refreshes the home list.
- Depth Anything saved rooms must keep `roomCoordinateFrame=depth_anything_image_depth_meters`. Do not let them fall back to SHARP/classic behavior on reopen.
- `RoomCoordinateFrame.depthAnythingImageDepthMeters` uses native meter scene units and a front-facing RealityKit camera. This is separate from SHARP classic PLY behavior.
- SHARP still saves `Room_*_classic.ply` plus sidecars and a binary `.splatcache`; `.splatcache` remains the fast path for SHARP rooms, with PLY parsing as fallback.
- `SwiftSharpMathRoomReconstructor` exists as a no-ML flat-plane prototype path, not the quality target.
- Room/furniture size and overlay scale details live in `docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md`.

Important caveats:

- The Depth Anything route is a single-image relief mesh, not a true metric 3D reconstruction. It is useful as a license-clean replacement prototype and visual path, but it should not be treated as fit-grade measurement.
- Old generated USDZ files keep whatever UVs/metadata they were exported with. Regenerate rooms after exporter/camera fixes before judging preview orientation.
- The long-term fit-grade replacement remains **LiDAR-first sweep fusion**: ARKit pose + `smoothedSceneDepth` capture, validator gate, then fused AR-world-meter output.

### LiDAR Sweep Fusion Status

LiDAR sweep fusion is the intended fit-grade commercial v1 path because it avoids SHARP licensing, monocular scale ambiguity, COLMAP/ICP complexity, and viewer downgrade.

Important files:

- `Furnit/Models/PosedFrameSweep.swift` — persists `frames/*.jpg`, `depth/*.bin`, `confidence/*.bin`, and `poses.json`.
- `Furnit/Views/Components/ARRoomSweepCaptureView.swift` — LiDAR-gated AR sweep capture using `smoothedSceneDepth`, ARKit camera poses, and keyframe filtering.
- `Furnit/Services/RoomReconstruction/PosedFrameSweepValidator.swift` — validates pose/depth/intrinsics convention, writes debug PLYs, reports NN and plane residuals.
- `Furnit/Services/RoomReconstruction/PosedFrameSweepFusion.swift` — fuses posed depth points into a saved AR-world-meter room artifact.
- `Furnit/Views/Components/LiDARRoomSweepCreationView.swift` — user-facing sweep capture, validation, fusion, and preview flow.

Current contract:

- Capture stores ARKit `camera.transform` as world-from-camera and depth in metric meters.
- Validator unprojects with ARKit convention `(x, y, -depth)` transformed by world-from-camera, scales RGB intrinsics to the depth grid, filters confidence/tracking, and can compare selectable frame indices.
- Nearest-neighbor residuals catch gross misregistration. Plane residuals catch normal-offset/double-wall drift. A pure in-plane tangential slide can still look numerically clean, so the overlay PLY remains the visual truth gate.
- LiDAR fused/saved rooms must use `roomCoordinateFrame=ar_world_meters`, native meter units, and no SHARP/classic Y/Z flip.
- Next real gate is on-device LiDAR capture and validation before treating fusion as production-quality.

## Current Android Architecture

Android SHARP work is centered on ExecuTorch debug flavors.

Important files/docs:

- `android/README.md` — Android overview, model staging, logs.
- `android/docs/TEST_AND_SETTINGS.md` — run/settings checklist.
- `android/docs/EXECUTORCH_VULKAN_KNOWN_GOOD_FLOW.md` — Vulkan known-good flow.
- `android/app/src/main/java/com/furnit/android/` — app source.
- `android/app/src/main/cpp/` — native SHARP/ExecuTorch integration.

Default agent compile check is both ExecuTorch flavors:

```bash
cd android && ./gradlew :app:compileEtCpuDebugKotlin :app:compileEtVulkanDebugKotlin --no-daemon
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
- For RTMDet class labels, `Furnit/Models/RTMDet/rtmdet-coco80-classes.json` is standard COCO-80 (`56 == chair`).
