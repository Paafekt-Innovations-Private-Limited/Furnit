# Furnit iOS - Recent Changes

## 1.2 — Offline Licenses and Attribution (2026-08-05)

- Bumped the iOS application target to marketing version **1.2**, build **87**.
- Added `Furnit/Licenses/THIRD_PARTY_NOTICES.txt` and made the existing full Apache
  License 2.0 text readable offline from Settings → Licenses.
- Recorded the exact Depth Anything V2 Metric Indoor Small variant, GeoCalib pinhole
  weights, RTMDet/COCO attribution, and Paafekt's converted Core ML/ONNX/LiteRT model
  formats.
- Added the bundled-notice link label to all 14 localization catalogs.
- Verified an unsigned generic-device build packages both legal text resources and the
  compiled Depth Anything, GeoCalib, and RTMDet models. App Store Connect reported the
  1.2 (87) upload as processed; review/publication is tracked separately.

## Docs & Diagrams — Android Room Creation / RTMDet Parity

- Updated Android room-creation docs and diagrams for immediate AI/manual picker display, off-main sampled EXIF decode, no artificial generation wait, JPEG GLB textures, saved depth metadata, and Swift-style floating `GLBRoomActivity` controls.
- Updated Android RTMDet docs and diagrams for the shared process-wide ONNX Runtime backend and the boxes-only identify fast path.
- Updated Swift docs and diagrams to describe the shared room-viewer control model: measurement/gesture helpers, inline brain, and full-video viewfinder flow.

## Docs & Diagrams — Brain / Full-Video Parity

- Updated `Furnit/diagrams/rtmdet-swift-flow.svg` for room-viewer inline brain, text.viewfinder full-video modes, and transparent cutout compositing over 3D room.
- Updated `Furnit/diagrams/room-generation-flow.svg` for **two-phase Photo → 3D**: instant `PreviewFast` preview (no ML) then GeoCalib + Depth Anything + RTMDet on first save.
- Updated `Furnit/docs/README.md`, `Furnit/diagrams/README.md`, `Furnit/Views/FurnitureFit/README.md`, `docs/IOS_FURNITURE_FIT_ONNX_STYLE_PIPELINE.md`, `docs/RTMDET_IOS_SWIFT_SPIKE.md`, `docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md`, `CONTEXT.md`, and root `README.md`.

## Room Generation Documentation Cleanup

### Current Photo → 3D Room Path
- **Location**: `SinglePhotoRoomViewer.swift`, `DepthAnythingRoomReconstructor.swift`, `CameraExifSidecar.swift`, `USDZModel.swift`
- Default single-photo room creation is documented as: home **Photo → 3D** → photo capture/library image → camera metadata sidecar → **instant preview** (`PreviewFast`, placeholder dims) → **first save** runs parallel GeoCalib, Depth Anything, and RTMDet object-anchor work → measurement grid → textured USDZ export → saved room in `ModelViewerView` with inline brain.
- Progress UI uses existing generic localized copy: "Generating 3D Model" and "Creating your 3D room..." so backend names are not user-facing.
- Added `Furnit/diagrams/room-generation-flow.svg` and updated the iOS docs index to point at it.
- The old model-backed splat generation service/model path is no longer part of the Swift active path. Saved PLY/splat viewing remains for existing room assets and LiDAR/sweep outputs.

## Thermal & Cadence Management

### Live Cadence Throttle
- **Location**: `FurnitureFitView.swift`
- RTMDet live-identify cadence raised to ~5fps (`rtmdetLiveTargetInterval = 200ms`) with genuine idle gaps between inference runs, reducing sustained thermal load vs the previous ~100% duty cycle.

### Placement Inference Pause
- **Location**: `FurnitureFitView.swift`
- When independent overlay items are placed (multi-select segmentation), inference is fully skipped — not just results-discarded. Camera preview stays alive for the room view.

### Background / Resign-Active Pause
- **Location**: `FurnitureFitView.swift`
- Added observers for `UIApplication.willResignActiveNotification` / `didBecomeActiveNotification` to stop/resume the capture session on app lifecycle transitions.

### Thermal-State Backoff
- **Location**: `FurnitureFitView.swift`
- Observes `ProcessInfo.thermalStateDidChangeNotification`. Maps thermal state to cadence:
  - `.nominal` / `.fair` → 200ms (~5fps)
  - `.serious` → 400ms (~2.5fps)
  - `.critical` → pause inference entirely, keep last-displayed boxes

### Camera Ownership (AR↔AVCapture)
- **Location**: `FurnitureFitView.swift`
- Added 150ms settle delay between AR session pause and AVCapture session start in `startClassicCameraPathIfNeeded`, reducing `-17281` camera contention errors.

## Cluster Bounding Boxes in Full Video Mode
- **Location**: `FurnitureFitView.swift`, `RTMDetImageInference.swift`
- Full video mode now displays union bounding boxes per affinity-group cluster instead of individual detection boxes.
- Tapping a cluster bbox selects all its members for segmentation.
- Mask affinity graph is always built when raw mask planes exist (not gated on `buildInstanceMasks`).

## Independent Per-Furniture Movement (Multi-Select)
- **Location**: `FurnitureFitView.swift`, `FurnitureFitOverlayScaling.swift`
- When multiple furniture items are selected and segmented, each gets an independent overlay with stable `UUID` identity (Regime A: freeze on selection).
- Items can be independently panned and pinched without affecting each other.
- Uses `.measuredPlacement` to keep items at detected positions (no auto-centering).
- Debug mode preserves user gestures while freezing only assisted scale.

## Onboarding Hint System
- **Location**: `SplatRoomView.swift`, `SettingsView.swift`
- Priority-ordered, one-at-a-time transient hints with `@AppStorage` persistence flags.
- Four hints: brain icon (browsing), pinch resize (furnitureFit), AR sizing (furnitureFit), pick another (furnitureFit/fullVideo).
- Mode-scoped eligibility: browsing → B only, furnitureFit → A′/E/G, fullVideo → G only.
- "?" button in toolbar shows all eligible hints on demand for 5 seconds.
- "Show tips again" option in Settings lets users restore onboarding hints.

## Room Viewer Measurement Controls
- **Location**: `SplatRoomView.swift`, `GLBRoomView.swift`, `MeshRoomView.swift`, `ModelViewerView.swift`
- Room measurements are surfaced through the current room-viewer top controls and persisted metadata when available.
- The older floating `roomDimensionsChipOverlay` is removed; measurement, pinch, tap, brain, viewfinder, recenter/save, and AR helpers live in the viewer controls.

## Debug Bounding Box Drawing
- **Location**: `FurnitureFitView.swift`, `RTMDetImageInference.swift`
- Single `drawDebugDetectionBboxes` helper with 4-color scheme: red (primary), orange (affinity group), yellow (explicit pin), cyan (unselected).
- Burns bounding boxes + legend into the CGImage in image space.
- Called from both live and cached segmentation paths.

## RTMDet / Furniture Fit Diagnostics

### Settings Image Scan Parity
- **Location**: `SettingsFurnitureFitImageScanView.swift`, `RTMDetImageInference.swift`
- Settings image scan now follows the same RTMDet still-image path as the live room flow:
  - no fixed detection cap (`maxDetectionCount: nil`)
  - fused `instanceMaskImages`
  - pixel-level RGBA mask union instead of Core Graphics alpha blending
  - no bbox-overlap-only clustering

### RTMDet Object-Piece Fusion
- **Location**: `RTMDetImageInference.swift`
- Added mask-affinity grouping over raw mask planes so disconnected pieces of the same object can be treated as one object.
- Grouping is class-agnostic and is reused by cached selected-mask rebuilds.

### Confidence-First NMS
- **Location**: `RTMDetImageInference.swift`
- Class-aware NMS is confidence-first. Area is only a tie-breaker.

### Segmented Overlay Gestures
- **Location**: `FurnitureFitView.swift`
- Pinch on a segmented cutout updates `userPinchScale` and is applied through `FurnitureFitOverlayScaling`.
- In USDZ / GLB / saved PLY room viewers, the FurnitureFit overlay must capture two-finger touches when a cutout is visible so the room viewer does not steal pinch zoom.

### Repeated Swift RTMDet Test Fixture
- **Location**: `RTMDetVideoIntegrationTests.swift`, `FurnitTests/rtmdet_repeated_chair_frame.jpg`
- Added a real Swift/Core ML repeated still-frame test path for detecting deterministic still-frame regressions.

## Room Viewer Enhancements

### Auto-Orbit Feature
- **Location**: `SettingsView.swift`, `SplatRoomView.swift`
- Added auto-orbit toggle in Settings
- Default is **OFF** (was previously ON)
- When enabled, camera oscillates ±30° around the room center when idle
- Setting persisted in `@AppStorage("roomViewer.oscillation")`

```swift
@AppStorage("roomViewer.oscillation") private var oscillationEnabled: Bool = false
```

### Grey Screen Fix (Warm-up Rendering)
- **Location**: `SplatRoomView.swift` (WebGL JavaScript)
- Added 5-second warm-up period for continuous rendering
- Fixes issue where room appeared grey when auto-orbit was disabled
- **Removed (2026-07):** SparkJS Gaussian splat bundle; saved PLY rooms use MetalSplatter + SplatIO natively.

```javascript
const WARMUP_DURATION = 5000;
const animationStartTime = performance.now();
// In animate loop:
const inWarmup = elapsed < WARMUP_DURATION;
if (inWarmup) { shouldRender = true; }
```

### Room Dimension Persistence
- **Location**: `USDZModel.swift`, `USDZModelManager.swift`, `SplatRoomView.swift`, `ContentView.swift`

#### USDZModel.swift
Added dimension fields:
```swift
let roomWidth: Float?
let roomHeight: Float?
let roomDepth: Float?
let photoOrientation: PhotoOrientation
```

#### USDZModelManager.swift
- `savePLY()` now accepts and saves room dimensions
- `loadPLYMetadata()` reads dimensions from metadata file
- Dimensions stored in `.metadata` file alongside PLY

```swift
func savePLY(from sourceURL: URL, name: String,
             photoOrientation: PhotoOrientation = .portrait,
             roomWidth: Float? = nil,
             roomHeight: Float? = nil,
             completion: @escaping (Bool, String?) -> Void)
```

#### SplatRoomView.swift
- Accepts `savedRoomWidth` and `savedRoomHeight` parameters
- Reports dimensions from WebGL via JavaScript message handler
- Multiple dimension reports (500ms, 1500ms, 3000ms) to ensure delivery
- Navigation title shows dimensions: prioritizes saved > JS-measured > defaults

```swift
init(plyURL: URL,
     allowSave: Bool = true,
     photoOrientation: PhotoOrientation = .portrait,
     savedRoomWidth: Float? = nil,
     savedRoomHeight: Float? = nil,
     savedRoomModel: USDZModel? = nil)
```

#### ContentView.swift
- Passes saved dimensions when opening rooms from home screen
- HomeViewModelRow displays actual room dimensions

### Custom Calibration Number Pad
- **Location**: `SplatRoomView.swift`
- Replaced system keyboard with custom number pad overlay
- Number pad rotates with the calibration overlay for landscape orientation
- Supports decimal input for room height calibration

### Orientation Labels
- **Location**: `SplatRoomView.swift`
- Shows orientation label for both portrait and landscape rooms
- Portrait: "held vertically - Portrait"
- Landscape: "held horizontally - Landscape"

### Room Icon
- **Location**: `ContentView.swift`
- PLY files (user-created rooms): `circle.grid.3x3.fill` (purple)
- USDZ files (bundled models): `cube.fill` (green)

## File Changes Summary

| File | Changes |
|------|---------|
| `USDZModel.swift` | Added `roomWidth`, `roomHeight`, `roomDepth` fields |
| `USDZModelManager.swift` | Save/load dimensions in metadata |
| `SettingsView.swift` | Auto-orbit toggle (default OFF) |
| `SplatRoomView.swift` | Warm-up, auto-orbit, calibration overlay, number pad, dimension handling |
| `ContentView.swift` | Pass dimensions when opening rooms |
| `SplatRoomViewTests.swift` | Unit tests for all new features |

## Metadata Format

Room metadata is stored in `{roomName}.metadata` file:

```
orientation=portrait
roomWidth=4.5
roomHeight=3.2
```

## JavaScript Message Handlers

SplatRoomView handles these messages from WebGL:

```swift
case "dimensionsMeasured":
    // Receives { width: Float, height: Float }
    // Updates jsFrontWallWidth/Height state variables
```

## Unit Tests

`SplatRoomViewTests.swift` includes 32 tests covering:
- Calibration overlay display
- Number pad functionality
- Auto-orbit toggle behavior
- Warm-up rendering period
- Dimension persistence and display
- Orientation label display

## Navigation Title Logic

The room dimensions shown in the title follow this priority:
1. Saved dimensions from metadata (`savedRoomWidth`, `savedRoomHeight`)
2. JS-measured dimensions from WebGL (`jsFrontWallWidth`, `jsFrontWallHeight`)
3. Default fallback (4.0 × 3.0 m)
