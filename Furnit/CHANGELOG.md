# Furnit iOS - Recent Changes

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
- In SHARP/GLB/Mesh room viewers, the FurnitureFit overlay must capture two-finger touches when a cutout is visible so the room viewer does not steal pinch zoom.

### Repeated Swift RTMDet Test Fixture
- **Location**: `RTMDetVideoIntegrationTests.swift`, `FurnitTests/rtmdet_repeated_chair_frame.jpg`
- Added a real Swift/Core ML repeated still-frame test path for detecting deterministic still-frame regressions.

## Room Viewer Enhancements

### Auto-Orbit Feature
- **Location**: `SettingsView.swift`, `SharpRoomView.swift`
- Added auto-orbit toggle in Settings
- Default is **OFF** (was previously ON)
- When enabled, camera oscillates ±30° around the room center when idle
- Setting persisted in `@AppStorage("roomViewer.oscillation")`

```swift
@AppStorage("roomViewer.oscillation") private var oscillationEnabled: Bool = false
```

### Grey Screen Fix (Warm-up Rendering)
- **Location**: `SharpRoomView.swift` (WebGL JavaScript)
- Added 5-second warm-up period for continuous rendering
- Fixes issue where room appeared grey when auto-orbit was disabled
- SparkJS Gaussian splat needs time to fully load before static rendering works

```javascript
const WARMUP_DURATION = 5000;
const animationStartTime = performance.now();
// In animate loop:
const inWarmup = elapsed < WARMUP_DURATION;
if (inWarmup) { shouldRender = true; }
```

### Room Dimension Persistence
- **Location**: `USDZModel.swift`, `USDZModelManager.swift`, `SharpRoomView.swift`, `ContentView.swift`

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

#### SharpRoomView.swift
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
- **Location**: `SharpRoomView.swift`
- Replaced system keyboard with custom number pad overlay
- Number pad rotates with the calibration overlay for landscape orientation
- Supports decimal input for room height calibration

### Orientation Labels
- **Location**: `SharpRoomView.swift`
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
| `SharpRoomView.swift` | Warm-up, auto-orbit, calibration overlay, number pad, dimension handling |
| `ContentView.swift` | Pass dimensions when opening rooms |
| `SharpRoomViewTests.swift` | Unit tests for all new features |

## Metadata Format

Room metadata is stored in `{roomName}.metadata` file:

```
orientation=portrait
roomWidth=4.5
roomHeight=3.2
```

## JavaScript Message Handlers

SharpRoomView handles these messages from WebGL:

```swift
case "dimensionsMeasured":
    // Receives { width: Float, height: Float }
    // Updates jsFrontWallWidth/Height state variables
```

## Unit Tests

`SharpRoomViewTests.swift` includes 32 tests covering:
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
