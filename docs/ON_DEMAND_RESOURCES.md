# On-Demand Resources (ODR) for ML Models

## Overview

The app uses Apple's **On-Demand Resources (ODR)** to deliver the large RTMDet
TFLite model separately from the initial download. This keeps the App Store /
TestFlight binary smaller and downloads the model only when first needed.

| Model | File | Size | ODR Tag | Used For |
|-------|------|------|---------|----------|
| RTMDet | `rtmdet-ins-m-raw-fp16.tflite` | ~55 MB source asset | `RTMDetModel` | Furniture detection + segmentation and room object anchor; bump tag only when intentionally breaking ODR cache |

**Bundled in the app (not ODR):**

| Model | Location | Used For |
|-------|----------|----------|
| Depth Anything V2 Metric Indoor Small | `Furnit/Models/DepthAnything/` | Single-photo metric depth → USDZ room mesh |
| GeoCalib Pinhole CNN | `Furnit/Models/GeoCalib/` | Camera focal length + gravity for Depth Anything sizing |

Default **iOS room creation** = **instant preview (no ML)** then **GeoCalib + Depth Anything +
RTMDet object anchor → USDZ on first save** (see `CONTEXT.md`, `SinglePhotoRoomViewer.swift`,
`DepthAnythingRoomReconstructor.swift`, and `Furnit/diagrams/room-generation-flow.svg`).
Depth Anything and GeoCalib are bundled; RTMDet runs at save time and for Furniture Fit in viewers.
No separate room-generation ODR tag is used by the active Swift path.

## How It Works

### App Store / TestFlight Distribution
- **Initial app download**: Depth Anything + GeoCalib are bundled; RTMDet may be ODR
- **First furniture detection use**: Downloads RTMDet model if not bundled
- **Subsequent uses**: Cached models load instantly

### Xcode Development
- Xcode stages the tagged RTMDet file as an asset pack beside the app for generic
  device builds; Run/TestFlight/App Store installation controls how that pack is made
  available on the device.
- Services skip ODR whenever the model is physically embedded in the app bundle

## Implementation Details

### 1. Xcode Project Configuration (`project.pbxproj`)

```
// Known asset tags for ODR
knownAssetTags = (RTMDetModel);

// Enable ODR in build settings
ENABLE_ON_DEMAND_RESOURCES = YES;
```

### 2. RTMDetModelService.swift (RTMDet model)

Singleton: `RTMDetModelService.shared`

Key properties:
```swift
@Published var model: RTMDetLiteRuntime?    // nil until loaded
@Published var isLoadingModel: Bool         // true during download + load
@Published var isDownloadingResources: Bool
@Published var downloadProgress: Double
@Published var resourcesAvailable: Bool
```

Key methods:
- `ensureModelLoaded()` - Call from room view `.onAppear`; triggers ODR download + LiteRT Metal load
- `modelForInference()` - Wait for and return the shared runtime for room-anchor work
- `releaseResources()` - Frees disk space and unloads model

One mandatory-Metal interpreter is shared by `ModelViewerView`, saved-room viewers, Settings image scan, and the one-shot object-anchor
step in room generation (**first save only**; preview skips RTMDet for room measurement).

### 3. Depth Anything + GeoCalib (bundled)

- Loaded directly from `Furnit/Models/` by `DepthAnythingRoomReconstructor` and `GeoCalibCalibrationService`
- No ODR download step for default room generation
- Export/refresh scripts: `scripts/export_geocalib_to_coreml.py`, `scripts/convert_depthanything_metric_indoor_small_to_coreml.py`

### 4. Development vs Production detection (RTMDet)

```swift
if Bundle.main.url(forResource: modelName, withExtension: modelExtension) != nil {
    // Development/local install or intentionally bundled resource: load directly.
} else {
    // App Store/TestFlight ODR resource: mount or download the tagged pack.
}
```

### 5. UI

**Depth Anything** (`SinglePhotoRoomViewer.swift`): Instant preview overlay during exploration; save
progress overlay when first-save ML runs. No separate large-model ODR for depth/GeoCalib when bundled.

**RTMDet**: Model loads in the background when any room view appears. Required for Furniture Fit brain overlay and chair-anchor depth calibration.

## Testing ODR

ODR can only be fully tested through **TestFlight** or **App Store**. For local testing:

1. Bundled models load directly from the app
2. You'll see logs like: `RTMDet-Ins-m embedded in app bundle — skipping ODR`

## Files

| File | Role |
|------|------|
| `Furnit.xcodeproj/project.pbxproj` | ODR tags and `ENABLE_ON_DEMAND_RESOURCES` |
| `RTMDetModelService.swift` | RTMDet ODR download + shared LiteRT runtime |
| `RTMDetLiteRuntime.swift` | Dedicated LiteRT worker + mandatory fully audited Metal delegate |
| `DepthAnythingRoomReconstructor.swift` | Bundled Depth Anything + metric pipeline |
| `GeoCalibCalibrationService.swift` | Bundled GeoCalib CNN + Swift LM |
| `SinglePhotoRoomViewer.swift` | Room generation UI |
| `ModelViewerView.swift` | USDZ preview + RTMDet preload |

## Logs

When ODR is working correctly for RTMDet:
```
RTMDet-Ins-m embedded in app bundle — skipping ODR                 // Dev / bundled
RTMDet-Ins-m ODR resources available (conditionallyBeginAccessingResources)
RTMDet-Ins-m ODR download complete
RTMDet-Ins-m loaded with verified full LiteRT Metal delegation ... cpuNodes=0 ...
```

Depth Anything / GeoCalib:
```
[GeoCalib][CNN] loaded model=...
[DepthAnythingRoom][InferenceRaw] ...
[DepthAnythingRoom][MetricCalib] ...
```

## Troubleshooting

### "No manifest found for bundle ID"
This error occurs when running from Xcode. ODR only works with App Store/TestFlight distribution.

### Model not loading after download
Ensure `resourceRequest` is kept alive (not released) after download completes.

### App size still large in development
Expected when Depth Anything, GeoCalib, and RTMDet are all bundled for local development.
