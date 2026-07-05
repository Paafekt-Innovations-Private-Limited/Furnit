# On-Demand Resources (ODR) for ML Models

## Overview

The app uses Apple's **On-Demand Resources (ODR)** to deliver large CoreML models separately from the initial download. This keeps the App Store / TestFlight binary smaller and downloads models only when the user first needs them.

| Model | File | Size | ODR Tag | Used For |
|-------|------|------|---------|----------|
| RTMDet | `rtmdet-ins-m.mlpackage` / `rtmdet-ins-m.mlmodelc` | export-dependent | `RTMDetModel` | Furniture detection + segmentation (FurnitureFit); bump tag only when intentionally breaking ODR cache |

**Bundled in the app (not ODR):**

| Model | Location | Used For |
|-------|----------|----------|
| Depth Anything V2 Metric Indoor Small | `Furnit/Models/DepthAnything/` | Single-photo metric depth → USDZ room mesh |
| GeoCalib Pinhole CNN | `Furnit/Models/GeoCalib/` | Camera focal length + gravity for Depth Anything sizing |

Default **iOS room creation** = **GeoCalib + Depth Anything** (see `CONTEXT.md`, `DepthAnythingRoomReconstructor.swift`). Legacy SHARP ODR tags are no longer used for the shipping room path.

## How It Works

### App Store / TestFlight Distribution
- **Initial app download**: Depth Anything + GeoCalib are bundled; RTMDet may be ODR
- **First furniture detection use**: Downloads RTMDet model if not bundled
- **Subsequent uses**: Cached models load instantly

### Xcode Development
- Models under `Furnit/Models/` are typically **bundled** in the app (ODR doesn't work locally for dev installs)
- RTMDet may still use ODR in production builds when not embedded
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
@Published var model: MLModel?              // nil until loaded
@Published var isLoadingModel: Bool         // true during download + load
@Published var isDownloadingResources: Bool
@Published var downloadProgress: Double
@Published var resourcesAvailable: Bool
```

Key methods:
- `ensureModelLoaded()` - Call from room view `.onAppear`; triggers ODR download + CoreML load
- `releaseResources()` - Frees disk space and unloads model

Shared by `ModelViewerView`, legacy room viewers, and Settings image scan.

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

**Depth Anything** (`SinglePhotoRoomViewer.swift`): Generation progress overlay; no separate large-model ODR for depth/GeoCalib when bundled.

**RTMDet**: Model loads in the background when any room view appears. Required for Furniture Fit brain overlay and chair-anchor depth calibration.

## Testing ODR

ODR can only be fully tested through **TestFlight** or **App Store**. For local testing:

1. Bundled models load directly from the app
2. You'll see logs like: `RTMDet-Ins-m embedded in app bundle — skipping ODR`

## Files

| File | Role |
|------|------|
| `Furnit.xcodeproj/project.pbxproj` | ODR tags and `ENABLE_ON_DEMAND_RESOURCES` |
| `RTMDetModelService.swift` | RTMDet ODR download + CoreML loading |
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
