# On-Demand Resources (ODR) for ML Models

## Overview

The app uses Apple's **On-Demand Resources (ODR)** to deliver large CoreML models separately from the initial download. This keeps the App Store / TestFlight binary small and downloads models only when the user first needs them.

| Model | File | Size | ODR Tag | Used For |
|-------|------|------|---------|----------|
| SHARP | `SHARP_fp32_1536.mlpackage` | ~1.2 GB | `SHARPModel` | AI room generation (3D Gaussian splats) |
| RTMDet | `rtmdet-ins-m.mlpackage` / `rtmdet-ins-m.mlmodelc` | export-dependent | `RTMDetModel` | Furniture detection + segmentation (FurnitureFit); bump tag only when intentionally breaking ODR cache |

## How It Works

### App Store / TestFlight Distribution
- **Initial app download**: Much smaller (models excluded)
- **First AI room use**: Downloads SHARP model (~1.2 GB) with progress UI
- **First furniture detection use**: Downloads RTMDet model — fast relative to SHARP
- **Subsequent uses**: Models are cached locally, load instantly

### Xcode Development
- Both models are **bundled** in the app (ODR doesn't work locally)
- No download required during development
- The services skip ODR whenever the model is physically embedded in the app bundle.

## Implementation Details

### 1. Xcode Project Configuration (`project.pbxproj`)

```
// Known asset tags for ODR
knownAssetTags = (SHARPModel, RTMDetModel);

// Enable ODR in build settings
ENABLE_ON_DEMAND_RESOURCES = YES;

// Tag each model file
SHARP_fp32_1536.mlpackage: settings = {ASSET_TAGS = (SHARPModel, ); };
Models/RTMDet/rtmdet-ins-m.mlpackage: assetTagsByRelativePath = { RTMDetModel };
```

### 2. SHARPService.swift (SHARP model)

Key properties:
```swift
@Published var isDownloadingResources: Bool = false
@Published var downloadProgress: Double = 0.0
@Published var resourcesAvailable: Bool = false
```

Key methods:
- `checkResourceAvailability()` - Checks if model is already downloaded
- `downloadResourcesIfNeeded()` - Downloads model with progress tracking
- `releaseResources()` - Frees disk space when model not needed

### 3. RTMDetModelService.swift (RTMDet model)

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

Shared by SharpRoomView, MeshRoomView, GLBRoomView, ModelViewerView, and Settings image scan.

### 4. Development vs Production detection (both services)

```swift
if Bundle.main.url(forResource: modelName, withExtension: modelExtension) != nil {
    // Development/local install or intentionally bundled resource: load directly.
} else {
    // App Store/TestFlight ODR resource: mount or download the tagged pack.
}
```

### 5. UI

**SHARP** (SinglePhotoRoomViewer.swift): Download progress overlay with circular indicator and percentage.

**RTMDet**: Model loads in the background when any room view appears. It is not required until the user taps the brain/FurnitureFit button or runs Settings image scan.

## Testing ODR

ODR can only be fully tested through **TestFlight** or **App Store**. For local testing:

1. Both models load directly from the bundle
2. You'll see logs like: `SHARP: Model embedded in app bundle (...) — skipping ODR`
3. And: `RTMDet-Ins-m embedded in app bundle — skipping ODR`

## Files

| File | Role |
|------|------|
| `Furnit.xcodeproj/project.pbxproj` | ODR tags (`KnownAssetTags`, `ASSET_TAGS`) and `ENABLE_ON_DEMAND_RESOURCES = YES` |
| `SHARPService.swift` | SHARP ODR download + CoreML loading (singleton) |
| `RTMDetModelService.swift` | RTMDet ODR download + CoreML loading (singleton) |
| `SinglePhotoRoomViewer.swift` | SHARP download progress UI |
| `SharpRoomView.swift` | Uses `RTMDetModelService.shared` |
| `MeshRoomView.swift` | Uses `RTMDetModelService.shared` |
| `GLBRoomView.swift` | Uses `RTMDetModelService.shared` |
| `ModelViewerView.swift` | Uses `RTMDetModelService.shared` |

## Logs

When ODR is working correctly, you'll see:
```
SHARP: Model embedded in app bundle (...) — skipping ODR          // Dev / bundled
SHARP: ODR conditionallyBeginAccessingResources: true              // Already downloaded
SHARP: Starting ODR download...                                    // First download
SHARP: ODR download complete                                       // Success

RTMDet-Ins-m embedded in app bundle — skipping ODR                 // Dev / bundled
RTMDet-Ins-m ODR resources available (conditionallyBeginAccessingResources)
RTMDet-Ins-m ODR download complete
```

## Troubleshooting

### "No manifest found for bundle ID"
This error occurs when running from Xcode. ODR only works with App Store/TestFlight distribution.

### Model not loading after download
Ensure `resourceRequest` is kept alive (not released) after download completes. The request must persist for the duration of resource use.

### App size still large in development
Expected behavior. Both models are bundled for local development. Size reduction only applies to App Store distribution.
