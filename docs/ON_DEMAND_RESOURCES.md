# On-Demand Resources (ODR) for ML Models

## Overview

The app uses Apple's **On-Demand Resources (ODR)** to deliver large CoreML models separately from the initial download. This keeps the App Store / TestFlight binary small and downloads models only when the user first needs them.

| Model | File | Size | ODR Tag | Used For |
|-------|------|------|---------|----------|
| SHARP | `SHARP_fp32_1536.mlpackage` | ~1.2 GB | `SHARPModel` | AI room generation (3D Gaussian splats) |
| YOLOE | `yoloe-26l-seg-pf.mlpackage` (or `yoloe-26l-seg-pf_seg_o2m`) | ~60 MB | `yoloe_model_v26` | Furniture detection (FurnitureFit); bump tag when model changes (ODR cache) |

## How It Works

### App Store / TestFlight Distribution
- **Initial app download**: Much smaller (models excluded)
- **First AI room use**: Downloads SHARP model (~1.2 GB) with progress UI
- **First furniture detection use**: Downloads YOLOE model (~60 MB) — fast
- **Subsequent uses**: Models are cached locally, load instantly

### Xcode Development
- Both models are **bundled** in the app (ODR doesn't work locally)
- No download required during development
- `#if DEBUG` flag skips ODR logic

## Implementation Details

### 1. Xcode Project Configuration (`project.pbxproj`)

```
// Known asset tags for ODR
KnownAssetTags = (SHARPModel, yoloe_model_v26);

// Enable ODR in build settings
ENABLE_ON_DEMAND_RESOURCES = YES;

// Tag each model file
SHARP_fp32_1536.mlpackage: settings = {ASSET_TAGS = (SHARPModel, ); };
yoloe-26l-seg-pf.mlpackage: settings = {ASSET_TAGS = (yoloe_model_v26, ); };
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

### 3. YOLOEModelService.swift (YOLOE model)

Singleton: `YOLOEModelService.shared`

Key properties:
```swift
@Published var model: MLModel?              // nil until loaded
@Published var isLoadingModel: Bool         // true during download + load
@Published var isDownloadingResources: Bool
@Published var downloadProgress: Double
```

Key methods:
- `ensureModelLoaded()` - Call from room view `.onAppear`; triggers ODR download + CoreML load
- `releaseResources()` - Frees disk space and unloads model

Replaces duplicated `loadMLModel()` functions previously in SharpRoomView, MeshRoomView, GLBRoomView, and ModelViewerView.

### 4. Development vs Production detection (both services)

```swift
private var isRunningFromXcode: Bool {
    #if DEBUG
    return true  // Skip ODR in debug builds
    #else
    return false // Use ODR in release builds
    #endif
}
```

### 5. UI

**SHARP** (SinglePhotoRoomViewer.swift): Download progress overlay with circular indicator and percentage.

**YOLOE**: Model loads in the background when any room view appears. No blocking UI needed since it's only ~60 MB and is not required until the user taps the brain/FurnitureFit button.

## Testing ODR

ODR can only be fully tested through **TestFlight** or **App Store**. For local testing:

1. Both models load directly from the bundle
2. You'll see logs like: `SHARP: Running from Xcode - model bundled locally, skipping ODR`
3. And: `YOLOE: Running from Xcode — model bundled locally, skipping ODR`

## Files

| File | Role |
|------|------|
| `Furnit.xcodeproj/project.pbxproj` | ODR tags (`KnownAssetTags`, `ASSET_TAGS`) and `ENABLE_ON_DEMAND_RESOURCES = YES` |
| `SHARPService.swift` | SHARP ODR download + CoreML loading (singleton) |
| `YOLOEModelService.swift` | YOLOE ODR download + CoreML loading (singleton) |
| `SinglePhotoRoomViewer.swift` | SHARP download progress UI |
| `SharpRoomView.swift` | Uses `YOLOEModelService.shared` |
| `MeshRoomView.swift` | Uses `YOLOEModelService.shared` |
| `GLBRoomView.swift` | Uses `YOLOEModelService.shared` |
| `ModelViewerView.swift` | Uses `YOLOEModelService.shared` |

## Logs

When ODR is working correctly, you'll see:
```
SHARP: Running from Xcode - model bundled locally, skipping ODR   // Dev
SHARP: ODR conditionallyBeginAccessingResources: true              // Already downloaded
SHARP: Starting ODR download...                                    // First download
SHARP: ODR download complete                                       // Success

YOLOE: Running from Xcode — model bundled locally, skipping ODR   // Dev
YOLOE: ODR conditionallyBeginAccessingResources: true              // Already downloaded
YOLOE: Starting ODR download…                                      // First download
YOLOE: ODR download complete                                       // Success
```

## Troubleshooting

### "No manifest found for bundle ID"
This error occurs when running from Xcode. ODR only works with App Store/TestFlight distribution.

### Model not loading after download
Ensure `resourceRequest` is kept alive (not released) after download completes. The request must persist for the duration of resource use.

### App size still large in development
Expected behavior. Both models are bundled for local development. Size reduction only applies to App Store distribution.
