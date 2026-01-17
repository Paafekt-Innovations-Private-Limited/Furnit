# On-Demand Resources (ODR) for SHARP Model

## Overview

The SHARP CoreML model (`SHARP_fp32_1536.mlpackage`) is 1.2GB, which would make the initial app download very large. To optimize this, we use Apple's **On-Demand Resources (ODR)** system to download the model only when the user first needs it.

## How It Works

### App Store / TestFlight Distribution
- **Initial app download**: ~300MB (without SHARP model)
- **First SHARP feature use**: Downloads 1.2GB model with progress UI
- **Subsequent uses**: Model is cached locally, loads instantly

### Xcode Development
- Model is **bundled** in the app (ODR doesn't work locally)
- No download required during development
- App size will be ~1.5GB+ when built from Xcode

## Implementation Details

### 1. Xcode Project Configuration (`project.pbxproj`)

```
// Asset tag for ODR
KnownAssetTags = (SHARPModel);

// Enable ODR in build settings
ENABLE_ON_DEMAND_RESOURCES = YES;

// Tag the model file
SHARP_fp32_1536.mlpackage: settings = {ASSET_TAGS = (SHARPModel, ); };
```

### 2. SHARPService.swift

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

Development vs Production detection:
```swift
private var isRunningFromXcode: Bool {
    #if DEBUG
    return true  // Skip ODR in debug builds
    #else
    return false // Use ODR in release builds
    #endif
}
```

### 3. UI (SinglePhotoRoomViewer.swift)

Download progress overlay shows:
- Circular progress indicator
- Download percentage
- "One-time download (~1.2 GB)" message

## Testing ODR

ODR can only be fully tested through **TestFlight** or **App Store**. For local testing:

1. The model loads directly from the bundle
2. You'll see log: `SHARP: Running from Xcode - model bundled locally, skipping ODR`

To test the download UI without App Store:
1. Temporarily set `isRunningFromXcode` to return `false`
2. The download will fail but you can see the UI

## Files Modified

| File | Changes |
|------|---------|
| `Furnit.xcodeproj/project.pbxproj` | Added ODR tags and enabled ODR |
| `SHARPService.swift` | Added ODR download logic |
| `SinglePhotoRoomViewer.swift` | Added download progress UI |

## Logs

When ODR is working correctly, you'll see:
```
SHARP: Running from Xcode - model bundled locally, skipping ODR  // Dev
SHARP: ODR conditionallyBeginAccessingResources: true            // Already downloaded
SHARP: Starting ODR download...                                   // First download
SHARP: ODR download complete                                      // Success
```

## Troubleshooting

### "No manifest found for bundle ID"
This error occurs when running from Xcode. ODR only works with App Store/TestFlight distribution.

### Model not loading after download
Ensure `resourceRequest` is kept alive (not released) after download completes. The request must persist for the duration of resource use.

### App size still large in development
Expected behavior. The model is bundled for local development. Size reduction only applies to App Store distribution.
