# On-Demand Resources (ODR) for iOS ML Models

## Current model delivery

The iOS app uses Apple's On-Demand Resources to make the RTMDet-Ins-m **Core ML**
model available under the `RTMDetModel` tag. Android separately ships an FP16
TFLite model; that file is not an iOS runtime dependency and is explicitly excluded
from the `Furnit` target's synchronized-folder membership.

| Model | Source file | ODR tag | Used for |
|---|---|---|---|
| RTMDet-Ins-m | `Furnit/Models/RTMDet/rtmdet-ins-m.mlpackage` | `RTMDetModel` | Furniture detection/segmentation and the room-generation object anchor |

The following Core ML models are bundled without a separate ODR tag:

| Model | Location | Used for |
|---|---|---|
| Depth Anything V2 Metric Indoor Small | `Furnit/Models/DepthAnything/` | Single-photo metric depth and projective USDZ generation |
| GeoCalib Pinhole CNN | `Furnit/Models/GeoCalib/` | Camera focal length and gravity calibration |

The default Photo → 3D path runs GeoCalib, Depth Anything, and the RTMDet object
anchor before opening the version-5 projective USDZ preview. Save promotes that
inspected artifact without rerunning reconstruction. Preview/save appearance remains
device-unconfirmed; see `Furnit/docs/README.md`.

## Xcode configuration

`Furnit.xcodeproj/project.pbxproj` owns the distribution contract:

```text
knownAssetTags = (RTMDetModel);
ENABLE_ON_DEMAND_RESOURCES = YES;
```

The project assigns `Models/RTMDet/rtmdet-ins-m.mlpackage` to `RTMDetModel`. Generic
device builds may stage tagged resources beside or inside the development product;
App Store/TestFlight installation controls production delivery.

## Runtime ownership

`RTMDetModelService.shared` owns one `MLModel` shared by live Furniture Fit, the
Settings image scan, and room-generation object anchoring.

Key state:

```swift
@Published var model: MLModel?
@Published var isLoadingModel: Bool
@Published var isDownloadingResources: Bool
@Published var downloadProgress: Double
@Published var resourcesAvailable: Bool
```

Key methods:

- `ensureModelLoaded()` starts a background load for room viewers.
- `modelForInference()` waits for the shared load when a caller cannot proceed
  without the model.
- `releaseResources()` unloads Core ML state and ends ODR access.

The service first checks whether a compiled model/package is already visible through
`Bundle.main`. If not, it mounts or downloads the `RTMDetModel` resource request. It
then loads Core ML with this compute-unit fallback order:

1. `.cpuAndNeuralEngine`
2. `.cpuAndGPU`
3. `.all`
4. `.cpuOnly`

`RTMDetImageInference` owns image preparation, raw-head decoding, class-aware NMS,
mask-affinity grouping, and cutout generation.

## Validation

For a local device build, verify that RTMDet loads and that Furniture Fit can identify
and segment an item. ODR download behavior must also be checked through TestFlight or
App Store distribution.

Expected service logs include one of these resource paths followed by a Core ML load:

```text
RTMDet-Ins-m embedded in app bundle — skipping ODR
RTMDet-Ins-m ODR resources available (conditionallyBeginAccessingResources)
RTMDet-Ins-m ODR download complete
RTMDet-Ins-m loaded with computeUnits=... location=...
```

A Release build must contain `rtmdet-ins-m.mlmodelc` (or the tagged source package in
its ODR product) and must not contain any `.tflite` file. The 2026-08-16 fresh unsigned
iPhoneOS Release build passed that inventory check.

## Owning files

| File | Role |
|---|---|
| `Furnit.xcodeproj/project.pbxproj` | ODR tag, synchronized-folder exclusions, and build settings |
| `Furnit/Services/OnDevice/RTMDetModelService.swift` | ODR access and shared Core ML model lifecycle |
| `Furnit/Services/OnDevice/RTMDetImageInference.swift` | Core ML inference and postprocess |
| `Furnit/Models/RTMDet/README.md` | Accepted model names and output contract |
| `Furnit/Views/FurnitureFit/README.md` | Live/still-image flow and overlay behavior |

Do not add an iOS TFLite model or revive the reverted LiteRT/Metal runtime from this
document. That experiment remains historical on `wip/litert-ios`.
