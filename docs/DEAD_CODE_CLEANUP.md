# Dead code cleanup audit (iOS / Swift)

**Date:** 2026-07-10 (updated after removals)  
**Periphery:** `2.21.2` — `periphery scan --format xcode` → **No unused code detected** (pre-removal)

---

## Removed (approved 2026-07-10)

| Category | Item | Proof |
|----------|------|-------|
| Orphan file | `QwenRoomReconstructor.swift` | Zero Swift call sites |
| Orphan file | `PlanarRoomReconstructor.swift` | Zero Swift call sites (`planarRoomMeters` coordinate frame enum kept in `USDZModel.swift`) |
| Orphan file | `DepthProMetricDepthService.swift` | `shared` / `generateMetricDepthIfPossible` never called |
| Orphan file | `WallMeasurementEstimator.swift` | `enum WallMeasurementEstimator` had no external Swift callers |
| Orphan file | `RoomGenerationImplementation.swift` | Single-case enum unused in Swift |
| Dead entry point | `SinglePhotoRoomReconstructor.processPhoto(_:)` | UI only calls `processPhotoWithBoundaries` |
| Orphan logger | `logDepthPro` + `AlwaysOnOSLog.depthPro` in `Logger.swift` | Only used by deleted `DepthProMetricDepthService` |
| Unused import | `import CoreML` in `SinglePhotoRoomReconstructor.swift` | No CoreML symbols in file |

**~3,700 lines removed.** Live paths unchanged: Depth Anything reconstructor, manual boundary pipeline, scale fusion, capture, USDZ export, AR/LiDAR.

---

## Kept (confirmed live)

| Item | Reason |
|------|--------|
| `SyntheticDepthEstimator.swift` | Used by `SinglePhotoRoomReconstructor.processPhotoWithBoundaries` |
| `logWallMeasurement` in `Logger.swift` | Used by `CameraExifSidecar`, `RoomGenerationCameraSidecar` |
| `RoomCoordinateFrame.planarRoomMeters` | Persisted metadata contract for saved splat rooms |

---

## Build verification

| Step | Result |
|------|--------|
| `xcodebuild` Debug iphoneos | **BUILD SUCCEEDED** (after removals) |
| Unit tests | No Simulator runtime on host — not run |
| On-device smoke | Required before release |

---

## MiDaS note

Legacy MiDaS `DepthEstimator` was already absent; replaced by `SyntheticDepthEstimator` (not removed).
