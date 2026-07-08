
# Furnit

Image-to-3D room generation and furniture fitment across iOS and Android.

## Current iOS Documentation

- `Furnit/docs/README.md` — iOS docs and diagram index.
- `Furnit/Views/FurnitureFit/README.md` — current RTMDet/Furniture Fit segmentation pipeline, Settings image scan parity, mask affinity grouping, pixel-level mask union, and overlay pinch ownership.
- `docs/RTMDET_IOS_SWIFT_SPIKE.md` — RTMDet Core ML loader/export expectations and Swift postprocess status.
- `docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md` — room/furniture sizing math and overlay gesture ownership notes.
- `Furnit/diagrams/room-generation-flow.svg` — current Photo to 3D Room flow.
- `Furnit/diagrams/rtmdet-swift-flow.svg` — current RTMDet live/still-image flow.
- `CONTEXT.md` — **GeoCalib + Depth Anything + RTMDet object anchor** single-photo room pipeline (metric depth, USDZ export, measurement metadata).

## iOS room generation (default)

Single-photo room creation uses **GeoCalib** (focal length + gravity from the full frame),
**Depth Anything V2 Metric Indoor** (per-pixel metric depth), and a one-shot **RTMDet** object anchor
for scale/measurement context, then exports a textured **USDZ** mesh. See
`Furnit/Services/RoomReconstruction/GeoCalibCalibrationService.swift`,
`DepthAnythingRoomReconstructor.swift`, and `Furnit/diagrams/room-generation-flow.svg`.

## Android

Android room generation has its own model stack under `android/docs/` (ExecuTorch / ONNX). iOS uses
the GeoCalib + Depth Anything USDZ path above.

https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
https://github.com/apple/ml-matrix3d
