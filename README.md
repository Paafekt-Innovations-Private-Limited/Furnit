
# Furnit

Image-to-3D room generation and furniture fitment across iOS and Android.

## Current iOS Documentation

- `Furnit/docs/README.md` — iOS docs and diagram index.
- `Furnit/Views/FurnitureFit/README.md` — current RTMDet/Furniture Fit segmentation pipeline, Settings image scan parity, mask affinity grouping, pixel-level mask union, and overlay pinch ownership.
- `docs/RTMDET_IOS_SWIFT_SPIKE.md` — RTMDet Core ML loader/export expectations and Swift postprocess status.
- `docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md` — room/furniture sizing math and overlay gesture ownership notes.
- `Furnit/diagrams/rtmdet-swift-flow.svg` — current RTMDet live/still-image flow.
- `CONTEXT.md` — **GeoCalib + Depth Anything** single-photo room pipeline (metric depth, USDZ export, chair-anchor calibration).

## iOS room generation (default)

Single-photo room creation uses **GeoCalib** (focal length + gravity from the full frame) and **Depth Anything V2 Metric Indoor** (per-pixel metric depth), then exports a textured **USDZ** mesh. See `Furnit/Services/RoomReconstruction/GeoCalibCalibrationService.swift`, `DepthAnythingRoomReconstructor.swift`, and `scripts/run_geocalib.py` / `scripts/export_geocalib_to_coreml.py`.

## Android

Android room generation still uses the legacy **SHARP** Gaussian-splat stack under `android/docs/` (ExecuTorch / ONNX). iOS does not use SHARP for the default room path.

https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
https://github.com/apple/ml-matrix3d
