
# Furnit

Image-to-3D room generation and furniture fitment across iOS and Android.

## Current iOS Documentation

- `Furnit/docs/README.md` — iOS docs, diagram index, and room-viewer smoke test.
- `Furnit/Views/FurnitureFit/README.md` — RTMDet/Furniture Fit segmentation pipeline, room-viewer brain/full-video flow, Settings image scan parity, mask affinity grouping, pixel-level mask union, and overlay pinch ownership.
- `docs/RTMDET_IOS_SWIFT_SPIKE.md` — RTMDet Core ML loader/export expectations and Swift postprocess status.
- `docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md` — room/furniture sizing math and overlay gesture ownership notes.
- `Furnit/diagrams/room-generation-flow.svg` — current **Photo → 3D** room flow.
- `Furnit/diagrams/rtmdet-swift-flow.svg` — RTMDet inline brain / full-video flow in room viewers.
- `CONTEXT.md` — **GeoCalib + Depth Anything + RTMDet object anchor** single-photo room pipeline (metric depth, USDZ export, measurement metadata).

## iOS room generation (default)

Home toolbar **Photo → 3D** → single-photo room creation uses **GeoCalib** (focal length + gravity from the full frame),
**Depth Anything V2 Metric Indoor** (per-pixel metric depth), and a one-shot **RTMDet** object anchor
for scale/measurement context, then exports a textured **USDZ** mesh. See
`Furnit/Services/RoomReconstruction/GeoCalibCalibrationService.swift`,
`DepthAnythingRoomReconstructor.swift`, and `Furnit/diagrams/room-generation-flow.svg`.

## Android

Android room generation exports an optimized **flat full-photo GLB** preview under `android/docs/`:
the method picker appears immediately, photo decode is EXIF-aware and sampled off the UI thread,
and GLB textures are embedded as JPEG for faster preview/save. iOS uses the GeoCalib + Depth
Anything USDZ path above. Both platforms share inline brain / full-video segmentation UX and the
room viewer top controls: floating back, center ruler/pinch/tap helpers, recenter/save, and AR.

https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
https://github.com/apple/ml-matrix3d
