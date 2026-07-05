# iOS — Room size, furniture size, ratios, and overlay scale

This document explains **how room dimensions, furniture dimensions, fitment ratios, and mask overlay scale relate** in the **Depth Anything USDZ** room + Furniture Fit flow. It is written so **non-specialists** can follow the ideas, with a **technical section** for engineers matching console logs to code.

Legacy Gaussian-splat / depth-raycast room viewers may still show different bound logs; the **default** iOS room path stores dimensions from **GeoCalib + Depth Anything** inference (see `DepthAnythingRoomReconstructor.swift`).

---

## Lay summary

### Room size on Depth Anything USDZ rooms

For rooms created from a single photo (default path):

1. **GeoCalib** estimates camera focal length on the **full frame** (letterboxed, `fx = fy` in working pixels).
2. **Depth Anything** produces a **metric depth map** on the same pixel grid.
3. **Chair anchor** (RTMDet cls 56) may scale the depth map when EXIF agrees focal is correct.
4. **Room W×H×D** shown in the preview nav bar comes from **depth-unprojected point spread** on that calibrated map — not from mesh bounds or “whole frame = one wall.”

Saved `.usdz.meta` sidecars store the same inference dimensions.

### Two different “room” sizes in legacy splat viewers (if opened)

After a legacy Gaussian-splat PLY runs in MetalSplatter, logs may show **two** width × height × depth values in **scene units (su)**:

1. **Metal splat AABB** — A large box around **all** splat points (the whole Gaussian cloud). It includes slack, outliers, and empty volume. Think *“outer shipping crate for the entire 3D cloud.”*

2. **Metal depth raycast room** — A **tighter** box from **depth-buffer raycasting** (legacy splat viewer): how wide/tall/deep the **occupied** volume looks from the render. Used for Furniture Fit overlay ratios when viewing splat rooms.

So: **splat AABB = coarse outer bounds; depth raycast = refined room extents.** Both can appear as `[PLY_BOUNDS]` lines; the **raycast** pair is the one used for **fit / overlay ratio** when available.

### How “furniture size” is estimated

The model gives a **2D box** on the image (pixels). Turning that into **meters** uses two ideas you will see in logs:

1. **Proportion (simple)** — “What fraction of the frame is the box?” multiplied by **room width/height in meters**. Fast to explain; **misleading on close-ups** (a big object can look “room-sized”).

2. **Pinhole (camera geometry)** — Treat the camera like a **pinhole**: **real size ≈ (size in pixels ÷ focal length in pixels) × distance** to the subject.  
   - **Distance** may come from **LiDAR / AR scene depth / plane raycast** when available.  
   - When not, the pipeline **`av_focal_room_depth_proxy`** uses a **stand-in distance**: roughly **`room depth in meters × 0.45`**.  
   - In logs, **`centerDepth_m=0.22`** is that **distance in meters**, **not** “furniture height.” The **pinhole width × height in meters** next to it **are** the estimated furniture extents from the formula.

### “Ratio” — comparing furniture to room

**Raycast room** and **pinhole furniture** start in different units (scene units vs meters). Before comparing, furniture meters are **mapped into the same scene-unit space as the raycast room** (`furniture_su` in logs).

Then:

- **Fitment ratios** = **furniture ÷ room** in that shared space (width, height, depth).  
  - Example: height ratio **9%** means “this detection’s **height in su** is about **9%** of the **room height in su**.”  
  - **Depth ratio** is often tiny: furniture “depth” from pinhole is a **thickness proxy**, not the room’s depth span — **low % is expected**.

### What actually scales the overlay (mask) on screen

The composited mask uses a **uniform** scale on the image view. Roughly:

- Compute **scaleX** and **scaleY** = **furniture_su ÷ room_su** (width and height).  
- **`autoScaleFromRoom`** uses the **average** of those two, **clamped** (see technical section).  
- **If AR-assisted sizing is valid**, the product uses **AR scale** for the metric path and **does not** multiply in the room ratio again (avoids double scaling).  
- **User pinch** still applies on top.  
- Console: **`[FurnitureFitOverlay]`** shows **`roomStored`** vs **`roomUsed`** (when AR is valid, **`roomUsed`** is 1 so only AR × pinch drives the scale).

---

## Technical reference

### Source locations (quick map)

| Topic | Primary code |
|--------|----------------|
| Depth Anything inference dims | `DepthAnythingRoomReconstructor.swift`, `SinglePhotoRoomViewer.swift` — GeoCalib focal, depth spread, chair anchor, nav-bar W×H×D |
| Pinhole / pipelines / `phase=all` | `Furnit/Views/FurnitureFit/FurnitureFitView.swift` — `primaryBboxMonocularSizeMeters`, `processFrameOnnxStyleCommon` logging |
| Map meters → raycast su | `FurnitureMonocularMeasurer.furnitureMetersMappedToRaycastSceneUnits` |
| Overlay ratios | `Furnit/Models/RoomFitmentMeasurement.swift` — `OverlayScale.ratios`, `OverlayScale.compute`, `FitmentCheck` |
| Room-based overlay scale + AR product | `FurnitureFitView.swift` — `updateAutoScaleFromRoom`, `applyCurrentOverlayScaleTransform`, `updateAssistedOverlayScale` |
| Segmented-overlay gestures | `FurnitureFitView.swift` — `handlePinch(_:)`, `handlePan(_:)`, `hitTest(_:with:)`, `gestureRecognizer(_:shouldReceive:)` |
| Depth raycast room + PLY bounds logs | `Furnit/Views/Components/GaussianSplatView.swift` — `measureRoomFromDepthBuffer`, `logPlyBoundsDiagnostic` |
| USDZ / Depth Anything preview dims | `SinglePhotoRoomViewer.swift` — `DepthAnythingPreviewRoomView`, saved `.usdz.meta` |
| Legacy Sharp room UI (raycast-first title) | `Furnit/Views/SharpRoomView.swift` — `navigationRoomMetersLine`, `raycastRoomDimensions` |

### Room dimensions

- **`room_display_m`** in `[FurnitureFitSize]` — Room width/height/depth in **meters** passed into Furniture Fit (saved meta / raycast-backed display / Depth Anything inference dims for USDZ rooms).  
- **`raycast_su`** — `RoomRaycastDimensions` from Metal depth raycast (W×H×D in **su**).  
- **`roomRaycastSceneDimensions`** on the view — same structure; drives **overlay ratio** when mapping succeeds.

### Furniture dimensions (pipelines)

`primaryBboxMonocularSizeMeters` returns **`(size, pipeline, distanceMeters)`**:

- **LiDAR / depth snapshot** paths use real depth in the bbox.  
- **`ar_frame_scene_depth_intrinsics`** uses live `ARFrame` scene depth when available.  
- **`av_focal_room_depth_proxy`** — AV-style focal + **`centerDepth = roomDepthMeters * 0.45`** (unless snapshot overrides), then `FurnitureMonocularMeasurer.estimateSize(...)`.

### Overlay math (current behavior)

1. **`OverlayScale.ratios(furniture:room:)`** — `scaleX = furn.width/room.width`, `scaleY = furn.height/room.height` in **su**.  
2. **`updateAutoScaleFromRoom`** — Maps pinhole **`FurnitureSceneSize`** (m) → **`furnSu`**, then `uniform = (scaleX + scaleY) / 2`, then clamps **`autoScaleFromRoom`** to **`[0.3, 3.0]`** (`minCombinedOverlayScale` / `maxCombinedOverlayScale` in `FurnitureFitView.swift`).  
3. **`applyCurrentOverlayScaleTransform`** —  
   - `arOn = hasARKitAssistedSizingPayload && arAssistedScaleValid`  
   - `roomFactor = arOn ? 1.0 : autoScaleFromRoom`  
   - `assistedScale = arOn ? autoScaleFromAR : 1.0`  
   - `product = roomFactor * assistedScale * userPinchScale`, clamped again to **`[0.3, 3.0]`**.

So if the **average ratio is below 0.3**, **`autoScaleFromRoom`** sits at **0.3** until AR or pinch changes the combined product.

### Gesture ownership in the main room flow

The segmented furniture cutout is a transparent `UIImageView` (`maskImageView`) over USDZ / GLB / legacy splat room content. The room layer underneath also has its own pinch zoom. Therefore:

- **Two-finger pinch while a cutout is visible** should be claimed by `FurnitureFitContainerView` so it updates `userPinchScale`.
- **Single-finger pan** stays stricter and is accepted only on visible mask pixels.
- Hit-testing must not return `nil` for active two-finger touches; otherwise the underlying `GaussianSplatView`/room viewer receives the pinch and zooms the room instead of the segmented furniture cluster.

If pinch seems broken in the main flow, confirm that `handlePinch(_:)` logs appear. If they do not, the issue is gesture ownership/hit-testing, not overlay scale math.

### Console log cheat sheet

| Prefix / tag | Meaning |
|----------------|---------|
| `[PLY_BOUNDS] Metal splat AABB (classic_ply …)` | Full splat bounding box in **su** (often much larger than the lived-in volume). |
| `[PLY_BOUNDS] Metal depth raycast room …` | Room W×H×D in **su** from depth raycast — **preferred** for fit/UI when present. |
| `[DepthAnythingRoom][MetricCalib]` | GeoCalib vs EXIF focal, depth scale factor, chair anchor before/after. |
| `[DepthAnythingRoom][InferenceDims]` | Final W×H×D, focal_px, depth spread, object bbox raw vs calibrated. |
| `[FurnitureFitSize] phase=fitment_abs` | Furniture and room in **su**, plus pinhole meters and pipeline name. |
| `📐 [Fitment]` | Human-readable **ratio** checks (furniture ÷ room). |
| `📐 [Overlay]` | Logged **scaleX × scaleY** from `OverlayScale.compute` (throttled fitment path). |
| `[FurnitureFitSize] phase=all` | Per-frame summary: **proportion**, **pinhole** (with **centerDepth_m** / proxy note), **ar=**, **tracking=**, **planes=**, **room_display_m**, **raycast_su**, bbox pixels. |
| `[FurnitureFitOverlay]` | Effective overlay: **`roomStored`**, **`roomUsed`**, **`ar`**, **`pinch`**, **`wantAR`**, **`arValid`**. |
| `📐 [PINCH]` | Overlay pinch started/changed/ended; if absent, the room layer probably captured the gesture. |

### Caveats for debugging

- **Primary detection** drives bbox and thus pinhole and ratios — a wrong primary (e.g. pet vs furniture) skews everything.  
- **Fig -17281** / **FigCaptureSourceRemote** messages are **system camera/XPC** noise; not used for sizing math.  
- **Managed CoreMotion plist** warnings in some environments are **permission/sandbox** on system files, unrelated to dimensions.

---

## Related reading

- `docs/IOS_FURNITURE_FIT_ONNX_STYLE_PIPELINE.md` — ONNX-style detection/mask pipeline (where `phase=all` is emitted).  
- `Furnit/Views/FurnitureFit/README.md` — Current RTMDet segmentation stages, Settings scan parity, mask affinity, pixel-union compositing, and overlay gesture ownership.
