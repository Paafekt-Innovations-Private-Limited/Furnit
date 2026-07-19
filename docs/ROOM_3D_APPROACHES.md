# Room 3D Reconstruction — Approaches Tried

## Project: Single-Image to 3D Room (Furnit)

**Goal:** Take a single photo of a room interior and produce a 3D model (GLB/OBJ/USDZ) viewable in standard 3D viewers and AR.

**Source image:** `room.jpeg` (1200×1600, indoor room with curtains, ceiling fan, office chair)

> Historical note (2026-07-18): the superseded experiment scripts and generated 3D outputs described below were removed from the app repository. They are archived under `Furnit_non_app_code/obsolete_cleanup_2026-07-18/FurnitTests/` in the external backup.

---

## Approach 1: VLM Spatial JSON → Box Primitives

**File:** `pipeline.py`

**Method:** Query Qwen2.5-VL (7B, local via Ollama) with a prompt asking for room dimensions and asset positions as JSON. Build 3D scene from colored box primitives using trimesh.

**What it produced:**
- Room structure as a bounding box
- Individual assets (chair, curtains) as colored cubes placed at VLM-specified coordinates

**Problems:**
- VLM returned inconsistent/hallucinated coordinates
- Extra phantom boxes appeared (VLM invented objects)
- No photo texture — just solid-colored cubes
- Room felt like a toy diorama, not a reconstruction

**Verdict:** VLM can estimate rough structure but is too unreliable for geometry.

---

## Approach 2: VLM JSON → Photo-Textured Box (Pinhole Camera)

**File:** `perfect_pipeline.py`

**Method:** Same VLM spatial query, but project the original photo onto the room box as a texture using a pinhole camera model with UV mapping.

**What it produced:**
- Room box with photo projected onto visible faces

**Problems:**
- Texture appeared on OUTSIDE of room (normals wrong) — fixed with `mesh.invert()`
- Back-facing surfaces got stretched texture — fixed with facing-angle dot-product check
- Image was horizontally mirrored — `u = 1 - u` fix
- Camera parameters were guesses, causing misalignment

**Verdict:** Photo projection works in theory but depends on correct camera intrinsics/extrinsics that VLM cannot provide accurately.

---

## Approach 3: Pure VLM Oracle → Homography Texturing

**File:** `pure_vlm_room_fixed.py`

**Method:** Ask Qwen for `image_quad_px` (pixel coordinates of each wall's 4 corners in the photo). Use `cv2.findHomography` to warp photo regions onto each 3D face directly — no camera model needed.

**What it produced:**
- Room with per-face texture mapping via homography

**Problems:**
- Qwen returned grid-based rectangles, not actual perspective corner traces
- Some coordinates were negative or identical (VLM hallucination)
- Fell back to OpenCV edge detection to find actual wall boundaries
- Ceiling/floor boundaries were inverted by VLM

**Verdict:** VLM cannot reliably provide pixel-accurate quad boundaries. Homography approach is sound but needs accurate corner data.

---

## Approach 4: Homography + Edge Extension for Unseen Walls

**File:** `final_room.py`

**Method:** Use homography for visible faces (back wall, floor, ceiling). For unseen faces (front wall, side walls), extend edge pixels outward instead of mirroring.

**What it produced:**
- Visible walls textured correctly
- Unseen walls filled with extended edge colors (no black gaps, no mirror)

**Problems:**
- Still dependent on VLM for initial wall boundaries
- Extension looked artificial on side walls
- Overall quality limited by VLM boundary accuracy

**Verdict:** Good technique for filling unseen surfaces, but upstream VLM errors propagate.

---

## Approach 5: Point Cloud (Qwen Depth Heuristic)

**File:** `point_cloud_pipeline.py`

**Method:** Query Qwen for depth constraints (room dimensions, object positions). Build a heuristic depth buffer (sigmoid function), unproject all pixels to 3D using pinhole camera, export as PLY point cloud.

**What it produced:**
- Colored point cloud of the room
- Sigmoid depth gave continuous depth transitions

**Problems:**
- PLY rendered as shimmering gray in web viewers (missing color channels)
- Fixed by exporting as ASCII PLY with explicit uint8 RGB
- Some viewers reported "no faces" (point-cloud-only PLY unsupported)
- Converted to GLB with billboard quads (4 verts + 2 faces per point) for universal compatibility

**Verdict:** Point clouds avoid texture distortion entirely but look grainy and need mesh conversion for most viewers.

---

## Approach 6: Advanced Point Cloud (Sigmoid + Filters)

**File:** `quantum_pointcloud_v2.py`

**Method:** Sigmoid depth buffer with bilinear 2×2 super-sampling for anti-aliased colors, voxel grid downsampling, statistical outlier removal (via scipy cKDTree), PCA normal estimation.

**What it produced:**
- Cleaner, denser point cloud with normals
- Reduced grain and shimmer

**Problems:**
- Sigmoid depth created cylindrical warping (walls curved like a cylinder)
- Voxel size tuning was sensitive

**Verdict:** Good filtering pipeline but sigmoid depth is fundamentally wrong for flat walls.

---

## Approach 7: Piecewise Flat Depth + Soft Edges

**File:** `flat_plane_pointcloud.py`

**Method:** Replace sigmoid with constant Z per region (back wall=2.8m, curtains=2.5m, chair=1.4m). Apply Gaussian blur (σ=3px) ONLY at region boundaries for smooth transitions.

**What it produced:**
- Flat walls (no cylindrical bending)
- Chair and curtains at correct depths
- Exported as PLY + GLB (billboard quads) + OBJ

**Problems:**
- Manual region masks (hardcoded pixel ranges for chair, curtains, ceiling)
- No real depth understanding — just human-specified planes
- OBJ/GLB alignment issues between formats (trimesh axis conventions)

**Verdict:** Best visual result from Qwen-era pipeline. But entirely manual depth — defeats the purpose of automation.

---

## Approach 8: Flat Plane (Alignment Debugging)

**File:** `simple_room_pointcloud.py`

**Method:** Project entire photo onto a flat XY plane at Z=0 (no depth at all). Used to isolate and fix OBJ/GLB viewer alignment issues.

**Findings:**
- trimesh auto-applies -90° X rotation for GLB (Y-up convention) but not for OBJ
- OBJ appeared flat on floor (vertical mapped to Z instead of Y)
- Fix: remove explicit rotation for OBJ export, ensure Y=vertical in source data
- Both GLB and OBJ consistent after fix (Size Y: 3.0, Size Z: 0.0)

**Verdict:** Not a reconstruction approach — purely a debugging exercise that resolved the axis convention confusion.

---

## Approach 9: Depth Anything V2 (Local .pth Checkpoint)

**File:** `depth_anything_pipeline.py` (first version)

**Method:** Load Depth Anything V2 Metric Indoor (ViT-S) from local `.pth` checkpoint on LaCie drive. Run inference on MPS GPU. Use per-pixel metric depth to build point cloud with billboard quads.

**Model:** `/Volumes/LaCie/apr8th2026depth/android/third_party/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth`

**What it produced:**
- 650K point cloud with real learned depth (1.42m–4.35m range)
- Per-pixel metric depth — no manual region masks

**Problems:**
- Billboard quads = scattered pieces, not one solid room
- Looked like confetti/particles instead of a surface

**Verdict:** Real ML depth is vastly better than VLM heuristics, but point cloud rendering is inadequate.

---

## Approach 10: Depth Anything V2 → Connected Grid Mesh (Pinhole Unprojection)

**File:** `depth_anything_pipeline.py` (second version)

**Method:** Same model, but connect adjacent pixels with triangles (grid mesh). Used full pinhole unprojection: `X = (col - cx) * depth / focal_length`.

**What it produced:**
- One solid mesh, 1.92M vertices, 3.8M faces
- Room with depth-aware geometry

**Problems:**
- **Massive stretching/distortion** — pixels at image edges with large depth got pushed far outward (funnel effect)
- Dimensions: X=5.35, Y=8.84, Z=3.97 — aspect ratio wrong
- Chair ended up behind the room (Z direction inverted)
- Looked like a crumpled sheet of paper

**Verdict:** Pinhole unprojection is mathematically correct for multi-view but WRONG for single-image relief display.

---

## Approach 11: Depth Anything V2 HF → Proportional XY + Depth-Only Z ✓

**File:** `depth_anything_pipeline.py` (final version)

**Model:** `depth-anything/Depth-Anything-V2-Small-hf` from HuggingFace (Apache 2.0)

**Method:**
- X = `(col - center) * pixel_scale` — proportional to pixel position only
- Y = `-(row - center) * pixel_scale` — proportional, Y-up
- Z = `-(max_depth - depth)` — depth relief only, near objects toward viewer
- Grid triangulation with 0.4m depth-discontinuity threshold (skip stretched edges)
- Physical scale: 4.5m wide room

**What it produced:**
- 480K vertices, 952K faces
- One solid connected mesh
- Correct proportions (X:4.5m, Y:6.0m, Z:4.0m relief)
- Ceiling fan, curtains, walls all visible and positioned correctly

**Remaining issues:**
- Chair Z-direction required multiple flips to get right
- Some edge fraying where depth discontinuity breaks triangles (acceptable)

**Verdict:** Best approach. Model handles perception; we handle geometry construction.

---

## Approach 12: GeoCalib + Depth Anything + RTMDet Anchor → USDZ (iOS production) ✓

**Files:** `Furnit/Services/RoomReconstruction/GeoCalibCalibrationService.swift`, `DepthAnythingRoomReconstructor.swift`, `Furnit/Utilities/ImageLetterboxLayout.swift`

**Models:**
- **GeoCalib** Pinhole CNN (`GeoCalibPinholeCNN.mlpackage`) — focal length + gravity from full frame (letterbox 320², `fx = fy` in working pixels); LM refinement in Swift.
- **Depth Anything V2 Metric Indoor Small** (`DepthAnythingV2MetricIndoorSmall.mlpackage`) — per-pixel metric depth on the same working grid.

**Method (two-phase):**

**Phase 1 — Instant preview (no ML):**
1. Downsample photo, write JPEG + camera sidecar.
2. `PreviewFast` opens `DepthAnythingPreviewRoomView` with placeholder dims (W=2 m, H=aspect×W, D=3 m).

**Phase 2 — First save (full ML):**
1. Downsample working frame (e.g. 1200×1600).
2. GeoCalib on letterboxed full frame → square-pixel focal in working grid (ARKit capture gravity/height override when present).
3. Depth Anything `scaleFit`, strip letterbox padding, upsample depth to working resolution.
4. RTMDet chair anchor (COCO cls 56, ~1.15 m) scales depth when EXIF agrees focal is correct.
5. Room W×H×D from **depth-unprojected point spread** (p5–p95), not mesh bounds.
6. Proportional-XY relief mesh + texture → **USDZ** export.

Vanishing-point gravity refiner (`VanishingPointGravity`) is stubbed (`vps=0`); GeoCalib + ARKit are the active gravity sources.

**Verdict:** Current iOS shipping path. Preview is instant; metric inference runs only on save.

---

## Summary: What Works

| Component | Best Solution | Why |
|-----------|--------------|-----|
| Camera intrinsics (iOS) | GeoCalib CNN + Swift LM | Focal/gravity from full frame; square pixels (`fx = fy`) |
| Depth source | Depth Anything V2 Metric Indoor (Core ML) | Accurate per-pixel metric depth, no VLM hallucination |
| XY positioning | Proportional to pixel (no depth multiplication) | Avoids funnel distortion |
| Z positioning | Depth relief only, negated for correct facing | Near objects in front, not behind |
| Mesh connectivity | Grid triangulation with depth-jump skip | One solid surface without rubber-banding |
| Export | textured USDZ | Native iOS room preview/save path |

## Models Available

| Model | Size | License | Type |
|-------|------|---------|------|
| GeoCalib Pinhole CNN (.mlpackage) | export-dependent | see `third_party/GeoCalib/` | Focal + gravity, Core ML + Swift LM |
| DA2 Metric Indoor Small (.mlpackage) | 50 MB | Apache 2.0 | Metric meters, CoreML |
| DA2 Small F16+INT8 (.mlpackage) | 27 MB | Apache 2.0 | Relative depth, quantized |
| DA2 Metric Hypersim ViTS (.pth) | ~50 MB | Apache 2.0 | Metric meters, PyTorch |
| DA2 Small HF (online) | ~100 MB | Apache 2.0 | Relative depth, HuggingFace |
| Apple Depth Pro (.mlpackage) | 1.0 GB | **Not used** (removed from ship) | Was evaluated historically; not in Paafekt |

## Key Learnings

1. **VLMs cannot provide pixel-accurate geometry.** They're good for high-level understanding ("there's a chair") but hallucinate coordinates.
2. **Pinhole unprojection causes funnel distortion** when depth range is wide. Only valid for multi-view reconstruction, not single-image relief.
3. **Depth direction matters enormously.** Wrong Z convention = objects behind walls. Took 3 iterations to get right.
4. **Point clouds look like confetti** in most viewers. Connected mesh is mandatory for solid appearance.
5. **Depth discontinuity threshold** is the key to avoiding rubber-sheet artifacts at object edges.
6. **trimesh axis conventions differ** between GLB (auto-rotates to Y-up) and OBJ (raw export). Must account for this.
7. **50MB CoreML model** is production-viable for iOS — no LiDAR needed, runs on Neural Engine.
