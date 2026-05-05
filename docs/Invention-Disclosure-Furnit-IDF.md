# Invention Disclosure — Furnit / Paafekt On-Device Spatial Computing

Technical disclosure describing the implemented system: single-image neural room reconstruction (SHARP-class), live furniture instance segmentation (YOLO-E-class), metric scaling and dimension extraction, and supporting mobile infrastructure. Intended as specification-ready background for patent drafting.

---

## 1. GENERAL INFORMATION

### 1.1 Title of the Invention

**Working title:** System and Method for On-Device Single-Image Room Reconstruction, Metric Scale Resolution, and Instance Segmentation for Augmented-Reality Furniture Visualization

### 1.2 Field of the Invention

| | |
|--|--|
| **Broad field** | Artificial intelligence; mobile computing; computer graphics; augmented reality |
| **Narrow field** | On-device neural 3D scene representation (3D Gaussian splatting inference); real-time instance segmentation on mobile; fusion of platform AR depth/tracking with monocular neural outputs for metric scaling; privacy-preserving spatial pipelines without room-model upload; differentiation from retailer **catalog AR** (IKEA Place–class, Amazon View in Your Room–class apps); technical adjacency to **virtual staging** and real-time **camera-backed compositing** used in broader **VFX / previz** workflows (mobile embodiment, not stage ICVFX) |

### 1.3 Keywords

on-device inference, 3D Gaussian splatting, SHARP, Core ML, ExecuTorch, ONNX Runtime, NCNN, YOLO-E, instance segmentation, open-vocabulary detection, prototype masks, ARCore, ARKit, metric scale estimation, monocular depth, coefficient of variation, room reconstruction, PLY export, Gaussian splat renderer, Metal, Vulkan delegate, furniture fitment, semantic mask compositing, EXIF focal length, plane fitting, RANSAC, on-demand resources, dual-camera contention, retailer AR furniture placement, IKEA Place, Amazon View in Your Room, Wayfair 3D Room Planner, virtual staging, virtual production adjacency, real-time compositing, depth-aware overlay

---

## 2. TECHNICAL DETAILS

### 2.1 Background & Prior Art (“Old Way”)

**Current practice.** Interior spatial apps typically combine one or more of: (i) multi-view photogrammetry or SLAM with dense depth fusion and mesh extraction; (ii) cloud-hosted reconstruction or segmentation with upload of imagery or features; (iii) LiDAR-only or depth-camera-heavy pipelines on premium devices; (iv) generic 2D object detectors without tight integration into mask-based furniture-fit overlays and room-scale metric workflows; (v) separate tools for 3D scanning, measurement, and placement rather than one on-device workflow.

Representative ecosystems (non-exhaustive): ARKit / ARCore; commercial room-scan products; Ultralytics-style detection and segmentation exports; Apple **SHARP**-style single-image 3D Gaussian splatting; ExecuTorch / Core ML mobile deployment; ONNX Runtime and NCNN on Android.

**Deficiencies motivating this work.**

- Cloud reliance conflicts with privacy expectations for home imagery and room geometry.
- Multi-view capture increases time and user skill versus a single photograph.
- Fixed semantic taxonomies often do not match open-world furniture and clutter in consumer rooms.
- Large neural models on mobile need memory-aware loading, backend choice (CPU/GPU/Vulkan), and sometimes refusal rather than crash on constrained devices.
- Metric scale from monocular neural reconstruction is ambiguous without disciplined fusion of platform metric cues with neural outputs and explicit validity gating.

**Comparison to retailer AR furniture apps (IKEA, Amazon, and similar).** Mass-market **“place in your room”** experiences (e.g. IKEA’s AR placement flows, **Amazon View in Your Room**, comparable Wayfair/Target-class tools) typically:

- Anchor a **retailer-supplied 3D SKU** to a **horizontal plane** (floor / table) detected by ARKit/ARCore, sometimes with basic occlusion against coarse depth.
- Treat the session as **sku-centric**: browse catalog → preview one item → minimal persistent **room model** or exportable geometry.
- Rely on **platform AR** for scale and tracking; they generally **do not** run a **single-image neural full-room reconstruction** (Gaussian splat / dense 3D room asset) on device.
- Often integrate **cloud catalog and asset streaming**; privacy posture is product-dependent but differs from an explicitly **on-device room reconstruction** pipeline.

**Furnit differs** along bundled technical axes (see §2.3–2.4): **(i)** optional **whole-room** neural recon from **one photograph** (SHARP-class) with **persisted splat/PLY** and splat rendering; **(ii)** **metric scale** workflows that **fuse AR depths with neural monocular signals** and **refuse** inconsistent estimates (CV gate); **(iii)** **live instance segmentation** (YOLO-E-class prototype masks) with **multi-candidate mask union**, memory-conscious decode, and Android/iOS parity paths—not only dropping a single USDZ on a plane; **(iv)** **manual five-plane** fallback compatible with downstream measurement/placement; **(v)** operational constraints (RAM gates, ODR for huge models, dual-camera contention handling) aimed at **consumer phones**, not only flagship-tier simplified demos.

Counsel should name exact competitor apps and publications for prior-art search; trade names above are illustrative.

**Relationship to virtual production (VP) and VFX.** In film and broadcast, **virtual production** usually denotes **stage-scale** pipelines: LED volumes, camera tracking, engine renders (e.g. Unreal), **ICVFX**, timecode-locked plates, and workstation/DCC export—not the mobile app architecture in this repository. **This codebase does not implement** a VP stage stack (no LED wall sync, nDisplay, or editorial shot-management layer).

The **shared technical idea** with VP/VFX and **virtual staging** is nonetheless real: **convincingly composite synthetic 3D with real-world imagery** using **consistent scale**, **contact with surfaces**, and often **segmentation / mattes**. Furnit’s embodiment targets **handheld mobile spatial computing**: neural **room reconstruction**, **depth- and plane-aware AR**, **instance-mask compositing** for furniture, and **exportable 3D artefacts** (PLY/GLB). That sits **adjacent** to marketing/previz **virtual staging** (populate a photographed room with CG furniture) and to **match-move–adjacent** scale problems, while remaining **distinct from** distributor VP tooling. If patent strategy is meant to reach **studio VP or offline compositing**, counsel should add a **separate embodiment** (e.g. export to EXR/FBX, lens grids, timecode, USD pipeline into Nuke/Resolve) once those features exist or are firmly specified—otherwise claims should stay scoped to **mobile on-device** methods actually implemented.

**Reference documents & similar art.**

| Kind | Reference | What it covers / how this invention differs |
|------|-----------|-----------------------------------------------|
| Retailer AR placement | IKEA AR / Amazon View in Your Room / comparable apps | Catalog SKU on floor plane + platform AR; generally **no** on-device single-image Gaussian room recon, CV-gated metric fusion, or Furnit segmentation union pipeline |
| Third-party model / stack | Ultralytics YOLO-E (segmentation exports) | Detection + mask coefficients; not Furnit-specific post-processing, AR fusion, or SHARP room pipeline |
| Third-party reconstruction | Apple ML SHARP / Gaussian-splat single-image lines of work | Single-image 3D GS inference; not full Furnit integration (scale estimators, ExecuTorch tiling, iOS ODR, dual-camera arbitration) |
| Runtime | Core ML, ExecuTorch, ONNX Runtime, NCNN | General frameworks; not Furnit orchestration and cross-platform parity paths |
| Platform AR | ARKit, ARCore | Tracking and depth; not Furnit-specific scale-ratio CV gate or dimension branches |
| Internal consistency check | `docs/paafekt-patent-audit.md` | Maps draft patent language to this codebase |

Counsel should supplement with patent/publication numbers, arXiv IDs/DOIs, and closest competitor products after a formal prior-art search.

**Runtime and third-party stack (FOT awareness).**

| Dependency | Typical licence | Role in Furnit |
|--------------|-----------------|----------------|
| YOLO-E / Ultralytics tooling | Ultralytics repo is AGPL-3.0; verify exposure for export scripts vs shipped artefacts | Training/export tooling; bundled weights exported to Core ML / ONNX / NCNN |
| Apple SHARP / Core ML | Verify Apple model and conversion terms | iOS SHARP inference |
| PyTorch / ExecuTorch | BSD-style (confirm version) | Android SHARP export and runtime |
| ONNX Runtime | MIT | Android YOLO-E inference path |
| NCNN | BSD | Alternate Android YOLO-E path |
| ARKit / ARCore | Platform SDK terms | Tracking, depth, planes |
| Firebase | Google terms | Authentication in app layer; spatial pipeline avoids uploading room photos/models per current architecture audit |

### 2.2 Summary of the Invention (“New Way”)

The invention is an **integrated, predominantly on-device** system for interior spatial computing on phones and tablets—**not** a thin wrapper around “drop retailer USDZ on the floor” AR. A **single photograph** (with intrinsics from EXIF or the platform) drives **neural 3D Gaussian splat reconstruction** (SHARP-class), yielding a persistent **splat/PLY representation** rendered in real time. **Metric scale** is recovered without uploading room imagery by combining **platform AR metric cues** (depth samples, plane raycasts, anchors) with **monocular depth or reconstruction-internal signals**, using a **robust ratio-based scale estimate** and an **invalidity gate** (coefficient of variation across ratios) when sources disagree. **Live furniture understanding** uses a **YOLO-E-class** segmenter with **instance masks** (prototype-mask decode, candidate pruning, optional Metal-fused compositing)—the same **real-time camera-backed compositing** problem family as **virtual staging** and consumer AR, but with **neural room capture** and **segmentation-centric** matting rather than catalog-only placement. **Fallbacks** include manual boundary capture into a **five-plane textured room**, **dual Android backends** (ONNX vs NCNN), **RAM pre-gates** for multi-gigabyte models, **iOS on-demand model delivery**, and **ARKit/AVCapture contention arbitration**. The result is low-friction capture, privacy-preserving core spatial processing, and robust behaviour across device tiers.

### 2.3 Detailed Description (“How”)

**2.3.1 Structure — major subsystems**

| Component name | Role | Representative implementation |
|----------------|------|-------------------------------|
| **Room reconstructor (neural)** | Single-image → Gaussian parameters → PLY / viewer | iOS: `SHARPService.swift`; Android: `ExecutorchInt8Sharp.kt`, `SharpService.kt` |
| **Room geometry / dimensions** | Plane statistics, EXIF-conditioned branches, AABB from renderer depth | `SharpRoomDimensionsV7.kt`, `RoomGeometryEngine.swift`, `GaussianSplatView` depth-grid measurement |
| **Metric scale estimator** | AR metric + monocular depth ratios; robust statistic; CV gate | `MetricScaleEstimator.kt`, `FurnitureFitArMetrics.kt` |
| **User scale calibration** | Two-point known distance on rendered reconstruction | `SplatDepthQuery.swift` (`TwoPointSplatCalibration`) |
| **Furniture segmentation** | Stretch/letterbox preprocess; Core ML / ONNX / NCNN; decode; NMS; mask union | `FurnitureFitView.swift`, `FurnitureFitOnnxStylePipeline.swift`, `YoloEDetectionParser.swift`, `FurnitureFitManager.kt`, `NcnnYoloe.kt` |
| **AR + camera orchestration** | World tracking, depth, throttling, contention | `FurnitureFitARSupport.swift`, `FurnitureFitArCameraController.kt` |
| **Placement / visualization** | Raycast, wall snap, 2D overlay, splat viewer | `RealityKitObjectPlacementManager.swift`, `GaussianSplatView.swift`, `FurnitureFitOverlayView.kt` |
| **Manual fallback reconstructor** | User polylines → five-plane GLB | `SinglePhotoRoomReconstructor` (iOS/Android), `RoomBoundaryActivity.kt`, `GlbGenerator.kt` |
| **Model delivery & lifecycle** | ODR / bundle resolution; RAM refusal; post-inference release | `SHARPService.swift`, `YOLOEModelService.swift` |

**2.3.2 Function — end-to-end flows**

**Flow A — Neural room from one photo**

1. User captures still; app reads **intrinsics** (EXIF focal / principal point or camera API).  
2. Image resized per model constraint (e.g. ~1536×1536 class inputs for SHARP — export-dependent).  
3. **On-device neural inference** emits Gaussian parameters (tensor names vary by export).  
4. System writes **structured 3D file** (e.g. PLY) and optional thumbnails/sidecars.  
5. **Renderer** displays splats (Metal on iOS); user measures / places content.  
6. **Scale:** Android fuses AR metric depths with **monocular depth** from the SHARP native path; per-sample **ratio** = metric / monocular; **median** (or trimmed) scale; if **coefficient of variation > threshold** (e.g. 0.5), mark **invalid** with reason. Alternatives: plane-separated dimensions with ceiling prior; two-point user calibration.  
7. **Dimensions:** **EXIF focal** and **straight-on vs corner** branches (`SharpRoomDimensionsV7`); iOS may use **quantile-trimmed depth grid** (e.g. 36×36, ~6% tail trim, minimum inlier count) from GPU depth readback for AABB extents.

**Flow B — Live furniture segmentation (Furniture Fit)**

1. **Video frame** from AVCapture; orientation from buffer geometry.  
2. Branch: **ONNX-style** stretch-to-square vs letterbox Core ML (`furniture_fit_use_onnx_runtime` flag — legacy name).  
3. Inference → **detection** + **prototype mask** tensors → boxes, classes, mask coefficients.  
4. **NMS** (IoU 0.5); **primary selection** (confidence / area / center); **candidate pruning** (primary intersection, mask overlap via SGEMV-style ops, reused scratch buffers).  
5. **Union mask**, alpha composite, optional **Metal-fused** path; overlay UI.  
6. **ARKit/ARCore** depth/planes for distance gating; **camera contention** handling between AR and capture sessions.

**Flow C — Manual room**

1. User aligns polylines on the photo for floor/ceiling/wall edges.  
2. System builds **five-plane** textured GLB compatible with viewers and downstream steps.

**Representative parameters (confirm per release branch)**

- YOLO-E input commonly **640×640**; NMS IoU **0.5**; primary confidence ~**0.57**, contributor parse thresholds lower (~**0.10** class).  
- Android ARCore frame throttle **55 ms** minimum interval (`FurnitureFitArCameraController.kt`).  
- SHARP iOS load refusal below ~**2 GB** physical RAM (`SHARPService.swift`).

### 2.4 Novel Elements (drafting focus; prior-art dependent)

1. **Single-image Gaussian-splat room reconstruction** persisted on-device without requiring upload of room imagery for that reconstruction path (network use for auth/catalog is out of scope for this technical summary).  
2. **Two-source metric scaling:** platform AR metric distances plus affine-invariant monocular depth from the neural pipeline, robust aggregation, **CV invalidity gate** instead of always emitting scale.  
3. **EXIF-focal and shot-type-conditioned** dimension extraction from splat/point data.  
4. **Quantile-trimmed depth-buffer grid** room extents from the **splat renderer** (outliers rejected; minimum inlier count).  
5. **Furniture-fit pipeline:** prototype masks, memory-bounded per-candidate ops, optional GPU fusion, **iOS/Android parity** (stretch vs letterbox).  
6. **Operational robustness:** RAM pre-gate; post-inference model release; **ONNX/NCNN** dual backend with device heuristics; **ARKit + AVCapture** contention arbitration; **manual five-plane** fallback interoperable with neural outputs.  
7. **iOS on-demand resources** for multi-gigabyte models with multi-path bundle resolution.  
8. **Differentiation from retailer catalog AR:** combined **room-scale neural recon + metric gating + instance-segmentation furniture composite**, versus typical **single-SKU floor-plane** preview apps.

### 2.5 Best Mode

- **iOS:** SHARP and YOLO-E via **Core ML** with **`cpuOnly`** in production paths for stability; large SHARP via **On-Demand Resources** when applicable; **Metal** splat rendering; **LiDAR scene depth** when present as optional refinement.  
- **Android:** SHARP via **ExecuTorch** (INT8 CPU and/or Vulkan builds); YOLO-E via **ONNX Runtime** (CPU default in reviewed configuration) or **NCNN** per backend flags; **SharpRoomDimensionsV7** and **MetricScaleEstimator** on ARCore-capable hardware.

### 2.6 Alternatives and Variations

- Multi-view or video accumulation (separate embodiment).  
- Letterbox vs stretch; multiple resolutions; FP16/INT8/FP32 exports; Vulkan vs CPU ExecuTorch.  
- Substitute another detector with **mask coefficient** heads reusing the same post-processing surface.  
- Scale via **door height prior** (legacy ~2.1 m iOS), **ceiling constant** (~2.44 m class), or **two-point** calibration only.  
- Cloud-assisted processing (not described here as the preferred embodiment).  
- **Studio / VP export:** bridge splat or GLB outputs into DCC or real-time engines for previz or marketing (separate embodiment if implemented).

### 2.7 Advantages

- **Privacy:** Spatial reconstruction and measurement paths avoid transmitting room photos and 3D room models to application servers (subject to verification of analytics and third-party SDK behaviour in each release).  
- **Low capture friction:** One photo versus lengthy guided scans for the neural room path.  
- **Honest metric output:** CV gating refuses inconsistent scale rather than returning a misleading single number.  
- **Memory discipline:** Reused pixel buffers and multi-arrays, per-candidate linear algebra instead of huge batched allocations where documented in Furniture Fit; RAM gates before loading multi-GB SHARP.  
- **Hardware reach:** CPU-stable neural paths and backend redundancy across Android vendors.  
- **Versus retailer AR demos:** Delivers a **persistent navigable room representation** and **segmentation-driven** furniture integration, not only **catalog geometry on a plane**.

Claims of **absolute centimetre accuracy** are not asserted here without a committed benchmark protocol and dataset.

### 2.8 Drawings / Sketches (for formal figures)

| Figure | Description |
|--------|-------------|
| **Fig. 1** | System context: Camera → {SHARP recon \| YOLO-E segment \| AR} → scale/dimensions → PLY/GLB storage → renderer / placement UI. |
| **Fig. 2** | Single-image flow: photo + intrinsics → inference → Gaussians → PLY → splat renderer → measurement. |
| **Fig. 3** | Metric fusion: AR samples + monocular samples → ratios → robust estimator + CV threshold → scale or invalid. |
| **Fig. 4** | Furniture Fit: frame → preprocess → inference → NMS → mask union → composite. |

Formal drawings should be redrawn to filing standards by counsel or a draftsman.

### 2.9 AI / ML Disclosure Checklist

- **Architectures:** SHARP-class 3D Gaussian splatting (single-image inference); YOLO-E-class detection with prototype mask heads. Novelty target is **system integration, gating, and mobile deployment**, not necessarily base network topology.  
- **Training:** Invention as implemented is primarily **inference-time** and export packaging; training recipes and custom datasets are out of scope unless future claims cover them.  
- **Pre-trained weights:** SHARP and YOLO-E checkpoints — maintain provenance and licence records per release.  
- **Training methodology:** Standard export/fine-tuning pipelines in `scripts/` and `android/pyfiles/` as used for deployment; no special claim here without attorney review.  
- **Evaluation:** Any performance or accuracy claim in filings should cite **reproducible** internal benchmarks (latency, memory, optional tape-measure study); none are embedded in this document.  
- **Inference targets:** Core ML (iOS), ExecuTorch + XNNPACK/Vulkan (Android SHARP), ONNX/NCNN (Android YOLO-E); explicit CPU-first choices where GPU/ANE unstable.

---

## 3. COMMERCIAL & MARKET VIABILITY

### 3.1 Application — Products and Adjacent Industries

**Current product:** **Furnit** — paired **iOS** (Swift, Xcode project `Furnit.xcodeproj`) and **Android** (`android/` Kotlin) applications implementing the spatial pipeline above; marketed/branded in related materials as **Paafekt** / Furniture Fit / Sharp room workflows.

**Future extensions (examples):** In-app commerce hand-off, measurement PDF/JSON export, white-label SDK for retailers — only where actually shipped.

**Adjacent industries:** Interior retail and visualization, **virtual staging** (marketing stills/video from real rooms), property inspection, remote warranty/claims viewing, smart-home layout tools. **Virtual production / episodic VFX:** relevant only as **adjacent** (shared compositing and scale problems); **stage ICVFX** tooling is **out of scope** for the current mobile codebase unless/until a studio export/pipeline embodiment is specified (see §2.1).

### 3.2 Infringement Detectability

| Observable | Hidden / backend |
|------------|------------------|
| App behaviour: single-photo room preview, Gaussian splat viewer, measurement UX, live furniture mask overlay | Exact thresholds, tensor naming, export graph details |
| Binary splits, iOS ODR manifest entries | Training data, cloud-side logic if added later |
| User-visible PLY/GLB or behaviour under test | Proprietary export scripts |

On-device **method** claims align with runtime steps; **system** claims may cite app + bundled model artefacts.

### 3.3 Development Stage

The codebase reflects a **shipped or ship-ready implementation**: full Swift and Kotlin targets, bundled model packages (Core ML `.mlpackage`, ExecuTorch `.pte`, ONNX, NCNN), AR integration, and internal documentation (`docs/`, `android/docs/`). OS version floor, device matrix, and field QA metrics are release-configuration details maintained outside this disclosure.

---

*End of disclosure. Technical background for patent preparation; not legal advice.*
