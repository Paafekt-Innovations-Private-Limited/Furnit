# Model & Weights License Audit — Paafekt

> **Not legal advice.** Primary-source research completed **2026-07-11**. Checked by: _[fill name]_. Lawyer review: _[date / firm / outcome]_.

---

## 🔴 License blockers (commercial ship — needs resolution or lawyer)

| # | Item | Verdict | Why (primary source) |
|---|------|---------|----------------------|
| 1 | **Depth Anything V2 Metric Indoor Small — training data** | **AMBIGUOUS — needs lawyer** | Metric Small weights are **Apache-2.0** per upstream README, but fine-tuning uses **Hypersim** (**CC-BY-SA 3.0**). Share-Alike may affect redistribution of derived weights; not resolved in any model card we found. |
| 2 | **GeoCalib pinhole weights — training data** | **AMBIGUOUS — needs lawyer** | Code + release weights have **no separate NC terms**, but training uses **OpenPano** (HDRMAPS + Poly Haven CC0 + Laval HDR). **HDRMAPS** and **Laval** commercial terms were **not found** in primary sources reviewed. |
| 3 | **HF license tag missing** | **Documentation gap** | Hugging Face API returns `"license": null` for `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` (queried 2026-07-11). Resolved for **Small** via GitHub README; still document in diligence. |

**Not blockers (confirmed):** Inria 3DGS **not** in ship path; MiDaS **removed**; SparkJS **removed** (iOS + Android, 2026-07-11); MetalSplatter / Three.js / ONNX Runtime / RTMDet code **permissive** with attribution.

---

## The rule that trips everyone up

Each model has **three separate licenses that can differ** — check all three; you ship on the **weights**:

1. **Code / architecture** (repo)
2. **Weights / checkpoint** (what you ship) ← most important
3. **Training data** (what the weights were trained on)

---

## What Furnit ships (repo inventory — 2026-07-11)

| Shipped | Artifact | iOS | Android |
|---------|----------|-----|---------|
| Yes | Depth Anything V2 Metric Indoor Small | `DepthAnythingV2MetricIndoorSmall.mlpackage` | `depth_anything_v2_metric_indoor_small.onnx` |
| Yes | GeoCalib Pinhole CNN | `GeoCalibPinholeCNN.mlpackage` | ONNX **not** in assets (`.gitkeep`) |
| Yes | RTMDet-Ins-m | `rtmdet-ins-m.mlpackage` (ODR) | `rtmdet-ins-m-raw.onnx` |
| Yes | MetalSplatter 1.0.1 | SPM | — |
| Yes | spz-swift 2.1.0 | SPM (transitive) | — |
| Yes | Three.js r170 | WebView GLB/Mesh | WebView GLB |
| Yes | Filament via SceneView 2.0.3 | — | Native GLB |
| Yes | ONNX Runtime 1.24.2 | — | Inference runtime |
| Removed | SparkJS (`spark.module.js`) | **Removed** — splats use MetalSplatter | **Removed** — unused legacy bundle |
| No | M-LSD, MiDaS, Whisper, SHARP, ExecuTorch, NCNN, Inria 3DGS code | — | — |

Splat rooms: user/imported PLY (`_3dgs.ply`). Renderer = **MetalSplatter (MIT)**. `scripts/depthanything_to_splat.py` uses INRIA-style PLY layout only; **no** Inria training code in repo (grep 2026-07-11).

---

## Model Bill of Materials (verified)

Format: **License (SPDX)** · **Commercial** · **Attribution** · **Source** · **Date**

### 1. Depth Anything V2 — Metric Indoor Small

| Layer | License | Commercial | Attribution | Primary source | Date |
|-------|---------|------------|-------------|----------------|------|
| **Code** | Apache-2.0 | YES | NOTICE + license copy on redistribution | https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE | 2026-07-11 |
| **Weights (Small family)** | Apache-2.0 | YES (per upstream; Small only) | Same as Apache-2.0 | https://github.com/DepthAnything/Depth-Anything-V2/blob/main/README.md — LICENSE section: *“Depth-Anything-V2-**Small** model is under the Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.”* | 2026-07-11 |
| **HF checkpoint tag** | *(none)* | — | — | HF API `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` → `"license": null` | 2026-07-11 |
| **HF model card (training)** | — | — | — | https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf — fine-tuned on **Hypersim** (synthetic indoor) | 2026-07-11 |
| **Training data (Hypersim)** | CC-BY-SA-3.0 | YES with **Share-Alike** on derivatives | Attribution + SA | https://github.com/apple/ml-hypersim/blob/main/README.md — *“licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License”* | 2026-07-11 |

**App claim cross-check:** iOS `licenses.depthAnything` = “Licensed under the Apache License 2.0.” — **Matches upstream for Small weights** per GitHub README. Does **not** mention Hypersim CC-BY-SA training data.

**Shipped checkpoint:** Small encoder (24.8M params); **do not** ship Base/Large/Giant (NC).

**Action:** Lawyer on **CC-BY-SA → exported CoreML/ONNX weights**. Add Hypersim attribution if counsel agrees.

---

### 2. GeoCalib — Pinhole CNN

| Layer | License | Commercial | Attribution | Primary source | Date |
|-------|---------|------------|-------------|----------------|------|
| **Code** | Apache-2.0 | YES | NOTICE + license copy | https://github.com/cvg/GeoCalib/blob/main/LICENSE — *Copyright 2024 ETH Zurich* | 2026-07-11 |
| **Vendored copy** | Apache-2.0 | YES | Same | `third_party/GeoCalib/LICENSE` in repo | 2026-07-11 |
| **Weights tarball** | *(no separate license text)* | **AMBIGUOUS — needs lawyer** | ETH Zurich / Apache if covered by code license | https://github.com/cvg/GeoCalib/releases/tag/v1.0 — `geocalib-pinhole.tar`; release notes contain **no** additional terms | 2026-07-11 |
| **Training (OpenPano)** | Mixed | **AMBIGUOUS — needs lawyer** | Varies by source | https://github.com/cvg/GeoCalib/blob/main/README.md#openpano-dataset — sources: **HDRMAPS**, **Poly Haven**, **Laval Photometric Indoor HDR** | 2026-07-11 |
| ↳ Poly Haven panos | CC0 | YES | Not required (appreciated) | https://polyhaven.com/license | 2026-07-11 |
| ↳ HDRMAPS panos | **Not found** | **AMBIGUOUS** | — | Referenced in GeoCalib README only; no license URL in repo | 2026-07-11 |
| ↳ Laval HDR | **Not found** | **AMBIGUOUS** | — | http://hdrdb.com/indoor-hdr-photometric/ (linked from GeoCalib README; terms not audited here) | 2026-07-11 |
| **Stanford2D3D** | Terms-of-use form | **Eval only in siclib** | — | GeoCalib README: must agree to Google Form before download — used for **benchmark eval**, not stated as sole training set for shipped pinhole weights | 2026-07-11 |

**App claim cross-check:** `licenses.geoCalib` = Apache-2.0, Copyright 2024 ETH Zurich — **Matches code LICENSE**. Does not disclose OpenPano training mix.

**Action:** Lawyer on weights + HDRMAPS/Laval. Confirm Android ONNX export uses same weights.

---

### 3. RTMDet-Ins-m (OpenMMLab)

| Layer | License | Commercial | Attribution | Primary source | Date |
|-------|---------|------------|-------------|----------------|------|
| **Code (MMDetection)** | Apache-2.0 | YES | NOTICE + copyright | https://github.com/open-mmlab/mmdetection/blob/main/LICENSE — *Copyright 2018-2023 OpenMMLab* | 2026-07-11 |
| **Checkpoint** | *(no separate terms on download page)* | **YES** (same as project; no NC found) | OpenMMLab + Apache | https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_m_8xb32-300e_coco/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth (linked from Furnit export scripts) | 2026-07-11 |
| **COCO annotations** | CC-BY-4.0 | YES | Attribution | https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm — *“annotations … licensed under a Creative Commons Attribution 4.0 License”* | 2026-07-11 |
| **COCO images** | Flickr ToU | **User responsibility** | — | Same page — *“COCO Consortium does not own the copyright of the images. Use … must abide by the Flickr Terms of Use.”* | 2026-07-11 |

**App claim cross-check:** `licenses.rtmdet` = Apache-2.0, OpenMMLab — **Matches code license**. COCO CC-BY attribution not explicit in app strings.

**Action:** Add COCO CC-BY-4.0 acknowledgment. Lawyer optional on pretrained-weight + Flickr-trained model in commercial app.

---

### 4. Gaussian splatting ship path

| Component | License | Commercial | In ship path? | Primary source | Date |
|-----------|---------|------------|---------------|----------------|------|
| **Inria gaussian-splatting** (training) | **Non-commercial research** | **NO** | **NO** — zero `graphdeco` / `diff-gaussian` refs in repo | https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md — *“non-commercially”*, *“THE USER CANNOT USE … FOR COMMERCIAL PURPOSES”* | 2026-07-11 |
| **MetalSplatter 1.0.1** | MIT | YES | **YES** (iOS renderer) | https://github.com/scier/MetalSplatter/blob/main/LICENSE — Copyright (c) 2026 Sean Cier | 2026-07-11 |
| **User PLY splats** | User content | N/A | YES | Generated/imported sidecars | — |
| **depthanything_to_splat.py** | Furnit script | N/A | Export tool only | INRIA-style PLY layout comment only; no Inria code | 2026-07-11 |

**Verdict:** Ship path = **MIT renderer + user PLY** — **OK** for commercial **if** no Inria code/weights added later.

---

### 5. SparkJS (legacy bundle) — **removed both platforms**

| Layer | License | Commercial | Attribution | Primary source | Date |
|-------|---------|------------|-------------|----------------|------|
| **Upstream @sparkjsdev/spark** | MIT | YES | N/A — not shipped | https://github.com/sparkjsdev/spark/blob/main/LICENSE | 2026-07-11 |
| **Furnit iOS** | **Removed** | N/A | **Cleared** | Deleted `Furnit/Resources/WebViewVendor/spark/` | 2026-07-11 |
| **Furnit Android** | **Removed** | N/A | **Cleared** | Deleted `android/app/src/main/assets/vendor/spark/` | 2026-07-11 |

**Verdict:** **Cleared** — inactive bundle removed; splats use MetalSplatter (iOS). No SparkJS in ship path on either platform.

---

### 6. MiDaS — removal confirmed

| Check | Result | Source | Date |
|-------|--------|--------|------|
| MiDaS weights / inference | **Absent** | No `MiDaS` / `midas` in Swift/Kotlin ship path | 2026-07-11 |
| Placeholder | `SyntheticDepthEstimator` — explicit *“not MiDaS”* | `Furnit/Services/RoomReconstruction/SyntheticDepthEstimator.swift` | 2026-07-11 |
| Depth Pro (non-commercial) | **Removed** | `docs/DEAD_CODE_CLEANUP.md` — `DepthProMetricDepthService.swift` deleted | 2026-07-11 |

---

### 7. Not shipped (no action for current build)

| Item | Status | Source |
|------|--------|--------|
| M-LSD | Scripts only (`scripts/mlsd_*.py`) | Repo grep |
| Whisper | Zero references | Repo grep |
| Apple SHARP | Doc mention only (historical) | was `docs/UI_AUDIT.md`; splats now MetalSplatter |
| ExecuTorch / NCNN / LiteRT | Gitignored / legacy | `android/README.md` |

---

## Inference libraries / runtimes (verified)

| Component | Version | License | Commercial | Attribution | Primary source | In app licences? | Date |
|-----------|---------|---------|------------|-------------|----------------|------------------|------|
| Core ML / RealityKit | System | Apple SDK terms | Per Apple agreement | — | Apple developer terms | — | — |
| ONNX Runtime | 1.24.2 | MIT | YES | Copyright notice | https://github.com/microsoft/onnxruntime/blob/v1.24.2/LICENSE | **No** (Android) | 2026-07-11 |
| Three.js | r170 | MIT | YES | Copyright notice | Bundled header: `SPDX-License-Identifier: MIT`; https://github.com/mrdoob/three.js/blob/r170/LICENSE | **Yes** | 2026-07-11 |
| Filament (via SceneView) | 2.0.3 | Apache-2.0 | YES | NOTICE | https://github.com/google/filament/blob/main/LICENSE | **No** | 2026-07-11 |
| MetalSplatter | 1.0.1 | MIT | YES | Copyright notice | https://github.com/scier/MetalSplatter/blob/main/LICENSE | **Yes** (iOS) | 2026-07-11 |
| spz-swift | 2.1.0 | MIT | YES | Niantic + Sean Cier notice | https://github.com/scier/spz-swift/blob/main/LICENSE | **No** | 2026-07-11 |
| Draco | — | Apache-2.0 (typical) | YES | NOTICE | Google Draco (loader support only; Android GLB avoids Draco compression) | **No** | — |
| Firebase | 12.11.0 (iOS SPM) | Apache-2.0 | Per Google ToS | Yes | Firebase SDK | **Yes** | 2026-07-11 |

**Attribution gaps to close:** ONNX Runtime, Filament/SceneView, spz-swift.

---

## `licenses.phase1Notice` — why it exists (git history)

| Commit | Date | What happened |
|--------|------|----------------|
| `4ef4e10` | 2026-02-26 | **Added** Phase 1 notice with YOLO-E / **SHARP** attributions and in-app Licenses screens. Message: *“Phase 1 release: … non-commercial use only.”* |
| `3c15aec` | 2026-06-16 | **Removed notice from UI** ahead of commercial release; kept string keys. Commit message: RTMDet attribution added; phase1 “unused”. |
| `e2aac05` | 2026-07-08 | **Re-displayed** notice; text changed to *“Current release: … non-commercial use only.”* |
| *(working tree)* | 2026-07-13 | **Updated** notice to *“Current release: This app is currently offered for commercial use.”* (iOS + Android string resources). |

**Displayed today:** iOS `Furnit/Views/ContentView.swift` (`LicensesView`); Android `LicensesActivity.kt`.

**Conclusion:** **Product / release-phase policy**, not required by Depth Anything, GeoCalib, or RTMDet licenses. Originally carried from early Phase 1 (SHARP/YOLO-era legal posture). **Updated 2026-07-13** to state commercial use; aligns with intended commercial launch. Lawyer should still align full terms with model audit above.

---

## Red-flag license types

**Non-commercial / NC / research-only / GPL-AGPL / RAIL** on weights or data → stop or get written permission.

**Confirmed NC in ecosystem (not shipped):** Inria `gaussian-splatting` LICENSE.md. DAV2 **Base/Large/Giant** = CC-BY-NC-4.0 per README (Furnit uses **Small only**).

---

## Process (unchanged)

1. Resolve 🔴 blockers with counsel.
2. Close attribution gaps in Settings → Licenses (both platforms).
3. Re-audit on **any checkpoint version bump**.
4. Keep this doc dated as diligence evidence.

---

## Lawyer priority (one fixed-fee review)

1. **Hypersim CC-BY-SA 3.0** → Depth Anything Metric **Small** exported weights in commercial app.
2. **GeoCalib OpenPano** training mix (HDRMAPS / Laval) → commercial weights.
3. **RTMDet + COCO** (optional; lower risk than #1–2).

---

## Re-audit trigger checklist

- [ ] New `.mlpackage` / `.onnx` / checkpoint ID change
- [ ] New SPM / Gradle ML dependency
- [ ] M-LSD, Whisper, SHARP, or ExecuTorch promoted on-device
- [x] `phase1Notice` updated to commercial use (2026-07-13)
- [x] SparkJS bundle removed — iOS + Android (2026-07-11)
