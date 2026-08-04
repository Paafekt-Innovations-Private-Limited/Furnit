# Model & Weights License Audit — Paafekt

> **Not legal advice.** Primary-source research completed **2026-07-11** (GeoCalib weights update **2026-07-19**; Android RTMDet artifact inventory updated **2026-08-04**). Checked by: _[fill name]_. Lawyer review: _[date / firm / outcome]_.

---

## For a second reader (share this file)

Path in repo: [`docs/MODEL_LICENSE_AUDIT.md`](MODEL_LICENSE_AUDIT.md).

**What we discussed / audited:** Depth Anything V2 Metric Indoor **Small**, **Hypersim** (training data), **GeoCalib** pinhole weights, **RTMDet** + **COCO**, plus what we explicitly **do not ship** (Inria 3DGS, DA Base/Large/Giant, MiDaS, SparkJS, etc.).

**Please double-check especially:**

1. **Depth Anything Small** — Upstream says Small weights are **Apache-2.0**. Metric Indoor Small was fine-tuned on **Hypersim (CC-BY-SA 3.0)**. Does Share-Alike create a material risk for redistributing our exported CoreML/ONNX weights in a commercial app? (Our working view: open lawyer question / tail risk, not an automatic ship-blocker.)
2. **GeoCalib** — Upstream README now states trained weights are **CC BY 4.0** (commercial OK with attribution); code **Apache-2.0**. Confirm that still matches the live [GeoCalib README](https://github.com/cvg/GeoCalib/blob/main/README.md) License section.
3. **RTMDet + COCO** — Code Apache-2.0; COCO **annotations** CC-BY-4.0 (we attribute in-app); images are Flickr ToU / not owned by COCO Consortium. Any leftover concern for shipping pretrained detection weights?
4. **Ship inventory** — Confirm we only ship Small / GeoCalib pinhole / RTMDet-ins-m paths listed below (not Base/Large/Giant, not Inria training code).
5. **In-app Licenses** — Attribution present on iOS + Android Settings → Licenses for the above.

**Not asking you to rewrite the whole BOM** — challenge the 🔴 / lawyer-priority items and flag anything we got wrong vs primary sources.

---

## 🔴 License blockers (commercial ship — needs resolution or lawyer)

| # | Item | Verdict | Why (primary source) |
|---|------|---------|----------------------|
| 1 | **Depth Anything V2 Metric Indoor Small — training data** | **AMBIGUOUS — needs lawyer** | Metric Small weights are **Apache-2.0** per upstream README, but fine-tuning uses **Hypersim** (**CC-BY-SA 3.0**). Share-Alike may affect redistribution of derived weights; not resolved in any model card we found. |
| 2 | ~~GeoCalib pinhole weights~~ | **CLEARED for commercial (2026-07-19)** | Upstream README: **code Apache-2.0**, **trained weights CC BY 4.0** (commercial OK; attribution). Thanks Laval authors for allowing that weight license. OpenPano mix remains background; published weight grant is CC BY 4.0. |
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
| Yes | GeoCalib Pinhole CNN | `GeoCalibPinholeCNN.mlpackage` | `geocalib_pinhole_cnn.onnx` |
| Yes | RTMDet-Ins-m | `rtmdet-ins-m.mlpackage` (ODR) | `rtmdet-ins-m-raw-fp16.tflite` |
| Yes | MetalSplatter 1.0.1 | SPM | — |
| Yes | spz-swift 2.1.0 | SPM (transitive) | — |
| Yes | Three.js r170 | WebView GLB/Mesh | WebView GLB |
| Yes | Filament via SceneView 2.0.3 | — | Native GLB |
| Yes | ONNX Runtime 1.24.2 | — | Inference runtime |
| Removed | SparkJS (`spark.module.js`) | **Removed** — splats use MetalSplatter | **Removed** — unused legacy bundle |
| No | M-LSD, MiDaS, Whisper, SHARP, ExecuTorch, NCNN, Inria 3DGS code | — | — |

Splat rooms: user/imported PLY (`_3dgs.ply`). Renderer = **MetalSplatter (MIT)**. There is **no** Inria training code in the repository (verified 2026-07-18).

---

## Model Bill of Materials (verified)

Format: **License (SPDX)** · **Commercial** · **Attribution** · **Source** · **Date**

### 1. Depth Anything V2 — Metric Indoor Small

| Layer | License | Commercial | Attribution | Primary source | Date |
|-------|---------|------------|-------------|----------------|------|
| **Code** | Apache-2.0 | YES | NOTICE + license copy on redistribution | https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE | 2026-07-11 |
| **Weights (Small family)** | Apache-2.0 | YES (per upstream; Small only) | Same as Apache-2.0 | https://github.com/DepthAnything/Depth-Anything-V2/blob/main/README.md — LICENSE section: *“Depth-Anything-V2-**Small** model is under the Apache-2.0 license.”* (Base/Large/Giant are **not shipped** in Paafekt.) | 2026-07-11 |
| **HF checkpoint tag** | *(none)* | — | — | HF API `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` → `"license": null` | 2026-07-11 |
| **HF model card (training)** | — | — | — | https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf — fine-tuned on **Hypersim** (synthetic indoor) | 2026-07-11 |
| **Training data (Hypersim)** | CC-BY-SA-3.0 | YES with **Share-Alike** on derivatives | Attribution + SA | https://github.com/apple/ml-hypersim/blob/main/README.md — *“licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License”* | 2026-07-11 |

**App claim cross-check:** In-app licenses/credits titles use **Depth Anything V2 Metric Indoor Small** (not generic “Depth Anything V2”). English license body states Small weights are Apache-2.0 and Base/Large/Giant are not used — **Matches upstream for Small weights** per GitHub README. Does **not** resolve Hypersim CC-BY-SA training-data Share-Alike (lawyer).

**Shipped checkpoint:** Small encoder (24.8M params); **do not** ship Base/Large/Giant.

**Action:** Lawyer on **CC-BY-SA → exported CoreML/ONNX weights**. Hypersim attribution was added to both in-app license screens on 2026-07-18; attribution does not resolve the Share-Alike legal question.

---

### 2. GeoCalib — Pinhole CNN

| Layer | License | Commercial | Attribution | Primary source | Date |
|-------|---------|------------|-------------|----------------|------|
| **Code** | Apache-2.0 | YES | NOTICE + license copy | https://github.com/cvg/GeoCalib/blob/main/LICENSE — *Copyright 2024 ETH Zurich* | 2026-07-19 |
| **Trained weights** | **CC BY 4.0** | YES | Attribution | https://github.com/cvg/GeoCalib/blob/main/README.md — License section: *“weights … under the Creative Commons Attribution 4.0 International Public License”*; thanks Laval authors | 2026-07-19 |
| **CC BY 4.0 legalcode** | CC-BY-4.0 | YES | BY | https://creativecommons.org/licenses/by/4.0/legalcode | 2026-07-19 |
| **Training (OpenPano)** | Mixed (dataset assembly) | Background | — | https://github.com/cvg/GeoCalib/blob/main/README.md#openpano-dataset — HDRMAPS, Poly Haven (CC0), Laval Indoor HDR | 2026-07-19 |
| ↳ Poly Haven panos | CC0 | YES | Not required | https://polyhaven.com/license | 2026-07-11 |
| **Stanford2D3D** | Terms-of-use form | **Eval only** | — | Benchmark eval; not the published weight license | 2026-07-11 |

**App claim cross-check (2026-07-19):** In-app licenses state **code Apache-2.0** + **shipped pinhole weights CC BY 4.0**, ETH Zurich copyright; UI links both license texts. Matches upstream README.

**Action:** Keep dual attribution links. No NC. Optional: credit OpenPano sources in docs if desired; not required by CC BY weight grant.

---

### 3. RTMDet-Ins-m (OpenMMLab)

| Layer | License | Commercial | Attribution | Primary source | Date |
|-------|---------|------------|-------------|----------------|------|
| **Code (MMDetection)** | Apache-2.0 | YES | NOTICE + copyright | https://github.com/open-mmlab/mmdetection/blob/main/LICENSE — *Copyright 2018-2023 OpenMMLab* | 2026-07-11 |
| **Checkpoint** | *(no separate terms on download page)* | **YES** (same as project; no commercial-restriction terms found) | OpenMMLab + Apache | https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_m_8xb32-300e_coco/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth (linked from Furnit export scripts) | 2026-07-11 |
| **COCO annotations** | CC-BY-4.0 | YES | Attribution | https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm — *“annotations … licensed under a Creative Commons Attribution 4.0 License”* | 2026-07-11 |
| **COCO images** | Flickr ToU | **User responsibility** | — | Same page — *“COCO Consortium does not own the copyright of the images. Use … must abide by the Flickr Terms of Use.”* | 2026-07-11 |

**App claim cross-check:** `licenses.rtmdet` = Apache-2.0, OpenMMLab — **Matches code license**. COCO CC-BY attribution not explicit in app strings.

**Action:** COCO CC-BY-4.0 acknowledgment added to both in-app license screens on 2026-07-18. Lawyer optional on pretrained-weight + Flickr-trained model in commercial app.

---

### 4. Gaussian splatting ship path

| Component | License | Commercial | In ship path? | Primary source | Date |
|-----------|---------|------------|---------------|----------------|------|
| **Inria gaussian-splatting** (training) | **Excluded from ship** | **NO** | **NO** — zero `graphdeco` / `diff-gaussian` refs in repo | https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md — upstream license is incompatible with this commercial product; not used | 2026-07-11 |
| **MetalSplatter 1.0.1** | MIT | YES | **YES** (iOS renderer) | https://github.com/scier/MetalSplatter/blob/main/LICENSE — Copyright (c) 2026 Sean Cier | 2026-07-11 |
| **User PLY splats** | User content | N/A | YES | Generated/imported sidecars | — |
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
| Depth Pro | **Removed** (not used in commercial ship) | `docs/DEAD_CODE_CLEANUP.md` — `DepthProMetricDepthService.swift` deleted | 2026-07-11 |

---

### 7. Not shipped (no action for current build)

| Item | Status | Source |
|------|--------|--------|
| M-LSD | Development scripts only (`scripts/mlsd_draw_room_lines.py`, `scripts/structure_box_measure_room.py`); no app runtime integration | Repo grep |
| Whisper | Zero references | Repo grep |
| Apple SHARP | Doc mention only (historical) | was `docs/UI_AUDIT.md`; splats now MetalSplatter |
| ExecuTorch / NCNN | No active runtime integration | Repo grep |

---

## Inference libraries / runtimes (verified)

| Component | Version | License | Commercial | Attribution | Primary source | In app licences? | Date |
|-----------|---------|---------|------------|-------------|----------------|------------------|------|
| Core ML / RealityKit | System | Apple SDK terms | Per Apple agreement | — | Apple developer terms | — | — |
| ONNX Runtime | 1.24.2 | MIT | YES | Copyright notice | https://github.com/microsoft/onnxruntime/blob/v1.24.2/LICENSE | **Yes** (Android) | 2026-07-18 |
| LiteRT | 1.4.2 | Apache-2.0 | YES | NOTICE + license copy | Published Maven POM for `com.google.ai.edge.litert:litert:1.4.2` | **No** | 2026-08-04 |
| Three.js | r170 | MIT | YES | Copyright notice | Bundled header: `SPDX-License-Identifier: MIT`; https://github.com/mrdoob/three.js/blob/r170/LICENSE | **Yes** | 2026-07-11 |
| Filament (via SceneView) | 2.0.3 | Apache-2.0 | YES | NOTICE | https://github.com/google/filament/blob/main/LICENSE | **Yes** (Android) | 2026-07-18 |
| MetalSplatter | 1.0.1 | MIT | YES | Copyright notice | https://github.com/scier/MetalSplatter/blob/main/LICENSE | **Yes** (iOS) | 2026-07-11 |
| spz-swift | 2.1.0 | MIT | YES | Niantic + Sean Cier notice | https://github.com/scier/spz-swift/blob/main/LICENSE | **Yes** (iOS) | 2026-07-18 |
| Draco | — | Apache-2.0 (typical) | YES | NOTICE | Google Draco (loader support only; Android GLB avoids Draco compression) | **No** | — |
| Firebase | 12.11.0 (iOS SPM) | Apache-2.0 | Per Google ToS | Yes | Firebase SDK | **Yes** | 2026-07-11 |

**Attribution gaps closed in app UI (2026-07-18):** ONNX Runtime, Filament/SceneView, spz-swift, and COCO annotations.

---

## `licenses.phase1Notice` — retired (git history)

| Commit | Date | What happened |
|--------|------|----------------|
| `4ef4e10` | 2026-02-26 | **Added** early Phase 1 notice with legacy model attributions and in-app Licenses screens (legacy restricted wording). |
| `3c15aec` | 2026-06-16 | **Removed notice from UI** ahead of commercial release; kept string keys temporarily. |
| `e2aac05` / later | 2026-07 | Notice wording updated toward commercial release, then removed entirely from license UI. |

**Displayed today:** No. Removed from both in-app license screens on 2026-07-18. Paafekt is a **commercial** product; release/policy wording belongs in product terms, not third-party attribution lists.

**Conclusion:** Phase-1 notice was **product policy**, not required by Depth Anything, GeoCalib, or RTMDet licenses. Current posture: commercial App Store / Play launch. Lawyer should still align full terms with the model audit above.

---

## Red-flag license types (commercial ship)

Licenses that **prohibit commercial distribution**, or research-only / GPL-AGPL / RAIL terms on weights or data → stop or get written permission before shipping.

**Excluded from this commercial product (not shipped):** Inria `gaussian-splatting`; Depth Anything V2 **Base/Large/Giant** (Paafekt ships **Small** only, Apache-2.0).

---

## Process (unchanged)

1. Resolve 🔴 blockers with counsel.
2. Close attribution gaps in Settings → Licenses (both platforms).
3. Re-audit on **any checkpoint version bump**.
4. Keep this doc dated as diligence evidence.

---

## Lawyer priority (one fixed-fee review)

1. **Hypersim CC-BY-SA 3.0** → Depth Anything Metric **Small** exported weights in commercial app (tail risk; commercial not blocked by SA).
2. ~~GeoCalib~~ — **resolved upstream as CC BY 4.0 weights** (2026-07-19); keep attribution.
3. **RTMDet + COCO** (optional; lower risk).

---

## Re-audit trigger checklist

- [ ] New `.mlpackage` / `.onnx` / checkpoint ID change
- [ ] New SPM / Gradle ML dependency
- [ ] M-LSD, Whisper, SHARP, or ExecuTorch promoted on-device
- [x] `phase1Notice` removed from license UI; commercial-release wording belongs in product terms (2026-07-18)
- [x] SparkJS bundle removed — iOS + Android (2026-07-11)
