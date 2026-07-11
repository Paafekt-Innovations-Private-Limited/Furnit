# Model & Weights License Audit — Paafekt

> **Not legal advice.** Practical due-diligence template for a startup with no legal team. Fill every row, resolve every ⚠️/ambiguous item **before launch**, then get a **one-time fixed-fee IP-lawyer review** of the completed table (cheap, and exactly what investors/acquirers ask for).

**Last repo inventory:** 2026-07-11 (from codebase search — not a legal sign-off).  
**Checked by:** _[name]_  
**Lawyer review:** _[date / firm / outcome]_

---

## The rule that trips everyone up

Each model has **three separate licenses that can differ** — check all three, and the one you ship on is the **weights**:

1. **Code / architecture** (repo)
2. **Weights / checkpoint** (what you ship) ← most important
3. **Training data** (what the weights were trained on)

Classic trap: permissive code (MIT/Apache) but **weights or training data are non-commercial/research-only.**

---

## What Furnit actually ships (repo inventory)

| Shipped in app | Model / library | iOS | Android |
|----------------|-----------------|-----|---------|
| **Yes** | Depth Anything V2 Metric Indoor Small | Core ML `.mlpackage` | ONNX in assets |
| **Yes** | GeoCalib Pinhole CNN | Core ML `.mlpackage` | ONNX export **pending** (`.gitkeep` only) |
| **Yes** | RTMDet-Ins-m | Core ML (ODR `RTMDetModel`) | ONNX `rtmdet-ins-m-raw.onnx` |
| **Yes** | MetalSplatter (renderer, not weights) | SPM 1.0.1 | — |
| **Yes** | Three.js r170 | WebView (GLB/Mesh) | WebView (GLB) |
| **Yes** | Filament (via SceneView 2.0.3) | — | Native GLB |
| **Yes** | ONNX Runtime 1.24.2 | — | Furniture Fit + room gen |
| **No** | M-LSD | — | Scripts only |
| **No** | MiDaS | Removed | Removed |
| **No** | Whisper | — | — |
| **No** | Apple SHARP | Not implemented | Not implemented |
| **No** | ExecuTorch / NCNN / LiteRT | Legacy/gitignored | Legacy/gitignored |
| **Bundled, inactive** | SparkJS (`spark.module.js`) | Legacy WebView | Legacy assets |

Gaussian splat **rooms** are user/imported PLY sidecars (`_3dgs.ply`). The app does **not** ship Inria 3DGS training weights; it **renders** splats via MetalSplatter (MIT). `scripts/depthanything_to_splat.py` builds deterministic Gaussians from depth (INRIA-style PLY layout only).

---

## Model Bill of Materials

Record the link + **date checked** + who checked for every VERIFY cell.

| Model / exact checkpoint | Source (repo / HF URL) | Code license | Weights license | Training data + its license | Commercial OK? | Attribution/NOTICE req? | Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Depth Anything V2 — Metric Indoor Small** ⚠️ | HF: [depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf). Export: `scripts/convert_depthanything_metric_indoor_small_to_coreml.py`. Shipped: `Furnit/Models/DepthAnything/DepthAnythingV2MetricIndoorSmall.mlpackage`, `android/.../depth_anything_v2_metric_indoor_small.onnx` | VERIFY on HF + Depth-Anything-V2 repo | ⚠️ **VERIFY** — DAV2 licenses differ by size/variant; confirm **this metric indoor small checkpoint** allows commercial redistribution | VERIFY (Hypersim / synthetic indoor mix per model card) | ⚠️ | App lists Apache-2.0 (`licenses.depthAnything`) — **confirm weights match** | Read HF license tag + model card **before ship** |
| **Gaussian splatting viewer** (MetalSplatter; not Inria training code) ⚠️ | Renderer: [scier/MetalSplatter](https://github.com/scier/MetalSplatter) (MIT). PLY layout: INRIA-style SH0 in `GaussianSplatView.swift` / `depthanything_to_splat.py` | MIT (MetalSplatter) — `third_party` N/A | **N/A** — no learned 3DGS checkpoint bundled; user PLY data | User content / export pipeline | ⚠️ | MIT attribution for MetalSplatter (iOS licenses screen ✅) | **Confirm** you do not ship Inria 3DGS **training** code or NC weights; renderer MIT is fine |
| **RTMDet-Ins-m** (OpenMMLab) | Checkpoint: [OpenMMLab RTMDet-Ins-m COCO](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_m_8xb32-300e_coco/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth). Export: `scripts/rtmdet_ins_coreml_raw_export.py`, `scripts/rtmdet_ins_onnx_raw_export.py` | VERIFY — MMDetection typically **Apache-2.0** | VERIFY — same checkpoint license on OpenMMLab | **COCO** (annotations **CC-BY-4.0**); images have separate terms | VERIFY | Yes — Apache NOTICE + COCO attribution if required | Confirm weights + [COCO terms](https://cocodataset.org/#termsofuse) |
| **GeoCalib** (pinhole CNN) | [cvg/GeoCalib](https://github.com/cvg/GeoCalib); weights: [geocalib-pinhole.tar v1.0](https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar). Vendored: `third_party/GeoCalib/` | **Apache-2.0** (`third_party/GeoCalib/LICENSE`) | ⚠️ VERIFY release tarball + paper supplement (research projects sometimes add use restrictions beyond code LICENSE) | VERIFY (Stanford2D3D etc. in `siclib` — see GeoCalib docs) | VERIFY | Yes — ETH Zurich / Apache (`licenses.geoCalib`) | Lawyer eye on **weights tarball** terms |
| **M-LSD** (Naver) | Scripts only: `scripts/mlsd_draw_room_lines.py`, `structure_box_measure_room.py`. Upstream: [navervision/mlsd](https://github.com/navervision/mlsd) — **not cited in repo** | VERIFY (commonly Apache-2.0) | VERIFY | VERIFY | N/A — **not shipped** | N/A | No action for app ship; audit if promoted to on-device |
| **MiDaS** (removed) | — | MIT (historical upstream) | — | — | — | — | **Confirm fully removed** from shipped app (`SyntheticDepthEstimator` is non-ML placeholder only) |
| **Whisper** | — | — | — | — | — | — | **Not used** in repo |
| **Apple SHARP** | Mentioned in `docs/UI_AUDIT.md` only | — | — | — | — | — | **Not implemented** — no action unless added |
| **SparkJS** (legacy bundle) | `Furnit/Resources/WebViewVendor/spark/`, `android/.../vendor/spark/` | ⚠️ VERIFY | N/A | N/A | VERIFY if re-enabled | VERIFY | **No LICENSE file in repo** — resolve or remove bundle |
| **ONNX Runtime** | `com.microsoft.onnxruntime:onnxruntime-android:1.24.2` | VERIFY (MIT) | N/A | N/A | VERIFY | Yes | Add to Android licenses screen if missing |
| **Firebase** | SPM / Gradle | Apache-2.0 | N/A | N/A | VERIFY Google ToS | Yes | Listed in app licenses ✅ |
| _(add any new checkpoint here)_ | | | | | | | |

### Inference libraries / runtimes

| Component | Version in repo | License | In app acknowledgements? | Attribution req? |
| --- | --- | --- | --- | --- |
| Core ML / RealityKit (Apple) | System | Apple SDK terms | — | — |
| ONNX Runtime | 1.24.2 (Android) | MIT (VERIFY) | ⚠️ **Not listed** in `LicensesActivity.kt` | Yes |
| Three.js | r170 (Android vendor) | MIT (VERIFY bundled file header) | Yes (`licenses.three`) | Yes |
| Filament / SceneView | SceneView 2.0.3 | Apache-2.0 (VERIFY) | ⚠️ **Not listed** separately | Yes |
| MetalSplatter | 1.0.1 (iOS SPM) | MIT | Yes (iOS only) | Yes |
| spz-swift | 2.1.1 (transitive SPM) | VERIFY | No | Likely yes |
| Draco | Three.js loader support only | Apache-2.0 (VERIFY) | No | Yes if shipping Draco-compressed assets (**currently avoided** on Android GLB) |
| ExecuTorch / NCNN / LiteRT | Gitignored / legacy | VERIFY | No | N/A for current ship |
| TFLite | Not shipped | VERIFY | No | N/A |

**In-app licence surfaces today**

- iOS: `Furnit/Views/ContentView.swift` (Licenses view) + `Furnit/en.lproj/Localizable.strings`
- Android: `android/app/src/main/java/com/furnit/android/LicensesActivity.kt` + `strings.xml`
- Full text: `Furnit/Licenses/APACHE-2.0.txt`, `android/app/src/main/assets/Licenses/APACHE-2.0.txt`

**Gaps to close:** ONNX Runtime, Filament/SceneView, spz-swift, SparkJS (if kept), MetalSplatter on Android (N/A today).

---

## Red-flag license types — stop and resolve before shipping

Any of these on **weights or data**: **non-commercial / "NC" / CC-BY-NC**, **research/academic-only**, **GPL/AGPL** (copyleft — can force open-sourcing), RAIL/"acceptable use" restrictions, or community licenses with **user thresholds**. → swap the model, get written permission, or lawyer-review.

**Repo-specific flags**

- `licenses.phase1Notice` in iOS strings: *"currently offered for non-commercial use only"* — align product terms with model licences.
- `docs/ROOM_3D_APPROACHES.md` notes Apple Depth Pro as non-commercial — **removed** from app; keep out of ship path.

---

## Where to find the truth

For each VERIFY row:

1. Repo `LICENSE` file
2. **Model card** on Hugging Face (commercial limits often here, not in LICENSE)
3. HuggingFace **license tag**
4. Separate "Terms of Use" / weight release notes (e.g. GeoCalib `.tar`, OpenMMLab model zoo)

Record: **URL + date checked + initials**.

---

## Process

1. Fill every row (code / weights / data for each model + each runtime lib).
2. Resolve all ⚠️ and "ambiguous" rows **before launch**.
3. Maintain in-app **"Licenses / Acknowledgements"** with every shipped model + library + required notice text (Apache NOTICE, BSD, CC-BY).
4. Keep this doc **dated** — diligence evidence for fundraising/acquisition.
5. **Re-audit on any model version bump** — a new checkpoint can carry a different license.

---

## Where to spend the one legal dollar

Prioritize a lawyer's eyes on:

1. **Depth Anything V2 Metric Indoor Small** — weights + HF model card (DAV2 family has variant-specific terms).
2. **Gaussian splatting** — confirm ship path is **viewer + user PLY only** (MetalSplatter MIT), not Inria 3DGS NC training stack.
3. **GeoCalib weights tarball** — beyond Apache code LICENSE.
4. **RTMDet + COCO** — commercial use of COCO-trained weights in a consumer app.
5. Anything marked ⚠️/NC/research above.

---

## Re-audit trigger checklist

- [ ] New `.mlpackage` / `.onnx` / `.pte` added to repo or assets
- [ ] Hugging Face checkpoint ID changes in export scripts
- [ ] New SPM / Gradle ML dependency
- [ ] M-LSD, Whisper, SHARP, or ExecuTorch promoted from scripts to on-device
- [ ] App licence strings change from "non-commercial only" to commercial launch
