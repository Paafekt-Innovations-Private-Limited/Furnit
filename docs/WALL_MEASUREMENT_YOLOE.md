# Wall measurement + YOLO-E (on save)

This document describes how **room width/height** are estimated when saving a SHARP **PLY** from **SharpRoomView**, and what **YOLO-E detection** means in this codebase.

Related code:

| Platform | Wall + YOLO entry |
|----------|-------------------|
| iOS | `Furnit/Services/OnDevice/WallMeasurementEstimator.swift`, `YoloEImageInference.swift` |
| Android | `android/.../WallMeasurementEstimator.kt`, `YoloEImageInference.kt` |

---

## What is “YOLO-E detection”?

**YOLO-E** here is the **open-vocabulary / LVIS-style** segmentation model shipped as:

- **iOS:** CoreML `yoloe-11l-seg-pf.mlmodelc` (loaded via `YOLOEModelService`).
- **Android:** NCNN `yoloe-11l-seg` (same role as Furniture Fit).

**Detection** means: for a **still image** (the room **thumbnail** next to the SHARP folder), the model outputs many **candidate boxes** (and mask coeffs). Each candidate has:

- **Class index** → mapped to a **name** in `classes.json` (LVIS-style labels: wall, playroom, wall lamp, …).
- **Bounding box** in **letterboxed model space**, then mapped back to **thumbnail pixel** coordinates.
- **Class score** (per-anchor best class probability in the old anchor layout, or objectness-style score in the “new” tensor layout).

Wall measurement **does not** use Furniture Fit’s `blacklist.json` — it passes an **empty class blacklist** so labels like **wall** / **door** are not stripped.

---

## Image sizes (two different “sizes”)

| Concept | Typical value | Role |
|--------|----------------|------|
| **Model input** | **1280×1280** letterboxed square (`yolo_side` / `modelInputSize`) | What CoreML/NCNN runs on. |
| **Source thumbnail** | e.g. **4284×5712** px | `reference_image size`; boxes are mapped **here** for wall math. |

So: inference is on a **1280** square; **metrics** use the **full thumbnail** width/height.

---

## Anchor class score floor (`0.05`) — not 0, not 0.25

The raw model output can contain **~tens of thousands of anchor rows** (iOS) or many NCNN candidates (Android). Two extremes:

- **Floor = 0:** Almost every anchor is kept → tier 1 can pick a **huge but meaningless** LVIS **571** box at **~0.01** score and **override** a better semantic tier-2 box → unstable wall height.
- **Floor = 0.25 (Furniture Fit default):** Tends to **drop** weak but valid **wall** candidates → tier 1 often empty; tier 2 semantic picks dominate.

**Wall measurement** uses a **middle floor: `0.05`** (constant `yoloWallMeasureClassScoreFloor` / `YOLO_WALL_MEASURE_CLASS_SCORE_FLOOR`):

- Drops **anchor noise** without applying the full Furniture Fit **0.25** gate.
- **Semantic / geometry tiers** still pick the actual wall.

---

## Wall bbox selection (priority order)

After filtering by the score floor, candidates are **not** “the wall” yet — we pick **one** rectangle using **tiers** (first tier with at least one candidate wins; **within tier**, **largest bbox area**):

1. **LVIS class 571** (`wall`) — preferred when present above the floor.
2. **Semantic labels** from `classes.json`: whole-word **wall** / **room**, venue/interior phrases (hotel, playroom, …), with negatives for things like **wall lamp** / **office chair** (not a furniture blacklist — wall measurement never uses `blacklist.json`).
3. **Heuristic wide strip:** wide/tall aspect ratios similar to a **wall panel** (see code for thresholds).
4. **Fallback:** large **wide** boxes (aspect + area fraction), **no** extra confidence gate (geometry only).

**Furniture `blacklist.json` is not used** for this path (logged explicitly).

---

## From wall rectangle to meters (on save)

- **Thumbnail EXIF** (if `camera_exif.json` exists) → **focal length in pixels**; else **fallback** `4.5 mm / sensor_width_mm × imageWidth` (pref `wall_measurement_sensor_width_mm`).
- **Monodepth** (`sharp_monodepth.bin`) if present → median depth in the wall rect → **metric** width/height via **door** or **ceiling** calibration prefs.
- If monodepth is **missing** → **assumed depth** `Z` from `wall_measurement_assumed_depth_m` (`wm = (wall_px_w / focal_px) * Z`, height uses geometry with **ceiling** / **strip** rules when bbox height is unreliable).

---

## Logging (grep-friendly)

- **iOS:** `print` lines prefixed with **`[WALL_MEAS]`** (`logWallMeasurement` in `Logger.swift`).
- **Android:** `LogUtil` tag **`WALL_MEAS`** (`adb logcat | grep WALL_MEAS`).

Useful lines:

- `reference_image`, `class_score_floor`, `yolo_detections count`
- `wall_pick_priority`, `wall_pick_tier_skip`, `yolo_wall_pick`, `yolo_wall_pick_reason`
- `measure_final` — final **width_m**, **height_m**, inputs, **wall_detection_source**

---

## UI vs saved dimensions (SharpRoomView)

When opening a saved room **from Home** (`allowSave == false`), **saved `.meta` width/height** (from YOLO measurement on save) take **priority** over the WebGL **Box3** estimate for the **navigation title**, so the **5×3.5 m cap** in JS does not overwrite measured values. Live capture **before save** still prefers WebGL until metadata exists.

---

## Console noise unrelated to YOLO

Messages such as **`RTIInputSystemClient`**, **variant selector cell**, **Reporter disconnected** come from **iOS keyboard / text input** and **Xcode**, not from this pipeline. They can be ignored unless you see an actual UI bug.
