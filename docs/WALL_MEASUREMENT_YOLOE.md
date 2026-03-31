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

- **iOS:** CoreML `yoloe-26l-seg-pf_seg_o2m` (loaded via `YOLOEModelService`; letterbox side usually **640**).
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
| **Model input** | **640×640** letterboxed square on iOS 26L PF (`yolo_side` / `modelInputSize`; NCNN Android may differ) | What CoreML/NCNN runs on. |
| **Source thumbnail** | e.g. **4284×5712** px | `reference_image size`; boxes are mapped **here** for wall math. |

So: inference is on the model square (e.g. **640**); **metrics** use the **full thumbnail** width/height.

---

## Anchor class score floor (`0.05`) — not 0, not 0.25

The raw model output can contain **~tens of thousands of anchor rows** (iOS) or many NCNN candidates (Android). Two extremes:

- **Floor = 0:** Almost every anchor is kept → a **low-score** LVIS **571** box can win tier 1 over a better **room** label → unstable sizes.
- **Floor = 0.25 (Furniture Fit default):** Tends to **drop** weak but valid **wall** candidates → tier 1 often empty; room/wall-word tiers pick up.

**Wall measurement** uses a **middle floor: `0.05`** (constant `yoloWallMeasureClassScoreFloor` / `YOLO_WALL_MEASURE_CLASS_SCORE_FLOOR`):

- Drops **anchor noise** without applying the full Furniture Fit **0.25** gate.
- **Semantic tiers** still pick a usable box when class 571 is missing.

---

## Wall bbox selection (four tiers)

After the score floor, we pick **one** rectangle: first tier with any candidate wins; **within tier**, **largest bbox area** (pixel `w × h` in source/thumbnail space).

1. **LVIS class 571** (`wall`).
2. **`\bwall\b` in label**, excluding wall-mounted object phrases (wall lamp, wallpaper, wall clock, …).
3. **`\broom\b` in label** (living room, bedroom, hospital room, …), excluding object-style negatives (e.g. kitchen cabinet, hospital bed) via the same substring list as iOS/Android code.  
   **Important:** Scene-style room boxes often span **floor + wall + ceiling** (a large fraction of the image). The bbox is **not** treated as the wall surface height directly. We **crop vertically inside that rect**: remove **~10% from the top** (ceiling band) and **~25% from the bottom** (floor band), then use the **remaining band** as the wall strip for geometry. This avoids inflated heights (e.g. ~3.8 m from a full-scene box when the real wall band is ~2.5 m).
4. **Full-image fallback** when no class-571 / wall-word / room-word match: a **conservative crop** of the frame — **5%** side margin, **y** from **10%** of image height, **height** **65%** of image (roughly the middle “wall band” when the user aimed at a wall).

**Furniture `blacklist.json` is not used** for this path. Tier 4 guarantees a rectangle so measurement does not abort solely for “no wall label.”

---

## From wall rectangle to meters (on save)

- **Thumbnail EXIF** (if `camera_exif.json` exists) → **focal length in pixels**; else **fallback** `4.5 mm / sensor_width_mm × imageWidth` (pref `wall_measurement_sensor_width_mm`).
- **Depth `Z` at the wall:** median of `sharp_monodepth.bin` over the chosen wall rect when the file exists and the sample is valid; otherwise **`wall_measurement_assumed_depth_m`** clamped to about **0.5…20 m** (`assumed_z`).
- **Pinhole raw size:** `rawW = (wall_px_w / focal_px) * Z`, `rawH = (wall_px_h / focal_px) * Z`.
- **Single calibration scale** (iOS `calibrationScale` / Android equivalent):  
  - If pref is **`auto`** or **`door`**: try **door** scale = `2.03 m / door_height_raw` using a **door** label bbox; door depth uses monodepth at the door rect if available, else the same **`Z`** as the wall.  
  - Else (or if door scale is unavailable): **ceiling** scale = `assumed_ceiling_m / rawH` with `wall_measurement_assumed_ceiling_m` clamped to **2.0…4.5 m**.  
  - If pref is **`door`** only and no usable door was found, the ceiling path is still used but the mode is logged as **`ceiling_fallback`**.
- **Final meters:** `width_m = rawW * scale`, `height_m = rawH * scale`, then clamps (currently about **1.5…12 m** width, **1.5…5 m** height).

---

## Logging (grep-friendly)

- **iOS:** `print` lines prefixed with **`[WALL_MEAS]`** (`logWallMeasurement` in `Logger.swift`).
- **Android:** `LogUtil` tag **`WALL_MEAS`** (`adb logcat | grep WALL_MEAS`).

Useful lines:

- `reference_image`, `class_score_floor`, `yolo_detections count`
- `wall_pick`, `wall_pick_skip`, `yolo_wall_pick`, `yolo_wall_pick_reason`, `room_scene_crop` (tier 3 vertical trim)
- `measure_final` — final **width_m**, **height_m**, **scale**, **depth** source (`monodepth` vs `assumed_z`), **wall_source** / **wall_detection_source**

---

## UI vs saved dimensions (SharpRoomView)

When opening a saved room **from Home** (`allowSave == false`), **saved `.meta` width/height** (from YOLO measurement on save) take **priority** over the WebGL **Box3** estimate for the **navigation title**, so the **5×3.5 m cap** in JS does not overwrite measured values. Live capture **before save** still prefers WebGL until metadata exists.

---

## Console noise unrelated to YOLO

Messages such as **`RTIInputSystemClient`**, **variant selector cell**, **Reporter disconnected** come from **iOS keyboard / text input** and **Xcode**, not from this pipeline. They can be ignored unless you see an actual UI bug.
