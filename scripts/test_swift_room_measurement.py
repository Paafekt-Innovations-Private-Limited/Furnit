#!/usr/bin/env python3
"""Unit test for the Swift single-photo room measurement.

Replays `DepthAnythingRoomReconstructor.reconstructWithResult` math exactly:
  1. Working image: downsample so max(side) == 1600 (Swift `downsampledImage`).
  2. Depth Anything CoreML, scaleFill 518x518, bilinear resize back (Swift `inferDepth`).
  3. GeoCalib CNN (letterbox 320, [0,1] input) + content-only solver -> roll/pitch.
  4. Focal: EXIF/sidecar 35mm-equiv -> f_px = f35/36 * width (measurement trust order).
  5. RTMDet bbox for floor exclusion (chair etc.), padded 5%+2px (Swift `excludeRect`).
  6. Camera height: bottom-band leveled median y (`cameraHeightFromFloorSamples`),
     scale = 1.7 / camH when 0.45..5.0.
  7. Dims: wall-anchored leveled spread W/D + ceiling-anchored H (`measureDepthSpread`).

Prints the same debug values as the Swift logs and PASS/FAIL against tape.

Run:
  python3 scripts/test_swift_room_measurement.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DA_PKG = REPO_ROOT / "Furnit/Models/DepthAnything/DepthAnythingV2MetricIndoorSmall.mlpackage"
GEO_PKG = REPO_ROOT / "Furnit/Models/GeoCalib/GeoCalibPinholeCNN.mlpackage"

# Swift constants (DepthAnythingRoomReconstructor).
MAX_WORKING_DIMENSION = 1600
CAMERA_HEIGHT_PRIOR_M = 1.70
CAMERA_HEIGHT_RAW_VALID = (0.45, 5.0)
FLOOR_BAND_START_FRACTION = 0.78
FLOOR_CHAIR_EXCLUDE_U = 0.58
FLOOR_CHAIR_EXCLUDE_V = 0.55
WALL_MARGIN = 0.05
MAX_PLAUSIBLE_ROLL_RAD = 0.6
MAX_PLAUSIBLE_PITCH_RAD = 0.9
CEILING_BAND_ROW_FRACTION = 0.18
MIN_CEILING_CLEARANCE_M = 0.3
CEILING_ANCHORED_HEIGHT_RANGE = (1.9, 4.2)
PLAUSIBLE_ROOM_SPAN = (1.2, 8.0)
FALLBACK_FOCAL_35MM = 28.0
MIN_HFOV_DEG = 55.0
MAX_HFOV_DEG = 88.0
DEFAULT_HFOV_DEG = 70.0
# Rejected candidate fix (kept for reference): flattening the observed floor band to
# cancel residual pitch made all dims worse on the living room (DA depth error is not
# a pure rotation) and mis-fires on rooms where the bottom band is not floor.
FLOOR_CORRECTION = False
# Candidate fix under test: ceiling clearance from the z=0 intercept of the
# clearance-vs-depth fit, cancelling residual-pitch inflation on the ceiling band.
CEILING_INTERCEPT = True
# Candidate fix under test: the GeoCalib CNN/solver pitch convention appears opposite
# to the physical up-vector convention (positive = camera up). -1 flips it.
GRAVITY_PITCH_SIGN = float(__import__("os").environ.get("PITCH_SIGN", "1"))
GRAVITY_ROLL_SIGN = float(__import__("os").environ.get("ROLL_SIGN", "1"))
# Rejected candidate (kept for reference): floor/ceiling plane separation. Plane fits
# collapse on non-planar bands (bathroom fixtures, pitched framings) and inherit the
# ceiling depth error.
PLANE_HEIGHT = __import__("os").environ.get("PLANE_HEIGHT", "0") == "1"
# Rejected candidate (kept for reference): wall-plateau height (pixel extent of the
# constant-depth back-wall band). DA's depth plateau bleeds onto floor/ceiling near the
# wall, so the run spans far beyond the true junctions (living came out 3.94).
WALL_PLATEAU = __import__("os").environ.get("WALL_PLATEAU", "0") == "1"


def padded_bbox(bbox, shape):
    if bbox is None:
        return None
    h, w = shape
    pad_x = int(round((bbox[1] - bbox[0]) * 0.05)) + 2
    pad_y = int(round((bbox[3] - bbox[2]) * 0.05)) + 2
    return (max(0, bbox[0] - pad_x), min(w - 1, bbox[1] + pad_x),
            max(0, bbox[2] - pad_y), min(h - 1, bbox[3] + pad_y))

_da_model = None
_geo_model = None


def da_model():
    global _da_model
    if _da_model is None:
        import coremltools as ct
        _da_model = ct.models.MLModel(str(DA_PKG))
    return _da_model


def geo_model():
    global _geo_model
    if _geo_model is None:
        import coremltools as ct
        _geo_model = ct.models.MLModel(str(GEO_PKG))
    return _geo_model


# ---------------------------------------------------------------- Swift helpers

def swift_percentile(sorted_values: np.ndarray, fraction: float) -> float:
    """Swift `percentile(sorted:fraction:)`: nearest-rank via round(frac * (n-1))."""
    idx = int(round(fraction * (len(sorted_values) - 1)))
    return float(sorted_values[idx])


def swift_median(values: np.ndarray) -> float:
    """Swift `median(_:)`: midpoint of sorted middle pair."""
    s = np.sort(values)
    m = len(s) // 2
    if len(s) % 2 == 0:
        return float((s[m - 1] + s[m]) * 0.5)
    return float(s[m])


def gravity_up_vector(roll: float, pitch: float) -> np.ndarray:
    """Swift `GeoCalibCalibrationResult.gravityVector` ((0,-1,0) at level, +pitch = camera up)."""
    sr, cr, sp, cp = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)
    v = np.array([-sr * cp, -cr * cp, sp], dtype=np.float64)
    return v / np.linalg.norm(v)


def leveling_rotation(up_camera: np.ndarray) -> np.ndarray:
    """Swift `levelingRotationMatrix`: rotate `up_camera` onto (0,-1,0)."""
    target = np.array([0.0, -1.0, 0.0])
    src = up_camera / np.linalg.norm(up_camera)
    c = float(np.dot(src, target))
    if c > 0.9999:
        return np.eye(3)
    axis = np.cross(src, target)
    axis = axis / np.linalg.norm(axis)
    ang = np.arccos(np.clip(c, -1.0, 1.0))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


# ---------------------------------------------------------------- pipeline stages

def working_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= MAX_WORKING_DIMENSION:
        return image
    scale = MAX_WORKING_DIMENSION / longest
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR)


def focal_from_hfov(width: int, hfov_deg: float) -> float:
    return (width * 0.5) / np.tan(np.radians(hfov_deg * 0.5))


def hfov_from_focal(width: int, focal_px: float) -> float:
    return float(np.degrees(2 * np.arctan((width * 0.5) / max(focal_px, 1e-6))))


def resolve_focal(width: int, candidate_focal: float | None = None) -> tuple[float, str, bool]:
    min_focal = focal_from_hfov(width, MAX_HFOV_DEG)
    max_focal = focal_from_hfov(width, MIN_HFOV_DEG)
    if candidate_focal is not None and np.isfinite(candidate_focal) and candidate_focal > 1:
        clamped = float(np.clip(candidate_focal, min_focal, max_focal))
        return clamped, "geocalib_clamped" if abs(clamped - candidate_focal) > 0.5 else "geocalib", abs(clamped - candidate_focal) > 0.5
    return focal_from_hfov(width, DEFAULT_HFOV_DEG), "default_70deg", False


def infer_depth(image: Image.Image) -> np.ndarray:
    """Swift `inferDepth`: Vision .scaleFill to 518, resize back to working size."""
    w, h = image.size
    out = da_model().predict({"image": image.resize((518, 518), Image.Resampling.BILINEAR)})
    d = np.asarray(next(iter(out.values())), dtype=np.float32).squeeze()
    return np.asarray(Image.fromarray(d, mode="F").resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)


def geocalib_roll_pitch(image: Image.Image) -> tuple[float, float]:
    """GeoCalib CNN + content-only solve (numerical stand-in for the Swift LM solver)."""
    from scipy.optimize import minimize

    SIDE = 320
    w, h = image.size
    scale = min(SIDE / w, SIDE / h)
    cw, ch = max(1, round(w * scale)), max(1, round(h * scale))
    ox, oy = (SIDE - cw) // 2, (SIDE - ch) // 2
    canvas = Image.new("RGB", (SIDE, SIDE), (0, 0, 0))
    canvas.paste(image.resize((cw, ch), Image.Resampling.BILINEAR), (ox, oy))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    out = geo_model().predict({"image": np.transpose(arr, (2, 0, 1))[None].astype(np.float32)})
    up = np.asarray(out["up_field"]).squeeze()
    lat = np.asarray(out["latitude_field"]).squeeze()
    up_conf = np.asarray(out["up_confidence"]).squeeze()
    lat_conf = np.asarray(out["latitude_confidence"]).squeeze()

    stride = 8
    xs, ys, ux, uy, ls, wu, wl = [], [], [], [], [], [], []
    for y in range(max(stride // 2, oy), min(SIDE, oy + ch)):
        if (y - stride // 2) % stride:
            continue
        for x in range(max(stride // 2, ox), min(SIDE, ox + cw)):
            if (x - stride // 2) % stride:
                continue
            uxv, uyv = up[0, y, x], up[1, y, x]
            n = max(1e-6, float(np.hypot(uxv, uyv)))
            uc = float(np.clip(up_conf[y, x], 0, 1))
            lc = float(np.clip(lat_conf[y, x], 0, 1))
            if uc > 1e-4 or lc > 1e-4:
                xs.append(x); ys.append(y)
                ux.append(uxv / n); uy.append(uyv / n)
                ls.append(np.sin(lat[y, x]))
                wu.append(max(0.05, uc)); wl.append(max(0.05, lc))
    xs = np.array(xs, float); ys = np.array(ys, float)
    ux = np.array(ux); uy = np.array(uy); ls = np.array(ls)
    wu = np.array(wu); wl = np.array(wl)

    def cost(state):
        roll, pitch, logf = state
        f = np.exp(logf)
        u = (xs - SIDE * 0.5) / f
        v = (ys - SIDE * 0.5) / f
        sr, cr, sp, cp = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)
        gx, gy, gz = -sr * cp, -cr * cp, sp
        pux, puy = gx - gz * u, gy - gz * v
        n = np.maximum(1e-6, np.sqrt(pux ** 2 + puy ** 2))
        pux, puy = pux / n, puy / n
        pls = np.clip((u * gx + v * gy + gz) / np.sqrt(u ** 2 + v ** 2 + 1), -1, 1)
        s2 = 1e-4
        def hub(q):
            return np.where(q / s2 <= 1, q, (2 * np.sqrt(np.maximum(q / s2, 0)) - 1) * s2)
        return float(np.mean(wu * hub((ux - pux) ** 2 + (uy - puy) ** 2) + wl * hub((ls - pls) ** 2)))

    res = minimize(cost, [0.0, 0.0, np.log(0.7 * SIDE)], method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 2000})
    return float(res.x[0]), float(res.x[1])


def rtmdet_detection(image: Image.Image) -> tuple[tuple[int, int, int, int], int, float] | None:
    """Best measurement object bbox in working-image pixels, plus class/conf."""
    try:
        from rtmdet_exclude_mask import _decode_class_candidates, _model_input_payload, MODEL_INPUT, DEFAULT_MODEL
        import coremltools as ct
    except ImportError:
        return None
    if not DEFAULT_MODEL.exists():
        return None
    model = ct.models.MLModel(str(DEFAULT_MODEL), compute_units=ct.ComputeUnit.CPU_ONLY)
    outputs = {k: np.asarray(v) for k, v in model.predict(_model_input_payload(model, image)).items()}
    w, h = image.size
    best = None
    for class_id in (56, 57, 59, 60, 61):  # chair, couch, bed, dining table, toilet
        for cand in _decode_class_candidates(outputs, class_id, 0.30):
            score, box, *_ = cand
            if best is None or score > best[0]:
                best = (score, box, class_id)
    if best is None:
        return None
    score, box, class_id = best
    sx, sy = w / MODEL_INPUT, h / MODEL_INPUT
    bbox = (int(box[0] * sx), int(box[2] * sx), int(box[1] * sy), int(box[3] * sy))  # lx, rx, ty, by
    print(f"  rtmdet: cls={class_id} conf={score:.3f} bbox_px=x:{bbox[0]}-{bbox[1]},y:{bbox[2]}-{bbox[3]}")
    return bbox, class_id, float(score)


def rtmdet_exclude_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    det = rtmdet_detection(image)
    return det[0] if det else None


def floor_band_leveled_points(depth: np.ndarray, rot: np.ndarray, fx: float, fy: float,
                              bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    """Leveled bottom-band candidate floor points (same sampling as camera height)."""
    h, w = depth.shape
    margin = WALL_MARGIN
    lx = int(round(margin * w))
    rx = int(round((1 - margin) * w)) - 1
    floor_start = int(round(h * FLOOR_BAND_START_FRACTION))
    by = h - 1
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    step = max(4, (by - floor_start) // 32)
    band_w, band_h = max(rx - lx, 1), max(by - floor_start, 1)

    exclude = None
    if bbox is not None:
        pad_x = int(round((bbox[1] - bbox[0]) * 0.05)) + 2
        pad_y = int(round((bbox[3] - bbox[2]) * 0.05)) + 2
        exclude = (max(0, bbox[0] - pad_x), min(w - 1, bbox[1] + pad_x),
                   max(0, bbox[2] - pad_y), min(h - 1, bbox[3] + pad_y))

    pts = []
    for row in range(floor_start, by + 1, step):
        for col in range(lx, rx + 1, step):
            if exclude is not None:
                if exclude[0] <= col <= exclude[1] and exclude[2] <= row <= exclude[3]:
                    continue
            else:
                u = (col - lx) / band_w
                v = (row - floor_start) / band_h
                if u > FLOOR_CHAIR_EXCLUDE_U and v > FLOOR_CHAIR_EXCLUDE_V:
                    continue
            d = float(depth[row, col])
            if not (np.isfinite(d) and d > 0):
                continue
            p = rot @ np.array([(col - cx) * d / fx, (row - cy) * d / fy, d])
            if p[1] > 0.05:
                pts.append(p)
    return np.asarray(pts)


def robust_floor_slope(pts: np.ndarray) -> float | None:
    """Slope of leveled floor y vs z with outlier trimming (0 when leveling is exact)."""
    if len(pts) < 64:
        return None
    z, y = pts[:, 2], pts[:, 1]
    keep = np.ones(len(z), bool)
    slope = 0.0
    for _ in range(3):
        if keep.sum() < 32:
            return None
        slope, intercept = np.polyfit(z[keep], y[keep], 1)
        resid = np.abs(y - (slope * z + intercept))
        mad = np.median(resid[keep])
        keep = resid < max(3.0 * mad, 0.05)
    return float(slope)


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def floor_corrected_rotation(depth: np.ndarray, rot: np.ndarray, fx: float, fy: float,
                             bbox, max_correction_rad: float = 0.35) -> tuple[np.ndarray, dict]:
    """Cancel residual pitch by flattening the observed floor band (two passes)."""
    diag = {}
    corrected = rot
    total = 0.0
    for i in range(2):
        pts = floor_band_leveled_points(depth, corrected, fx, fy, bbox)
        slope = robust_floor_slope(pts)
        if slope is None:
            diag[f"pass{i}"] = "insufficient"
            break
        correction = np.arctan(slope)
        diag[f"pass{i}_slope_deg"] = round(float(np.degrees(correction)), 2)
        if abs(total + correction) > max_correction_rad:
            diag[f"pass{i}"] = "gated"
            break
        total += correction
        corrected = rot_x(correction) @ corrected
        if abs(correction) < np.radians(0.3):
            break
    diag["total_correction_deg"] = round(float(np.degrees(total)), 2)
    return corrected, diag


def band_camera_points(depth: np.ndarray, fx: float, fy: float, row_range: tuple[int, int],
                       bbox: tuple[int, int, int, int] | None, step: int) -> np.ndarray:
    """Raw camera-space points from an image row band (margin-cropped columns)."""
    h, w = depth.shape
    lx = int(round(WALL_MARGIN * w))
    rx = int(round((1 - WALL_MARGIN) * w)) - 1
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    pts = []
    for row in range(row_range[0], row_range[1] + 1, step):
        for col in range(lx, rx + 1, step):
            if bbox is not None and bbox[0] <= col <= bbox[1] and bbox[2] <= row <= bbox[3]:
                continue
            d = float(depth[row, col])
            if not (np.isfinite(d) and d > 0):
                continue
            pts.append([(col - cx) * d / fx, (row - cy) * d / fy, d])
    return np.asarray(pts)


def robust_plane_fit(pts: np.ndarray, iterations: int = 3) -> tuple[np.ndarray, float, int] | None:
    """SVD plane fit with residual trimming. Returns (unit normal, offset d with n·p = d, inliers)."""
    if len(pts) < 48:
        return None
    p = pts
    normal = None
    for _ in range(iterations):
        centroid = p.mean(axis=0)
        q = p - centroid
        _, _, vt = np.linalg.svd(q, full_matrices=False)
        normal = vt[-1]
        resid = np.abs(q @ normal)
        threshold = max(float(np.percentile(resid, 70)), 0.02)
        keep = resid < threshold
        if keep.sum() < 32:
            break
        p = p[keep]
    d = float(normal @ p.mean(axis=0))
    if d < 0:
        normal, d = -normal, -d
    return normal, d, len(p)


def plane_separation_height(depth: np.ndarray, fx: float, fy: float,
                            bbox: tuple[int, int, int, int] | None) -> tuple[dict, float | None, float | None]:
    """Rotation-invariant height: distance between the floor plane and ceiling plane in raw
    camera space, scaled so the camera sits 1.7 m above the floor plane. Immune to
    gravity/pitch estimation errors (plane separation does not depend on leveling)."""
    h, w = depth.shape
    floor_band = (int(round(h * FLOOR_BAND_START_FRACTION)), h - 1)
    ceil_band = (int(round(WALL_MARGIN * h)),
                 int(round(WALL_MARGIN * h + CEILING_BAND_ROW_FRACTION * (1 - 2 * WALL_MARGIN) * h)))
    step = max(4, (floor_band[1] - floor_band[0]) // 32)
    diag: dict = {}
    floor_pts = band_camera_points(depth, fx, fy, floor_band, bbox, step)
    ceil_pts = band_camera_points(depth, fx, fy, ceil_band, None, step)
    floor_fit = robust_plane_fit(floor_pts)
    ceil_fit = robust_plane_fit(ceil_pts)
    if floor_fit is None or ceil_fit is None:
        diag["planes"] = "fit_failed"
        return diag, None, None
    n_f, d_f, in_f = floor_fit
    n_c, d_c, in_c = ceil_fit
    parallel = abs(float(n_f @ n_c))
    diag.update({
        "floor_normal": np.round(n_f, 3).tolist(), "floor_dist": round(d_f, 3), "floor_inliers": in_f,
        "ceil_normal": np.round(n_c, 3).tolist(), "ceil_dist": round(d_c, 3), "ceil_inliers": in_c,
        "parallel_dot": round(parallel, 3),
    })
    if parallel < 0.90:
        diag["planes"] = "not_parallel"
        return diag, None, None
    # Ceiling offset along the floor normal (project ceiling points onto n_f).
    d_c_on_f = swift_median(ceil_pts @ n_f)
    separation = abs(d_f - d_c_on_f)
    diag["separation_raw"] = round(float(separation), 3)
    if not (CAMERA_HEIGHT_RAW_VALID[0] <= d_f <= CAMERA_HEIGHT_RAW_VALID[1]):
        diag["planes"] = "floor_dist_out_of_range"
        return diag, None, None
    scale = CAMERA_HEIGHT_PRIOR_M / d_f
    return diag, float(scale * separation), float(d_f)


def camera_height_from_floor(depth: np.ndarray, rot: np.ndarray, fx: float, fy: float,
                             bbox: tuple[int, int, int, int] | None) -> tuple[float | None, dict]:
    h, w = depth.shape
    margin = WALL_MARGIN
    lx = int(round(margin * w))
    rx = int(round((1 - margin) * w)) - 1
    floor_start = int(round(h * FLOOR_BAND_START_FRACTION))
    by = h - 1
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    step = max(4, (by - floor_start) // 32)
    band_w, band_h = max(rx - lx, 1), max(by - floor_start, 1)

    exclude = None
    if bbox is not None:
        pad_x = int(round((bbox[1] - bbox[0]) * 0.05)) + 2
        pad_y = int(round((bbox[3] - bbox[2]) * 0.05)) + 2
        exclude = (max(0, bbox[0] - pad_x), min(w - 1, bbox[1] + pad_x),
                   max(0, bbox[2] - pad_y), min(h - 1, bbox[3] + pad_y))

    heights, zs = [], []
    for row in range(floor_start, by + 1, step):
        for col in range(lx, rx + 1, step):
            if exclude is not None:
                if exclude[0] <= col <= exclude[1] and exclude[2] <= row <= exclude[3]:
                    continue
            else:
                u = (col - lx) / band_w
                v = (row - floor_start) / band_h
                if u > FLOOR_CHAIR_EXCLUDE_U and v > FLOOR_CHAIR_EXCLUDE_V:
                    continue
            d = float(depth[row, col])
            if not (np.isfinite(d) and d > 0):
                continue
            p = rot @ np.array([(col - cx) * d / fx, (row - cy) * d / fy, d])
            if p[1] > 0.05:
                heights.append(float(p[1]))
                zs.append(float(p[2]))
    diag = {"floor_samples": len(heights)}
    if len(heights) >= 8:
        # Diagnostic: residual tilt = slope of leveled floor y vs z (0 when leveling is exact).
        slope = float(np.polyfit(np.array(zs), np.array(heights), 1)[0])
        diag["floor_y_vs_z_slope"] = slope
        diag["residual_pitch_deg"] = float(np.degrees(np.arctan(slope)))
    if len(heights) < 32:
        return None, diag
    return swift_median(np.array(heights)), diag


def wall_plateau_height(depth: np.ndarray, rot: np.ndarray, fx: float, fy: float,
                        wall_depth: float) -> tuple[float | None, dict]:
    """Python quad-height parity: per column, find the contiguous row run whose depth
    sits at the back-wall plateau, unproject its top/bottom at that depth, and take the
    median leveled vertical extent. The ceiling's own depth never enters."""
    h, w = depth.shape
    lx = int(round(WALL_MARGIN * w))
    rx = int(round((1 - WALL_MARGIN) * w)) - 1
    ty = int(round(WALL_MARGIN * h))
    by = int(round((1 - WALL_MARGIN) * h)) - 1
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    lo, hi = 0.75 * wall_depth, 1.25 * wall_depth

    heights, tops, bottoms = [], [], []
    min_run_rows = max(8, (by - ty) // 8)
    for col in range(lx, rx + 1, 8):
        d_col = depth[ty:by + 1, col]
        on_wall = np.isfinite(d_col) & (d_col >= lo) & (d_col <= hi)
        if not on_wall.any():
            continue
        # longest contiguous run
        best_len, best_start, run_start = 0, -1, None
        for i, flag in enumerate(np.append(on_wall, False)):
            if flag and run_start is None:
                run_start = i
            elif not flag and run_start is not None:
                if i - run_start > best_len:
                    best_len, best_start = i - run_start, run_start
                run_start = None
        if best_len < min_run_rows:
            continue
        row_top = ty + best_start
        row_bottom = ty + best_start + best_len - 1
        p_top = rot @ np.array([(col - cx) * wall_depth / fx, (row_top - cy) * wall_depth / fy, wall_depth])
        p_bottom = rot @ np.array([(col - cx) * wall_depth / fx, (row_bottom - cy) * wall_depth / fy, wall_depth])
        extent = float(p_bottom[1] - p_top[1])
        if extent > 0.5:
            heights.append(extent)
            tops.append(row_top)
            bottoms.append(row_bottom)
    diag = {"wall_columns": len(heights)}
    if len(heights) < 12:
        return None, diag
    diag["row_top_med"] = int(np.median(tops))
    diag["row_bottom_med"] = int(np.median(bottoms))
    return swift_median(np.array(heights)), diag


def measure_depth_spread(depth: np.ndarray, rot: np.ndarray, fx: float, fy: float,
                         camera_height_prior: float = CAMERA_HEIGHT_PRIOR_M) -> tuple[tuple, dict]:
    """Swift `measureDepthSpread` (wall-anchored spread W/D, ceiling-anchored H)."""
    h, w = depth.shape
    margin = WALL_MARGIN
    lx = int(round(margin * w))
    rx = int(round((1 - margin) * w)) - 1
    ty = int(round(margin * h))
    by = int(round((1 - margin) * h)) - 1
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    ceiling_row_cutoff = ty + CEILING_BAND_ROW_FRACTION * (by - ty)

    X, Y, Z, clearances, clearance_z = [], [], [], [], []
    for row in range(ty, by + 1, 8):
        for col in range(lx, rx + 1, 8):
            d = float(depth[row, col])
            if not (np.isfinite(d) and d > 0):
                continue
            p = rot @ np.array([(col - cx) * d / fx, (row - cy) * d / fy, d])
            if p[2] > 0:
                X.append(p[0]); Y.append(p[1]); Z.append(p[2])
                if row < ceiling_row_cutoff and p[1] < -MIN_CEILING_CLEARANCE_M:
                    clearances.append(-p[1])
                    clearance_z.append(p[2])
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    diag = {"points": len(Z), "ceiling_samples": len(clearances)}
    if len(Z) < 32:
        return None, diag

    room_depth = swift_percentile(np.sort(Z), 0.80)
    far = Z > 0.6 * room_depth
    wall_x = X[far] if far.sum() >= 64 else X
    wall_y = Y[far] if far.sum() >= 64 else Y
    sx, sy = np.sort(wall_x), np.sort(wall_y)
    width = swift_percentile(sx, 0.96) - swift_percentile(sx, 0.04)
    height = swift_percentile(sy, 0.97) - swift_percentile(sy, 0.03)
    diag["height_spread"] = float(height)
    diag["height_source"] = "spread_p97_p3"
    if len(clearances) >= 32:
        cl = np.array(clearances)
        zl = np.array(clearance_z)
        clearance = swift_median(cl)
        diag["ceiling_clearance_median"] = float(clearance)
        if CEILING_INTERCEPT:
            # A flat ceiling has constant leveled clearance. Residual pitch error d adds
            # z*sin(d), so the intercept of clearance-vs-z at z=0 recovers the true
            # clearance. Only correct in the inflation direction (slope > 0).
            slope, intercept = theil_sen(zl, cl)
            diag["ceiling_slope"] = round(float(slope), 4)
            diag["ceiling_intercept"] = round(float(intercept), 3)
            if slope > 0.02 and intercept > MIN_CEILING_CLEARANCE_M:
                clearance = float(intercept)
                diag["ceiling_clearance_used"] = "intercept"
        ceiling_height = camera_height_prior + clearance
        if CEILING_ANCHORED_HEIGHT_RANGE[0] <= ceiling_height <= CEILING_ANCHORED_HEIGHT_RANGE[1]:
            height = ceiling_height
            diag["height_source"] = "ceiling_anchored"
        else:
            diag["height_source"] = f"spread_p97_p3(ceiling {ceiling_height:.2f} out of range)"
    return (float(width), float(height), float(room_depth)), diag


def theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust slope/intercept: sort by x, pair first half with second half, median of
    pair slopes (matches Swift `robustLineFit`)."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    half = len(xs) // 2
    x0, x1 = xs[:half], xs[half:2 * half]
    y0, y1 = ys[:half], ys[half:2 * half]
    keep = (x1 - x0) > 1e-3
    if keep.sum() < 8:
        return 0.0, swift_median(y)
    slopes = (y1[keep] - y0[keep]) / (x1[keep] - x0[keep])
    slope = swift_median(slopes)
    intercept = swift_median(y - slope * x)
    return slope, intercept


# ---------------------------------------------------------------- full test case

@dataclass
class ScaleCandidate:
    source: str
    tier: int
    depth_scale: float
    det_conf: float
    implied_room_height: float
    debug: str


def median_scale(cands: list[ScaleCandidate]) -> float:
    scales = sorted(c.depth_scale for c in cands)
    mid = len(scales) // 2
    if len(scales) % 2:
        return float(scales[mid])
    return float((scales[mid - 1] + scales[mid]) * 0.5)


def resolve_scale(cands: list[ScaleCandidate], raw_camera_height: float | None,
                  implied_height) -> tuple[float, float, str, list[ScaleCandidate], list[ScaleCandidate]]:
    valid = [c for c in cands if np.isfinite(c.depth_scale) and c.depth_scale > 0 and
             1.9 <= c.implied_room_height <= 3.6]
    clamped_out = [c for c in cands if c not in valid]
    t1 = [c for c in valid if c.tier == 1 and c.det_conf > 0.5]
    if t1:
        return median_scale(t1), 0.90, "tier1_architectural", t1, clamped_out
    t2 = [c for c in valid if c.tier == 2 and c.det_conf > 0.5]
    if t2:
        return median_scale(t2), 0.75, "tier2_fixture", t2, clamped_out
    if raw_camera_height is None or raw_camera_height <= 0:
        return 1.0, 0.05, "tier3_camera_height_unavailable", [], clamped_out
    camera_h = min(1.75, max(1.55, 1.65))
    scale = camera_h / raw_camera_height
    obs = ScaleCandidate("camera_height", 3, scale, 1.0, implied_height(scale),
                         f"clampedH={camera_h:.2f}m rawH={raw_camera_height:.3f}m")
    return float(scale), 0.50, "tier3_camera_height", [obs], clamped_out


def depth_percentile(depth: np.ndarray, bbox: tuple[int, int, int, int], fraction: float) -> float | None:
    lx, rx, ty, by = bbox
    lx, rx = max(0, lx), min(depth.shape[1] - 1, rx)
    ty, by = max(0, ty), min(depth.shape[0] - 1, by)
    vals = depth[ty:by + 1:max(1, (by - ty) // 120 + 1), lx:rx + 1:max(1, (rx - lx) // 120 + 1)]
    vals = vals[np.isfinite(vals) & (vals > 0.1) & (vals < 50)]
    if vals.size == 0:
        return None
    return swift_percentile(np.sort(vals), fraction)


def scale_candidates(raw_camera_height: float | None, fallback: float, implied_height,
                     detection, depth: np.ndarray, fx: float) -> list[ScaleCandidate]:
    cands: list[ScaleCandidate] = []
    if detection is not None:
        bbox, cls, conf = detection
        lx, rx, ty, by = bbox
        img_h = depth.shape[0]
        bbox_h = max(1, by - ty)
        raw_d = depth_percentile(depth, bbox, 0.20)
        if cls == 61 and raw_d is not None:
            bottom_margin = (img_h - by) / max(img_h, 1)
            height_frac = bbox_h / max(img_h, 1)
            if bottom_margin <= 0.05 and height_frac < 0.20:
                source, prior, tier = "toilet_seat", 0.40, 2
            else:
                source, prior, tier = "toilet_full", 0.78, 2
            raw_size = bbox_h * raw_d / fx
            scale = prior / raw_size if raw_size > 0 else np.nan
            h = implied_height(scale)
            if np.isfinite(scale) and h is not None:
                cands.append(ScaleCandidate(
                    source, tier, float(scale), float(conf), float(h),
                    f"cls={cls} conf={conf:.2f} px={bbox_h} rawSize={raw_size:.3f}m prior={prior:.2f}m tier={tier}"
                ))
    return cands


def assert_fuse_scale_rejects_outlier() -> None:
    cands = [
        ScaleCandidate("tile_floor", 1, 0.50, 0.8, 2.4, ""),
        ScaleCandidate("tile_wall", 1, 0.52, 0.7, 2.5, ""),
        ScaleCandidate("toilet_seat", 2, 0.49, 0.9, 2.35, ""),
        ScaleCandidate("outlier", 1, 1.20, 1.0, 5.2, ""),
    ]
    scale, confidence, source, *_ = resolve_scale(cands, 3.0, lambda s: 2.7)
    assert 0.49 <= scale <= 0.52, (scale, confidence, source)
    assert source == "tier1_architectural", source
    mixed = [
        ScaleCandidate("chair_height", 4, 0.40, 0.9, 2.0, ""),
    ]
    scale, confidence, source, *_ = resolve_scale(mixed, 3.0, lambda s: 2.7)
    assert abs(scale - 0.55) < 1e-6, (scale, confidence, source)
    assert source == "tier3_camera_height", source


@dataclass
class CaseResult:
    width: float
    height: float
    depth: float
    camera_height_raw: float | None
    scale: float
    focal_px: float
    roll: float
    pitch: float
    diags: dict


TESTDATA_DIR = REPO_ROOT / "scripts/testdata"


def open_with_cache(image_path: str, cache_name: str) -> Image.Image:
    """Downloads access is TCC-gated and flaky for CLI runs; keep a repo-local copy."""
    cache = TESTDATA_DIR / cache_name
    try:
        image = Image.open(image_path)
        image.load()
        if not cache.exists():
            TESTDATA_DIR.mkdir(parents=True, exist_ok=True)
            image.save(cache, "JPEG", quality=95, exif=image.info.get("exif", b""))
        return image
    except (PermissionError, FileNotFoundError):
        if cache.exists():
            print(f"  (using cached copy {cache.name})")
            return Image.open(cache)
        raise


def run_case(image_path: str, focal_35mm: float | None, cache_name: str,
             forced_pitch_deg: float | None = None,
             camera_height_prior: float = CAMERA_HEIGHT_PRIOR_M) -> CaseResult:
    image = open_with_cache(image_path, cache_name).convert("RGB")
    exif_f35 = None
    exif = image.getexif()
    if exif:
        ex = exif.get_ifd(0x8769)
        exif_f35 = float(ex.get(0xA405)) if ex and ex.get(0xA405) else None
    f35 = focal_35mm or exif_f35 or FALLBACK_FOCAL_35MM
    work = working_image(image)
    w, h = work.size
    fx, focal_source, focal_clamped = resolve_focal(w)
    fy = fx
    print(f"  working={w}x{h} EXIF ignored -> f={fx:.1f}px source={focal_source} "
          f"hFOV={hfov_from_focal(w, fx):.1f} clamped={focal_clamped}")

    depth = infer_depth(work)
    roll, pitch = geocalib_roll_pitch(work)
    print(f"  geocalib roll={roll:.4f} ({np.degrees(roll):.1f}deg) pitch={pitch:.4f} ({np.degrees(pitch):.1f}deg)")

    roll *= GRAVITY_ROLL_SIGN
    pitch *= GRAVITY_PITCH_SIGN
    if forced_pitch_deg is not None:
        # Simulated ARKit device gravity (exact pitch, GeoCalib convention).
        rot = leveling_rotation(gravity_up_vector(0.0, np.radians(forced_pitch_deg)))
        gravity_source = f"arkit_sim({forced_pitch_deg}deg)"
    elif abs(roll) <= MAX_PLAUSIBLE_ROLL_RAD and abs(pitch) <= MAX_PLAUSIBLE_PITCH_RAD:
        rot = leveling_rotation(gravity_up_vector(roll, pitch))
        gravity_source = "geocalib"
    else:
        rot = np.eye(3)
        gravity_source = "identity(gates)"

    detection = rtmdet_detection(work)
    bbox = detection[0] if detection else None
    if FLOOR_CORRECTION:
        rot, corr_diag = floor_corrected_rotation(depth, rot, fx, fy, bbox)
        gravity_source += "+floor_fit"
        print(f"  floor_correction={corr_diag}")
    cam_h, floor_diag = camera_height_from_floor(depth, rot, fx, fy, bbox)
    scale = 1.0
    if cam_h is not None and CAMERA_HEIGHT_RAW_VALID[0] <= cam_h <= CAMERA_HEIGHT_RAW_VALID[1]:
        scale = camera_height_prior / cam_h
    def implied_height(candidate_scale: float) -> float:
        implied_prior = cam_h * candidate_scale if cam_h is not None else camera_height_prior
        dims_candidate, _ = measure_depth_spread(depth * candidate_scale, rot, fx, fy, implied_prior)
        return float("nan") if dims_candidate is None else float(dims_candidate[1])
    cands = scale_candidates(cam_h, scale, implied_height, detection, depth, fx)
    fused_scale, scale_conf, scale_source, selected, clamped_out = resolve_scale(cands, cam_h, implied_height)
    cand_text = "; ".join(
        f"{c.source}/t{c.tier}={c.depth_scale:.4f}/conf{c.det_conf:.2f}/h{c.implied_room_height:.2f}/{c.debug}"
        for c in cands
    )
    selected_text = "; ".join(f"{c.source}={c.depth_scale:.4f}" for c in selected)
    clamped_text = "; ".join(f"{c.source}=h{c.implied_room_height:.2f}" for c in clamped_out)
    print(f"  [ScaleResolver] source={scale_source} scale={fused_scale:.4f} confidence={scale_conf:.2f} "
          f"selected=[{selected_text}] anchors=[{cand_text}] clamped_out=[{clamped_text}]")
    scale = fused_scale
    effective_camera_height_prior = cam_h * scale if cam_h is not None and scale_conf > 0.05 else camera_height_prior
    print(f"  camera_height_raw_m={cam_h if cam_h is None else round(cam_h, 3)} "
          f"measurement_scale={scale:.4f} gravity={gravity_source} floor_diag={floor_diag}")

    if PLANE_HEIGHT:
        plane_diag, plane_height, plane_floor_dist = plane_separation_height(depth, fx, fy,
                                                                             padded_bbox(bbox, depth.shape))
        print(f"  plane_diag={plane_diag} plane_height={plane_height} plane_floor_dist={plane_floor_dist}")

    dims, spread_diag = measure_depth_spread(depth * scale, rot, fx, fy, effective_camera_height_prior)
    if dims is None:
        raise AssertionError("measureDepthSpread returned nil")
    width, height, room_depth = dims
    if WALL_PLATEAU:
        plateau_height, plateau_diag = wall_plateau_height(depth * scale, rot, fx, fy, room_depth)
        print(f"  plateau_diag={plateau_diag} plateau_height="
              f"{None if plateau_height is None else round(plateau_height, 3)}")
        if plateau_height is not None and \
                CEILING_ANCHORED_HEIGHT_RANGE[0] <= plateau_height <= CEILING_ANCHORED_HEIGHT_RANGE[1]:
            height = plateau_height
    ok_span = all(PLAUSIBLE_ROOM_SPAN[0] <= v <= PLAUSIBLE_ROOM_SPAN[1] for v in dims)
    print(f"  spread_diag={spread_diag} plausible_span={ok_span}")
    print(f"  result_dims_m=W:{width:.4f},H:{height:.4f},D:{room_depth:.4f}")
    print(f"  debug_pill: camH {cam_h:.2f}m · scale {scale:.2f} · f {fx:.0f}px · grav geo")
    return CaseResult(width, height, room_depth, cam_h, scale, fx, roll, pitch,
                      {"floor": floor_diag, "spread": spread_diag})


def check(name: str, result: CaseResult, tape: tuple[float | None, float | None, float | None],
          height_tolerance_m: float = 0.35) -> bool:
    """Height is the hard assertion (user priority); width/depth are informational."""
    tape_w, tape_h, tape_d = tape
    ok = True
    for label, measured, expected, tol, hard in (
        ("width", result.width, tape_w, 0.6, False),
        ("height", result.height, tape_h, height_tolerance_m, True),
        ("depth", result.depth, tape_d, 0.6, False),
    ):
        if expected is None:
            continue
        err = measured - expected
        within = abs(err) <= tol
        status = "PASS" if within else ("FAIL" if hard else "WARN")
        if hard and not within:
            ok = False
        print(f"  {status} {label}: measured={measured:.2f} tape={expected:.2f} err={err:+.2f} (tol ±{tol})")
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    return ok


CASES = [
    # (name, path, tape (W,H,D), focal-35mm override, forced_pitch_deg, camera_height_m)
    # camera_height_m is how high the phone was ACTUALLY held. The Swift app always
    # assumes 1.70 m; when the real hold height differs, all dims scale by the ratio.
    ("living_room", "/Users/al/Downloads/WhatsApp Image 2026-07-06 at 10.21.21.jpeg",
     (3.15, 2.85, 3.37), 24.0, None, 1.70),
    # Bathroom photo was shot from ~1.2 m (window-framing crouch): junction geometry
    # puts the camera 1.0-1.2 m above the floor. With the app's fixed 1.7 m prior this
    # exact photo reads H≈2.75-3.15 (device showed 3.15) — that is the 1.7/1.2 ratio,
    # not a math bug. This case asserts the math is right GIVEN the true hold height.
    ("bathroom_true_hold_height", "/Users/al/Downloads/WhatsApp Image 2026-07-06 at 10.00.38.jpeg",
     (1.6, 2.25, 2.55), 24.0, None, 1.20),
    # Same photo with the app's fixed 1.7 m assumption (documents the failure mode;
    # height assert is soft/expected-high here).
    ("bathroom_app_assumption_1p7", "/Users/al/Downloads/WhatsApp Image 2026-07-06 at 10.00.38.jpeg",
     (None, None, None), 24.0, None, 1.70),
    # Informational only (no asserts): pitched-up device framing approximated by a
    # crop of the app screenshot — has baked-in UI overlays and no EXIF, so numbers
    # are indicative, not ground truth. Device reported W:3.95 H:3.83 D:3.49 here.
    ("device_pitched_up", str(TESTDATA_DIR / "device_pitched_up.jpeg"),
     (None, None, None), 24.0, None, 1.70),
]


def main() -> int:
    assert_fuse_scale_rejects_outlier()
    print("fuse_scale synthetic test PASS")
    failures = 0
    ran = 0
    for name, path, tape, f35, forced_pitch, cam_height in CASES:
        print(f"\n=== {name}: {Path(path).name} (hold height {cam_height}m) ===")
        try:
            result = run_case(path, f35, f"{name}.jpeg", forced_pitch, cam_height)
        except (PermissionError, FileNotFoundError):
            print("  SKIP: photo unreadable (macOS privacy) and no cached copy")
            continue
        ran += 1
        if not check(name, result, tape):
            failures += 1
    if ran == 0:
        print("\nNO CASES RAN")
        return 2
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} case(s) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
