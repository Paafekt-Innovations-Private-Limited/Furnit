#!/usr/bin/env python3
"""Structure-box lines + Depth Anything metric depth → room W×H×D.

Pipeline:
  1. M-LSD: green = wall verticals; red∩blue at top (blue y wins); green∩red at floor.
  2. Lines define the back-wall quadrilateral (pixel box) only — no depth from lines.
  3. Depth Anything (+ GeoCalib focal, camera-height scale) supplies all metric depth.
  4. RG quad: fit back-wall plane from interior samples (skip curtain/chair) → ray–plane corners.
  5. Width/height from leveled corner spans; depth = ray hit at wall center.

Example:
  python3 scripts/run_geocalib.py --image room.jpg --json-out /tmp/room_geocalib.json
  python3 scripts/structure_box_measure_room.py \\
    --image room.jpg \\
    --geocalib-json /tmp/room_geocalib.json \\
    --tape-height 2.85
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from depthanything_measure_room import (  # noqa: E402
    DEFAULT_ONNX,
    focal_from_exif,
    leveled_points,
    median_at,
    read_rgb,
    run_depth_onnx,
)
from mlsd_draw_room_lines import (  # noqa: E402
    ANGLE_TOLERANCE_DEG,
    classify_segment_color,
    dominant_families,
    line_angle,
    line_length as mlsd_line_length,
)

DEFAULT_OUT_DIR = Path("/tmp/structure_box_measure")
CAMERA_HEIGHT_MIN_M = 1.0
CAMERA_HEIGHT_MAX_M = 1.8
ROOM_HEIGHT_MIN_M = 1.8
ROOM_HEIGHT_MAX_M = 4.5
RG_HEIGHT_MIN_M = 2.0
RG_HEIGHT_MAX_M = 3.2


@dataclass
class StructureBox:
    left_x: float
    right_x: float
    ceiling_y: float
    floor_y: float
    source: str
    red_lines: int
    green_lines: int
    dark_blue_lines: int
    ceiling_y_left: float | None = None
    ceiling_y_right: float | None = None
    floor_y_left: float | None = None
    floor_y_right: float | None = None

    def corner_px(self) -> dict[str, tuple[float, float]]:
        cl = self.ceiling_y_left if self.ceiling_y_left is not None else self.ceiling_y
        cr = self.ceiling_y_right if self.ceiling_y_right is not None else self.ceiling_y
        fl = self.floor_y_left if self.floor_y_left is not None else self.floor_y
        fr = self.floor_y_right if self.floor_y_right is not None else self.floor_y
        return {
            "top_left": (self.left_x, cl),
            "top_right": (self.right_x, cr),
            "bottom_left": (self.left_x, fl),
            "bottom_right": (self.right_x, fr),
        }


@dataclass
class Plane:
    normal: np.ndarray  # unit
    offset: float  # n·p + offset = 0

    def signed_distance(self, point: np.ndarray) -> float:
        return float(np.dot(self.normal, point) + self.offset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D structure box + guided plane intersections for room W×H×D."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--geocalib-json", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--camera-height-prior", type=float, default=1.40)
    parser.add_argument("--region-inset-px", type=int, default=12)
    parser.add_argument("--max-samples-per-region", type=int, default=2500)
    parser.add_argument("--score-thr", type=float, default=0.10, help="M-LSD score threshold.")
    parser.add_argument("--dist-thr", type=float, default=20.0, help="M-LSD distance threshold.")
    parser.add_argument("--tape-width", type=float, default=None)
    parser.add_argument("--tape-height", type=float, default=None)
    parser.add_argument("--tape-depth", type=float, default=None)
    return parser.parse_args()


def detect_mlsd_lines(rgb: np.ndarray, score_thr: float, dist_thr: float) -> tuple[np.ndarray, str]:
    try:
        import mlsd as mlsd_pkg

        interpreter = mlsd_pkg.load_interpreter("M-LSD_512_large_fp32.tflite")
        lines = mlsd_pkg.get_lines(rgb, interpreter, score_thr=score_thr, dist_thr=dist_thr)
        return np.asarray(lines, dtype=np.float32).reshape(-1, 4), "mlsd_512_large"
    except Exception as exc:
        print(f"[WARN] M-LSD failed ({exc}); using OpenCV LSD")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines_raw, _, _, _ = lsd.detect(gray)
        if lines_raw is None:
            return np.zeros((0, 4), dtype=np.float32), "fallback_opencv_lsd"
        return lines_raw.reshape(-1, 4).astype(np.float32), "fallback_opencv_lsd"


def line_length(x1: float, y1: float, x2: float, y2: float) -> float:
    return mlsd_line_length(x1, y1, x2, y2)


def spans_between_verticals(
    x1: float,
    x2: float,
    left_x: float,
    right_x: float,
    min_span_frac: float = 0.50,
) -> bool:
    seg_left = min(x1, x2)
    seg_right = max(x1, x2)
    wall_span = max(right_x - left_x, 1.0)
    overlap = min(seg_right, right_x) - max(seg_left, left_x)
    return overlap >= wall_span * min_span_frac


def classify_line_segments(
    lines: np.ndarray,
    families: list[float],
    min_length: float,
) -> list[dict]:
    tolerance = math.radians(ANGLE_TOLERANCE_DEG)
    segments: list[dict] = []
    for index, (x1, y1, x2, y2) in enumerate(lines):
        length = line_length(x1, y1, x2, y2)
        if length < min_length:
            continue
        angle = line_angle(x1, y1, x2, y2)
        _, label, family_index = classify_segment_color(angle, families, tolerance)
        segments.append({
            "id": int(index),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "length_px": float(length),
            "angle_deg": round(math.degrees(angle), 2),
            "color_label": label,
            "family_index": family_index,
            "x_min": float(min(x1, x2)),
            "x_max": float(max(x1, x2)),
            "y_min": float(min(y1, y2)),
            "y_max": float(max(y1, y2)),
            "mid_x": float((x1 + x2) * 0.5),
            "mid_y": float((y1 + y2) * 0.5),
        })
    return segments


def segment_endpoints(seg: dict) -> list[tuple[float, float]]:
    return [(seg["x1"], seg["y1"]), (seg["x2"], seg["y2"])]


def point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx = bx - ax
    aby = by - ay
    length_sq = abx * abx + aby * aby
    if length_sq <= 1e-6:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length_sq))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(px - closest_x, py - closest_y)


def segments_touch(seg_a: dict, seg_b: dict, tolerance_px: float) -> tuple[bool, tuple[float, float] | None, tuple[float, float] | None]:
    for point_a in segment_endpoints(seg_a):
        for point_b in segment_endpoints(seg_b):
            if math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]) <= tolerance_px:
                return True, point_a, point_b
    return False, None, None


def segment_endpoint_near_segment(seg_a: dict, seg_b: dict, tolerance_px: float) -> bool:
    for px, py in segment_endpoints(seg_a):
        if point_to_segment_distance(px, py, seg_b["x1"], seg_b["y1"], seg_b["x2"], seg_b["y2"]) <= tolerance_px:
            return True
    for px, py in segment_endpoints(seg_b):
        if point_to_segment_distance(px, py, seg_a["x1"], seg_a["y1"], seg_a["x2"], seg_a["y2"]) <= tolerance_px:
            return True
    return False


def chromatic_blue_connections(
    chromatic: dict,
    blues_on_box: list[dict],
    endpoint_tol: float,
    near_tol: float,
) -> list[int]:
    connected: list[int] = []
    for blue in blues_on_box:
        if segments_touch(chromatic, blue, endpoint_tol)[0]:
            connected.append(int(blue["id"]))
        elif segment_endpoint_near_segment(chromatic, blue, near_tol):
            connected.append(int(blue["id"]))
    return connected


def keep_chromatic_segment(
    chromatic: dict,
    blues_on_box: list[dict],
    width: int,
    endpoint_tol: float,
    near_tol: float,
) -> bool:
    connected = chromatic_blue_connections(chromatic, blues_on_box, endpoint_tol, near_tol)
    if not connected:
        return False
    mid_x = chromatic["mid_x"]
    curtain_zone = width * 0.28 < mid_x < width * 0.76
    floor_diagonal_only = set(connected) == {10}
    if curtain_zone and floor_diagonal_only:
        return False
    if chromatic["color_label"] == "green" and curtain_zone and 10 in connected and len(connected) == 1:
        return False
    return True


def junctions_near_x(junctions: list[dict], anchor_x: float, band_px: float) -> list[dict]:
    return [j for j in junctions if abs(j["x"] - anchor_x) <= band_px]


def floor_y_from_wall_green(
    green: dict,
    reds: list[dict],
    touch_tol: float,
    height: int,
    ceiling_y: float,
) -> float | None:
    junctions: list[float] = []
    for red in reds:
        if red["mid_y"] < height * 0.38:
            continue
        touches, point_a, point_b = segments_touch(green, red, touch_tol)
        if not touches or point_a is None or point_b is None:
            continue
        junctions.append(max(point_a[1], point_b[1], green["y_max"], red["y_max"]))
    if not junctions:
        return None
    floor_cap = height * 0.80
    floor_min = ceiling_y + height * 0.28
    plausible = [y for y in junctions if floor_min <= y <= floor_cap]
    if not plausible:
        return None
    return float(np.percentile(plausible, 78))


def is_spanning_wall_green(green: dict, ceiling_y: float, height: int) -> bool:
    span = green["y_max"] - green["y_min"]
    return span >= height * 0.30 and green["y_min"] <= ceiling_y + height * 0.24


def extract_wall_quad_line_junctions(
    segments: list[dict],
    width: int,
    height: int,
) -> tuple[StructureBox, dict]:
    """Build wall quad: green verticals, red∩blue ceiling (blue wins), green∩red floor."""
    greens = [s for s in segments if s["color_label"] == "green"]
    reds = [s for s in segments if s["color_label"] == "red"]
    blues = [s for s in segments if s["color_label"] == "dark_blue"]
    touch_tol = max(35.0, width * 0.035)

    meta: dict = {
        "ceiling_junctions": [],
        "floor_junctions": [],
        "blue_only_ceiling": [],
    }

    if len(greens) >= 2:
        left_x = float(np.percentile([s["x_min"] for s in greens], 8))
        right_x = float(np.percentile([s["x_max"] for s in greens], 92))
        source = "line_junction_quad"
    else:
        source = "image_margin_fallback"
        left_x = width * 0.08
        right_x = width * 0.92

    left_x = float(np.clip(left_x, 0, width - 2))
    right_x = float(np.clip(right_x, left_x + 8, width - 1))

    # --- CEILING: red touching blue; blue y takes precedence ---
    ceiling_candidates: list[float] = []
    for red in reds:
        for blue in blues:
            touches, point_a, point_b = segments_touch(red, blue, touch_tol)
            if not touches or point_a is None or point_b is None:
                continue
            junction_x = (point_a[0] + point_b[0]) * 0.5
            junction_y = min(blue["y_min"], red["y_min"], point_a[1], point_b[1])
            if junction_y > height * 0.42:
                continue
            if junction_x < left_x - width * 0.12 or junction_x > right_x + width * 0.08:
                continue
            ceiling_candidates.append(junction_y)
            meta["ceiling_junctions"].append({
                "y": round(junction_y, 1),
                "x": round(junction_x, 1),
                "red_id": red["id"],
                "blue_id": blue["id"],
                "rule": "red_touches_blue_blue_y_wins",
            })

    for blue in blues:
        if blue["y_min"] > height * 0.38:
            continue
        if blue["mid_x"] < left_x - width * 0.15 or blue["mid_x"] > right_x + width * 0.1:
            continue
        ceiling_candidates.append(blue["y_min"])
        meta["blue_only_ceiling"].append({
            "y": round(blue["y_min"], 1),
            "blue_id": blue["id"],
            "rule": "blue_precedence_topmost_in_wall_band",
        })

    if ceiling_candidates:
        side_band = max(55.0, width * 0.16)
        left_ceiling_junctions = junctions_near_x(meta["ceiling_junctions"], left_x, side_band)
        right_ceiling_junctions = junctions_near_x(meta["ceiling_junctions"], right_x, side_band)
        ceiling_y_left = min(j["y"] for j in left_ceiling_junctions) if left_ceiling_junctions else None
        ceiling_y_right = min(j["y"] for j in right_ceiling_junctions) if right_ceiling_junctions else None
        fallback_ceiling = min(ceiling_candidates)
        ceiling_y_left = ceiling_y_left if ceiling_y_left is not None else fallback_ceiling
        ceiling_y_right = ceiling_y_right if ceiling_y_right is not None else fallback_ceiling
        ceiling_y = min(ceiling_y_left, ceiling_y_right)
    else:
        spanning_reds = [
            s for s in reds
            if spans_between_verticals(s["x1"], s["x2"], left_x, right_x)
        ]
        upper_reds = [s for s in spanning_reds if s["mid_y"] <= height * 0.55]
        ceiling_y = min(s["y_min"] for s in upper_reds) if upper_reds else height * 0.08
        ceiling_y_left = ceiling_y_right = ceiling_y
        source = f"{source}_no_red_blue_fallback"

    # --- FLOOR: each wall green rolls down to meet red floor (chair ignored via band cap) ---
    wall_green_band = max(40.0, width * 0.09)
    left_wall_greens = sorted(
        [
            g for g in greens
            if abs(g["x_min"] - left_x) <= wall_green_band and is_spanning_wall_green(g, ceiling_y, height)
        ],
        key=lambda g: (abs(g["x_min"] - left_x), g["y_min"]),
    )
    right_wall_greens = sorted(
        [
            g for g in greens
            if abs(g["x_max"] - right_x) <= wall_green_band and is_spanning_wall_green(g, ceiling_y, height)
        ],
        key=lambda g: (abs(g["x_max"] - right_x), g["y_min"]),
    )
    # Partial greens on the right wall (chair may break the long vertical).
    if not right_wall_greens:
        right_wall_greens = sorted(
            [
                g for g in greens
                if abs(g["x_max"] - right_x) <= wall_green_band and g["y_max"] - g["y_min"] >= height * 0.14
            ],
            key=lambda g: (abs(g["x_max"] - right_x), -g["y_min"]),
        )
    floor_touch_tol = touch_tol + 12.0
    floor_y_left = None
    floor_y_right = None
    for green in left_wall_greens[:3]:
        floor_y_left = floor_y_from_wall_green(green, reds, floor_touch_tol, height, ceiling_y)
        if floor_y_left is not None:
            meta["floor_junctions"].append({
                "y": round(floor_y_left, 1),
                "green_id": green["id"],
                "side": "left",
                "rule": "wall_green_meets_red_floor",
            })
            break
    for green in right_wall_greens[:3]:
        floor_y_right = floor_y_from_wall_green(green, reds, floor_touch_tol, height, ceiling_y)
        if floor_y_right is not None:
            meta["floor_junctions"].append({
                "y": round(floor_y_right, 1),
                "green_id": green["id"],
                "side": "right",
                "rule": "wall_green_meets_red_floor",
            })
            break

    spanning_reds = [
        s for s in reds
        if spans_between_verticals(s["x1"], s["x2"], left_x, right_x)
    ]
    lower_reds = [s for s in spanning_reds if s["mid_y"] >= height * 0.42]
    floor_fallback = None
    if lower_reds:
        capped = [s["y_max"] for s in lower_reds if s["y_max"] <= height * 0.80]
        floor_fallback = float(np.percentile(capped or [s["y_max"] for s in lower_reds], 75))
    elif left_wall_greens or right_wall_greens:
        pool = [g["y_max"] for g in (left_wall_greens + right_wall_greens)]
        floor_fallback = float(np.percentile(pool, 85))

    floor_y_left = floor_y_left if floor_y_left is not None else floor_fallback
    floor_y_right = floor_y_right if floor_y_right is not None else floor_fallback
    if floor_y_left is None and floor_y_right is None:
        floor_y = height - 1 - height * 0.08
        floor_y_left = floor_y_right = floor_y
        source = f"{source}_floor_fallback"
    else:
        floor_y_left = floor_y_left if floor_y_left is not None else floor_y_right
        floor_y_right = floor_y_right if floor_y_right is not None else floor_y_left
        floor_y = max(floor_y_left, floor_y_right)

    # --- Re-lock verticals: greens that span ceiling→floor define L/R ---
    wall_greens = [
        g for g in greens
        if g["y_min"] <= ceiling_y + height * 0.1 and g["y_max"] >= floor_y - height * 0.08
    ]
    if wall_greens:
        left_x = float(np.clip(min(g["x_min"] for g in wall_greens), 0, width - 2))
        right_x = float(np.clip(max(g["x_max"] for g in wall_greens), left_x + 8, width - 1))

    ceiling_y = float(np.clip(ceiling_y, 0, height - 2))
    floor_y = float(np.clip(floor_y, ceiling_y + 8, height - 1))

    box = StructureBox(
        left_x=left_x,
        right_x=right_x,
        ceiling_y=ceiling_y,
        floor_y=floor_y,
        source=source,
        red_lines=len(reds),
        green_lines=len(greens),
        dark_blue_lines=len(blues),
        ceiling_y_left=ceiling_y_left,
        ceiling_y_right=ceiling_y_right,
        floor_y_left=floor_y_left,
        floor_y_right=floor_y_right,
    )
    corners = box.corner_px()
    meta["quad_px"] = {
        "top_left": [round(corners["top_left"][0], 1), round(corners["top_left"][1], 1)],
        "top_right": [round(corners["top_right"][0], 1), round(corners["top_right"][1], 1)],
        "bottom_left": [round(corners["bottom_left"][0], 1), round(corners["bottom_left"][1], 1)],
        "bottom_right": [round(corners["bottom_right"][0], 1), round(corners["bottom_right"][1], 1)],
    }
    return box, meta


def patch_median_leveled(
    leveled: np.ndarray,
    px: float,
    py: float,
    radius: int = 10,
) -> np.ndarray | None:
    h, w = leveled.shape[:2]
    ix = int(round(px))
    iy = int(round(py))
    x0 = max(0, ix - radius)
    x1 = min(w, ix + radius + 1)
    y0 = max(0, iy - radius)
    y1 = min(h, iy + radius + 1)
    patch = leveled[y0:y1, x0:x1].reshape(-1, 3)
    good = np.isfinite(patch).all(axis=1)
    patch = patch[good]
    if patch.shape[0] < 8:
        return None
    return np.median(patch, axis=0).astype(np.float32)


def measure_wall_quad_depthanything(
    depth: np.ndarray,
    leveled: np.ndarray,
    box: StructureBox,
    inset_px: int,
) -> dict:
    """Metric W×H×D from line quad corners + Depth Anything (no line-derived depth)."""
    inset = max(6, inset_px)
    corners_px = {
        name: (px + (inset if "left" in name else -inset), py + (inset if "top" in name else -inset))
        for name, (px, py) in box.corner_px().items()
    }
    corners_3d: dict[str, list[float] | None] = {}
    for name, (px, py) in corners_px.items():
        pt = patch_median_leveled(leveled, px, py)
        corners_3d[name] = None if pt is None else [round(float(v), 4) for v in pt]

    center_x = int(round((box.left_x + box.right_x) * 0.5))
    center_y = int(round((box.ceiling_y + box.floor_y) * 0.5))
    depth_m = median_at(depth, center_x, center_y, radius=12)

    bl = patch_median_leveled(leveled, *corners_px["bottom_left"])
    br = patch_median_leveled(leveled, *corners_px["bottom_right"])
    tl = patch_median_leveled(leveled, *corners_px["top_left"])
    tr = patch_median_leveled(leveled, *corners_px["top_right"])

    width_m = None
    height_m = None
    if bl is not None and br is not None:
        width_m = float(np.linalg.norm(br - bl))
    heights: list[float] = []
    if bl is not None and tl is not None:
        heights.append(float(bl[1] - tl[1]))
    if br is not None and tr is not None:
        heights.append(float(br[1] - tr[1]))
    if heights:
        height_m = float(np.mean(heights))

    return {
        "source": "depthanything_quad_corners",
        "width_m": None if width_m is None else round(width_m, 3),
        "height_m": None if height_m is None else round(height_m, 3),
        "depth_m": None if depth_m is None else round(depth_m, 3),
        "depth_source": "depthanything_median_at_wall_center",
        "corners_px": {k: [round(v[0], 1), round(v[1], 1)] for k, v in corners_px.items()},
        "corners_leveled": corners_3d,
        "wall_center_px": [center_x, center_y],
    }


def leveled_ray_through_pixel(
    px: float,
    py: float,
    fx: float,
    fy: float,
    width: int,
    height: int,
    level_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    cam_dir = np.array([(px - cx) / fx, (py - cy) / fy, 1.0], dtype=np.float32)
    cam_dir /= max(float(np.linalg.norm(cam_dir)), 1e-6)
    leveled_dir = cam_dir @ level_rot.T
    leveled_dir /= max(float(np.linalg.norm(leveled_dir)), 1e-6)
    return np.zeros(3, dtype=np.float32), leveled_dir


def horizontal_distance_3d(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(float(a[0] - b[0]), float(a[2] - b[2])))


def quad_interior_sample_mask(
    box: StructureBox,
    width: int,
    height: int,
) -> np.ndarray:
    """Pixels inside the wall quad, skipping curtain center and chair bottom-right."""
    corners = box.corner_px()
    polygon = np.array([
        corners["top_left"],
        corners["top_right"],
        corners["bottom_right"],
        corners["bottom_left"],
    ], dtype=np.float32)
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    x_min = float(np.min(x_coords))
    x_max = float(np.max(x_coords))
    y_min = float(np.min(y_coords))
    y_max = float(np.max(y_coords))
    quad_width = max(x_max - x_min, 1.0)
    quad_height = max(y_max - y_min, 1.0)

    poly_int = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
    inside = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(inside, [poly_int], 1)

    yy, xx = np.nonzero(inside)
    mask = np.zeros((height, width), dtype=bool)
    for y_idx, x_idx in zip(yy, xx):
        u = (x_idx - x_min) / quad_width
        v = (y_idx - y_min) / quad_height
        if u < 0.10 or u > 0.90 or v < 0.08 or v > 0.92:
            continue
        if 0.22 < u < 0.78 and 0.18 < v < 0.62:
            continue
        if u > 0.62 and v > 0.58:
            continue
        mask[y_idx, x_idx] = True
    return mask


def measure_wall_quad_plane_fit(
    depth: np.ndarray,
    leveled: np.ndarray,
    level_rot: np.ndarray,
    box: StructureBox,
    fx: float,
    fy: float,
    width: int,
    height: int,
    floor_mask: np.ndarray,
    camera_height_scaled: float | None,
    scale_trusted: bool,
    rng: np.random.Generator,
) -> dict:
    """Fit back-wall plane from quad interior; ray–plane intersect corners (occlusion-safe)."""
    interior_mask = quad_interior_sample_mask(box, width, height)
    sample_points = leveled[interior_mask]
    good = np.isfinite(sample_points).all(axis=1) & (sample_points[:, 2] > 0.05)
    sample_points = sample_points[good]
    if sample_points.shape[0] > 6000:
        idx = rng.choice(sample_points.shape[0], size=6000, replace=False)
        sample_points = sample_points[idx]

    expected_wall = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    wall_plane, inlier_count = fit_plane_ransac(sample_points, expected_wall, rng=rng)
    if wall_plane is None:
        return {
            "source": "plane_fit_failed",
            "error": "back_wall_plane_fit_failed",
            "interior_samples": int(sample_points.shape[0]),
        }
    wall_plane = orient_plane(wall_plane, expected_wall)

    corner_names = ("top_left", "top_right", "bottom_left", "bottom_right")
    corners_px = box.corner_px()
    corners_3d: dict[str, list[float] | None] = {}
    for name in corner_names:
        px, py = corners_px[name]
        origin, direction = leveled_ray_through_pixel(px, py, fx, fy, width, height, level_rot)
        hit = ray_plane_intersection(origin, direction, wall_plane)
        corners_3d[name] = None if hit is None else [round(float(v), 4) for v in hit]

    tl = np.asarray(corners_3d["top_left"], dtype=np.float32) if corners_3d["top_left"] else None
    tr = np.asarray(corners_3d["top_right"], dtype=np.float32) if corners_3d["top_right"] else None
    bl = np.asarray(corners_3d["bottom_left"], dtype=np.float32) if corners_3d["bottom_left"] else None
    br = np.asarray(corners_3d["bottom_right"], dtype=np.float32) if corners_3d["bottom_right"] else None

    width_m = None
    height_m = None
    if tl is not None and tr is not None and bl is not None and br is not None:
        width_m = 0.5 * (horizontal_distance_3d(tl, tr) + horizontal_distance_3d(bl, br))
        height_m = 0.5 * (float(bl[1] - tl[1]) + float(br[1] - tr[1]))

    center_x = int(round((box.left_x + box.right_x) * 0.5))
    center_y = int(round((box.ceiling_y + box.floor_y) * 0.5))
    origin, direction = leveled_ray_through_pixel(
        float(center_x), float(center_y), fx, fy, width, height, level_rot,
    )
    depth_hit = ray_plane_intersection(origin, direction, wall_plane)
    depth_m = None if depth_hit is None else float(depth_hit[2])

    camera_height_m = camera_height_from_floor_samples(leveled, floor_mask)
    if camera_height_m is None:
        camera_height_m = camera_height_scaled

    confidence = "low"
    confidence_flags: list[str] = []
    if scale_trusted:
        confidence = "medium"
    if camera_height_m is not None:
        if not (CAMERA_HEIGHT_MIN_M <= camera_height_m <= CAMERA_HEIGHT_MAX_M):
            confidence_flags.append("camera_height_out_of_range")
    else:
        confidence_flags.append("camera_height_unknown")
    if height_m is not None:
        if not (RG_HEIGHT_MIN_M <= height_m <= RG_HEIGHT_MAX_M):
            confidence_flags.append("height_out_of_range")
        elif scale_trusted and not confidence_flags:
            confidence = "high"
    else:
        confidence_flags.append("height_missing")

    residuals = np.abs(sample_points @ wall_plane.normal + wall_plane.offset) if sample_points.size else np.array([])
    return {
        "source": "plane_fit_ray_intersect",
        "depth_source": "ray_plane_wall_center",
        "width_m": None if width_m is None else round(width_m, 3),
        "height_m": None if height_m is None else round(height_m, 3),
        "depth_m": None if depth_m is None else round(depth_m, 3),
        "corners_px": {
            name: [round(corners_px[name][0], 1), round(corners_px[name][1], 1)]
            for name in corner_names
        },
        "corners_leveled": corners_3d,
        "wall_plane": {
            "normal": wall_plane.normal.tolist(),
            "offset": round(float(wall_plane.offset), 5),
            "inliers": int(inlier_count),
            "interior_samples": int(sample_points.shape[0]),
            "residual_median_m": None if residuals.size == 0 else round(float(np.median(residuals)), 4),
        },
        "wall_center_px": [center_x, center_y],
        "camera_height_m": None if camera_height_m is None else round(camera_height_m, 3),
        "confidence": confidence,
        "confidence_flags": confidence_flags,
    }


def region_masks(
    box: StructureBox,
    width: int,
    height: int,
    inset_px: int,
) -> dict[str, np.ndarray]:
    inset = max(4, inset_px)
    left = int(round(box.left_x)) + inset
    right = int(round(box.right_x)) - inset
    ceiling_row = int(round(box.ceiling_y)) + inset
    floor_row = int(round(box.floor_y)) - inset
    left = max(0, min(left, width - 3))
    right = max(left + 2, min(right, width - 1))
    ceiling_row = max(0, min(ceiling_row, height - 3))
    floor_row = max(ceiling_row + 2, min(floor_row, height - 1))

    wall_strip = max(inset * 2, int((right - left) * 0.10))
    full = np.zeros((height, width), dtype=bool)

    # Floor surface is below the floor-wall junction (larger image y).
    floor = full.copy()
    floor_start = min(height - 1, int(round(box.floor_y)) + inset)
    floor[floor_start:height, left:right] = True

    # Ceiling surface is above the ceiling-wall junction (smaller image y).
    ceiling = full.copy()
    ceiling_end = max(1, int(round(box.ceiling_y)) - inset)
    ceiling[0:ceiling_end, left:right] = True

    back_wall = full.copy()
    back_wall[ceiling_row:floor_row, left:right] = True

    left_wall = full.copy()
    lx1 = max(0, left - wall_strip)
    left_wall[ceiling_row:floor_row, lx1:left] = True

    right_wall = full.copy()
    rx1 = min(width, right + wall_strip)
    right_wall[ceiling_row:floor_row, right:rx1] = True

    return {
        "floor": floor,
        "ceiling": ceiling,
        "back_wall": back_wall,
        "left_wall": left_wall,
        "right_wall": right_wall,
    }


def sample_region_points(
    leveled: np.ndarray,
    mask: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = leveled[ys, xs]
    good = np.isfinite(pts).all(axis=1)
    pts = pts[good]
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if pts.shape[0] > max_samples:
        idx = rng.choice(pts.shape[0], size=max_samples, replace=False)
        pts = pts[idx]
    return pts.astype(np.float32)


def fit_plane_svd(points: np.ndarray) -> Plane | None:
    if points.shape[0] < 32:
        return None
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1].astype(np.float32)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-6:
        return None
    normal /= norm
    offset = -float(np.dot(normal, centroid))
    return Plane(normal=normal, offset=offset)


def fit_plane_ransac(
    points: np.ndarray,
    expected_normal: np.ndarray,
    iterations: int = 120,
    inlier_threshold: float = 0.06,
    rng: np.random.Generator | None = None,
) -> tuple[Plane | None, int]:
    if points.shape[0] < 48:
        plane = fit_plane_svd(points)
        return plane, 0 if plane is None else points.shape[0]

    rng = rng or np.random.default_rng(0)
    expected = expected_normal / max(float(np.linalg.norm(expected_normal)), 1e-6)
    best_plane: Plane | None = None
    best_inliers = -1
    best_points = points

    for _ in range(iterations):
        idx = rng.choice(points.shape[0], size=3, replace=False)
        trial = fit_plane_svd(points[idx])
        if trial is None:
            continue
        if float(np.dot(trial.normal, expected)) < 0:
            trial = Plane(normal=-trial.normal, offset=-trial.offset)
        distances = np.abs(points @ trial.normal + trial.offset)
        inlier_mask = distances <= inlier_threshold
        count = int(np.count_nonzero(inlier_mask))
        if count > best_inliers:
            best_inliers = count
            best_points = points[inlier_mask]

    plane = fit_plane_svd(best_points)
    if plane is None:
        return None, 0
    if float(np.dot(plane.normal, expected)) < 0:
        plane = Plane(normal=-plane.normal, offset=-plane.offset)
    inlier_count = max(best_inliers, int(best_points.shape[0]))
    return plane, inlier_count


def orient_plane(plane: Plane, expected_normal: np.ndarray) -> Plane:
    expected = expected_normal / max(float(np.linalg.norm(expected_normal)), 1e-6)
    if float(np.dot(plane.normal, expected)) < 0:
        return Plane(normal=-plane.normal, offset=-plane.offset)
    return plane


def intersect_three_planes(p1: Plane, p2: Plane, p3: Plane) -> np.ndarray | None:
    matrix = np.stack([p1.normal, p2.normal, p3.normal], axis=0).astype(np.float64)
    rhs = -np.array([p1.offset, p2.offset, p3.offset], dtype=np.float64)
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-5:
        return None
    try:
        point = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(point).all():
        return None
    return point.astype(np.float32)


def ray_plane_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    plane: Plane,
) -> np.ndarray | None:
    direction = direction.astype(np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    denom = float(np.dot(plane.normal, direction))
    if abs(denom) < 1e-6:
        return None
    t = -(float(np.dot(plane.normal, origin)) + plane.offset) / denom
    if t <= 0:
        return None
    return (origin + direction * t).astype(np.float32)


def refit_plane_offset(points: np.ndarray, normal: np.ndarray) -> Plane | None:
    if points.shape[0] < 16:
        return None
    normal = normal.astype(np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1e-6)
    signed = -(points @ normal)
    good = signed[np.isfinite(signed)]
    if good.size < 16:
        return None
    offset = float(np.median(good))
    return Plane(normal=normal, offset=offset)


def snap_manhattan_planes(
    plane_info: dict[str, dict],
    leveled: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Snap fitted planes to gravity/Manhattan axes; re-fit offset from interior samples."""
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    down = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    for name, normal in (
        ("floor", up),
        ("ceiling", down),
    ):
        points = sample_region_points(leveled, masks[name], 4000, np.random.default_rng(1))
        snapped = refit_plane_offset(points, normal)
        if snapped is not None:
            plane_info[name] = {
                **plane_info.get(name, {}),
                "normal": snapped.normal.tolist(),
                "offset": round(float(snapped.offset), 5),
                "samples": int(points.shape[0]),
                "snap": "horizontal_gravity",
                "plane": snapped,
            }

    back_points = sample_region_points(leveled, masks["back_wall"], 4000, np.random.default_rng(2))
    if back_points.shape[0] >= 32:
        raw = plane_info.get("back_wall", {}).get("plane")
        if raw is not None:
            xz = raw.normal[[0, 2]].astype(np.float64)
            norm = float(np.linalg.norm(xz))
            if norm > 0.15:
                xz /= norm
                wall_normal = np.array([-float(xz[0]), 0.0, -float(xz[1])], dtype=np.float32)
                snapped = refit_plane_offset(back_points, wall_normal)
                if snapped is not None:
                    plane_info["back_wall"] = {
                        **plane_info.get("back_wall", {}),
                        "normal": snapped.normal.tolist(),
                        "offset": round(float(snapped.offset), 5),
                        "samples": int(back_points.shape[0]),
                        "snap": "vertical_xz",
                        "plane": snapped,
                    }

    back = plane_info.get("back_wall", {}).get("plane")
    if back is not None:
        side_x = np.array([-back.normal[2], 0.0, back.normal[0]], dtype=np.float32)
        side_norm = float(np.linalg.norm(side_x))
        if side_norm > 1e-4:
            side_x /= side_norm
            for name, sign in (("left_wall", 1.0), ("right_wall", -1.0)):
                points = sample_region_points(leveled, masks[name], 4000, np.random.default_rng(3))
                snapped = refit_plane_offset(points, side_x * sign)
                if snapped is not None:
                    plane_info[name] = {
                        **plane_info.get(name, {}),
                        "normal": snapped.normal.tolist(),
                        "offset": round(float(snapped.offset), 5),
                        "samples": int(points.shape[0]),
                        "snap": "side_wall_xz",
                        "plane": snapped,
                    }
    return plane_info


def camera_height_from_floor_samples(leveled: np.ndarray, floor_mask: np.ndarray) -> float | None:
    ys = leveled[floor_mask, 1]
    ys = ys[np.isfinite(ys) & (ys > 0)]
    if ys.size < 32:
        return None
    return float(np.median(ys))


def measure_room_from_planes(
    floor_plane: Plane,
    ceiling_plane: Plane,
    back_wall_plane: Plane,
    left_wall_plane: Plane | None,
    right_wall_plane: Plane | None,
) -> dict:
    left = left_wall_plane
    right = right_wall_plane
    if left is None:
        left = Plane(normal=np.array([1.0, 0.0, 0.0], dtype=np.float32), offset=0.0)
    if right is None:
        right = Plane(normal=np.array([-1.0, 0.0, 0.0], dtype=np.float32), offset=0.0)

    bl = intersect_three_planes(floor_plane, back_wall_plane, left)
    br = intersect_three_planes(floor_plane, back_wall_plane, right)
    tl = intersect_three_planes(ceiling_plane, back_wall_plane, left)
    tr = intersect_three_planes(ceiling_plane, back_wall_plane, right)

    corners = {
        "floor_back_left": None if bl is None else bl.tolist(),
        "floor_back_right": None if br is None else br.tolist(),
        "ceiling_back_left": None if tl is None else tl.tolist(),
        "ceiling_back_right": None if tr is None else tr.tolist(),
    }

    width_m = None
    height_m = None
    if bl is not None and br is not None:
        width_m = float(np.linalg.norm(br - bl))
    if bl is not None and tl is not None:
        height_m = float(bl[1] - tl[1])
    elif br is not None and tr is not None:
        height_m = float(br[1] - tr[1])

    forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    depth_hit = ray_plane_intersection(np.zeros(3, dtype=np.float32), forward, back_wall_plane)
    depth_m = None if depth_hit is None else float(depth_hit[2])

    return {
        "width_m": None if width_m is None else round(width_m, 3),
        "height_m": None if height_m is None else round(height_m, 3),
        "depth_m": None if depth_m is None else round(depth_m, 3),
        "corners_leveled": corners,
    }


def parallel_plane_separation(plane_a: Plane, plane_b: Plane) -> float | None:
    """Distance between parallel planes (unit normals)."""
    dot = float(np.dot(plane_a.normal, plane_b.normal))
    if abs(abs(dot) - 1.0) > 0.2:
        return None
    if dot > 0:
        return abs(float(plane_a.offset - plane_b.offset))
    return abs(float(plane_a.offset + plane_b.offset))


def measure_manhattan_cuboid_from_planes(plane_info: dict[str, dict]) -> dict:
    """Width × height × depth from snapped floor/ceiling/left/right/back planes."""
    floor = plane_info.get("floor", {}).get("plane")
    ceiling = plane_info.get("ceiling", {}).get("plane")
    left = plane_info.get("left_wall", {}).get("plane")
    right = plane_info.get("right_wall", {}).get("plane")
    back = plane_info.get("back_wall", {}).get("plane")
    front = Plane(normal=np.array([0.0, 0.0, 1.0], dtype=np.float32), offset=0.0)

    width_m = parallel_plane_separation(left, right) if left and right else None
    height_m = parallel_plane_separation(floor, ceiling) if floor and ceiling else None
    depth_m = parallel_plane_separation(front, back) if back else None

    corner_dims = measure_room_from_planes(
        floor_plane=floor or Plane(normal=np.array([0.0, -1.0, 0.0]), offset=0.0),
        ceiling_plane=ceiling or Plane(normal=np.array([0.0, 1.0, 0.0]), offset=0.0),
        back_wall_plane=back or Plane(normal=np.array([0.0, 0.0, -1.0]), offset=0.0),
        left_wall_plane=left,
        right_wall_plane=right,
    )

    if width_m is None:
        width_m = corner_dims.get("width_m")
    if height_m is None:
        height_m = corner_dims.get("height_m")
    if depth_m is None:
        depth_m = corner_dims.get("depth_m")

    return {
        "source": "manhattan_cuboid_planes",
        "width_m": None if width_m is None else round(float(width_m), 3),
        "height_m": None if height_m is None else round(float(height_m), 3),
        "depth_m": None if depth_m is None else round(float(depth_m), 3),
        "plane_separations": {
            "width_from_walls": None if left is None or right is None else round(float(width_m), 3) if width_m else None,
            "height_from_floor_ceiling": None if floor is None or ceiling is None else round(float(height_m), 3) if height_m else None,
            "depth_from_front_back": None if back is None else round(float(depth_m), 3) if depth_m else None,
        },
        "corners_leveled": corner_dims.get("corners_leveled", {}),
        "planes": {
            name: {
                "normal": info.get("normal"),
                "offset": info.get("offset"),
                "samples": info.get("samples"),
                "inliers": info.get("inliers"),
                "snap": info.get("snap"),
                "error": info.get("error"),
            }
            for name, info in plane_info.items()
        },
    }


def measure_room_cuboid(
    image: Image.Image,
    depth: np.ndarray,
    gravity: list[float],
    fx: float,
    fy: float,
    camera_height_prior: float,
    region_inset_px: int,
    max_samples: int,
    score_thr: float,
    dist_thr: float,
    rng: np.random.Generator,
    exclude_mask: np.ndarray | None = None,
    rtmdet_meta: dict | None = None,
) -> dict:
    """Full cuboid: M-LSD RG quad → RTMDet-masked plane fits → Manhattan W×H×D."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    min_len = max(24.0, 0.04 * min(w, h))
    lines, detector = detect_mlsd_lines(rgb, score_thr, dist_thr)
    families = dominant_families(lines, min_length=min_len)
    segments = classify_line_segments(lines, families, min_len)

    rg_box, rg_meta = extract_red_green_intersection_quad(segments, w, h)
    if rg_box is None:
        return {
            "error": "rg_quad_failed",
            "rg_intersection_quad": rg_meta,
            "detector": detector,
        }

    masks = region_masks(rg_box, w, h, region_inset_px)
    if exclude_mask is not None and exclude_mask.any():
        from rtmdet_exclude_mask import apply_exclude_mask

        masks = apply_exclude_mask(masks, exclude_mask)

    leveled, level_rot = leveled_points(depth, fx, fy, gravity)
    floor_mask = masks["floor"]
    camera_height_raw = camera_height_from_floor_samples(leveled, floor_mask)
    depth_scale = 1.0
    scale_trusted = False
    if camera_height_raw is not None and 0.45 <= camera_height_raw <= 5.0:
        depth_scale = camera_height_prior / camera_height_raw
        depth = rescale_depth(depth, depth_scale)
        leveled, level_rot = leveled_points(depth, fx, fy, gravity)
        camera_height_scaled = camera_height_from_floor_samples(leveled, floor_mask)
        scale_trusted = (
            camera_height_scaled is not None
            and CAMERA_HEIGHT_MIN_M <= camera_height_scaled <= CAMERA_HEIGHT_MAX_M
        )
    else:
        camera_height_scaled = camera_height_raw

    plane_info = fit_guided_planes(leveled, masks, max_samples, rng)
    plane_info = snap_manhattan_planes(plane_info, leveled, masks)
    cuboid = measure_manhattan_cuboid_from_planes(plane_info)

    # RG plane-fit back wall (occlusion-safe) for comparison
    rg_plane = measure_wall_quad_plane_fit(
        depth=depth,
        leveled=leveled,
        level_rot=level_rot,
        box=rg_box,
        fx=fx,
        fy=fy,
        width=w,
        height=h,
        floor_mask=floor_mask,
        camera_height_scaled=camera_height_scaled,
        scale_trusted=scale_trusted,
        rng=rng,
    )

    width_m = cuboid.get("width_m")
    height_m = cuboid.get("height_m")
    depth_m = cuboid.get("depth_m")

    confidence = "low"
    confidence_flags: list[str] = []
    if scale_trusted:
        confidence = "medium"
    if camera_height_scaled is not None:
        if not (CAMERA_HEIGHT_MIN_M <= camera_height_scaled <= CAMERA_HEIGHT_MAX_M):
            confidence_flags.append("camera_height_out_of_range")
    if height_m is not None:
        if not (RG_HEIGHT_MIN_M <= height_m <= RG_HEIGHT_MAX_M):
            confidence_flags.append("height_out_of_range")
        elif scale_trusted and not confidence_flags:
            confidence = "high"
    else:
        confidence_flags.append("height_missing")

    w_h_d = None
    if width_m is not None and height_m is not None and depth_m is not None:
        w_h_d = f"{width_m:.3f} × {height_m:.3f} × {depth_m:.3f} m"

    return {
        "source": "cuboid_mlsd_geocalib_depthanything",
        "detector": detector,
        "line_segments": int(lines.shape[0]),
        "rg_intersection_quad": rg_meta,
        "rg_intersection_quad_px": rg_meta.get("quad_px"),
        "rtmdet_exclude": rtmdet_meta,
        "depth_scale": round(depth_scale, 4),
        "scale_trusted": scale_trusted,
        "camera_height_raw_m": None if camera_height_raw is None else round(camera_height_raw, 3),
        "camera_height_scaled_m": None if camera_height_scaled is None else round(camera_height_scaled, 3),
        "camera_height_prior_m": round(camera_height_prior, 3),
        "plane_fits": cuboid.get("planes", {}),
        "cuboid_measurement": cuboid,
        "rg_back_wall_plane_fit": rg_plane,
        "width_m": width_m,
        "height_m": height_m,
        "depth_m": depth_m,
        "w_h_d": w_h_d,
        "confidence": confidence,
        "confidence_flags": confidence_flags,
        "level_rotation": level_rot.tolist(),
        "_rg_box": rg_box,
        "_segments": segments,
    }


def fit_guided_planes(
    leveled: np.ndarray,
    masks: dict[str, np.ndarray],
    max_samples: int,
    rng: np.random.Generator,
) -> dict[str, dict]:
    expectations = {
        "floor": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "ceiling": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "back_wall": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "left_wall": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "right_wall": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    }
    out: dict[str, dict] = {}
    for name, expected in expectations.items():
        points = sample_region_points(leveled, masks[name], max_samples, rng)
        plane, inliers = fit_plane_ransac(points, expected, rng=rng)
        if plane is None:
            out[name] = {"error": "fit_failed", "samples": int(points.shape[0])}
            continue
        plane = orient_plane(plane, expected)
        residuals = np.abs(points @ plane.normal + plane.offset) if points.shape[0] else np.array([])
        out[name] = {
            "normal": plane.normal.tolist(),
            "offset": round(float(plane.offset), 5),
            "samples": int(points.shape[0]),
            "inliers": int(inliers),
            "residual_median_m": None if residuals.size == 0 else round(float(np.median(residuals)), 4),
            "plane": plane,
        }
    return out


def rescale_depth(depth: np.ndarray, scale: float) -> np.ndarray:
    out = depth.astype(np.float32, copy=True)
    out[np.isfinite(out) & (out > 0)] *= scale
    return out


def quad_edge_segments(box: StructureBox) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    corners = box.corner_px()
    order = ("top_left", "top_right", "bottom_right", "bottom_left")
    points = [corners[name] for name in order]
    return [(points[index], points[(index + 1) % 4]) for index in range(4)]


def segment_near_quad_top_edge(seg: dict, box: StructureBox, tolerance_px: float) -> bool:
    corners = box.corner_px()
    top_left = corners["top_left"]
    top_right = corners["top_right"]
    span_x = top_right[0] - top_left[0]
    if span_x <= 1.0:
        return False
    for px, py in (
        *segment_endpoints(seg),
        ((seg["x1"] + seg["x2"]) * 0.5, (seg["y1"] + seg["y2"]) * 0.5),
    ):
        if px < top_left[0] - tolerance_px or px > top_right[0] + tolerance_px:
            continue
        edge_t = (px - top_left[0]) / span_x
        edge_t = max(0.0, min(1.0, edge_t))
        y_on_edge = top_left[1] + edge_t * (top_right[1] - top_left[1])
        if abs(py - y_on_edge) <= tolerance_px:
            return True
    return False


def segment_touches_quad(seg: dict, box: StructureBox, tolerance_px: float) -> bool:
    if segment_near_quad_top_edge(seg, box, tolerance_px):
        return True
    probe_points = [
        *segment_endpoints(seg),
        ((seg["x1"] + seg["x2"]) * 0.5, (seg["y1"] + seg["y2"]) * 0.5),
    ]
    for px, py in probe_points:
        for (ax, ay), (bx, by) in quad_edge_segments(box):
            if (
                math.hypot(px - ax, py - ay) <= tolerance_px
                or math.hypot(px - bx, py - by) <= tolerance_px
                or point_to_segment_distance(px, py, ax, ay, bx, by) <= tolerance_px
            ):
                return True
    return False


def filter_box_connected_segments(
    segments: list[dict],
    box: StructureBox,
    width: int,
    height: int | None = None,
) -> tuple[list[dict], dict]:
    """Keep blue lines touching the frame, then red/green that genuinely touch those blues."""
    box_touch_tol = max(42.0, width * 0.042)
    endpoint_tol = max(10.0, width * 0.010)
    near_tol = max(12.0, width * 0.012)
    blues = [s for s in segments if s["color_label"] == "dark_blue"]
    chromatic = [s for s in segments if s["color_label"] in {"red", "green"}]

    blues_on_box = [s for s in blues if segment_touches_quad(s, box, box_touch_tol)]
    chromatic_on_blue: list[dict] = []
    for seg in chromatic:
        if keep_chromatic_segment(seg, blues_on_box, width, endpoint_tol, near_tol):
            chromatic_on_blue.append(seg)

    kept = blues_on_box + chromatic_on_blue
    counts = {"dark_blue": len(blues_on_box), "red": 0, "green": 0}
    for seg in chromatic_on_blue:
        counts[seg["color_label"]] += 1

    return kept, {
        "box_touch_tol_px": round(box_touch_tol, 1),
        "chromatic_endpoint_tol_px": round(endpoint_tol, 1),
        "chromatic_near_tol_px": round(near_tol, 1),
        "blue_touching_box_ids": [s["id"] for s in blues_on_box],
        "red_green_touching_blue_ids": [s["id"] for s in chromatic_on_blue],
        "kept_segment_counts": counts,
        "total_kept": len(kept),
        "total_input": len(segments),
    }


def extend_segment_to_frame(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_box: StructureBox,
) -> tuple[float, float, float, float]:
    """Extend a line segment along its direction until it meets the cyan frame edges."""
    corners = frame_box.corner_px()
    left = corners["top_left"][0]
    right = corners["top_right"][0]
    top = corners["top_left"][1]
    bottom = corners["bottom_left"][1]

    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return x1, y1, x2, y2

    ts: list[float] = []
    if abs(dx) > 1e-6:
        for x_edge in (left, right):
            t_param = (x_edge - x1) / dx
            y_hit = y1 + t_param * dy
            if top - 0.5 <= y_hit <= bottom + 0.5:
                ts.append(t_param)
    if abs(dy) > 1e-6:
        for y_edge in (top, bottom):
            t_param = (y_edge - y1) / dy
            x_hit = x1 + t_param * dx
            if left - 0.5 <= x_hit <= right + 0.5:
                ts.append(t_param)

    if len(ts) < 2:
        return x1, y1, x2, y2

    t_min = min(ts)
    t_max = max(ts)
    ex1 = float(np.clip(x1 + t_min * dx, left, right))
    ey1 = float(np.clip(y1 + t_min * dy, top, bottom))
    ex2 = float(np.clip(x1 + t_max * dx, left, right))
    ey2 = float(np.clip(y1 + t_max * dy, top, bottom))
    return ex1, ey1, ex2, ey2


def infinite_line_intersection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> tuple[float, float] | None:
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-6:
        return None
    t_param = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom
    return x1 + t_param * dx1, y1 + t_param * dy1


def extended_segment(seg: dict, frame_box: StructureBox) -> tuple[float, float, float, float]:
    return extend_segment_to_frame(seg["x1"], seg["y1"], seg["x2"], seg["y2"], frame_box)


def pick_ceiling_red(reds: list[dict], height: int) -> dict | None:
    upper = [r for r in reds if r["mid_y"] < height * 0.42]
    if not upper:
        return None
    return min(upper, key=lambda r: (r["mid_y"], -r["length_px"]))


def pick_floor_red(reds: list[dict], height: int) -> dict | None:
    lower = [r for r in reds if height * 0.52 < r["mid_y"] < height * 0.84]
    if not lower:
        lower = [r for r in reds if r["mid_y"] > height * 0.52]
    if not lower:
        return None
    return min(lower, key=lambda r: (r["mid_y"], -r["length_px"]))


def pick_wall_green(greens: list[dict], side: str, width: int) -> dict | None:
    if side == "left":
        candidates = [g for g in greens if g["mid_x"] < width * 0.28]
    else:
        candidates = [g for g in greens if g["mid_x"] > width * 0.62]
    if not candidates:
        return None
    return max(candidates, key=lambda g: g["length_px"])


def extract_red_green_intersection_quad(
    segments: list[dict],
    width: int,
    height: int,
    frame_margin_px: int = 2,
) -> tuple[StructureBox | None, dict]:
    """Inner wall quad from extended red×green line intersections (box-connected set)."""
    frame_box = full_image_frame_box(width, height, margin_px=frame_margin_px)
    filtered, filter_meta = filter_box_connected_segments(segments, frame_box, width, height)
    reds = [s for s in filtered if s["color_label"] == "red"]
    greens = [s for s in filtered if s["color_label"] == "green"]

    meta: dict = {
        "filter": filter_meta,
        "picked_lines": {},
        "errors": [],
    }
    if len(reds) < 2 or len(greens) < 2:
        meta["errors"].append("need_at_least_two_red_and_two_green")
        return None, meta

    ceiling_red = pick_ceiling_red(reds, height)
    floor_red = pick_floor_red(reds, height)
    left_green = pick_wall_green(greens, "left", width)
    right_green = pick_wall_green(greens, "right", width)
    if not ceiling_red or not floor_red or not left_green or not right_green:
        meta["errors"].append("failed_to_pick_ceiling_floor_red_or_wall_green")
        return None, meta

    meta["picked_lines"] = {
        "ceiling_red_id": ceiling_red["id"],
        "floor_red_id": floor_red["id"],
        "left_green_id": left_green["id"],
        "right_green_id": right_green["id"],
    }

    top_line = extended_segment(ceiling_red, frame_box)
    bottom_line = extended_segment(floor_red, frame_box)
    left_line = extended_segment(left_green, frame_box)
    right_line = extended_segment(right_green, frame_box)

    corner_names = ("top_left", "top_right", "bottom_left", "bottom_right")
    corner_lines = {
        "top_left": (top_line, left_line),
        "top_right": (top_line, right_line),
        "bottom_left": (bottom_line, left_line),
        "bottom_right": (bottom_line, right_line),
    }
    corners: dict[str, tuple[float, float]] = {}
    for name in corner_names:
        line_a, line_b = corner_lines[name]
        hit = infinite_line_intersection(*line_a, *line_b)
        if hit is None:
            meta["errors"].append(f"parallel_lines_at_{name}")
            return None, meta
        px, py = hit
        if not (0 <= px <= width - 1 and 0 <= py <= height - 1):
            meta["errors"].append(f"corner_outside_image_{name}")
            return None, meta
        corners[name] = (float(px), float(py))

    top_left = corners["top_left"]
    top_right = corners["top_right"]
    bottom_left = corners["bottom_left"]
    bottom_right = corners["bottom_right"]
    box = StructureBox(
        left_x=top_left[0],
        right_x=top_right[0],
        ceiling_y=min(top_left[1], top_right[1]),
        floor_y=max(bottom_left[1], bottom_right[1]),
        source="red_green_intersection_quad",
        red_lines=len(reds),
        green_lines=len(greens),
        dark_blue_lines=sum(1 for s in filtered if s["color_label"] == "dark_blue"),
        ceiling_y_left=top_left[1],
        ceiling_y_right=top_right[1],
        floor_y_left=bottom_left[1],
        floor_y_right=bottom_right[1],
    )
    meta["quad_px"] = {
        name: [round(corners[name][0], 1), round(corners[name][1], 1)]
        for name in corner_names
    }
    return box, meta


def draw_rg_intersection_quad_on_canvas(
    canvas: np.ndarray,
    box: StructureBox,
    *,
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: int = 5,
) -> None:
    corners = box.corner_px()
    quad_pts = np.array([
        [int(round(corners["top_left"][0])), int(round(corners["top_left"][1]))],
        [int(round(corners["top_right"][0])), int(round(corners["top_right"][1]))],
        [int(round(corners["bottom_right"][0])), int(round(corners["bottom_right"][1]))],
        [int(round(corners["bottom_left"][0])), int(round(corners["bottom_left"][1]))],
    ], dtype=np.int32)
    cv2.polylines(canvas, [quad_pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    for corner in quad_pts:
        cv2.circle(canvas, tuple(corner), 7, color, -1, cv2.LINE_AA)


def full_image_frame_box(width: int, height: int, margin_px: int = 2) -> StructureBox:
    """Cyan frame covering the full photo (image border inset by a few px)."""
    inset = max(0, margin_px)
    left = float(inset)
    right = float(width - 1 - inset)
    top = float(inset)
    bottom = float(height - 1 - inset)
    return StructureBox(
        left_x=left,
        right_x=right,
        ceiling_y=top,
        floor_y=bottom,
        source="full_image_frame",
        red_lines=0,
        green_lines=0,
        dark_blue_lines=0,
        ceiling_y_left=top,
        ceiling_y_right=top,
        floor_y_left=bottom,
        floor_y_right=bottom,
    )


def draw_box_connected_overlay(
    rgb: np.ndarray,
    segments: list[dict],
    out_path: Path,
    *,
    box_thickness: int = 5,
    frame_margin_px: int = 2,
) -> dict:
    """Full-image cyan frame + only blues on the frame + red/green touching those blues."""
    from mlsd_draw_room_lines import FAMILY_COLORS_RGB, UNASSIGNED_RGB

    height, width = rgb.shape[:2]
    frame_box = full_image_frame_box(width, height, margin_px=frame_margin_px)
    filtered, filter_meta = filter_box_connected_segments(segments, frame_box, width, height)
    canvas = rgb.copy()
    color_map = {
        "red": FAMILY_COLORS_RGB[0],
        "green": FAMILY_COLORS_RGB[1],
        "dark_blue": UNASSIGNED_RGB,
    }
    line_thickness = 3
    for seg in filtered:
        color = color_map.get(seg["color_label"], (200, 200, 200))
        x1, y1, x2, y2 = seg["x1"], seg["y1"], seg["x2"], seg["y2"]
        if seg["color_label"] in {"red", "green"}:
            x1, y1, x2, y2 = extend_segment_to_frame(x1, y1, x2, y2, frame_box)
        cv2.line(
            canvas,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            line_thickness,
            cv2.LINE_AA,
        )

    corners = frame_box.corner_px()
    quad_pts = np.array([
        [int(round(corners["top_left"][0])), int(round(corners["top_left"][1]))],
        [int(round(corners["top_right"][0])), int(round(corners["top_right"][1]))],
        [int(round(corners["bottom_right"][0])), int(round(corners["bottom_right"][1]))],
        [int(round(corners["bottom_left"][0])), int(round(corners["bottom_left"][1]))],
    ], dtype=np.int32)
    cv2.polylines(
        canvas,
        [quad_pts],
        isClosed=True,
        color=(0, 220, 255),
        thickness=box_thickness,
        lineType=cv2.LINE_AA,
    )
    for corner in quad_pts:
        cv2.circle(canvas, tuple(corner), 7, (0, 220, 255), -1, cv2.LINE_AA)

    rg_box, rg_meta = extract_red_green_intersection_quad(segments, width, height, frame_margin_px)
    if rg_box is not None:
        draw_rg_intersection_quad_on_canvas(canvas, rg_box)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    filter_meta["frame_px"] = {
        "top_left": [round(corners["top_left"][0], 1), round(corners["top_left"][1], 1)],
        "top_right": [round(corners["top_right"][0], 1), round(corners["top_right"][1], 1)],
        "bottom_right": [round(corners["bottom_right"][0], 1), round(corners["bottom_right"][1], 1)],
        "bottom_left": [round(corners["bottom_left"][0], 1), round(corners["bottom_left"][1], 1)],
        "image_size": [width, height],
        "margin_px": frame_margin_px,
    }
    if rg_box is not None:
        filter_meta["rg_intersection_quad_px"] = rg_meta.get("quad_px", {})
        filter_meta["rg_intersection_picked_lines"] = rg_meta.get("picked_lines", {})
    else:
        filter_meta["rg_intersection_errors"] = rg_meta.get("errors", [])
    return filter_meta


def structure_box_from_measurement(result: dict, width: int, height: int) -> StructureBox:
    px = result.get("structure_box_px", {})
    junction = result.get("line_junction_quad", {})
    quad = junction.get("quad_px", {})

    def corner(name: str, index: int, default: float) -> float:
        point = quad.get(name)
        return float(point[index]) if point else default

    left_x = float(px.get("left_x", width * 0.08))
    right_x = float(px.get("right_x", width * 0.92))
    ceiling_y = float(px.get("ceiling_y", height * 0.08))
    floor_y = float(px.get("floor_y", height * 0.92))
    counts = result.get("line_family_counts", {})

    return StructureBox(
        left_x=left_x,
        right_x=right_x,
        ceiling_y=ceiling_y,
        floor_y=floor_y,
        source=str(px.get("box_source", "unknown")),
        red_lines=int(counts.get("red", 0)),
        green_lines=int(counts.get("green", 0)),
        dark_blue_lines=int(counts.get("dark_blue", 0)),
        ceiling_y_left=corner("top_left", 1, ceiling_y),
        ceiling_y_right=corner("top_right", 1, ceiling_y),
        floor_y_left=corner("bottom_left", 1, floor_y),
        floor_y_right=corner("bottom_right", 1, floor_y),
    )


def draw_debug_overlay(
    rgb: np.ndarray,
    box: StructureBox,
    segments: list[dict],
    out_path: Path,
) -> None:
    from mlsd_draw_room_lines import FAMILY_COLORS_RGB, UNASSIGNED_RGB

    canvas = rgb.copy()
    color_map = {
        "red": FAMILY_COLORS_RGB[0],
        "green": FAMILY_COLORS_RGB[1],
        "dark_blue": UNASSIGNED_RGB,
    }
    for seg in segments:
        color = color_map.get(seg["color_label"], (200, 200, 200))
        thickness = 2 if seg["color_label"] == "dark_blue" else 1
        cv2.line(
            canvas,
            (int(seg["x1"]), int(seg["y1"])),
            (int(seg["x2"]), int(seg["y2"])),
            color,
            thickness,
            cv2.LINE_AA,
        )

    left = int(round(box.left_x))
    right = int(round(box.right_x))
    corners = box.corner_px()
    quad_pts = np.array([
        [int(round(corners["top_left"][0])), int(round(corners["top_left"][1]))],
        [int(round(corners["top_right"][0])), int(round(corners["top_right"][1]))],
        [int(round(corners["bottom_right"][0])), int(round(corners["bottom_right"][1]))],
        [int(round(corners["bottom_left"][0])), int(round(corners["bottom_left"][1]))],
    ], dtype=np.int32)
    cv2.polylines(canvas, [quad_pts], isClosed=True, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)
    for corner in quad_pts:
        cv2.circle(canvas, tuple(corner), 6, (0, 220, 255), -1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def tape_error(measured: float | None, tape: float | None) -> dict | None:
    if measured is None or tape is None or tape <= 0:
        return None
    err = measured - tape
    return {
        "tape_m": round(tape, 3),
        "measured_m": round(measured, 3),
        "error_m": round(err, 3),
        "error_pct": round(100.0 * err / tape, 1),
    }


def measure_structure_box_room(
    image: Image.Image,
    depth: np.ndarray,
    gravity: list[float],
    fx: float,
    fy: float,
    camera_height_prior: float,
    region_inset_px: int,
    max_samples: int,
    score_thr: float,
    dist_thr: float,
    rng: np.random.Generator,
) -> dict:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    min_len = max(24.0, 0.04 * min(w, h))
    lines, detector = detect_mlsd_lines(rgb, score_thr, dist_thr)
    families = dominant_families(lines, min_length=min_len)
    segments = classify_line_segments(lines, families, min_len)
    box, junction_meta = extract_wall_quad_line_junctions(segments, w, h)
    masks = region_masks(box, w, h, region_inset_px)

    leveled, level_rot = leveled_points(depth, fx, fy, gravity)
    floor_mask = masks["floor"]
    camera_height_raw = camera_height_from_floor_samples(leveled, floor_mask)
    depth_scale = 1.0
    scale_trusted = False
    if camera_height_raw is not None and 0.45 <= camera_height_raw <= 5.0:
        depth_scale = camera_height_prior / camera_height_raw
        depth = rescale_depth(depth, depth_scale)
        leveled, level_rot = leveled_points(depth, fx, fy, gravity)
        camera_height_scaled = camera_height_from_floor_samples(leveled, floor_mask)
        scale_trusted = (
            camera_height_scaled is not None
            and CAMERA_HEIGHT_MIN_M <= camera_height_scaled <= CAMERA_HEIGHT_MAX_M
        )
    else:
        camera_height_scaled = camera_height_raw

    dims = measure_wall_quad_depthanything(
        depth=depth,
        leveled=leveled,
        box=box,
        inset_px=region_inset_px,
    )

    rg_box, rg_meta = extract_red_green_intersection_quad(segments, w, h)
    rg_dims: dict | None = None
    if rg_box is not None:
        rg_masks = region_masks(rg_box, w, h, region_inset_px)
        rg_dims = measure_wall_quad_plane_fit(
            depth=depth,
            leveled=leveled,
            level_rot=level_rot,
            box=rg_box,
            fx=fx,
            fy=fy,
            width=w,
            height=h,
            floor_mask=rg_masks["floor"],
            camera_height_scaled=camera_height_scaled,
            scale_trusted=scale_trusted,
            rng=rng,
        )
        rg_dims["box_source"] = rg_box.source
        rg_dims["picked_lines"] = rg_meta.get("picked_lines", {})
        rg_w = rg_dims.get("width_m")
        rg_h = rg_dims.get("height_m")
        rg_d = rg_dims.get("depth_m")
        if rg_w is not None and rg_h is not None and rg_d is not None:
            rg_dims["w_h_d"] = f"{rg_w:.3f} × {rg_h:.3f} × {rg_d:.3f} m"

    width_m = dims.get("width_m")
    height_m = dims.get("height_m")
    depth_m = dims.get("depth_m")
    depth_source = dims.get("depth_source", "depthanything_median_at_wall_center")

    min_expected_height = (
        (camera_height_scaled or camera_height_prior) + 0.85
        if (camera_height_scaled or camera_height_prior)
        else None
    )
    height_sanity_flag = None
    if height_m is not None and min_expected_height is not None and height_m < min_expected_height:
        height_sanity_flag = "ceiling_line_maybe_too_low"

    confidence = "low"
    if scale_trusted and height_m is not None:
        if ROOM_HEIGHT_MIN_M <= height_m <= ROOM_HEIGHT_MAX_M:
            confidence = "medium"
        if width_m and depth_m:
            confidence = "high" if scale_trusted and height_sanity_flag is None else "medium"

    family_counts = {
        "red": sum(1 for s in segments if s["color_label"] == "red"),
        "green": sum(1 for s in segments if s["color_label"] == "green"),
        "dark_blue": sum(1 for s in segments if s["color_label"] == "dark_blue"),
    }

    return {
        "source": "line_quad_depthanything",
        "detector": detector,
        "line_segments": int(lines.shape[0]),
        "line_family_counts": family_counts,
        "family_angles_deg": [round(math.degrees(fa), 2) for fa in families],
        "line_junction_quad": junction_meta,
        "structure_box_px": {
            "left_x": round(box.left_x, 1),
            "right_x": round(box.right_x, 1),
            "ceiling_y": round(box.ceiling_y, 1),
            "floor_y": round(box.floor_y, 1),
            "box_source": box.source,
            "quad_px": junction_meta.get("quad_px", {}),
        },
        "quad_measurement": dims,
        "rg_intersection_quad": rg_meta,
        "rg_intersection_quad_px": rg_meta.get("quad_px"),
        "rg_intersection_measurement": rg_dims,
        "rg_width_m": None if rg_dims is None else rg_dims.get("width_m"),
        "rg_height_m": None if rg_dims is None else rg_dims.get("height_m"),
        "rg_depth_m": None if rg_dims is None else rg_dims.get("depth_m"),
        "rg_w_h_d": None if rg_dims is None else rg_dims.get("w_h_d"),
        "camera_height_raw_m": None if camera_height_raw is None else round(camera_height_raw, 3),
        "camera_height_scaled_m": None if camera_height_scaled is None else round(camera_height_scaled, 3),
        "camera_height_prior_m": round(camera_height_prior, 3),
        "depth_scale": round(depth_scale, 4),
        "scale_trusted": scale_trusted,
        "scale_gate": f"trust when camera_height in [{CAMERA_HEIGHT_MIN_M}, {CAMERA_HEIGHT_MAX_M}] m",
        "width_m": width_m,
        "height_m": height_m,
        "depth_m": depth_m,
        "depth_source": depth_source,
        "height_sanity_flag": height_sanity_flag,
        "w_h_d": (
            None
            if width_m is None or height_m is None or depth_m is None
            else f"{width_m:.3f} × {height_m:.3f} × {depth_m:.3f} m"
        ),
        "confidence": confidence,
        "level_rotation": level_rot.tolist(),
        "_segments_for_overlay": segments,
    }


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    geocalib = json.loads(args.geocalib_json.expanduser().read_text(encoding="utf-8"))
    gravity = geocalib.get("gravity")
    if not gravity:
        raise SystemExit("GeoCalib JSON missing gravity vector")

    image = read_rgb(image_path)
    width, height = image.size
    exif_fx, exif_fy, focal_source = focal_from_exif(image_path, width, height)
    fx = float(geocalib.get("focal_x_px") or geocalib.get("focalLengthPx") or exif_fx)
    fy = float(geocalib.get("focal_y_px") or geocalib.get("focalLengthYPx") or fx)

    print(f"Image: {image_path} ({width}×{height})")
    print(f"Focal: fx={fx:.1f} fy={fy:.1f} ({focal_source})")
    print("Running Depth Anything…")
    depth = run_depth_onnx(image, args.onnx, args.input_size)

    rng = np.random.default_rng(42)
    result = measure_structure_box_room(
        image=image,
        depth=depth,
        gravity=gravity,
        fx=fx,
        fy=fy,
        camera_height_prior=args.camera_height_prior,
        region_inset_px=args.region_inset_px,
        max_samples=args.max_samples_per_region,
        score_thr=args.score_thr,
        dist_thr=args.dist_thr,
        rng=rng,
    )

    tape_compare = {}
    for key, tape_val in (
        ("width", args.tape_width),
        ("height", args.tape_height),
        ("depth", args.tape_depth),
    ):
        err = tape_error(result.get(f"{key}_m"), tape_val)
        if err:
            tape_compare[key] = err
    if tape_compare:
        result["tape_comparison"] = tape_compare

    rg_tape_compare = {}
    for key, tape_val, result_key in (
        ("width", args.tape_width, "rg_width_m"),
        ("height", args.tape_height, "rg_height_m"),
        ("depth", args.tape_depth, "rg_depth_m"),
    ):
        err = tape_error(result.get(result_key), tape_val)
        if err:
            rg_tape_compare[key] = err
    if rg_tape_compare:
        result["rg_tape_comparison"] = rg_tape_compare

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    segments = result.pop("_segments_for_overlay", [])
    box = structure_box_from_measurement(result, width, height)

    overlay_path = out_dir / f"{stem}_structure_box.png"
    lines_overlay_path = out_dir / f"{stem}_structure_box_lines.png"
    connected_overlay_path = out_dir / f"{stem}_structure_box_connected.png"
    draw_debug_overlay(rgb, box, segments, lines_overlay_path)
    draw_debug_overlay(rgb, box, [], overlay_path)
    connected_filter = draw_box_connected_overlay(rgb, segments, connected_overlay_path)

    payload = {
        "image": str(image_path),
        "geocalib_json": str(args.geocalib_json.expanduser().resolve()),
        "focal_px": round(fx, 2),
        "structure_box_measurement": result,
        "overlay_png": str(overlay_path),
        "lines_overlay_png": str(lines_overlay_path),
        "box_connected_overlay_png": str(connected_overlay_path),
        "box_connected_segment_ids": connected_filter,
    }

    json_path = out_dir / f"{stem}_structure_box.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\noverlay:   {overlay_path}")
    print(f"lines:     {lines_overlay_path}")
    print(f"connected: {connected_overlay_path}")
    print(f"json:      {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
