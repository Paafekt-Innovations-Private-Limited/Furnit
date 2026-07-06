#!/usr/bin/env python3
"""Structure-box + scaled depth → plane-intersection room measurement.

Pipeline:
  1. M-LSD (or OpenCV LSD) finds 2D line segments on the RGB photo.
  2. Dominant horizontal/vertical families form a structure box (floor, ceiling, walls).
  3. Depth Anything + GeoCalib unproject interior region samples (inset from edges).
  4. Fit one plane per labeled region (SVD + light RANSAC).
  5. Intersect planes for corners; measure W×H×D in gravity-leveled metric space.
  6. Validate depth scale via camera-height-to-floor-plane (reject if implausible).

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
    read_rgb,
    run_depth_onnx,
)

DEFAULT_OUT_DIR = Path("/tmp/structure_box_measure")
CAMERA_HEIGHT_MIN_M = 1.0
CAMERA_HEIGHT_MAX_M = 1.8
ROOM_HEIGHT_MIN_M = 1.8
ROOM_HEIGHT_MAX_M = 4.5


@dataclass
class StructureBox:
    left_x: float
    right_x: float
    ceiling_y: float
    floor_y: float
    source: str
    horizontal_lines: int
    vertical_lines: int


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


def line_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def line_length(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def cluster_1d_weighted(
    items: list[tuple[float, float]],
    merge_distance: float,
) -> list[tuple[float, float]]:
    """Cluster (value, weight) pairs; return (center, total_weight) per cluster."""
    if not items:
        return []
    items = sorted(items, key=lambda item: item[0])
    clusters: list[list[tuple[float, float]]] = [[items[0]]]
    for value, weight in items[1:]:
        center = sum(v * w for v, w in clusters[-1]) / max(sum(w for _, w in clusters[-1]), 1e-6)
        if abs(value - center) <= merge_distance:
            clusters[-1].append((value, weight))
        else:
            clusters.append([(value, weight)])
    out: list[tuple[float, float]] = []
    for cluster in clusters:
        total_w = sum(w for _, w in cluster)
        center = sum(v * w for v, w in cluster) / max(total_w, 1e-6)
        out.append((center, total_w))
    return out


def extract_structure_box(lines: np.ndarray, width: int, height: int) -> StructureBox:
    min_len = max(24.0, 0.04 * min(width, height))
    horizontals: list[tuple[float, float]] = []
    verticals: list[tuple[float, float]] = []

    for x1, y1, x2, y2 in lines:
        length = line_length(x1, y1, x2, y2)
        if length < min_len:
            continue
        angle = line_angle_deg(x1, y1, x2, y2)
        if angle <= 18.0 or angle >= 162.0:
            y_repr = max(y1, y2)
            horizontals.append((y_repr, length))
            y_repr_top = min(y1, y2)
            horizontals.append((y_repr_top, length * 0.5))
        elif 72.0 <= angle <= 108.0:
            x_repr = min(x1, x2)
            verticals.append((x_repr, length))
            x_repr_right = max(x1, x2)
            verticals.append((x_repr_right, length * 0.5))

    merge_y = max(18.0, height * 0.02)
    merge_x = max(18.0, width * 0.02)
    h_clusters = cluster_1d_weighted(horizontals, merge_y)
    v_clusters = cluster_1d_weighted(verticals, merge_x)

    source = "mlsd_line_clusters"
    if len(h_clusters) >= 2 and len(v_clusters) >= 2:
        lower = [c for c in h_clusters if c[0] >= height * 0.45]
        upper = [c for c in h_clusters if c[0] <= height * 0.55]
        floor_cluster = max(lower or h_clusters, key=lambda item: item[1])
        ceiling_cluster = min(upper or h_clusters, key=lambda item: item[0])
        floor_y = floor_cluster[0]
        ceiling_y = ceiling_cluster[0]
        v_sorted = sorted(v_clusters, key=lambda item: item[0])
        left_x = v_sorted[0][0]
        right_x = v_sorted[-1][0]
    else:
        source = "image_margin_fallback"
        margin_x = width * 0.08
        margin_y = height * 0.08
        left_x = margin_x
        right_x = width - 1 - margin_x
        ceiling_y = margin_y
        floor_y = height - 1 - margin_y

    left_x = float(np.clip(left_x, 0, width - 2))
    right_x = float(np.clip(right_x, left_x + 8, width - 1))
    ceiling_y = float(np.clip(ceiling_y, 0, height - 2))
    floor_y = float(np.clip(floor_y, ceiling_y + 8, height - 1))

    return StructureBox(
        left_x=left_x,
        right_x=right_x,
        ceiling_y=ceiling_y,
        floor_y=floor_y,
        source=source,
        horizontal_lines=len(horizontals),
        vertical_lines=len(verticals),
    )


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


def draw_debug_overlay(
    rgb: np.ndarray,
    box: StructureBox,
    out_path: Path,
) -> None:
    canvas = rgb.copy()
    left = int(round(box.left_x))
    right = int(round(box.right_x))
    top = int(round(box.ceiling_y))
    bottom = int(round(box.floor_y))
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 220, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (left, bottom), (right, bottom), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (left, top), (right, top), (255, 128, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (left, top), (left, bottom), (255, 0, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (right, top), (right, bottom), (255, 0, 255), 2, cv2.LINE_AA)
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
    lines, detector = detect_mlsd_lines(rgb, score_thr, dist_thr)
    box = extract_structure_box(lines, w, h)
    masks = region_masks(box, w, h, region_inset_px)

    leveled, level_rot = leveled_points(depth, fx, fy, gravity)
    plane_info = fit_guided_planes(leveled, masks, max_samples, rng)
    plane_info = snap_manhattan_planes(plane_info, leveled, masks)

    floor_mask = masks["floor"]
    camera_height_raw = camera_height_from_floor_samples(leveled, floor_mask)
    depth_scale = 1.0
    scale_trusted = False
    if camera_height_raw is not None and 0.45 <= camera_height_raw <= 5.0:
        depth_scale = camera_height_prior / camera_height_raw
        depth = rescale_depth(depth, depth_scale)
        leveled, level_rot = leveled_points(depth, fx, fy, gravity)
        plane_info = fit_guided_planes(leveled, masks, max_samples, rng)
        plane_info = snap_manhattan_planes(plane_info, leveled, masks)
        camera_height_scaled = camera_height_from_floor_samples(leveled, floor_mask)
        scale_trusted = (
            camera_height_scaled is not None
            and CAMERA_HEIGHT_MIN_M <= camera_height_scaled <= CAMERA_HEIGHT_MAX_M
        )
    else:
        camera_height_scaled = camera_height_raw

    floor_plane: Plane | None = plane_info.get("floor", {}).get("plane")
    if floor_plane is None:
        return {
            "error": "floor_plane_fit_failed",
            "detector": detector,
            "structure_box": asdict(box),
            "planes": {k: {key: val for key, val in v.items() if key != "plane"} for k, v in plane_info.items()},
            "camera_height_raw_m": None if camera_height_raw is None else round(camera_height_raw, 3),
        }

    ceiling_plane = plane_info.get("ceiling", {}).get("plane")
    back_wall_plane = plane_info.get("back_wall", {}).get("plane")
    left_wall_plane = plane_info.get("left_wall", {}).get("plane")
    right_wall_plane = plane_info.get("right_wall", {}).get("plane")

    if floor_plane is None or ceiling_plane is None or back_wall_plane is None:
        return {
            "error": "required_plane_missing",
            "structure_box": asdict(box),
            "planes": {k: {key: val for key, val in v.items() if key != "plane"} for k, v in plane_info.items()},
            "camera_height_raw_m": None if camera_height_raw is None else round(camera_height_raw, 3),
            "depth_scale": round(depth_scale, 4),
            "scale_trusted": scale_trusted,
        }

    dims = measure_room_from_planes(
        floor_plane=floor_plane,
        ceiling_plane=ceiling_plane,
        back_wall_plane=back_wall_plane,
        left_wall_plane=left_wall_plane,
        right_wall_plane=right_wall_plane,
    )

    confidence = "low"
    if scale_trusted and dims.get("height_m") is not None:
        if ROOM_HEIGHT_MIN_M <= dims["height_m"] <= ROOM_HEIGHT_MAX_M:
            confidence = "medium"
        if dims.get("width_m") and dims.get("depth_m"):
            confidence = "high" if scale_trusted else "medium"

    planes_json = {
        name: {key: val for key, val in entry.items() if key != "plane"}
        for name, entry in plane_info.items()
    }

    return {
        "source": "structure_box_plane_intersections",
        "detector": detector,
        "line_segments": int(lines.shape[0]),
        "structure_box_px": {
            "left_x": round(box.left_x, 1),
            "right_x": round(box.right_x, 1),
            "ceiling_y": round(box.ceiling_y, 1),
            "floor_y": round(box.floor_y, 1),
            "box_source": box.source,
        },
        "planes": planes_json,
        "camera_height_raw_m": None if camera_height_raw is None else round(camera_height_raw, 3),
        "camera_height_scaled_m": None if camera_height_scaled is None else round(camera_height_scaled, 3),
        "camera_height_prior_m": round(camera_height_prior, 3),
        "depth_scale": round(depth_scale, 4),
        "scale_trusted": scale_trusted,
        "scale_gate": f"trust when camera_height in [{CAMERA_HEIGHT_MIN_M}, {CAMERA_HEIGHT_MAX_M}] m",
        "width_m": dims.get("width_m"),
        "height_m": dims.get("height_m"),
        "depth_m": dims.get("depth_m"),
        "w_h_d": (
            None
            if dims.get("width_m") is None or dims.get("height_m") is None or dims.get("depth_m") is None
            else f"{dims['width_m']:.3f} × {dims['height_m']:.3f} × {dims['depth_m']:.3f} m"
        ),
        "corners_leveled": dims.get("corners_leveled"),
        "confidence": confidence,
        "level_rotation": level_rot.tolist(),
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

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    box = StructureBox(
        left_x=result.get("structure_box_px", {}).get("left_x", width * 0.08),
        right_x=result.get("structure_box_px", {}).get("right_x", width * 0.92),
        ceiling_y=result.get("structure_box_px", {}).get("ceiling_y", height * 0.08),
        floor_y=result.get("structure_box_px", {}).get("floor_y", height * 0.92),
        source=result.get("structure_box_px", {}).get("box_source", "unknown"),
        horizontal_lines=0,
        vertical_lines=0,
    )
    overlay_path = out_dir / f"{stem}_structure_box.png"
    draw_debug_overlay(rgb, box, overlay_path)

    payload = {
        "image": str(image_path),
        "geocalib_json": str(args.geocalib_json.expanduser().resolve()),
        "focal_px": round(fx, 2),
        "structure_box_measurement": result,
        "overlay_png": str(overlay_path),
    }

    json_path = out_dir / f"{stem}_structure_box.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\noverlay: {overlay_path}")
    print(f"json:    {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
