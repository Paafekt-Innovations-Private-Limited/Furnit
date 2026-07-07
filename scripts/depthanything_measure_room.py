#!/usr/bin/env python3
"""Estimate depth and projected frame dimensions from Depth Anything V2 metric depth.

Offline reference for the iOS pipeline: GeoCalib focal (see scripts/run_geocalib.py)
+ Depth Anything metric depth + pinhole unprojection. Pair with geocalib sidecar when
EXIF/chair-anchor calibration is needed on device.

No manual measurements required. Auto-writes:
  - measurements JSON
  - depth preview PNG
  - textured room mesh GLB

Example:
  python3 scripts/depthanything_measure_room.py --image /Users/al/Downloads/USRoom.jpeg
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image, ImageOps

DEFAULT_IMAGE = Path("/Users/al/Downloads/USRoom.jpeg")
DEFAULT_ONNX = (
    Path(__file__).resolve().parents[1]
    / "Furnit/Models/DepthAnything/DepthAnythingV2MetricIndoorSmall.onnx"
)
DEFAULT_OUT_DIR = Path("/tmp/depthanything_splat")
DEFAULT_INPUT_SIZE = 518
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class RoomMeasurements:
    width_m: float
    height_m: float
    depth_m: float
    image_width: int
    image_height: int
    intrinsics_source: str
    focal_px: float
    depth_median_m: float
    depth_min_m: float
    depth_max_m: float
    wall_rect_norm: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single photo -> Depth Anything metric room dimensions (meters)."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument(
        "--wall-margin",
        type=float,
        default=0.05,
        help="Fraction inset from each image edge when estimating visible wall (default 5%%).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Folder for auto-written outputs (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Override measurements JSON path (default: <out-dir>/<image>_measurements.json).",
    )
    parser.add_argument(
        "--depth-png",
        type=Path,
        default=None,
        help="Override depth PNG path (default: <out-dir>/<image>_depth.png).",
    )
    parser.add_argument("--no-depth-png", action="store_true", help="Skip writing depth preview PNG.")
    parser.add_argument("--no-glb", action="store_true", help="Skip writing textured room GLB.")
    parser.add_argument(
        "--mesh-step",
        type=int,
        default=2,
        help="Pixel stride when building GLB mesh grid (default 2).",
    )
    parser.add_argument(
        "--max-depth-jump",
        type=float,
        default=0.15,
        help="Skip mesh triangles across depth edges larger than this many meters.",
    )
    parser.add_argument(
        "--flat-mesh",
        action="store_true",
        help="Put the photo on a single flat plane (no depth relief). Cleanest texture, no drag.",
    )
    parser.add_argument(
        "--depth-smooth",
        type=int,
        default=7,
        help="Median-filter kernel for depth before meshing (0=off). Reduces wall rippling.",
    )
    parser.add_argument(
        "--geocalib-json",
        type=Path,
        default=None,
        help="Optional JSON from scripts/run_geocalib.py for gravity/focal room-box prototype.",
    )
    parser.add_argument(
        "--room-box-prototype",
        action="store_true",
        help="Also run gravity-leveled Manhattan room-box prototype instead of only frustum wall math.",
    )
    parser.add_argument(
        "--camera-height-prior",
        type=float,
        default=1.70,
        help="Handheld camera-height prior used to rescale depth in prototype (default 1.70m).",
    )
    parser.add_argument(
        "--tile-grid-prototype",
        action="store_true",
        help="Detect floor tile/grid lines and estimate dimensions from an assumed tile size.",
    )
    parser.add_argument(
        "--tile-size-m",
        type=float,
        default=0.60,
        help="Assumed square tile size for --tile-grid-prototype (default 0.60m).",
    )
    parser.add_argument(
        "--tile-width-count",
        type=float,
        default=None,
        help="Manual/visual tile count across room width. Produces authoritative tile-count W×H×D when all three counts are supplied.",
    )
    parser.add_argument(
        "--tile-height-count",
        type=float,
        default=None,
        help="Manual/visual tile count from floor to ceiling.",
    )
    parser.add_argument(
        "--tile-depth-count",
        type=float,
        default=None,
        help="Manual/visual tile count from camera/front side to back wall.",
    )
    return parser.parse_args()


def default_output_paths(image_path: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    stem = image_path.expanduser().resolve().stem
    folder = out_dir.expanduser().resolve()
    return (
        folder / f"{stem}_measurements.json",
        folder / f"{stem}_depth.png",
        folder / f"{stem}_room.glb",
    )


def read_rgb(path: Path) -> Image.Image:
    image = Image.open(path.expanduser().resolve())
    return ImageOps.exif_transpose(image).convert("RGB")


def focal_from_exif(path: Path, image_width: int, image_height: int) -> tuple[float, float, str]:
    fallback_35mm = 28.0
    try:
        exif = Image.open(path.expanduser().resolve()).getexif()
    except Exception:
        exif = {}
    tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
    focal_35mm = tags.get("FocalLengthIn35mmFilm") or tags.get("FocalLenIn35mmFilm")
    if focal_35mm and float(focal_35mm) > 1:
        focal_px = (float(focal_35mm) / 36.0) * image_width
        return focal_px, focal_px, f"exif_35mm_equiv_{float(focal_35mm):.1f}mm"
    focal_px = (fallback_35mm / 36.0) * image_width
    return focal_px, focal_px, f"fallback_35mm_equiv_{fallback_35mm:.1f}mm"


def preprocess(image: Image.Image, input_size: int) -> np.ndarray:
    resized = image.resize((input_size, input_size), Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def run_depth_onnx(image: Image.Image, onnx_path: Path, input_size: int) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path.expanduser().resolve()),
        providers=["CPUExecutionProvider"],
    )
    inp = session.get_inputs()[0].name
    out = session.get_outputs()[0].name
    raw = session.run([out], {inp: preprocess(image, input_size)})[0]
    depth = np.asarray(raw, dtype=np.float32).squeeze()
    w, h = image.size
    depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
    depth_img = depth_img.resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(depth_img, dtype=np.float32)


def wall_rect_px(image_width: int, image_height: int, margin: float) -> tuple[float, float, float, float]:
    m = max(0.0, min(margin, 0.45))
    x = m * image_width
    y = m * image_height
    w = (1.0 - 2.0 * m) * image_width
    h = (1.0 - 2.0 * m) * image_height
    return x, y, w, h


def median_at(depth: np.ndarray, px: int, py: int, radius: int = 5) -> float | None:
    h, w = depth.shape
    patch = depth[max(0, py - radius): min(h, py + radius + 1), max(0, px - radius): min(w, px + radius + 1)]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(valid)) if valid.size else None


def measure_wall(
    depth: np.ndarray,
    rect: tuple[float, float, float, float],
    fx: float,
    fy: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float]:
    x, y, rw, rh = rect
    cx = intrinsics_cx = (image_width - 1) * 0.5
    cy = intrinsics_cy = (image_height - 1) * 0.5
    left_x = int(np.clip(round(x), 0, image_width - 1))
    right_x = int(np.clip(round(x + rw - 1), 0, image_width - 1))
    top_y = int(np.clip(round(y), 0, image_height - 1))
    bottom_y = int(np.clip(round(y + rh - 1), 0, image_height - 1))
    center_x = int(np.clip(round(x + rw * 0.5), 0, image_width - 1))
    center_y = int(np.clip(round(y + rh * 0.5), 0, image_height - 1))

    center_depth = median_at(depth, center_x, center_y)
    if center_depth is None:
        raise ValueError("Could not sample depth at image center — depth map may be empty.")

    left_plane = (left_x - cx) * center_depth / fx
    right_plane = (right_x - cx) * center_depth / fx
    top_plane = (top_y - cy) * center_depth / fy
    bottom_plane = (bottom_y - cy) * center_depth / fy
    return (
        abs(right_plane - left_plane),
        abs(bottom_plane - top_plane),
        center_depth,
    )


def depth_summary(depth: np.ndarray) -> tuple[float, float, float]:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return math.nan, math.nan, math.nan
    return float(valid.min()), float(np.median(valid)), float(valid.max())


def rotation_matrix_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source.astype(np.float32)
    target = target.astype(np.float32)
    source = source / max(float(np.linalg.norm(source)), 1e-6)
    target = target / max(float(np.linalg.norm(target)), 1e-6)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 0.9999:
        return np.eye(3, dtype=np.float32)
    if dot < -0.9999:
        return np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    axis = np.cross(source, target)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-6)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + math.sin(math.acos(dot)) * k + (1.0 - dot) * (k @ k)


def unproject_depth(depth: np.ndarray, fx: float, fy: float) -> np.ndarray:
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    z = depth.astype(np.float32)
    x = (xx - cx) * z / float(fx)
    y = (yy - cy) * z / float(fy)
    points = np.stack([x, y, z], axis=-1)
    points[~(np.isfinite(z) & (z > 0))] = np.nan
    return points


def leveled_points(depth: np.ndarray, fx: float, fy: float, gravity: list[float]) -> tuple[np.ndarray, np.ndarray]:
    points = unproject_depth(depth, fx, fy)
    # GeoCalib gravity vector is the same leveling reference used in Swift: at rest it is (0, -1, 0).
    rot = rotation_matrix_from_to(np.asarray(gravity, dtype=np.float32), np.array([0, -1, 0], dtype=np.float32))
    leveled = points @ rot.T
    return leveled, rot


def grid_normals(points: np.ndarray) -> np.ndarray:
    h, w, _ = points.shape
    normals = np.full_like(points, np.nan, dtype=np.float32)
    left = points[1:-1, :-2]
    right = points[1:-1, 2:]
    up = points[:-2, 1:-1]
    down = points[2:, 1:-1]
    dx = right - left
    dy = down - up
    normal = np.cross(dx, dy)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    good = np.isfinite(norm[..., 0]) & (norm[..., 0] > 1e-5)
    normal = np.divide(normal, np.maximum(norm, 1e-6))
    normals[1:-1, 1:-1][good] = normal[good]
    return normals


def strongest_peak(values: np.ndarray, bin_size: float = 0.05, minimum_count: int = 20) -> tuple[float, int] | None:
    values = values[np.isfinite(values)]
    if values.size < minimum_count:
        return None
    lo, hi = np.percentile(values, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    bins = max(8, min(512, int(math.ceil((hi - lo) / bin_size)) + 1))
    hist, edges = np.histogram(values[(values >= lo) & (values <= hi)], bins=bins, range=(lo, hi))
    smooth = np.convolve(hist, np.ones(3, dtype=np.int32), mode="same")
    idx = int(np.argmax(smooth))
    count = int(smooth[idx])
    if count < minimum_count:
        return None
    return float((edges[idx] + edges[idx + 1]) * 0.5), count


def axis_extent(values: np.ndarray, require_two_sides: bool) -> tuple[float, int] | None:
    values = values[np.isfinite(values)]
    if values.size < 24:
        return None
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        return None
    hist, edges = np.histogram(values[(values >= lo) & (values <= hi)], bins=max(8, min(512, int(math.ceil((hi - lo) / 0.05)) + 1)))
    smooth = np.convolve(hist, np.ones(3, dtype=np.int32), mode="same")
    nz = smooth[smooth > 0]
    if nz.size == 0:
        return None
    threshold = max(8, int(values.size * 0.03), int(np.median(nz) * 2))
    peaks = np.flatnonzero(smooth >= threshold)
    if peaks.size == 0:
        return None
    first, last = int(peaks[0]), int(peaks[-1])
    if require_two_sides and first == last:
        return None
    low = float((edges[first] + edges[first + 1]) * 0.5)
    high = float((edges[last] + edges[last + 1]) * 0.5)
    extent = abs(high - low) if require_two_sides else max(abs(high), abs(high - low))
    if not np.isfinite(extent) or extent <= 0:
        return None
    return extent, int(smooth[first] + (0 if last == first else smooth[last]))


def prototype_room_box(
    depth: np.ndarray,
    fx: float,
    fy: float,
    gravity: list[float],
    camera_height_prior: float,
) -> dict:
    leveled, _ = leveled_points(depth, fx, fy, gravity)
    normals = grid_normals(leveled)
    valid = np.isfinite(leveled[..., 2])

    horizontal = valid & np.isfinite(normals[..., 1]) & (np.abs(normals[..., 1]) >= 0.85)
    vertical = valid & np.isfinite(normals[..., 1]) & (np.abs(normals[..., 1]) <= 0.45)
    y_values = leveled[..., 1]
    floor_candidates = y_values[horizontal & (y_values > 0)]
    floor_peak = strongest_peak(floor_candidates, bin_size=0.05, minimum_count=20)
    camera_height_raw = floor_peak[0] if floor_peak else math.nan
    scale = 1.0
    if math.isfinite(camera_height_raw) and 0.45 <= camera_height_raw <= 5.0:
        scale = camera_height_prior / camera_height_raw
        depth = depth * scale
        leveled, _ = leveled_points(depth, fx, fy, gravity)
        normals = grid_normals(leveled)
        valid = np.isfinite(leveled[..., 2])
        horizontal = valid & np.isfinite(normals[..., 1]) & (np.abs(normals[..., 1]) >= 0.85)
        vertical = valid & np.isfinite(normals[..., 1]) & (np.abs(normals[..., 1]) <= 0.45)
        y_values = leveled[..., 1]
        floor_candidates = y_values[horizontal & (y_values > 0)]
        floor_peak = strongest_peak(floor_candidates, bin_size=0.05, minimum_count=20)

    ceiling_peak = strongest_peak(y_values[horizontal & (y_values < 0)], bin_size=0.05, minimum_count=20)
    height_source = "prior"
    height = 2.40
    if floor_peak and ceiling_peak:
        measured_height = abs(floor_peak[0] - ceiling_peak[0])
        if 1.8 <= measured_height <= 4.5:
            height = measured_height
            height_source = "measured_floor_ceiling"
    elif floor_peak:
        height = camera_height_prior + 1.0
        height_source = "camera_height_plus_headroom_estimate"

    wall_points = leveled[vertical]
    wall_normals = normals[vertical]
    floor_footprint = None
    if floor_peak:
        floor_band = horizontal & (np.abs(y_values - floor_peak[0]) <= 0.12)
        floor_points = leveled[floor_band]
        if floor_points.shape[0] >= 64:
            x_lo, x_hi = np.percentile(floor_points[:, 0], [2, 98])
            z_lo, z_hi = np.percentile(floor_points[:, 2], [2, 98])
            floor_footprint = {
                "width_m": float(abs(x_hi - x_lo)),
                "depth_m": float(abs(z_hi - z_lo)),
                "samples": int(floor_points.shape[0]),
            }
    width = math.nan
    room_depth = math.nan
    yaw = math.nan
    yaw_source = "none"
    wall_support = int(wall_points.shape[0])
    if wall_points.shape[0] >= 32:
        hn = wall_normals[:, [0, 2]]
        hn_norm = np.linalg.norm(hn, axis=1)
        good = hn_norm > 1e-4
        wall_points = wall_points[good]
        hn = hn[good] / hn_norm[good, None]
        angles = np.mod(np.arctan2(hn[:, 1], hn[:, 0]), math.pi / 2)
        hist, edges = np.histogram(angles, bins=36, range=(0, math.pi / 2))
        if hist.max() >= max(8, angles.size // 6):
            yaw = float((edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]) * 0.5)
            yaw_source = "dominant_normal_histogram"
        else:
            yaw = 0.0
            yaw_source = "fallback_camera_axes"
        if math.isfinite(yaw):
            c, s = math.cos(-yaw), math.sin(-yaw)
            xz = wall_points[:, [0, 2]]
            rx = c * xz[:, 0] - s * xz[:, 1]
            rz = s * xz[:, 0] + c * xz[:, 1]
            x_extent = axis_extent(rx, require_two_sides=True)
            z_extent = axis_extent(rz, require_two_sides=False)
            if x_extent:
                width = x_extent[0]
            if z_extent:
                room_depth = z_extent[0]

    if floor_footprint:
        if not math.isfinite(width) or width < 0.6:
            width = floor_footprint["width_m"]
        if not math.isfinite(room_depth) or room_depth < 0.6:
            room_depth = floor_footprint["depth_m"]

    return {
        "width_m": None if not math.isfinite(width) else round(width, 3),
        "height_m": round(height, 3),
        "depth_m": None if not math.isfinite(room_depth) else round(room_depth, 3),
        "height_source": height_source,
        "depth_scale_from_camera_height": round(scale, 4),
        "camera_height_raw_m": None if not math.isfinite(camera_height_raw) else round(camera_height_raw, 3),
        "floor_peak": None if not floor_peak else {"y": round(floor_peak[0], 3), "count": floor_peak[1]},
        "ceiling_peak": None if not ceiling_peak else {"y": round(ceiling_peak[0], 3), "count": ceiling_peak[1]},
        "horizontal_samples": int(np.count_nonzero(horizontal)),
        "vertical_samples": wall_support,
        "manhattan_yaw_rad": None if not math.isfinite(yaw) else round(yaw, 4),
        "manhattan_yaw_source": yaw_source,
        "floor_footprint": None if not floor_footprint else {
            "width_m": round(floor_footprint["width_m"], 3),
            "depth_m": round(floor_footprint["depth_m"], 3),
            "samples": floor_footprint["samples"],
        },
    }


def angle_distance_mod_pi(a: float, b: float) -> float:
    d = abs((a - b + math.pi / 2) % math.pi - math.pi / 2)
    return float(d)


def cluster_offsets(values: list[float], merge_distance: float) -> list[tuple[float, int]]:
    if not values:
        return []
    values = sorted(v for v in values if math.isfinite(v))
    clusters: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - float(np.mean(clusters[-1]))) <= merge_distance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [(float(np.mean(cluster)), len(cluster)) for cluster in clusters]


def tile_grid_from_bev(
    rgb: np.ndarray,
    leveled: np.ndarray,
    floor_band: np.ndarray,
    tile_size_m: float,
    debug_path: Path | None = None,
) -> dict:
    """Rectify floor pixels into a top-down XZ raster and estimate tile spacing there."""
    try:
        import cv2
    except Exception as exc:
        return {"error": f"opencv_unavailable: {exc}"}

    coords = leveled[floor_band]
    colors = rgb[floor_band]
    if coords.shape[0] < 512:
        return {"error": "not_enough_floor_pixels_for_bev", "samples": int(coords.shape[0])}

    x_lo, x_hi = np.percentile(coords[:, 0], [1, 99])
    z_lo, z_hi = np.percentile(coords[:, 2], [1, 99])
    x_span = float(x_hi - x_lo)
    z_span = float(z_hi - z_lo)
    if not (math.isfinite(x_span) and math.isfinite(z_span)) or x_span <= 0 or z_span <= 0:
        return {"error": "invalid_bev_extent"}

    max_side = 900
    pixels_per_raw_meter = min(max_side / max(x_span, z_span), 280.0)
    bev_w = max(32, int(math.ceil(x_span * pixels_per_raw_meter)))
    bev_h = max(32, int(math.ceil(z_span * pixels_per_raw_meter)))
    bev_accum = np.zeros((bev_h, bev_w, 3), dtype=np.uint32)
    count = np.zeros((bev_h, bev_w), dtype=np.uint16)

    ix = np.clip(((coords[:, 0] - x_lo) * pixels_per_raw_meter).astype(np.int32), 0, bev_w - 1)
    iz = np.clip(((coords[:, 2] - z_lo) * pixels_per_raw_meter).astype(np.int32), 0, bev_h - 1)
    # Flip Z for display so farther floor is visually upward in the BEV image.
    iy = bev_h - 1 - iz
    np.add.at(bev_accum, (iy, ix), colors.astype(np.uint32))
    np.add.at(count, (iy, ix), 1)
    mask = count > 0
    bev = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    bev[mask] = (bev_accum[mask].astype(np.float32) / count[mask, None]).astype(np.uint8)
    mask_u8 = (mask.astype(np.uint8) * 255)
    if np.count_nonzero(mask_u8) > 0:
        bev = cv2.inpaint(bev, 255 - mask_u8, 3, cv2.INPAINT_TELEA)

    gray = cv2.cvtColor(bev, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 140, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=35,
        minLineLength=max(24, min(bev_w, bev_h) // 16),
        maxLineGap=14,
    )

    debug = bev.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(debug, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
    if debug_path is not None:
        debug_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(debug).save(debug_path.expanduser().resolve())

    if lines is None:
        return {
            "error": "no_bev_hough_lines",
            "bev_size_px": [bev_w, bev_h],
            "floor_samples": int(coords.shape[0]),
            "debug_bev_png": str(debug_path.expanduser().resolve()) if debug_path else None,
        }

    segments = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < 20:
            continue
        angle = math.atan2(dy, dx) % math.pi
        midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        segments.append({"angle": angle, "midpoint": midpoint, "length": length})

    if len(segments) < 4:
        return {
            "error": "not_enough_bev_segments",
            "hough_lines": int(lines.shape[0]),
            "usable_segments": len(segments),
            "debug_bev_png": str(debug_path.expanduser().resolve()) if debug_path else None,
        }

    angle_mod = np.array([segment["angle"] % (math.pi / 2) for segment in segments], dtype=np.float32)
    hist, angle_edges = np.histogram(angle_mod, bins=36, range=(0, math.pi / 2))
    yaw = float((angle_edges[int(np.argmax(hist))] + angle_edges[int(np.argmax(hist)) + 1]) * 0.5)
    u = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    v = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
    offsets_u: list[float] = []
    offsets_v: list[float] = []
    for segment in segments:
        angle = segment["angle"]
        midpoint = segment["midpoint"]
        if angle_distance_mod_pi(angle, yaw) <= math.radians(18):
            offsets_v.append(float(np.dot(midpoint, v)))
        elif angle_distance_mod_pi(angle, yaw + math.pi / 2) <= math.radians(18):
            offsets_u.append(float(np.dot(midpoint, u)))

    clusters_u = cluster_offsets(offsets_u, merge_distance=10.0)
    clusters_v = cluster_offsets(offsets_v, merge_distance=10.0)

    def spacing_px(clusters: list[tuple[float, int]]) -> float | None:
        if len(clusters) < 2:
            return None
        centers = [center for center, _ in clusters]
        diffs = [b - a for a, b in zip(centers, centers[1:]) if 8 <= (b - a) <= 260]
        return float(np.median(diffs)) if diffs else None

    spacing_u = spacing_px(clusters_u)
    spacing_v = spacing_px(clusters_v)
    spacings = [value for value in (spacing_u, spacing_v) if value is not None]
    if not spacings:
        return {
            "error": "bev_tile_spacing_not_found",
            "usable_segments": len(segments),
            "clusters_u": len(clusters_u),
            "clusters_v": len(clusters_v),
            "debug_bev_png": str(debug_path.expanduser().resolve()) if debug_path else None,
        }

    raw_tile_spacing_px = float(np.median(spacings))
    meters_per_pixel = tile_size_m / raw_tile_spacing_px
    return {
        "source": "bev_rectified_tile_grid",
        "bev_size_px": [bev_w, bev_h],
        "hough_lines": int(lines.shape[0]),
        "usable_segments": len(segments),
        "yaw_rad": round(yaw, 4),
        "tile_spacing_px": round(raw_tile_spacing_px, 2),
        "meters_per_bev_pixel": round(meters_per_pixel, 5),
        "floor_axis_u_m": round(bev_w * meters_per_pixel, 3),
        "floor_axis_v_m": round(bev_h * meters_per_pixel, 3),
        "clusters_u": [{"offset_px": round(c, 1), "count": n} for c, n in clusters_u],
        "clusters_v": [{"offset_px": round(c, 1), "count": n} for c, n in clusters_v],
        "debug_bev_png": str(debug_path.expanduser().resolve()) if debug_path else None,
    }


def manual_tile_count_measurement(
    width_count: float | None,
    height_count: float | None,
    depth_count: float | None,
    tile_size_m: float,
) -> dict | None:
    if width_count is None or height_count is None or depth_count is None:
        return None
    if min(width_count, height_count, depth_count, tile_size_m) <= 0:
        return {
            "source": "manual_tile_count",
            "error": "tile counts and tile size must be positive",
        }
    width_m = width_count * tile_size_m
    height_m = height_count * tile_size_m
    depth_m = depth_count * tile_size_m
    return {
        "source": "manual_tile_count",
        "order": "width_height_depth",
        "tile_size_m": round(tile_size_m, 3),
        "tile_counts": {
            "width": round(width_count, 3),
            "height": round(height_count, 3),
            "depth": round(depth_count, 3),
        },
        "width_m": round(width_m, 3),
        "height_m": round(height_m, 3),
        "depth_m": round(depth_m, 3),
        "w_h_d": f"{width_m:.3f} × {height_m:.3f} × {depth_m:.3f} m",
        "note": "Direct tile-count estimate: dimensions = visible tile count × assumed tile size. This bypasses Depth Anything scale.",
    }


def tile_grid_prototype(
    image: Image.Image,
    depth: np.ndarray,
    fx: float,
    fy: float,
    gravity: list[float],
    tile_size_m: float,
    debug_bev_path: Path | None = None,
) -> dict:
    try:
        import cv2
    except Exception as exc:
        return {"error": f"opencv_unavailable: {exc}", "source": "measured_from_tile_grid"}

    leveled_raw, _ = leveled_points(depth, fx, fy, gravity)
    normals_raw = grid_normals(leveled_raw)
    valid = np.isfinite(leveled_raw[..., 2])
    horizontal = valid & np.isfinite(normals_raw[..., 1]) & (np.abs(normals_raw[..., 1]) >= 0.85)
    y_values = leveled_raw[..., 1]
    floor_peak = strongest_peak(y_values[horizontal & (y_values > 0)], bin_size=0.05, minimum_count=20)
    if not floor_peak:
        return {"error": "floor_peak_not_found", "source": "measured_from_tile_grid"}

    floor_band = horizontal & (np.abs(y_values - floor_peak[0]) <= 0.22)
    floor_points = leveled_raw[floor_band]
    if floor_points.shape[0] < 64:
        return {
            "error": "not_enough_floor_points",
            "source": "measured_from_tile_grid",
            "floor_samples": int(floor_points.shape[0]),
        }

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    floor_mask = (floor_band.astype(np.uint8) * 255)
    floor_mask = cv2.dilate(floor_mask, np.ones((7, 7), np.uint8), iterations=1)
    edges = cv2.bitwise_and(edges, edges, mask=floor_mask)

    h, w = depth.shape
    min_line_length = max(35, int(min(w, h) * 0.04))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=40,
        minLineLength=min_line_length,
        maxLineGap=18,
    )
    if lines is None:
        return {
            "error": "no_hough_lines",
            "source": "measured_from_tile_grid",
            "floor_samples": int(floor_points.shape[0]),
        }

    segments: list[dict] = []
    for raw_line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in raw_line]
        if not (0 <= x1 < w and 0 <= x2 < w and 0 <= y1 < h and 0 <= y2 < h):
            continue

        sample_count = 13
        xs = np.linspace(x1, x2, sample_count).round().astype(np.int32)
        ys = np.linspace(y1, y2, sample_count).round().astype(np.int32)
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        mask_hits = floor_mask[ys, xs] > 0
        if np.count_nonzero(mask_hits) < max(5, sample_count // 2):
            continue

        samples = leveled_raw[ys, xs]
        finite = np.isfinite(samples).all(axis=1)
        near_floor = np.abs(samples[:, 1] - floor_peak[0]) <= 0.35
        good = mask_hits & finite & near_floor
        samples = samples[good]
        if samples.shape[0] < 5:
            continue

        xz = samples[:, [0, 2]].astype(np.float32)
        midpoint = np.mean(xz, axis=0)
        centered = xz - midpoint
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        direction = vh[0].astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        projections = centered @ direction
        length = float(np.percentile(projections, 95) - np.percentile(projections, 5))
        if length < 0.20 or length > 4.0:
            continue
        angle = math.atan2(float(direction[1]), float(direction[0])) % math.pi
        segments.append({"angle": angle, "midpoint": midpoint, "length": length, "sample_count": int(samples.shape[0])})

    if len(segments) < 4:
        return {
            "error": "not_enough_floor_grid_segments",
            "source": "measured_from_tile_grid",
            "hough_lines": int(lines.shape[0]),
            "usable_segments": len(segments),
            "floor_samples": int(floor_points.shape[0]),
        }

    angle_mod = np.array([segment["angle"] % (math.pi / 2) for segment in segments], dtype=np.float32)
    hist, edges_angle = np.histogram(angle_mod, bins=36, range=(0, math.pi / 2))
    best_index = int(np.argmax(hist))
    yaw = float((edges_angle[best_index] + edges_angle[best_index + 1]) * 0.5)
    u = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    v = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)

    offsets_u: list[float] = []
    offsets_v: list[float] = []
    for segment in segments:
        angle = segment["angle"]
        midpoint = segment["midpoint"]
        if angle_distance_mod_pi(angle, yaw) <= math.radians(20):
            # Line is parallel to u, so adjacent tiles change along v.
            offsets_v.append(float(np.dot(midpoint, v)))
        elif angle_distance_mod_pi(angle, yaw + math.pi / 2) <= math.radians(20):
            # Line is parallel to v, so adjacent tiles change along u.
            offsets_u.append(float(np.dot(midpoint, u)))

    clusters_u = cluster_offsets(offsets_u, merge_distance=0.10)
    clusters_v = cluster_offsets(offsets_v, merge_distance=0.10)

    def cluster_spacing(clusters: list[tuple[float, int]]) -> float | None:
        if len(clusters) < 2:
            return None
        centers = [center for center, count in clusters if count >= 1]
        diffs = [b - a for a, b in zip(centers, centers[1:]) if 0.12 <= (b - a) <= 3.0]
        return float(np.median(diffs)) if diffs else None

    spacing_u = cluster_spacing(clusters_u)
    spacing_v = cluster_spacing(clusters_v)
    spacings = [value for value in (spacing_u, spacing_v) if value is not None and value > 0]
    if not spacings:
        return {
            "error": "tile_spacing_not_found",
            "source": "measured_from_tile_grid",
            "usable_segments": len(segments),
            "clusters_u": len(clusters_u),
            "clusters_v": len(clusters_v),
        }

    raw_tile_spacing = float(np.median(spacings))
    tile_scale = tile_size_m / raw_tile_spacing

    def tiled_extent(clusters: list[tuple[float, int]], raw_spacing: float) -> float | None:
        if len(clusters) >= 2:
            return (clusters[-1][0] - clusters[0][0] + raw_spacing) * tile_scale
        return None

    width_from_lines = tiled_extent(clusters_u, raw_tile_spacing)
    depth_from_lines = tiled_extent(clusters_v, raw_tile_spacing)
    x_lo, x_hi = np.percentile(floor_points[:, 0], [2, 98])
    z_lo, z_hi = np.percentile(floor_points[:, 2], [2, 98])
    width_from_floor = abs(float(x_hi - x_lo)) * tile_scale
    depth_from_floor = abs(float(z_hi - z_lo)) * tile_scale
    bev_grid = tile_grid_from_bev(
        rgb=rgb,
        leveled=leveled_raw,
        floor_band=floor_band,
        tile_size_m=tile_size_m,
        debug_path=debug_bev_path,
    )

    height_raw = None
    ceiling_peak = strongest_peak(y_values[horizontal & (y_values < 0)], bin_size=0.05, minimum_count=20)
    if ceiling_peak:
        height_raw = abs(floor_peak[0] - ceiling_peak[0])
    height = height_raw * tile_scale if height_raw else None

    return {
        "source": "measured_from_tile_grid",
        "tile_size_m": tile_size_m,
        "raw_tile_spacing_m": round(raw_tile_spacing, 3),
        "tile_depth_scale": round(tile_scale, 4),
        "floor_axis_u_m": None if width_from_lines is None else round(width_from_lines, 3),
        "floor_axis_v_m": None if depth_from_lines is None else round(depth_from_lines, 3),
        "vertical_scaled_candidate_m": None if height is None else round(height, 3),
        "note": "Tile grid measures floor axes only. Do not use vertical_scaled_candidate_m as room height unless independently validated.",
        "floor_footprint_scaled": {
            "width_m": round(width_from_floor, 3),
            "depth_m": round(depth_from_floor, 3),
        },
        "bev_rectified_grid": bev_grid,
        "floor_peak_raw_y": round(floor_peak[0], 3),
        "ceiling_peak_raw_y": None if not ceiling_peak else round(ceiling_peak[0], 3),
        "hough_lines": int(lines.shape[0]),
        "usable_segments": len(segments),
        "yaw_rad": round(yaw, 4),
        "clusters_u": [{"offset": round(c, 3), "count": n} for c, n in clusters_u],
        "clusters_v": [{"offset": round(c, 3), "count": n} for c, n in clusters_v],
    }


def save_depth_png(depth: np.ndarray, path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    lo, hi = np.percentile(valid, [2, 98])
    scaled = np.clip((depth - lo) / max(float(hi - lo), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(scaled).save(path)


def smooth_depth_for_mesh(depth: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return depth
    from scipy.ndimage import median_filter

    valid = np.isfinite(depth) & (depth > 0)
    smoothed = median_filter(np.where(valid, depth, 0.0), size=kernel)
    counts = median_filter(valid.astype(np.float32), size=kernel)
    out = depth.copy()
    mask = counts > 0.5
    out[mask] = smoothed[mask]
    out[~valid] = np.nan
    return out


def flatten_depth_for_mesh(depth: np.ndarray) -> np.ndarray:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return depth
    plane_depth = float(np.median(valid))
    out = depth.copy()
    out[np.isfinite(out) & (out > 0)] = plane_depth
    return out


def build_textured_mesh_glb(
    rgb_image: Image.Image,
    depth: np.ndarray,
    room_width_m: float,
    glb_path: Path,
    mesh_step: int,
    max_depth_jump: float,
) -> tuple[int, int]:
    """Build a flat/relief mesh aligned with DepthAnythingRoomReconstructor on iOS.

    XY is proportional to pixel position. Z is depth relief only (or flat if depth is constant).
    Uses UV texture mapping so colors stay pinned to pixels instead of smearing via vertex colors.
    """
    import trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    img_np = np.asarray(rgb_image, dtype=np.uint8)
    img_h, img_w = depth.shape
    step = max(1, mesh_step)

    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    depth_max = float(np.max(valid_depth))
    if not math.isfinite(depth_max) or depth_max <= 0:
        raise ValueError("Depth map has no valid samples for mesh export.")

    pixel_scale = room_width_m / float(img_w)
    center_x = img_w * 0.5
    center_y = img_h * 0.5

    rows = np.arange(0, img_h, step, dtype=np.int32)
    cols = np.arange(0, img_w, step, dtype=np.int32)
    grid_h, grid_w = len(rows), len(cols)

    vertex_indices = np.full(grid_h * grid_w, -1, dtype=np.int32)
    vertices: list[list[float]] = []
    uvs: list[list[float]] = []

    for ri, row in enumerate(rows):
        for ci, col in enumerate(cols):
            d = float(depth[row, col])
            if not math.isfinite(d) or d <= 0:
                continue
            vertex_indices[ri * grid_w + ci] = len(vertices)
            vertices.append([
                -(float(col) - center_x) * pixel_scale,
                (float(row) - center_y) * pixel_scale,
                -(depth_max - d),
            ])
            uvs.append([
                float(col) / max(img_w - 1, 1),
                1.0 - float(row) / max(img_h - 1, 1),
            ])

    if not vertices:
        raise ValueError("No valid depth vertices for mesh export.")

    faces: list[list[int]] = []
    for ri in range(grid_h - 1):
        for ci in range(grid_w - 1):
            i00 = ri * grid_w + ci
            i10 = ri * grid_w + (ci + 1)
            i01 = (ri + 1) * grid_w + ci
            i11 = (ri + 1) * grid_w + (ci + 1)
            v00 = int(vertex_indices[i00])
            v10 = int(vertex_indices[i10])
            v01 = int(vertex_indices[i01])
            v11 = int(vertex_indices[i11])
            if min(v00, v10, v01, v11) < 0:
                continue

            d00 = float(depth[rows[ri], cols[ci]])
            d10 = float(depth[rows[ri], cols[ci + 1]])
            d01 = float(depth[rows[ri + 1], cols[ci]])
            d11 = float(depth[rows[ri + 1], cols[ci + 1]])
            depths = (d00, d10, d01, d11)
            if not all(math.isfinite(v) and v > 0 for v in depths):
                continue
            if max(abs(a - b) for i, a in enumerate(depths) for b in depths[i + 1:]) > max_depth_jump:
                continue

            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    if not faces:
        raise ValueError("Could not build any mesh faces from depth map.")

    material = PBRMaterial(baseColorTexture=rgb_image, metallicFactor=0.0, roughnessFactor=1.0)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        process=False,
    )
    mesh.visual = TextureVisuals(uv=np.asarray(uvs, dtype=np.float32), material=material)
    glb_path = glb_path.expanduser().resolve()
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="room")
    scene.export(glb_path)
    return len(vertices), len(faces)


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    json_path, depth_png_path, glb_path = default_output_paths(image_path, args.out_dir)
    if args.out_json:
        json_path = args.out_json.expanduser().resolve()
    if args.depth_png:
        depth_png_path = args.depth_png.expanduser().resolve()

    image = read_rgb(image_path)
    width, height = image.size
    fx, fy, focal_source = focal_from_exif(image_path, width, height)
    depth = run_depth_onnx(image, args.onnx, args.input_size)
    d_min, d_median, d_max = depth_summary(depth)

    margin = args.wall_margin
    rect = wall_rect_px(width, height, margin)
    width_m, height_m, depth_m = measure_wall(depth, rect, fx, fy, width, height)
    mesh_room_width_m = max(width_m, 2.0)

    result = RoomMeasurements(
        width_m=round(width_m, 3),
        height_m=round(height_m, 3),
        depth_m=round(depth_m, 3),
        image_width=width,
        image_height=height,
        intrinsics_source=focal_source,
        focal_px=round(fx, 2),
        depth_median_m=round(d_median, 3),
        depth_min_m=round(d_min, 3),
        depth_max_m=round(d_max, 3),
        wall_rect_norm=[margin, margin, 1.0 - 2 * margin, 1.0 - 2 * margin],
    )

    payload = asdict(result)
    payload["image_path"] = str(image_path)
    payload["onnx_path"] = str(args.onnx.expanduser().resolve())
    outputs: dict[str, str] = {"measurements_json": str(json_path)}
    payload["note"] = (
        "Depth Anything outputs per-pixel metric depth, not room width/height. "
        "Width/height here are projection estimates that assume the visible frame is one wall surface; "
        "depth is camera-to-wall at center. EXIF missing -> 28mm-equiv fallback affects X/Y scale."
    )
    manual_tiles = manual_tile_count_measurement(
        width_count=args.tile_width_count,
        height_count=args.tile_height_count,
        depth_count=args.tile_depth_count,
        tile_size_m=args.tile_size_m,
    )
    if manual_tiles is not None:
        payload["manual_tile_count"] = manual_tiles
    if args.room_box_prototype:
        geocalib_payload = None
        if args.geocalib_json and args.geocalib_json.expanduser().exists():
            geocalib_payload = json.loads(args.geocalib_json.expanduser().read_text(encoding="utf-8"))
        if geocalib_payload is None:
            payload["room_box_prototype_error"] = "missing --geocalib-json"
        else:
            proto_fx = float(geocalib_payload.get("focal_x_px") or geocalib_payload.get("focalLengthPx") or fx)
            proto_fy = float(geocalib_payload.get("focal_y_px") or geocalib_payload.get("focalLengthYPx") or proto_fx)
            gravity = geocalib_payload.get("gravity")
            if not gravity:
                payload["room_box_prototype_error"] = "geocalib JSON has no gravity vector"
            else:
                payload["room_box_prototype"] = prototype_room_box(
                    depth=depth,
                    fx=proto_fx,
                    fy=proto_fy,
                    gravity=gravity,
                    camera_height_prior=args.camera_height_prior,
                )
                payload["room_box_prototype"]["focal_px"] = round(proto_fx, 2)
                payload["room_box_prototype"]["geocalib_json"] = str(args.geocalib_json.expanduser().resolve()) if args.geocalib_json else None
                if args.tile_grid_prototype:
                    debug_bev_path = json_path.with_name(f"{json_path.stem}_tile_bev.png")
                    payload["tile_grid_prototype"] = tile_grid_prototype(
                        image=image,
                        depth=depth,
                        fx=proto_fx,
                        fy=proto_fy,
                        gravity=gravity,
                        tile_size_m=args.tile_size_m,
                        debug_bev_path=debug_bev_path,
                    )
                    tile = payload["tile_grid_prototype"]
                    box = payload["room_box_prototype"]
                    axis_u = tile.get("floor_axis_u_m")
                    axis_v = tile.get("floor_axis_v_m")
                    vertical = box.get("height_m")
                    if axis_u is not None and axis_v is not None and vertical is not None:
                        payload["hybrid_interpretation"] = {
                            "source": "tile_grid_floor_axes_plus_room_box_vertical",
                            "w_h_d": {
                                "width_m": axis_u,
                                "height_m": vertical,
                                "depth_m": axis_v,
                            },
                            "swapped_floor_axes_w_h_d": {
                                "width_m": axis_v,
                                "height_m": vertical,
                                "depth_m": axis_u,
                            },
                            "note": "Use the swapped candidate when the CV grid axes are tilted/swapped relative to semantic room width/depth.",
                        }
            if manual_tiles is not None and "error" not in manual_tiles:
                payload["recommended_dimensions"] = {
                    "source": "manual_tile_count",
                    "order": "width_height_depth",
                    "width_m": manual_tiles["width_m"],
                    "height_m": manual_tiles["height_m"],
                    "depth_m": manual_tiles["depth_m"],
                    "w_h_d": manual_tiles["w_h_d"],
                    "note": "Using manual/visual tile counts as the most reliable estimate for tiled rooms.",
                }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_depth_png:
        save_depth_png(depth, depth_png_path)
        outputs["depth_png"] = str(depth_png_path)

    mesh_vertices = 0
    mesh_faces = 0
    if not args.no_glb:
        mesh_depth = depth.copy()
        if args.flat_mesh:
            mesh_depth = flatten_depth_for_mesh(mesh_depth)
        elif args.depth_smooth > 1:
            mesh_depth = smooth_depth_for_mesh(mesh_depth, args.depth_smooth)
        mesh_vertices, mesh_faces = build_textured_mesh_glb(
            rgb_image=image,
            depth=mesh_depth,
            room_width_m=mesh_room_width_m,
            glb_path=glb_path,
            mesh_step=args.mesh_step,
            max_depth_jump=args.max_depth_jump,
        )
        outputs["room_glb"] = str(glb_path)
        payload["mesh_vertices"] = mesh_vertices
        payload["mesh_faces"] = mesh_faces
        payload["mesh_mode"] = "flat" if args.flat_mesh else "relief"

    payload["outputs"] = outputs

    text = json.dumps(payload, indent=2)
    json_path.write_text(text + "\n", encoding="utf-8")

    print(text)
    print()
    print("=== OUTPUT FILES ===")
    print(f"measurements_json: {json_path}")
    if not args.no_depth_png:
        print(f"depth_png:         {depth_png_path}")
    if not args.no_glb:
        print(f"room_glb:          {glb_path}  ({mesh_vertices:,} verts, {mesh_faces:,} faces)")
    print(f"open output folder: open {json_path.parent}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
