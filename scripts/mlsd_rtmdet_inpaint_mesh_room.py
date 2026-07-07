#!/usr/bin/env python3
"""M-LSD planes + RTMDet object removal + line-guided fill → scaled depth → 3D mesh.

Pipeline:
  1. M-LSD line detection; draw structural planes (cyan frame, red/green/dark-blue, yellow RG quad).
  2. RTMDet-Ins: detect furniture/clutter and build an exclude mask.
  3. Fill removed regions using horizontal scanlines bounded by extended green (vertical) guides.
  4. Depth Anything metric depth on the filled RGB; scale depth so camera height ≈ 1.7 m.
  5. Derive room height from leveled floor/ceiling samples; width from quad aspect (approximate).
  6. Export textured GLB mesh (flat relief, same convention as depthanything_measure_room / iOS).

Example:
  python3 scripts/mlsd_rtmdet_inpaint_mesh_room.py \\
    --image "/Users/al/Downloads/WhatsApp Image 2026-07-06 at 10.21.21.jpeg" \\
    --geocalib-json /tmp/structure_box_measure/living_geocalib.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from depthanything_measure_room import (  # noqa: E402
    DEFAULT_ONNX,
    build_textured_mesh_glb,
    focal_from_exif,
    leveled_points,
    read_rgb,
    run_depth_onnx,
    smooth_depth_for_mesh,
)
from mlsd_draw_room_lines import dominant_families  # noqa: E402
from rtmdet_exclude_mask import build_rtmdet_exclude_mask  # noqa: E402
from structure_box_measure_room import (  # noqa: E402
    CAMERA_HEIGHT_MAX_M,
    CAMERA_HEIGHT_MIN_M,
    classify_line_segments,
    detect_mlsd_lines,
    draw_box_connected_overlay,
    extract_red_green_intersection_quad,
    extend_segment_to_frame,
    full_image_frame_box,
    region_masks,
    rescale_depth,
    camera_height_from_floor_samples,
)

DEFAULT_OUT_DIR = Path("/tmp/mlsd_rtmdet_mesh")
DEFAULT_CAMERA_HEIGHT_M = 1.70
DEFAULT_GRAVITY = [0.0, -1.0, 0.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M-LSD + RTMDet inpaint + camera-height-scaled 3D room mesh."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--geocalib-json", type=Path, default=None)
    parser.add_argument("--camera-height", type=float, default=DEFAULT_CAMERA_HEIGHT_M)
    parser.add_argument("--score-thr", type=float, default=0.10)
    parser.add_argument("--dist-thr", type=float, default=20.0)
    parser.add_argument("--region-inset-px", type=int, default=8)
    parser.add_argument("--mesh-step", type=int, default=4)
    parser.add_argument("--max-depth-jump", type=float, default=0.35)
    parser.add_argument("--depth-smooth-kernel", type=int, default=3)
    parser.add_argument("--skip-rtmdet", action="store_true")
    parser.add_argument("--skip-mesh", action="store_true")
    return parser.parse_args()


def load_intrinsics(
    image_path: Path,
    width: int,
    height: int,
    geocalib_json: Path | None,
) -> tuple[float, float, list[float], str]:
    exif_fx, exif_fy, focal_source = focal_from_exif(image_path, width, height)
    fx, fy = float(exif_fx), float(exif_fy)
    gravity = list(DEFAULT_GRAVITY)
    source = focal_source

    if geocalib_json and geocalib_json.expanduser().is_file():
        payload = json.loads(geocalib_json.expanduser().read_text(encoding="utf-8"))
        fx = float(payload.get("focal_x_px") or payload.get("focalLengthPx") or fx)
        fy = float(payload.get("focal_y_px") or payload.get("focalLengthYPx") or fy)
        if payload.get("gravity"):
            gravity = [float(v) for v in payload["gravity"]]
        source = f"geocalib+{focal_source}"

    return fx, fy, gravity, source


def vertical_guide_columns(
    segments: list[dict],
    width: int,
    frame_box,
) -> list[int]:
    """X positions from extended green (vertical) M-LSD segments."""
    xs: list[int] = []
    for seg in segments:
        if seg["color_label"] != "green":
            continue
        x1, y1, x2, y2 = extend_segment_to_frame(
            seg["x1"], seg["y1"], seg["x2"], seg["y2"], frame_box,
        )
        xs.extend([
            int(round(x1)),
            int(round(x2)),
            int(round(seg["mid_x"])),
        ])

    if not xs:
        return [0, width - 1]

    uniq = sorted({max(0, min(width - 1, x)) for x in xs})
    if uniq[0] != 0:
        uniq.insert(0, 0)
    if uniq[-1] != width - 1:
        uniq.append(width - 1)
    return uniq


def horizontal_guide_rows(
    segments: list[dict],
    height: int,
    frame_box,
) -> list[int]:
    """Y positions from extended red (horizontal) M-LSD segments."""
    ys: list[int] = []
    for seg in segments:
        if seg["color_label"] != "red":
            continue
        x1, y1, x2, y2 = extend_segment_to_frame(
            seg["x1"], seg["y1"], seg["x2"], seg["y2"], frame_box,
        )
        ys.extend([
            int(round(y1)),
            int(round(y2)),
            int(round(seg["mid_y"])),
        ])

    if not ys:
        return [0, height - 1]

    uniq = sorted({max(0, min(height - 1, y)) for y in ys})
    if uniq[0] != 0:
        uniq.insert(0, 0)
    if uniq[-1] != height - 1:
        uniq.append(height - 1)
    return uniq


def _interpolate_row(
    row_rgb: np.ndarray,
    row_depth: np.ndarray,
    row_exclude: np.ndarray,
    x_start: int,
    x_end: int,
) -> None:
    """Fill excluded pixels on one scanline by linear RGB/depth interpolation."""
    band_rgb = row_rgb[x_start : x_end + 1]
    band_depth = row_depth[x_start : x_end + 1]
    band_exclude = row_exclude[x_start : x_end + 1]
    if not band_exclude.any():
        return

    good = ~band_exclude & np.isfinite(band_depth) & (band_depth > 0)
    if good.sum() < 2:
        return

    indices = np.arange(band_rgb.shape[0], dtype=np.int32)
    for channel in range(3):
        values = band_rgb[:, channel].astype(np.float32)
        band_rgb[:, channel] = np.interp(indices, indices[good], values[good]).astype(np.uint8)

    depth_values = band_depth.astype(np.float32)
    band_depth[:] = np.interp(indices, indices[good], depth_values[good])


def _interpolate_column(
    col_rgb: np.ndarray,
    col_depth: np.ndarray,
    col_exclude: np.ndarray,
    y_start: int,
    y_end: int,
) -> None:
    """Fill excluded pixels on one column by linear RGB/depth interpolation."""
    band_rgb = col_rgb[y_start : y_end + 1]
    band_depth = col_depth[y_start : y_end + 1]
    band_exclude = col_exclude[y_start : y_end + 1]
    if not band_exclude.any():
        return

    good = ~band_exclude & np.isfinite(band_depth) & (band_depth > 0)
    if good.sum() < 2:
        return

    indices = np.arange(band_rgb.shape[0], dtype=np.int32)
    for channel in range(3):
        values = band_rgb[:, channel].astype(np.float32)
        band_rgb[:, channel] = np.interp(indices, indices[good], values[good]).astype(np.uint8)

    depth_values = band_depth.astype(np.float32)
    band_depth[:] = np.interp(indices, indices[good], depth_values[good])


HORIZONTAL_PLANES = frozenset({"floor", "ceiling"})
VERTICAL_PLANES = frozenset({"back_wall", "left_wall", "right_wall"})


def fill_excluded_plane_regions(
    rgb: np.ndarray,
    depth: np.ndarray,
    exclude_mask: np.ndarray,
    plane_masks: dict[str, np.ndarray],
    segments: list[dict],
    frame_box,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fill RTMDet holes per M-LSD plane (floor/ceiling/walls) using line-bounded scanlines."""
    if not exclude_mask.any():
        return rgb.copy(), depth.copy(), {"filled_pixels": 0, "method": "none"}

    filled_rgb = rgb.copy()
    filled_depth = depth.copy()
    height, width = rgb.shape[:2]
    guide_xs = vertical_guide_columns(segments, width, frame_box)
    guide_ys = horizontal_guide_rows(segments, height, frame_box)

    plane_stats: dict[str, dict] = {}
    total_filled = 0

    for plane_name, plane_mask in plane_masks.items():
        holes = plane_mask & exclude_mask
        if not holes.any():
            continue

        if plane_name in HORIZONTAL_PLANES:
            for y in range(height):
                if not holes[y].any():
                    continue
                for x0, x1 in zip(guide_xs[:-1], guide_xs[1:]):
                    _interpolate_row(
                        filled_rgb[y],
                        filled_depth[y],
                        holes[y],
                        x0,
                        x1,
                    )
        elif plane_name in VERTICAL_PLANES:
            for x in range(width):
                if not holes[:, x].any():
                    continue
                for y0, y1 in zip(guide_ys[:-1], guide_ys[1:]):
                    _interpolate_column(
                        filled_rgb[:, x],
                        filled_depth[:, x],
                        holes[:, x],
                        y0,
                        y1,
                    )

        structural = plane_mask & ~exclude_mask
        valid_struct = structural & np.isfinite(filled_depth) & (filled_depth > 0)
        remaining = plane_mask & exclude_mask
        if valid_struct.any():
            plane_rgb = np.median(filled_rgb[valid_struct], axis=0).astype(np.uint8)
            plane_depth_val = float(np.median(filled_depth[valid_struct]))
            unfilled = remaining & ~(
                np.isfinite(filled_depth) & (filled_depth > 0)
            )
            if unfilled.any():
                filled_rgb[unfilled] = plane_rgb
                filled_depth[unfilled] = plane_depth_val
            # Snap all hole depth to the plane so the mesh stays flat (no chair blob).
            filled_depth[remaining] = plane_depth_val
            # For RGB, keep scanline-interpolated tiles where available; median only for gaps.
            gap_rgb = remaining & np.all(filled_rgb == rgb, axis=-1)
            if gap_rgb.any():
                filled_rgb[gap_rgb] = plane_rgb

        filled_count = int(holes.sum())
        total_filled += filled_count
        plane_depth_stat = None
        if valid_struct.any():
            plane_depth_stat = round(float(np.median(filled_depth[valid_struct])), 4)
        plane_stats[plane_name] = {
            "holes_px": filled_count,
            "plane_depth_m": plane_depth_stat,
        }

    # Final pass: any remaining excluded pixels (e.g. chair straddling plane boundaries).
    for y in range(height):
        if not exclude_mask[y].any():
            continue
        for x0, x1 in zip(guide_xs[:-1], guide_xs[1:]):
            _interpolate_row(
                filled_rgb[y],
                filled_depth[y],
                exclude_mask[y],
                x0,
                x1,
            )

    floor_struct = plane_masks["floor"] & ~exclude_mask
    floor_depth = None
    if np.any(floor_struct & np.isfinite(filled_depth) & (filled_depth > 0)):
        floor_depth = float(np.median(filled_depth[floor_struct & np.isfinite(filled_depth) & (filled_depth > 0)]))
    floor_row = int(round(frame_box.floor_y)) if hasattr(frame_box, "floor_y") else height // 2
    if floor_depth is not None:
        below_floor = exclude_mask.copy()
        below_floor[: min(height, floor_row + 1), :] = False
        filled_depth[below_floor] = floor_depth

    return filled_rgb, filled_depth, {
        "method": "mlsd_plane_scanline",
        "vertical_guides": guide_xs,
        "horizontal_guides": guide_ys,
        "filled_pixels": total_filled,
        "exclude_pixels": int(exclude_mask.sum()),
        "planes": plane_stats,
    }


def fill_excluded_line_guided(
    rgb: np.ndarray,
    depth: np.ndarray,
    exclude_mask: np.ndarray,
    segments: list[dict],
    frame_box,
    plane_masks: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fill RTMDet holes; prefer per-plane M-LSD fill when plane masks are available."""
    if plane_masks:
        return fill_excluded_plane_regions(
            rgb, depth, exclude_mask, plane_masks, segments, frame_box,
        )

    filled_rgb = rgb.copy()
    filled_depth = depth.copy()
    height, width = rgb.shape[:2]
    guide_xs = vertical_guide_columns(segments, width, frame_box)
    guide_ys = horizontal_guide_rows(segments, height, frame_box)

    filled_count = 0

    # Primary pass: horizontal scanlines between vertical guides.
    for y in range(height):
        if not exclude_mask[y].any():
            continue
        for x0, x1 in zip(guide_xs[:-1], guide_xs[1:]):
            _interpolate_row(
                filled_rgb[y],
                filled_depth[y],
                exclude_mask[y],
                x0,
                x1,
            )
        filled_count += int(exclude_mask[y].sum())

    # Secondary pass: vertical interpolation between horizontal guides for stubborn holes.
    still_missing = exclude_mask & ~(
        np.isfinite(filled_depth) & (filled_depth > 0)
    )
    if still_missing.any():
        for x in range(width):
            col_exclude = still_missing[:, x]
            if not col_exclude.any():
                continue
            for y0, y1 in zip(guide_ys[:-1], guide_ys[1:]):
                band_exclude = col_exclude[y0 : y1 + 1]
                if not band_exclude.any():
                    continue
                band_depth = filled_depth[y0 : y1 + 1, x]
                good = ~band_exclude & np.isfinite(band_depth) & (band_depth > 0)
                if good.sum() < 2:
                    continue
                indices = np.arange(band_depth.shape[0], dtype=np.int32)
                band_depth[:] = np.interp(
                    indices, indices[good], band_depth[good].astype(np.float32),
                )
                for channel in range(3):
                    band_rgb = filled_rgb[y0 : y1 + 1, x, channel].astype(np.float32)
                    filled_rgb[y0 : y1 + 1, x, channel] = np.interp(
                        indices, indices[good], band_rgb[good],
                    ).astype(np.uint8)

    # Fallback: OpenCV inpaint for any remaining RGB holes (depth stays interpolated).
    remaining = exclude_mask.copy()
    remaining &= ~(
        np.isfinite(filled_depth) & (filled_depth > 0)
    )
    if remaining.any():
        inpaint_mask = (remaining | exclude_mask).astype(np.uint8) * 255
        filled_rgb = cv2.inpaint(filled_rgb, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return filled_rgb, filled_depth, {
        "method": "line_guided_scanline",
        "vertical_guides": guide_xs,
        "horizontal_guides": guide_ys,
        "filled_pixels": filled_count,
        "exclude_pixels": int(exclude_mask.sum()),
    }


def scale_depth_camera_height(
    depth: np.ndarray,
    fx: float,
    fy: float,
    gravity: list[float],
    floor_mask: np.ndarray,
    camera_height_m: float,
) -> tuple[np.ndarray, dict]:
    leveled, _ = leveled_points(depth, fx, fy, gravity)
    camera_height_raw = camera_height_from_floor_samples(leveled, floor_mask)
    depth_scale = 1.0
    scale_trusted = False
    camera_height_scaled = camera_height_raw

    if camera_height_raw is not None and 0.45 <= camera_height_raw <= 5.0:
        depth_scale = camera_height_m / camera_height_raw
        depth = rescale_depth(depth, depth_scale)
        leveled, _ = leveled_points(depth, fx, fy, gravity)
        camera_height_scaled = camera_height_from_floor_samples(leveled, floor_mask)
        scale_trusted = (
            camera_height_scaled is not None
            and CAMERA_HEIGHT_MIN_M <= camera_height_scaled <= CAMERA_HEIGHT_MAX_M
        )

    return depth, {
        "depth_scale": round(depth_scale, 4),
        "camera_height_prior_m": round(camera_height_m, 3),
        "camera_height_raw_m": None if camera_height_raw is None else round(camera_height_raw, 3),
        "camera_height_scaled_m": None if camera_height_scaled is None else round(camera_height_scaled, 3),
        "scale_trusted": scale_trusted,
    }


def estimate_room_dims(
    depth: np.ndarray,
    fx: float,
    fy: float,
    gravity: list[float],
    rg_box,
    width: int,
    height: int,
    region_inset_px: int,
) -> dict:
    """Height from leveled floor/ceiling; width from quad aspect; depth at center."""
    masks = region_masks(rg_box, width, height, region_inset_px)
    leveled, _ = leveled_points(depth, fx, fy, gravity)

    floor_ys = leveled[masks["floor"], 1]
    floor_ys = floor_ys[np.isfinite(floor_ys) & (floor_ys > 0)]
    ceiling_ys = leveled[masks["ceiling"], 1]
    ceiling_ys = ceiling_ys[np.isfinite(ceiling_ys)]

    room_height_m = None
    if floor_ys.size >= 16 and ceiling_ys.size >= 16:
        room_height_m = float(np.median(floor_ys) - np.median(ceiling_ys))
        if room_height_m <= 0.5:
            room_height_m = None

    quad_w_px = max(float(rg_box.right_x - rg_box.left_x), 1.0)
    quad_h_px = max(float(rg_box.floor_y - rg_box.ceiling_y), 1.0)
    room_width_m = None
    if room_height_m is not None:
        room_width_m = room_height_m * (quad_w_px / quad_h_px)

    cx = int(round((rg_box.left_x + rg_box.right_x) * 0.5))
    cy = int(round((rg_box.ceiling_y + rg_box.floor_y) * 0.5))
    cx = max(0, min(width - 1, cx))
    cy = max(0, min(height - 1, cy))
    depth_m = float(depth[cy, cx]) if np.isfinite(depth[cy, cx]) and depth[cy, cx] > 0 else None

    return {
        "width_m": None if room_width_m is None else round(room_width_m, 3),
        "height_m": None if room_height_m is None else round(room_height_m, 3),
        "depth_m": None if depth_m is None else round(depth_m, 3),
        "mesh_width_m": round(max(room_width_m or 2.5, 1.5), 3),
        "quad_px": {
            "left_x": round(rg_box.left_x, 1),
            "right_x": round(rg_box.right_x, 1),
            "ceiling_y": round(rg_box.ceiling_y, 1),
            "floor_y": round(rg_box.floor_y, 1),
            "width_px": round(quad_w_px, 1),
            "height_px": round(quad_h_px, 1),
        },
    }


def save_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    path: Path,
    left_label: str,
    right_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gap = 12
    canvas_h = max(left.shape[0], right.shape[0]) + 28
    canvas_w = left.shape[1] + right.shape[1] + gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[28 : 28 + left.shape[0], 0 : left.shape[1]] = left
    canvas[28 : 28 + right.shape[0], left.shape[1] + gap : left.shape[1] + gap + right.shape[1]] = right
    cv2.putText(canvas, left_label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        right_label,
        (left.shape[1] + gap + 8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    Image.fromarray(canvas).save(path)


def draw_plane_fill_overlay(
    rgb: np.ndarray,
    exclude_mask: np.ndarray,
    plane_masks: dict[str, np.ndarray],
    filled_rgb: np.ndarray,
) -> np.ndarray:
    """Visualize RTMDet exclude (red) and plane-filled regions (green tint)."""
    canvas = rgb.copy()
    canvas[exclude_mask] = (
        canvas[exclude_mask] * 0.4 + np.array([255, 60, 60], dtype=np.float32) * 0.6
    ).astype(np.uint8)
    changed = exclude_mask & np.any(filled_rgb != rgb, axis=-1)
    canvas[changed] = (
        canvas[changed] * 0.45 + np.array([60, 220, 80], dtype=np.float32) * 0.55
    ).astype(np.uint8)
    return canvas


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    image = read_rgb(image_path)
    width, height = image.size
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    fx, fy, gravity, focal_source = load_intrinsics(image_path, width, height, args.geocalib_json)

    print(f"Image: {image_path} ({width}×{height})")
    print(f"Intrinsics: fx={fx:.1f} fy={fy:.1f} ({focal_source})")

    min_len = max(24.0, 0.04 * min(width, height))
    lines, detector = detect_mlsd_lines(rgb, args.score_thr, args.dist_thr)
    families = dominant_families(lines, min_length=min_len)
    segments = classify_line_segments(lines, families, min_len)
    frame_box = full_image_frame_box(width, height)

    lines_overlay_path = out_dir / f"{stem}_mlsd_planes.png"
    line_meta = draw_box_connected_overlay(rgb, segments, lines_overlay_path)
    print(f"M-LSD: {detector}, segments={len(segments)}, overlay={lines_overlay_path}")

    rg_box, rg_meta = extract_red_green_intersection_quad(segments, width, height)
    if rg_box is None:
        print("[WARN] RG intersection quad failed; using full-image frame for regions")
        rg_box = frame_box
        rg_meta = {"fallback": "full_image_frame"}

    exclude_mask = np.zeros((height, width), dtype=bool)
    rtmdet_meta: dict = {"source": "skipped"}
    if not args.skip_rtmdet:
        exclude_mask, rtmdet_meta = build_rtmdet_exclude_mask(rgb)
        if exclude_mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            exclude_mask = cv2.dilate(exclude_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        print(
            f"RTMDet: {rtmdet_meta.get('source')} "
            f"exclude={rtmdet_meta.get('exclude_pixels', 0)} px "
            f"({rtmdet_meta.get('exclude_fraction', 0):.1%})"
        )

    plane_masks = region_masks(rg_box, width, height, args.region_inset_px)

    if exclude_mask.any():
        exclude_vis = rgb.copy()
        exclude_vis[exclude_mask] = (exclude_vis[exclude_mask] * 0.35 + np.array([255, 80, 80]) * 0.65).astype(np.uint8)
        exclude_path = out_dir / f"{stem}_rtmdet_exclude.png"
        Image.fromarray(exclude_vis).save(exclude_path)
    else:
        exclude_path = None

    print("Running Depth Anything…")
    depth = run_depth_onnx(image, args.onnx, args.input_size)

    filled_rgb, filled_depth, fill_meta = fill_excluded_line_guided(
        rgb, depth, exclude_mask, segments, frame_box, plane_masks=plane_masks,
    )
    filled_path = out_dir / f"{stem}_line_filled.png"
    Image.fromarray(filled_rgb).save(filled_path)
    if exclude_mask.any():
        plane_fill_path = out_dir / f"{stem}_plane_filled.png"
        Image.fromarray(draw_plane_fill_overlay(rgb, exclude_mask, plane_masks, filled_rgb)).save(plane_fill_path)
    save_side_by_side(
        rgb,
        filled_rgb,
        out_dir / f"{stem}_before_after_fill.png",
        "original",
        "plane-filled (chair removed)",
    )
    print(
        f"Plane fill: {fill_meta.get('filled_pixels', 0)} px, "
        f"planes={list(fill_meta.get('planes', {}).keys())}"
    )

    floor_mask = plane_masks["floor"]
    if exclude_mask.any():
        floor_mask = floor_mask & ~exclude_mask

    filled_depth, scale_meta = scale_depth_camera_height(
        filled_depth, fx, fy, gravity, floor_mask, args.camera_height,
    )
    print(
        f"Depth scale: {scale_meta['depth_scale']:.4f} "
        f"camera_height={scale_meta.get('camera_height_scaled_m')} m "
        f"(prior {args.camera_height} m)"
    )

    dims = estimate_room_dims(
        filled_depth, fx, fy, gravity, rg_box, width, height, args.region_inset_px,
    )
    w_h_d = dims.get("width_m"), dims.get("height_m"), dims.get("depth_m")
    print(f"Room estimate: {w_h_d[0]} × {w_h_d[1]} × {w_h_d[2]} m (width approximate)")

    mesh_depth = smooth_depth_for_mesh(filled_depth, args.depth_smooth_kernel)
    mesh_path = out_dir / f"{stem}_room.glb"
    mesh_info: dict = {}
    if not args.skip_mesh:
        filled_image = Image.fromarray(filled_rgb)
        vertex_count, face_count = build_textured_mesh_glb(
            filled_image,
            mesh_depth,
            dims["mesh_width_m"],
            mesh_path,
            mesh_step=args.mesh_step,
            max_depth_jump=args.max_depth_jump,
        )
        mesh_info = {
            "glb_path": str(mesh_path),
            "vertices": vertex_count,
            "faces": face_count,
            "mesh_width_m": dims["mesh_width_m"],
        }
        print(f"Mesh: {mesh_path} ({vertex_count} verts, {face_count} faces)")

    result = {
        "image_path": str(image_path),
        "image_size": [width, height],
        "detector": detector,
        "focal_source": focal_source,
        "fx_px": round(fx, 2),
        "fy_px": round(fy, 2),
        "camera_height_prior_m": args.camera_height,
        "scale": scale_meta,
        "dimensions_m": {
            "width": dims.get("width_m"),
            "height": dims.get("height_m"),
            "depth": dims.get("depth_m"),
        },
        "rg_quad": rg_meta,
        "mlsd_overlay": str(lines_overlay_path),
        "rtmdet": rtmdet_meta,
        "line_fill": fill_meta,
        "outputs": {
            "mlsd_planes_png": str(lines_overlay_path),
            "line_filled_png": str(filled_path),
            "plane_filled_png": str(out_dir / f"{stem}_plane_filled.png") if exclude_mask.any() else None,
            "before_after_png": str(out_dir / f"{stem}_before_after_fill.png"),
            "rtmdet_exclude_png": None if exclude_path is None else str(exclude_path),
            "mesh_glb": mesh_info.get("glb_path"),
        },
        "mesh": mesh_info,
        "line_filter_meta": line_meta,
    }

    json_path = out_dir / f"{stem}_mlsd_rtmdet_mesh.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
