#!/usr/bin/env python3
"""Detect line segments on a room photo using M-LSD and draw overlays.

Uses the mlsd package (M-LSD: Towards Light-weight and Real-time Line Segment
Detection, AAAI 2022). Falls back to OpenCV LSD if mlsd import fails.

This is a standalone experiment script. It does NOT use Depth Anything or
GeoCalib. It only detects lines, groups them into two dominant grid families,
clusters their offsets, and reports candidate tile interval counts.

Example:
  python3 scripts/mlsd_draw_room_lines.py \
    --image "/Users/al/Downloads/WhatsApp Image 2026-07-06 at 10.21.21.jpeg" \
    --tile-size-m 0.60
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

DEFAULT_OUT_DIR = Path("/tmp/mlsd_room_lines")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M-LSD room line detection experiment.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--tile-size-m", type=float, default=0.60)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--score-thr", type=float, default=0.10)
    parser.add_argument("--dist-thr", type=float, default=20.0)
    parser.add_argument(
        "--box-connected",
        action="store_true",
        help="Draw thick wall quad + only blues on the box + red/green touching those blues.",
    )
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    image = Image.open(path.expanduser().resolve())
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def detect_mlsd(rgb: np.ndarray, score_thr: float, dist_thr: float) -> tuple[np.ndarray, str]:
    try:
        import mlsd as mlsd_pkg
        interpreter = mlsd_pkg.load_interpreter("M-LSD_512_large_fp32.tflite")
        lines = mlsd_pkg.get_lines(rgb, interpreter, score_thr=score_thr, dist_thr=dist_thr)
        return np.asarray(lines, dtype=np.float32).reshape(-1, 4), "mlsd_512_large"
    except Exception as exc:
        print(f"[WARN] M-LSD import/inference failed ({exc}), falling back to OpenCV LSD")
        return detect_opencv_lsd(rgb), "fallback_opencv_lsd"


def detect_opencv_lsd(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines_raw, _, _, _ = lsd.detect(gray)
    if lines_raw is None:
        return np.zeros((0, 4), dtype=np.float32)
    return lines_raw.reshape(-1, 4).astype(np.float32)


def angle_mod_pi(angle: float) -> float:
    return angle % math.pi


def angle_distance_mod_half_pi(a: float, b: float) -> float:
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def line_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    return angle_mod_pi(math.atan2(y2 - y1, x2 - x1))


def line_length(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def dominant_families(lines: np.ndarray, min_length: float = 20.0) -> list[float]:
    angles = []
    weights = []
    for x1, y1, x2, y2 in lines:
        length = line_length(x1, y1, x2, y2)
        if length < min_length:
            continue
        angles.append(line_angle(x1, y1, x2, y2) % (math.pi / 2))
        weights.append(length)
    if not angles:
        return []
    hist, edges = np.histogram(
        np.array(angles), bins=36, range=(0, math.pi / 2), weights=np.array(weights)
    )
    first = int(np.argmax(hist))
    first_angle = float((edges[first] + edges[first + 1]) * 0.5)
    return [first_angle, (first_angle + math.pi / 2) % math.pi]


def cluster_1d(values: list[float], threshold: float) -> list[dict]:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return []
    clusters: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - float(np.mean(clusters[-1]))) <= threshold:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [{"center": round(float(np.mean(c)), 2), "count": len(c)} for c in clusters]


def family_analysis(
    lines: np.ndarray,
    family_angle: float,
    angle_tolerance_deg: float = 14.0,
    min_length: float = 20.0,
    cluster_merge_px: float = 18.0,
) -> dict:
    tolerance = math.radians(angle_tolerance_deg)
    normal = np.array([-math.sin(family_angle), math.cos(family_angle)], dtype=np.float32)
    offsets: list[float] = []
    selected_count = 0
    for x1, y1, x2, y2 in lines:
        length = line_length(x1, y1, x2, y2)
        if length < min_length:
            continue
        angle = line_angle(x1, y1, x2, y2)
        if angle_distance_mod_half_pi(angle, family_angle) <= tolerance:
            midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            offsets.append(float(np.dot(midpoint, normal)))
            selected_count += 1

    clusters = cluster_1d(offsets, cluster_merge_px)
    strong = [c for c in clusters if c["count"] >= 2]

    spacing = None
    intervals = None
    if len(strong) >= 2:
        centers = [c["center"] for c in strong]
        diffs = [b - a for a, b in zip(centers, centers[1:]) if 15 <= (b - a) <= 350]
        if diffs:
            spacing = round(float(np.median(diffs)), 2)
            span = centers[-1] - centers[0]
            intervals = round(span / spacing, 2) if spacing > 0 else None

    return {
        "family_angle_deg": round(math.degrees(family_angle), 1),
        "selected_segments": selected_count,
        "clusters": clusters,
        "strong_clusters": len(strong),
        "spacing_px": spacing,
        "tile_intervals": intervals,
    }


FAMILY_COLORS_RGB = [(255, 0, 0), (0, 200, 0)]
UNASSIGNED_RGB = (0, 0, 139)  # dark blue — third family (non red/green angles)
ANGLE_TOLERANCE_DEG = 14.0
MIN_DRAW_LENGTH = 20.0

COLOR_LEGEND = {
    "red": {
        "rgb": list(FAMILY_COLORS_RGB[0]),
        "label": "family_0",
        "meaning": "Segment angle within tolerance of dominant family 0 (histogram peak)",
    },
    "green": {
        "rgb": list(FAMILY_COLORS_RGB[1]),
        "label": "family_1",
        "meaning": "Segment angle within tolerance of dominant family 1 (perpendicular to family 0)",
    },
    "dark_blue": {
        "rgb": list(UNASSIGNED_RGB),
        "label": "family_2_unassigned",
        "meaning": "Segment angle does not match family 0 or family 1 within angle tolerance",
    },
}


def classify_segment_color(
    angle: float,
    families: list[float],
    tolerance_rad: float = math.radians(ANGLE_TOLERANCE_DEG),
) -> tuple[tuple[int, int, int], str, int | None]:
    for i, family_angle in enumerate(families):
        if angle_distance_mod_half_pi(angle, family_angle) <= tolerance_rad:
            color = FAMILY_COLORS_RGB[i % 2]
            label = "red" if i == 0 else "green"
            return color, label, i
    return UNASSIGNED_RGB, "dark_blue", None


def segment_metadata(
    lines: np.ndarray,
    families: list[float],
    min_length: float = MIN_DRAW_LENGTH,
) -> tuple[list[dict], dict]:
    segments: list[dict] = []
    counts = {"red": 0, "green": 0, "dark_blue": 0}
    tolerance_rad = math.radians(ANGLE_TOLERANCE_DEG)

    for index, (x1, y1, x2, y2) in enumerate(lines):
        length = line_length(x1, y1, x2, y2)
        angle_rad = line_angle(x1, y1, x2, y2)
        angle_deg = math.degrees(angle_rad)
        color, color_label, family_index = classify_segment_color(angle_rad, families, tolerance_rad)
        drawn = length >= min_length
        if drawn:
            counts[color_label] += 1

        family_angles_deg = [round(math.degrees(fa), 2) for fa in families]
        dist_to_families = [
            round(math.degrees(angle_distance_mod_half_pi(angle_rad, fa)), 2) for fa in families
        ]

        segments.append({
            "id": int(index),
            "x1": round(float(x1), 1),
            "y1": round(float(y1), 1),
            "x2": round(float(x2), 1),
            "y2": round(float(y2), 1),
            "length_px": round(float(length), 1),
            "angle_deg": round(float(angle_deg), 2),
            "angle_deg_mod_90": round(float(angle_deg % 90.0), 2),
            "midpoint": [round(float((x1 + x2) * 0.5), 1), round(float((y1 + y2) * 0.5), 1)],
            "drawn_on_overlay": bool(drawn),
            "color_label": str(color_label),
            "color_rgb": [int(color[0]), int(color[1]), int(color[2])],
            "family_index": None if family_index is None else int(family_index),
            "family_angles_deg": [float(round(math.degrees(float(fa)), 2)) for fa in families],
            "angle_distance_to_family_deg": [float(d) for d in dist_to_families],
            "angle_tolerance_deg": float(ANGLE_TOLERANCE_DEG),
        })

    summary = {
        "color_legend": COLOR_LEGEND,
        "family_angles_deg": [float(round(math.degrees(float(fa)), 2)) for fa in families],
        "angle_tolerance_deg": ANGLE_TOLERANCE_DEG,
        "min_draw_length_px": min_length,
        "drawn_segment_counts": counts,
        "total_segments": int(lines.shape[0]),
    }
    return segments, summary


def draw_overlay(
    rgb: np.ndarray,
    lines: np.ndarray,
    families: list[float],
    out_path: Path,
    min_length: float = MIN_DRAW_LENGTH,
) -> None:
    canvas = rgb.copy()
    tolerance = math.radians(ANGLE_TOLERANCE_DEG)

    for x1, y1, x2, y2 in lines:
        length = line_length(x1, y1, x2, y2)
        if length < min_length:
            continue
        angle = line_angle(x1, y1, x2, y2)
        color, label, _ = classify_segment_color(angle, families, tolerance)
        thickness = 2 if label == "dark_blue" else 1
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    rgb = read_rgb(image_path)
    h, w = rgb.shape[:2]
    out_dir = args.out_dir.expanduser().resolve()
    stem = image_path.stem

    print(f"Image: {image_path}  ({w}x{h})")
    lines, detector = detect_mlsd(rgb, args.score_thr, args.dist_thr)
    print(f"Detector: {detector}")
    print(f"Raw segments: {lines.shape[0]}")

    families = dominant_families(lines)
    print(f"Dominant families: {[round(math.degrees(a), 1) for a in families]} deg")

    segments, color_summary = segment_metadata(lines, families)
    analyses = [family_analysis(lines, fa) for fa in families]

    overlay_path = out_dir / f"{stem}_lines.png"
    if args.box_connected:
        from structure_box_measure_room import (
            classify_line_segments,
            draw_box_connected_overlay,
        )

        min_len = max(24.0, 0.04 * min(w, h))
        classified = classify_line_segments(lines, families, min_len)
        overlay_path = out_dir / f"{stem}_box_connected.png"
        connected_filter = draw_box_connected_overlay(rgb, classified, overlay_path)
        color_summary["box_connected_filter"] = connected_filter
    else:
        draw_overlay(rgb, lines, families, overlay_path)

    payload: dict = {
        "image": str(image_path),
        "image_size": [w, h],
        "detector": detector,
        "total_segments": int(lines.shape[0]),
        "color_summary": color_summary,
        "segments": segments,
        "families": analyses,
        "tile_size_m": args.tile_size_m,
        "dimension_candidates": [],
        "overlay": str(overlay_path),
    }

    for analysis in analyses:
        intervals = analysis.get("tile_intervals")
        if intervals is not None and intervals > 0:
            payload["dimension_candidates"].append({
                "family_angle_deg": analysis["family_angle_deg"],
                "tile_intervals": intervals,
                "meters": round(intervals * args.tile_size_m, 3),
                "spacing_px": analysis["spacing_px"],
                "strong_clusters": analysis["strong_clusters"],
            })

    json_path = out_dir / f"{stem}_lines.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\noverlay: {overlay_path}")
    print(f"json:    {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
