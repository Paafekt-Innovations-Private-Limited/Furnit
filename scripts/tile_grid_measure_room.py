#!/usr/bin/env python3
"""Estimate room dimensions by visually counting tile/grid intervals.

This is intentionally separate from depth/GeoCalib experiments. It uses only
classical CV on the RGB image:

  image -> Canny -> Hough line segments -> dominant line families -> line-offset
  clustering -> tile interval counts -> count * assumed tile size

It is a prototype for "what ChatGPT is doing visually": count visible tile
intervals and multiply by an assumed tile size. It also writes debug overlays so
bad counts can be inspected instead of trusted blindly.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


DEFAULT_OUT_DIR = Path("/tmp/tile_grid_measure")


@dataclass
class LineSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    angle: float
    length: float
    midpoint: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical-CV tile grid room estimator.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--tile-size-m", type=float, default=0.60)
    parser.add_argument(
        "--mode",
        choices=["auto", "bathroom", "living"],
        default="auto",
        help="Controls which ROI interpretation is preferred.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    image = Image.open(path.expanduser().resolve())
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def angle_mod_pi(angle: float) -> float:
    value = angle % math.pi
    return value + math.pi if value < 0 else value


def angle_distance_mod_pi(a: float, b: float) -> float:
    return abs((a - b + math.pi / 2.0) % math.pi - math.pi / 2.0)


def detect_segments(
    rgb: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    min_length_frac: float = 0.035,
) -> tuple[list[LineSegment], np.ndarray]:
    import cv2

    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = roi
    x0 = int(np.clip(x0, 0, w - 1))
    x1 = int(np.clip(x1, x0 + 1, w))
    y0 = int(np.clip(y0, 0, h - 1))
    y1 = int(np.clip(y1, y0 + 1, h))

    crop = rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 140, apertureSize=3)

    min_line_length = max(25, int(min(crop.shape[:2]) * min_length_frac))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=35,
        minLineLength=min_line_length,
        maxLineGap=18,
    )
    segments: list[LineSegment] = []
    if lines is None:
        return segments, edges

    for lx1, ly1, lx2, ly2 in lines[:, 0, :]:
        gx1 = float(lx1 + x0)
        gy1 = float(ly1 + y0)
        gx2 = float(lx2 + x0)
        gy2 = float(ly2 + y0)
        dx = gx2 - gx1
        dy = gy2 - gy1
        length = math.hypot(dx, dy)
        if length < min_line_length:
            continue
        angle = angle_mod_pi(math.atan2(dy, dx))
        midpoint = np.array([(gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5], dtype=np.float32)
        segments.append(LineSegment(gx1, gy1, gx2, gy2, angle, length, midpoint))
    return segments, edges


def dominant_angle_families(segments: list[LineSegment]) -> list[float]:
    if not segments:
        return []
    angles = np.array([segment.angle % (math.pi / 2.0) for segment in segments], dtype=np.float32)
    weights = np.array([segment.length for segment in segments], dtype=np.float32)
    hist, edges = np.histogram(angles, bins=36, range=(0, math.pi / 2.0), weights=weights)
    first = int(np.argmax(hist))
    first_angle = float((edges[first] + edges[first + 1]) * 0.5)
    return [first_angle, (first_angle + math.pi / 2.0) % math.pi]


def cluster_1d(values: list[float], threshold: float) -> list[dict]:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return []
    clusters: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        center = float(np.mean(clusters[-1]))
        if abs(value - center) <= threshold:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [
        {"center": float(np.mean(cluster)), "count": len(cluster)}
        for cluster in clusters
    ]


def family_offsets(segments: list[LineSegment], family_angle: float) -> tuple[list[float], list[LineSegment]]:
    selected: list[LineSegment] = []
    offsets: list[float] = []
    normal = np.array([-math.sin(family_angle), math.cos(family_angle)], dtype=np.float32)
    for segment in segments:
        if angle_distance_mod_pi(segment.angle, family_angle) <= math.radians(14):
            selected.append(segment)
            offsets.append(float(np.dot(segment.midpoint, normal)))
    return offsets, selected


def estimate_intervals_from_offsets(offsets: list[float], *, cluster_threshold_px: float = 18.0) -> dict | None:
    clusters = cluster_1d(offsets, cluster_threshold_px)
    # Ignore singleton clutter for spacing if possible, but keep all clusters for span.
    strong = [cluster for cluster in clusters if cluster["count"] >= 2]
    usable = strong if len(strong) >= 2 else clusters
    if len(usable) < 2:
        return None

    centers = [cluster["center"] for cluster in usable]
    diffs = [
        b - a
        for a, b in zip(centers, centers[1:])
        if 15.0 <= (b - a) <= 350.0
    ]
    if not diffs:
        return None
    spacing = float(np.median(diffs))
    span = float(centers[-1] - centers[0])
    intervals = span / max(spacing, 1e-6)
    # If visible boundaries include both outer grout lines, intervals is enough; if they
    # are interior lines only, +1 may be closer. Expose both.
    return {
        "clusters": [{"center_px": round(c["center"], 1), "count": c["count"]} for c in clusters],
        "strong_cluster_count": len(strong),
        "spacing_px": round(spacing, 2),
        "intervals_between_outer_lines": round(intervals, 2),
        "intervals_with_outer_margin": round(intervals + 1.0, 2),
    }


def analyze_roi(rgb: np.ndarray, roi: tuple[int, int, int, int], label: str) -> dict:
    segments, _ = detect_segments(rgb, roi)
    families = dominant_angle_families(segments)
    result: dict = {
        "label": label,
        "roi": list(roi),
        "segments": len(segments),
        "families": [],
    }
    for angle in families:
        offsets, selected = family_offsets(segments, angle)
        estimate = estimate_intervals_from_offsets(offsets)
        result["families"].append(
            {
                "angle_deg": round(math.degrees(angle), 1),
                "selected_segments": len(selected),
                "interval_estimate": estimate,
            }
        )
    return result


def draw_debug(rgb: np.ndarray, analyses: list[dict], out_path: Path) -> None:
    import cv2

    canvas = rgb.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255)]
    for index, analysis in enumerate(analyses):
        x0, y0, x1, y1 = analysis["roi"]
        color = colors[index % len(colors)]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            canvas,
            analysis["label"],
            (x0 + 8, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def choose_dimension_candidates(analysis: dict, tile_size_m: float) -> list[dict]:
    candidates: list[dict] = []
    for family in analysis.get("families", []):
        estimate = family.get("interval_estimate")
        if not estimate:
            continue
        for key in ("intervals_between_outer_lines", "intervals_with_outer_margin"):
            count = estimate[key]
            candidates.append(
                {
                    "family_angle_deg": family["angle_deg"],
                    "count_type": key,
                    "tile_count": count,
                    "meters": round(count * tile_size_m, 3),
                    "selected_segments": family["selected_segments"],
                    "strong_cluster_count": estimate["strong_cluster_count"],
                }
            )
    return candidates


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    rgb = read_rgb(image_path)
    h, w = rgb.shape[:2]
    out_dir = args.out_dir.expanduser().resolve()
    out_json = args.json_out or (out_dir / f"{image_path.stem}_tile_count_cv.json")
    debug_path = out_dir / f"{image_path.stem}_tile_count_debug.png"

    # ROIs are deliberately simple and inspectable. This is tile counting, not
    # depth-based geometry.
    rois = [
        (0, int(h * 0.45), w, h, "floor_lower"),
        (0, 0, w, int(h * 0.72), "wall_upper"),
        (int(w * 0.10), int(h * 0.15), int(w * 0.90), int(h * 0.85), "center_room"),
    ]
    analyses = [analyze_roi(rgb, roi[:4], roi[4]) for roi in rois]
    draw_debug(rgb, analyses, debug_path)

    payload = {
        "image": str(image_path),
        "tile_size_m": args.tile_size_m,
        "mode": args.mode,
        "analyses": analyses,
        "dimension_candidates": {
            analysis["label"]: choose_dimension_candidates(analysis, args.tile_size_m)
            for analysis in analyses
        },
        "debug_overlay": str(debug_path),
        "note": (
            "This script estimates tile interval counts from Canny/Hough line clusters. "
            "It does not use Depth Anything. Choose the family/count candidate that matches "
            "the semantic room axis (width/height/depth)."
        ),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\njson:  {out_json}")
    print(f"debug: {debug_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
