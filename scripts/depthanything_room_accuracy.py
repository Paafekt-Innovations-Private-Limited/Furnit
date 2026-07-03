#!/usr/bin/env python3
"""
Run the cheap Depth Anything V2 metric-depth accuracy gate before adding iOS UI.

Input is a JSON file with one entry per photographed room. Each room must include
the measured wall width, wall height, camera-to-wall distance, a wall rectangle
around the measured wall surface, and real camera intrinsics. The script runs Depth
Anything, ports the current WallMeasurementEstimator metric projection math, and
prints pass/marginal/fail stats.

Example:
  python3 scripts/depthanything_room_accuracy.py \
    --rooms-json room_measurements.json \
    --onnx /Volumes/LaCie/apr8th2026depth/android/depthanything_metric_handoff/DepthAnythingV2MetricIndoorSmall.onnx \
    --out-dir /tmp/depth_accuracy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


DEFAULT_ONNX = Path(
    "/Volumes/LaCie/apr8th2026depth/android/depthanything_metric_handoff/"
    "DepthAnythingV2MetricIndoorSmall.onnx"
)
DEFAULT_PTH = Path(
    "/Volumes/LaCie/apr8th2026depth/android/third_party/Depth-Anything-V2/"
    "metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
)
DEFAULT_DEPTH_REPO = Path(
    "/Volumes/LaCie/apr8th2026depth/android/third_party/Depth-Anything-V2/metric_depth"
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    source: str


@dataclass
class Prediction:
    room_id: str
    wall_type: str
    predicted_width_m: float
    predicted_height_m: float
    predicted_depth_m: float
    measured_width_m: float
    measured_height_m: float
    measured_depth_m: float
    width_error_pct: float
    height_error_pct: float
    depth_error_pct: float
    median_error_pct: float
    worst_error_pct: float
    intrinsics_source: str
    calibration: str
    depth_shape: tuple[int, int]
    image_shape: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Depth Anything V2 Metric Indoor room measurement accuracy."
    )
    parser.add_argument("--rooms-json", type=Path, required=True, help="JSON measurement file.")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Depth Anything ONNX file.")
    parser.add_argument("--pth", type=Path, default=DEFAULT_PTH, help="Official metric-depth .pth checkpoint.")
    parser.add_argument("--depth-repo", type=Path, default=DEFAULT_DEPTH_REPO, help="Depth Anything metric_depth repo root.")
    parser.add_argument("--backend", choices=["auto", "onnx", "pth"], default="auto")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--out-dir", type=Path, default=Path("depth_accuracy_out"))
    parser.add_argument("--save-depth-npy", action="store_true")
    parser.add_argument("--save-depth-png", action="store_true")
    parser.add_argument(
        "--allow-fallback-intrinsics",
        action="store_true",
        help="Allow guessed focal/sensor defaults. Smoke tests only; not valid for PASS/FAIL accuracy.",
    )
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    return parser.parse_args()


def load_rooms(path: Path) -> list[dict[str, Any]]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        rooms = data.get("rooms")
    else:
        rooms = data
    if not isinstance(rooms, list) or not rooms:
        raise ValueError("rooms JSON must be a non-empty list or {\"rooms\": [...]}.")
    return rooms


def read_image(path: Path) -> Image.Image:
    image = Image.open(path.expanduser().resolve())
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def preprocess_onnx(image: Image.Image, input_size: int) -> np.ndarray:
    resized = image.resize((input_size, input_size), Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def infer_onnx(image: Image.Image, model_path: Path, input_size: int) -> np.ndarray:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for --backend onnx") from exc

    session = ort.InferenceSession(str(model_path.expanduser().resolve()), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: preprocess_onnx(image, input_size)})[0]
    depth = np.asarray(output, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth output, got shape {output.shape}")
    return resize_depth(depth, image.size)


def infer_pth(image_path: Path, pth_path: Path, depth_repo: Path, input_size: int, max_depth: float) -> np.ndarray:
    try:
        import cv2
        import torch
    except ImportError as exc:
        raise RuntimeError("torch and opencv-python are required for --backend pth") from exc

    sys.path.insert(0, str(depth_repo.expanduser().resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    config = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "max_depth": max_depth}
    model = DepthAnythingV2(**config)
    model.load_state_dict(torch.load(str(pth_path.expanduser().resolve()), map_location="cpu"))
    model = model.to(device).eval()

    raw_image = cv2.imread(str(image_path.expanduser().resolve()))
    if raw_image is None:
        raise ValueError(f"Could not read image with OpenCV: {image_path}")
    return np.asarray(model.infer_image(raw_image, input_size), dtype=np.float32)


def resize_depth(depth: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    image = Image.fromarray(depth.astype(np.float32), mode="F")
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def rect_from_room(room: dict[str, Any], key: str, image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    norm_key = f"{key}_norm"
    px_key = f"{key}_px"
    if norm_key in room:
        x, y, w, h = [float(v) for v in room[norm_key]]
        return (x * image_width, y * image_height, w * image_width, h * image_height)
    if px_key in room:
        x, y, w, h = [float(v) for v in room[px_key]]
        return (x, y, w, h)
    return None


def resolve_intrinsics(room: dict[str, Any], image_width: int, image_height: int) -> Intrinsics:
    if float(room.get("focal_length_px", 0) or 0) > 1:
        focal = float(room["focal_length_px"])
        return Intrinsics(focal, focal, (image_width - 1) * 0.5, (image_height - 1) * 0.5, "focal_length_px")

    focal_mm = float(room.get("focal_length_mm", 0) or 0)
    sensor_width_mm = float(room.get("sensor_width_mm", 0) or 0)
    focal_35mm = float(room.get("focal_length_35mm_equiv_mm", 0) or 0)
    source = "focal_length_mm"

    if focal_mm <= 0:
        raise ValueError("Missing real intrinsics: provide focal_length_px or focal_length_mm")

    if sensor_width_mm <= 0 and focal_35mm > 0:
        sensor_width_mm = 36.0 * focal_mm / focal_35mm
        source += "_derived_sensor"

    if sensor_width_mm <= 0:
        raise ValueError(
            "Missing real intrinsics: provide sensor_width_mm or focal_length_35mm_equiv_mm"
        )

    fx = (focal_mm / sensor_width_mm) * image_width
    sensor_height_mm = sensor_width_mm * image_height / image_width
    fy = (focal_mm / sensor_height_mm) * image_height
    return Intrinsics(fx, fy, (image_width - 1) * 0.5, (image_height - 1) * 0.5, source)


def median_at(depth: np.ndarray, pixel_x: int, pixel_y: int, radius: int = 3) -> float | None:
    height, width = depth.shape
    x0 = max(0, pixel_x - radius)
    x1 = min(width, pixel_x + radius + 1)
    y0 = max(0, pixel_y - radius)
    y1 = min(height, pixel_y + radius + 1)
    samples = depth[y0:y1, x0:x1]
    samples = samples[np.isfinite(samples) & (samples > 0)]
    if samples.size == 0:
        return None
    return float(np.median(samples))


def measure_from_depth(
    depth: np.ndarray,
    wall_rect: tuple[float, float, float, float],
    intrinsics: Intrinsics,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float]:
    x, y, width, height = wall_rect
    left_x = int(np.clip(round(x), 0, image_width - 1))
    right_x = int(np.clip(round(x + width - 1), 0, image_width - 1))
    top_y = int(np.clip(round(y), 0, image_height - 1))
    bottom_y = int(np.clip(round(y + height - 1), 0, image_height - 1))
    center_x = int(np.clip(round(x + width * 0.5), 0, image_width - 1))
    center_y = int(np.clip(round(y + height * 0.5), 0, image_height - 1))

    center_depth = median_at(depth, center_x, center_y)
    if center_depth is None:
        raise ValueError("Could not sample center wall depth")

    left_x_center_plane = (left_x - intrinsics.cx) * center_depth / intrinsics.fx
    right_x_center_plane = (right_x - intrinsics.cx) * center_depth / intrinsics.fx
    top_y_center_plane = (top_y - intrinsics.cy) * center_depth / intrinsics.fy
    bottom_y_center_plane = (bottom_y - intrinsics.cy) * center_depth / intrinsics.fy
    return (
        abs(right_x_center_plane - left_x_center_plane),
        abs(bottom_y_center_plane - top_y_center_plane),
        center_depth,
    )


def apply_calibration(
    prediction: tuple[float, float, float],
    depth: np.ndarray,
    room: dict[str, Any],
    intrinsics: Intrinsics,
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float, float], str]:
    width, height, center_depth = prediction
    calibration = str(room.get("calibration", "none")).lower()
    if calibration == "door":
        door_rect = rect_from_room(room, "door_rect", image_width, image_height)
        door_height_m = float(room.get("door_height_m", 2.03))
        if door_rect is None:
            return prediction, "door_missing_rect"
        _, door_height_pred, _ = measure_from_depth(depth, door_rect, intrinsics, image_width, image_height)
        if door_height_pred <= 0:
            return prediction, "door_invalid"
        scale = door_height_m / door_height_pred
        return (width * scale, height * scale, center_depth * scale), "door"
    if calibration == "ceiling":
        ceiling_height_m = float(room.get("ceiling_height_m", 2.5))
        if height <= 0:
            return prediction, "ceiling_invalid"
        scale = ceiling_height_m / height
        return (width * scale, height * scale, center_depth * scale), "ceiling"
    return prediction, "none"


def error_pct(predicted: float, measured: float) -> float:
    if measured <= 0:
        return math.nan
    return abs(predicted - measured) / measured * 100.0


def evaluate_room(args: argparse.Namespace, room: dict[str, Any]) -> Prediction:
    room_id = str(room.get("id") or room.get("name") or Path(room["image_path"]).stem)
    image_path = Path(room["image_path"])
    image = read_image(image_path)
    image_width, image_height = image.size

    backend = args.backend
    if backend == "auto":
        backend = "onnx" if args.onnx.exists() else "pth"

    if backend == "onnx":
        depth = infer_onnx(image, args.onnx, args.input_size)
    else:
        depth = infer_pth(image_path, args.pth, args.depth_repo, args.input_size, args.max_depth)
        if depth.shape != (image_height, image_width):
            depth = resize_depth(depth, image.size)

    wall_rect = rect_from_room(room, "wall_rect", image_width, image_height)
    if wall_rect is None:
        raise ValueError(f"{room_id}: missing wall_rect_norm or wall_rect_px")

    try:
        intrinsics = resolve_intrinsics(room, image_width, image_height)
    except ValueError:
        if not args.allow_fallback_intrinsics:
            raise
        intrinsics = fallback_intrinsics(room, image_width, image_height)
    raw_prediction = measure_from_depth(depth, wall_rect, intrinsics, image_width, image_height)
    width_m, height_m, depth_m = raw_prediction
    (width_m, height_m, depth_m), calibration = apply_calibration(
        raw_prediction, depth, room, intrinsics, image_width, image_height
    )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_depth_npy:
        np.save(out_dir / f"{room_id}_depth_m.npy", depth.astype(np.float32))
    if args.save_depth_png:
        write_depth_png(depth, out_dir / f"{room_id}_depth.png")

    measured_width = float(room["measured_width_m"])
    measured_height = float(room["measured_height_m"])
    measured_depth = float(room["measured_depth_m"])
    errors = [
        error_pct(width_m, measured_width),
        error_pct(height_m, measured_height),
        error_pct(depth_m, measured_depth),
    ]
    return Prediction(
        room_id=room_id,
        wall_type=str(room.get("wall_type", "unknown")),
        predicted_width_m=width_m,
        predicted_height_m=height_m,
        predicted_depth_m=depth_m,
        measured_width_m=measured_width,
        measured_height_m=measured_height,
        measured_depth_m=measured_depth,
        width_error_pct=errors[0],
        height_error_pct=errors[1],
        depth_error_pct=errors[2],
        median_error_pct=float(np.nanmedian(errors)),
        worst_error_pct=float(np.nanmax(errors)),
        intrinsics_source=intrinsics.source,
        calibration=calibration,
        depth_shape=(int(depth.shape[1]), int(depth.shape[0])),
        image_shape=(image_width, image_height),
    )


def fallback_intrinsics(room: dict[str, Any], image_width: int, image_height: int) -> Intrinsics:
    focal_mm = float(room.get("focal_length_mm", 4.5) or 4.5)
    sensor_width_mm = float(room.get("sensor_width_mm_fallback", 6.4) or 6.4)
    fx = (focal_mm / sensor_width_mm) * image_width
    sensor_height_mm = sensor_width_mm * image_height / image_width
    fy = (focal_mm / sensor_height_mm) * image_height
    return Intrinsics(
        fx,
        fy,
        (image_width - 1) * 0.5,
        (image_height - 1) * 0.5,
        "fallback_intrinsics_smoke_test_only",
    )


def write_depth_png(depth: np.ndarray, path: Path) -> None:
    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        scaled = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(valid, [2, 98])
        denom = max(float(hi - lo), 1e-6)
        scaled = np.clip((depth - lo) / denom * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(scaled, mode="L").save(path)


def verdict(median_error: float) -> str:
    if median_error <= 5:
        return "PASS"
    if median_error <= 10:
        return "MARGINAL"
    return "FAIL"


def print_report(predictions: list[Prediction]) -> None:
    header = (
        "room,wall_type,pred_w,meas_w,err_w%,pred_h,meas_h,err_h%,"
        "pred_d,meas_d,err_d%,median%,worst%,verdict,calibration,intrinsics"
    )
    print(header)
    for item in predictions:
        print(
            f"{item.room_id},{item.wall_type},"
            f"{item.predicted_width_m:.3f},{item.measured_width_m:.3f},{item.width_error_pct:.2f},"
            f"{item.predicted_height_m:.3f},{item.measured_height_m:.3f},{item.height_error_pct:.2f},"
            f"{item.predicted_depth_m:.3f},{item.measured_depth_m:.3f},{item.depth_error_pct:.2f},"
            f"{item.median_error_pct:.2f},{item.worst_error_pct:.2f},{verdict(item.median_error_pct)},"
            f"{item.calibration},{item.intrinsics_source}"
        )

    all_medians = [item.median_error_pct for item in predictions]
    all_worsts = [item.worst_error_pct for item in predictions]
    print()
    print(f"OVERALL_MEDIAN_ERROR_PCT={np.median(all_medians):.2f}")
    print(f"OVERALL_WORST_ROOM_ERROR_PCT={np.max(all_worsts):.2f}")
    print(f"OVERALL_VERDICT={verdict(float(np.median(all_medians)))}")

    for wall_type in sorted({item.wall_type for item in predictions}):
        group = [item for item in predictions if item.wall_type == wall_type]
        med = float(np.median([item.median_error_pct for item in group]))
        worst = float(np.max([item.worst_error_pct for item in group]))
        print(f"{wall_type.upper()}_MEDIAN_ERROR_PCT={med:.2f}")
        print(f"{wall_type.upper()}_WORST_ERROR_PCT={worst:.2f}")
        print(f"{wall_type.upper()}_VERDICT={verdict(med)}")

    distance_bins = [
        ("NEAR_LE_3M", lambda item: item.measured_depth_m <= 3.0),
        ("MID_3_TO_4M", lambda item: 3.0 < item.measured_depth_m <= 4.0),
        ("FAR_GT_4M", lambda item: item.measured_depth_m > 4.0),
    ]
    for label, predicate in distance_bins:
        group = [item for item in predictions if predicate(item)]
        if not group:
            continue
        med = float(np.median([item.median_error_pct for item in group]))
        worst = float(np.max([item.worst_error_pct for item in group]))
        print(f"{label}_COUNT={len(group)}")
        print(f"{label}_MEDIAN_ERROR_PCT={med:.2f}")
        print(f"{label}_WORST_ERROR_PCT={worst:.2f}")
        print(f"{label}_VERDICT={verdict(med)}")


def write_csv(predictions: list[Prediction], path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0].__dict__.keys()) + ["verdict"])
        writer.writeheader()
        for item in predictions:
            row = item.__dict__.copy()
            row["verdict"] = verdict(item.median_error_pct)
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    rooms = load_rooms(args.rooms_json)
    predictions = [evaluate_room(args, room) for room in rooms]
    print_report(predictions)
    if args.csv:
        write_csv(predictions, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
