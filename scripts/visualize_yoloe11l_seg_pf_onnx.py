#!/usr/bin/env python3
"""
Run YOLOE 11L seg PF ONNX on one image and draw bbox + confidence labels.

Defaults are wired for the Furnit repo:
  - ONNX:  android/app/src/main/assets/yoloe-11l-seg-pf.onnx
  - Image: test_images/alchair.jpeg
  - Output: test_images/alchair_yoloe11l_onnx_boxes.png

Example:
  python3 scripts/visualize_yoloe11l_seg_pf_onnx.py

  python3 scripts/visualize_yoloe11l_seg_pf_onnx.py \
    --conf 0.20 --max-dets 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "yoloe-11l-seg-pf.onnx"
DEFAULT_IMAGE = REPO_ROOT / "test_images" / "alchair.jpeg"
DEFAULT_OUTPUT = REPO_ROOT / "test_images" / "alchair_yoloe11l_onnx_boxes.png"
DEFAULT_CLASSES = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "classes.json"


def load_class_names(classes_path: Path) -> dict[int, str]:
    with classes_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(key): str(value) for key, value in raw.items()}


def stretch_rgb(image: Image.Image, model_side: int) -> np.ndarray:
    resized_image = image.convert("RGB").resize((model_side, model_side), Image.Resampling.BILINEAR)
    array = np.asarray(resized_image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


def letterbox_rgb(image: Image.Image, model_side: int) -> tuple[np.ndarray, float, int, int]:
    rgb_image = image.convert("RGB")
    scale = min(model_side / rgb_image.width, model_side / rgb_image.height)
    resized_width = max(1, int(round(rgb_image.width * scale)))
    resized_height = max(1, int(round(rgb_image.height * scale)))
    resized_image = rgb_image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    pad_x = (model_side - resized_width) // 2
    pad_y = (model_side - resized_height) // 2
    canvas = Image.new("RGB", (model_side, model_side), (114, 114, 114))
    canvas.paste(resized_image, (pad_x, pad_y))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None, ...]
    return tensor, scale, pad_x, pad_y


def stretch_xywh_to_source(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    model_side: int,
    source_width: int,
    source_height: int,
) -> tuple[float, float, float, float]:
    scale_x = source_width / model_side
    scale_y = source_height / model_side
    x1 = (cx - bw / 2) * scale_x
    y1 = (cy - bh / 2) * scale_y
    x2 = (cx + bw / 2) * scale_x
    y2 = (cy + bh / 2) * scale_y
    return clamp_xyxy(x1, y1, x2, y2, source_width, source_height)


def letterbox_xywh_to_source(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    source_width: int,
    source_height: int,
) -> tuple[float, float, float, float]:
    x1 = (cx - bw / 2 - pad_x) / scale
    y1 = (cy - bh / 2 - pad_y) / scale
    x2 = (cx + bw / 2 - pad_x) / scale
    y2 = (cy + bh / 2 - pad_y) / scale
    return clamp_xyxy(x1, y1, x2, y2, source_width, source_height)


def clamp_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    source_width: int,
    source_height: int,
) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(source_width - 1.0, x1))
    x2 = max(0.0, min(source_width - 1.0, x2))
    y1 = max(0.0, min(source_height - 1.0, y1))
    y2 = max(0.0, min(source_height - 1.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def iou_xyxy(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter_area / (area_a + area_b - inter_area + 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLOE 11L seg PF ONNX detections.")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Path to yoloe-11l-seg-pf.onnx")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input image path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output image path")
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES, help="classes.json path")
    parser.add_argument("--conf", type=float, default=0.20, help="Min class confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="Class-agnostic NMS IoU threshold")
    parser.add_argument("--max-dets", type=int, default=40, help="Max detections after NMS")
    parser.add_argument(
        "--letterbox",
        action="store_true",
        help="Use 640 letterbox instead of stretch. Default uses stretch for 11L PF parity.",
    )
    args = parser.parse_args()

    import onnxruntime as ort

    if not args.onnx.is_file():
        raise SystemExit(f"Missing ONNX: {args.onnx}")
    if not args.image.is_file():
        raise SystemExit(f"Missing image: {args.image}")
    if not args.classes.is_file():
        raise SystemExit(f"Missing classes.json: {args.classes}")

    class_names = load_class_names(args.classes)
    source_image = Image.open(args.image).convert("RGB")
    source_width, source_height = source_image.size

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    model_side = int(input_meta.shape[2])

    if args.letterbox:
        input_tensor, letterbox_scale, pad_x, pad_y = letterbox_rgb(source_image, model_side)
    else:
        input_tensor = stretch_rgb(source_image, model_side)
        letterbox_scale, pad_x, pad_y = 1.0, 0, 0

    detections_output, proto_output = session.run(None, {input_name: input_tensor})
    _ = proto_output

    predictions = detections_output[0].T
    num_channels = predictions.shape[1]
    num_classes = num_channels - 4 - 32
    if num_classes <= 0:
        raise SystemExit(f"Unexpected ONNX det channels: {num_channels}")

    score_block = predictions[:, 4 : 4 + num_classes]
    max_scores = np.max(score_block, axis=1)
    max_class_indices = np.argmax(score_block, axis=1)
    valid_indices = np.where(max_scores >= args.conf)[0]
    valid_indices = valid_indices[np.argsort(-max_scores[valid_indices])]

    candidates: list[tuple[float, int, tuple[float, float, float, float]]] = []
    for anchor_index in valid_indices:
        class_id = int(max_class_indices[anchor_index])
        confidence = float(max_scores[anchor_index])
        cx, cy, bw, bh = (float(predictions[anchor_index, offset]) for offset in range(4))
        if args.letterbox:
            xyxy = letterbox_xywh_to_source(
                cx, cy, bw, bh, letterbox_scale, pad_x, pad_y, source_width, source_height
            )
        else:
            xyxy = stretch_xywh_to_source(cx, cy, bw, bh, model_side, source_width, source_height)
        if xyxy[2] - xyxy[0] < 2 or xyxy[3] - xyxy[1] < 2:
            continue
        candidates.append((confidence, class_id, xyxy))

    kept: list[tuple[float, int, tuple[float, float, float, float]]] = []
    for candidate in candidates:
        if any(iou_xyxy(candidate[2], prior[2]) > args.iou for prior in kept):
            continue
        kept.append(candidate)
        if len(kept) >= args.max_dets:
            break

    visualization = source_image.copy()
    draw = ImageDraw.Draw(visualization)
    try:
        label_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    except OSError:
        label_font = ImageFont.load_default()

    palette = [
        (255, 80, 80),
        (80, 220, 80),
        (80, 140, 255),
        (255, 190, 50),
        (210, 80, 210),
        (60, 210, 210),
        (255, 120, 190),
    ]

    for index, (confidence, class_id, (x1, y1, x2, y2)) in enumerate(kept):
        color = palette[index % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        class_name = class_names.get(class_id, f"id:{class_id}")
        label = f"{class_name} {confidence:.2f}"
        text_bounds = draw.textbbox((0, 0), label, font=label_font)
        text_width = text_bounds[2] - text_bounds[0]
        text_height = text_bounds[3] - text_bounds[1]
        label_y = max(0.0, y1 - text_height - 6)
        draw.rectangle([x1, label_y, x1 + text_width + 8, label_y + text_height + 6], fill=color)
        draw.text((x1 + 4, label_y + 3), label, fill=(0, 0, 0), font=label_font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    visualization.save(args.out, quality=95)

    print(f"ONNX:   {args.onnx}")
    print(f"Image:  {args.image} ({source_width}x{source_height})")
    print(f"Input:  {input_name} {list(input_meta.shape)} preprocess={'letterbox' if args.letterbox else 'stretch'}")
    print(f"Output: {args.out}")
    print(f"Boxes:  {len(kept)} kept after NMS (conf>={args.conf}, iou<={args.iou})")
    for rank, (confidence, class_id, (x1, y1, x2, y2)) in enumerate(kept[:15], start=1):
        class_name = class_names.get(class_id, f"id:{class_id}")
        print(
            f"{rank:02d}. {class_name:<24} conf={confidence:.3f} "
            f"box=({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})"
        )


if __name__ == "__main__":
    main()
