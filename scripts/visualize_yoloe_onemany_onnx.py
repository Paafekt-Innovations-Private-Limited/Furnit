#!/usr/bin/env python3
"""
Draw bounding boxes + class names from the one-to-many YOLOE ONNX (det_output [1, C, 8400]).

Inverse-letterboxes from 640 model space into original pixels. Uses simple class-agnostic NMS.

Example:
  python3 scripts/visualize_yoloe_onemany_onnx.py \\
    --onnx .build/yoloe-26l-seg-pf_seg_o2m.onnx \\
    --image FurnitTests/room.jpeg \\
    --out .build/room_onemany_boxes.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX = REPO_ROOT / ".build" / "yoloe-26l-seg-pf_seg_o2m.onnx"
DEFAULT_CLASSES = REPO_ROOT / "Furnit" / "Views" / "FurnitureFit" / "classes.json"


def load_class_names(classes_path: Path) -> dict[int, str]:
    with classes_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(key): str(value) for key, value in raw.items()}


def letterbox_image(source_image: Image.Image, model_side: int) -> tuple[Image.Image, float, int, int]:
    rgb_image = source_image.convert("RGB")
    scale = min(model_side / rgb_image.width, model_side / rgb_image.height)
    resized_width = max(1, int(round(rgb_image.width * scale)))
    resized_height = max(1, int(round(rgb_image.height * scale)))
    resized_image = rgb_image.resize((resized_width, resized_height), Image.BILINEAR)
    pad_x = (model_side - resized_width) // 2
    pad_y = (model_side - resized_height) // 2
    canvas = Image.new("RGB", (model_side, model_side), (114, 114, 114))
    canvas.paste(resized_image, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


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
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    x1 = (x1 - pad_x) / scale
    x2 = (x2 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    y2 = (y2 - pad_y) / scale
    x1 = max(0, min(source_width - 1, x1))
    x2 = max(0, min(source_width - 1, x2))
    y1 = max(0, min(source_height - 1, y1))
    y2 = max(0, min(source_height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter + 1e-6
    return inter / union


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one-to-many YOLOE ONNX detections.")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Path to seg ONNX (det_output).")
    parser.add_argument("--image", type=Path, required=True, help="Source image path.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES, help="classes.json path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Min class score.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--max-dets", type=int, default=25, help="Max boxes after NMS.")
    parser.add_argument("--side", type=int, default=640, help="Model input side length.")
    args = parser.parse_args()

    import onnxruntime as ort

    class_names = load_class_names(args.classes)

    def cname(cid: int) -> str:
        return str(class_names.get(cid, f"id:{cid}"))

    source = Image.open(args.image).convert("RGB")
    source_width, source_height = source.size
    canvas, scale, pad_x, pad_y = letterbox_image(source, args.side)
    tensor = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    det, _ = session.run(None, {"images": tensor})

    num_features, num_anchors = det.shape[1], det.shape[2]
    num_classes = num_features - 4 - 32
    scores_block = det[0, 4 : 4 + num_classes, :]
    box_block = det[0, 0:4, :]

    max_scores = np.max(scores_block, axis=0)
    max_idx = np.argmax(scores_block, axis=0)

    mask = max_scores >= args.conf
    indices = np.where(mask)[0]
    indices = indices[np.argsort(-max_scores[indices])]

    candidates: list[tuple[float, int, tuple[float, float, float, float]]] = []
    for anchor_index in indices:
        class_id = int(max_idx[anchor_index])
        confidence = float(max_scores[anchor_index])
        cx, cy, bw, bh = (float(box_block[row, anchor_index]) for row in range(4))
        xyxy = letterbox_xywh_to_source(
            cx, cy, bw, bh, scale, pad_x, pad_y, source_width, source_height
        )
        if xyxy[2] - xyxy[0] < 2 or xyxy[3] - xyxy[1] < 2:
            continue
        candidates.append((confidence, class_id, xyxy))

    candidates.sort(key=lambda item: -item[0])
    kept: list[tuple[float, int, tuple[float, float, float, float]]] = []
    for confidence, class_id, box in candidates:
        if any(iou_xyxy(box, prior[2]) > args.iou for prior in kept):
            continue
        kept.append((confidence, class_id, box))
        if len(kept) >= args.max_dets:
            break

    vis = source.copy()
    draw = ImageDraw.Draw(vis)
    try:
        label_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        label_font = ImageFont.load_default()

    colors = [
        (255, 60, 60),
        (60, 220, 60),
        (60, 120, 255),
        (255, 180, 40),
        (200, 60, 200),
        (40, 200, 200),
        (255, 120, 200),
    ]
    for index, (confidence, class_id, (x1, y1, x2, y2)) in enumerate(kept):
        color = colors[index % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{cname(class_id)} {confidence * 100:.1f}%"
        text_bounds = draw.textbbox((0, 0), label, font=label_font)
        text_width = text_bounds[2] - text_bounds[0]
        text_height = text_bounds[3] - text_bounds[1]
        label_y = max(0.0, y1 - text_height - 4)
        draw.rectangle([x1, label_y, x1 + text_width + 6, label_y + text_height + 4], fill=color)
        draw.text((x1 + 3, label_y + 2), label, fill=(0, 0, 0), font=label_font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vis.save(args.out, quality=95)
    print(f"Saved: {args.out} ({len(kept)} boxes after NMS)")


if __name__ == "__main__":
    main()
