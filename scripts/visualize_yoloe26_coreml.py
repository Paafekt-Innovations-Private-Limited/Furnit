#!/usr/bin/env python3
"""
Draw top-k raw detection rows from yoloe-26l-seg-pf Core ML on the **source** image.

Boxes are inverse-letterboxed from model space into original pixel coordinates so you
can open the PNG in Preview / Xcode and see what the mlpackage is proposing.

Example:
  python3 scripts/visualize_yoloe26_coreml.py \\
    --image FurnitTests/bus.jpg \\
    --out FurnitTests/yoloe26_coreml_overlay.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "yoloe-26l-seg-pf.mlpackage"
DEFAULT_CLASSES = REPO_ROOT / "Furnit" / "Views" / "FurnitureFit" / "classes.json"


def load_coremltools():
    import coremltools as ct

    return ct


def load_class_names(classes_path: Path) -> dict[int, str]:
    with classes_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(key): str(value) for key, value in raw.items()}


def letterbox_image(source_image: Image.Image, model_side: int) -> tuple[Image.Image, float, int, int]:
    rgb_image = source_image.convert("RGB")
    scale = min(model_side / rgb_image.width, model_side / rgb_image.height)
    resized_width = int(round(rgb_image.width * scale))
    resized_height = int(round(rgb_image.height * scale))
    resized_image = rgb_image.resize((resized_width, resized_height), Image.BILINEAR)
    pad_x = (model_side - resized_width) // 2
    pad_y = (model_side - resized_height) // 2
    canvas = Image.new("RGB", (model_side, model_side), (114, 114, 114))
    canvas.paste(resized_image, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def model_input_size(model) -> int:
    spec = model.get_spec()
    for feature in spec.description.input:
        if feature.name != "image":
            continue
        image_type = feature.type.imageType
        if image_type.width > 0:
            return int(image_type.width)
    raise RuntimeError("Could not determine image input size from Core ML model.")


def letterbox_box_to_source(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    sx1 = (x1 - pad_x) / scale
    sy1 = (y1 - pad_y) / scale
    sx2 = (x2 - pad_x) / scale
    sy2 = (y2 - pad_y) / scale
    margin = 0
    return (
        int(max(0, min(source_width - 1, round(sx1 - margin)))),
        int(max(0, min(source_height - 1, round(sy1 - margin)))),
        int(max(0, min(source_width - 1, round(sx2 + margin)))),
        int(max(0, min(source_height - 1, round(sy2 + margin)))),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize YOLOE-26L Core ML boxes on source image.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    ct = load_coremltools()
    class_names = load_class_names(args.classes)
    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_ONLY)
    side = model_input_size(model)

    source_image = Image.open(args.image).convert("RGB")
    letterboxed, scale, pad_x, pad_y = letterbox_image(source_image, side)
    outputs = model.predict({"image": letterboxed})
    detection_tensor = np.asarray(outputs["var_2346"])
    rows = detection_tensor[0]
    order = np.argsort(rows[:, 4])[::-1][: args.top_k]

    overlay = source_image.copy()
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    colors = (
        (255, 80, 80),
        (80, 200, 120),
        (80, 120, 255),
        (255, 180, 60),
        (200, 80, 255),
    )
    for rank, row_index in enumerate(order):
        row = rows[row_index]
        x1, y1, x2, y2 = row[:4]
        confidence = float(row[4])
        class_identifier = int(round(float(row[5])))
        class_name = class_names.get(class_identifier, f"id{class_identifier}")
        bx1, by1, bx2, by2 = letterbox_box_to_source(
            float(x1), float(y1), float(x2), float(y2), scale, pad_x, pad_y, source_image.width, source_image.height
        )
        color = colors[rank % len(colors)]
        draw.rectangle([bx1, by1, bx2, by2], outline=color, width=2)
        label = f"#{rank + 1} {class_name} {confidence:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([bx1, max(0, by1 - th - 4), bx1 + tw + 4, by1], fill=color)
        draw.text((bx1 + 2, max(0, by1 - th - 2)), label, fill=(0, 0, 0), font=font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.out, format="PNG")
    print(f"Wrote {args.out.resolve()} ({overlay.width}x{overlay.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
