#!/usr/bin/env python3
"""
Inspect YOLOE-26L Core ML outputs on a real image.

Default behavior:
- loads `yoloe-26l-seg-pf.mlpackage`
- letterboxes `FurnitTests/bus.jpg` to the model's required image size
- runs `MLModel.predict(...)`
- prints output metadata plus a row-by-row interpretation of the
  `[x1, y1, x2, y2, confidence, class_id, 32 mask coeffs]` layout

This script is meant to validate the exported Core ML package structure,
not to perform final post-processing or NMS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "yoloe-26l-seg-pf.mlpackage"
DEFAULT_IMAGE = REPO_ROOT / "FurnitTests" / "bus.jpg"
DEFAULT_CLASSES = REPO_ROOT / "Furnit" / "Views" / "FurnitureFit" / "classes.json"


def load_coremltools():
    import coremltools as ct

    return ct


def load_class_names(classes_path: Path) -> dict[int, str]:
    if not classes_path.is_file():
        return {}
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


def describe_outputs(outputs: dict[str, object]) -> None:
    print("Output tensors:")
    for output_name, output_value in outputs.items():
        output_array = np.asarray(output_value)
        print(
            f"  {output_name}: shape={tuple(output_array.shape)} "
            f"dtype={output_array.dtype} "
            f"min={float(output_array.min()):.4f} "
            f"max={float(output_array.max()):.4f}"
        )


def print_row_analysis(
    detection_tensor: np.ndarray,
    class_names: dict[int, str],
    top_k: int,
) -> None:
    rows = detection_tensor[0]
    confidence_order = np.argsort(rows[:, 4])[::-1]
    print(f"Top {min(top_k, len(confidence_order))} rows by confidence:")
    for rank, row_index in enumerate(confidence_order[:top_k], start=1):
        row = rows[row_index]
        x1, y1, x2, y2 = row[:4]
        confidence = float(row[4])
        class_identifier = int(round(float(row[5])))
        class_name = class_names.get(class_identifier, f"<unknown:{class_identifier}>")
        coeffs = row[6:]
        print(
            f"  #{rank:02d} row={row_index:03d} "
            f"box=({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f}) "
            f"wh=({x2 - x1:.1f},{y2 - y1:.1f}) "
            f"conf={confidence:.4f} "
            f"class_id={class_identifier} "
            f"class_name={class_name}"
        )
        print(
            f"       coeff_stats: min={float(coeffs.min()):.4f} "
            f"max={float(coeffs.max()):.4f} "
            f"mean={float(coeffs.mean()):.4f}"
        )


def print_structure_checks(detection_tensor: np.ndarray) -> None:
    rows = detection_tensor[0]
    class_column = rows[:, 5]
    confidence_column = rows[:, 4]
    near_integer_count = int(np.sum(np.isclose(class_column, np.round(class_column), atol=1e-3)))
    unique_class_ids = sorted({int(round(float(value))) for value in class_column[:50]})
    print("Structure checks:")
    print(f"  confidence column range: {float(confidence_column.min()):.4f} .. {float(confidence_column.max()):.4f}")
    print(f"  class-id column near-integer rows: {near_integer_count}/{len(class_column)}")
    print(f"  sample class ids (first 50 rows): {unique_class_ids[:20]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe YOLOE-26L Core ML output structure.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to yoloe-26l-seg-pf.mlpackage")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Path to test image")
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES, help="Path to classes.json")
    parser.add_argument("--top-k", type=int, default=20, help="How many detection rows to print")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Missing model package: {args.model}")
    if not args.image.exists():
        raise FileNotFoundError(f"Missing image: {args.image}")

    ct = load_coremltools()
    class_names = load_class_names(args.classes)
    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_ONLY)
    side = model_input_size(model)
    source_image = Image.open(args.image)
    letterboxed_image, scale, pad_x, pad_y = letterbox_image(source_image, side)

    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"Source size: {source_image.width}x{source_image.height}")
    print(f"Letterbox: {side}x{side} scale={scale:.6f} pad=({pad_x},{pad_y})")

    outputs = model.predict({"image": letterboxed_image})
    describe_outputs(outputs)

    if "var_2346" not in outputs:
        raise RuntimeError(f"Expected var_2346 in outputs, found {list(outputs.keys())}")

    detection_tensor = np.asarray(outputs["var_2346"])
    print_structure_checks(detection_tensor)
    print_row_analysis(detection_tensor, class_names, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
