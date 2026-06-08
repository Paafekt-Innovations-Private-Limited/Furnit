#!/usr/bin/env python3
"""
Probe an RTMDet-Ins Core ML package on a real image.

This is a lightweight inspection script for the Swift RTMDet still-image spike.
It does not assume exact output tensor names. Instead, it:
  1. loads a Core ML package
  2. preprocesses an input image with stretch or letterbox
  3. runs prediction on CPU
  4. prints output names, shapes, dtypes, and value ranges
  5. tries to identify box / label / mask tensors heuristically

Example:
  python3 scripts/probe_rtmdet_coreml.py \
    --model /path/to/rtmdet-ins-m.mlpackage \
    --image test_images/chair.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_CANDIDATES = (
    REPO_ROOT / "test_images" / "chair.jpeg",
    REPO_ROOT / "bus.jpg",
    REPO_ROOT / "FurnitTests" / "bus.jpg",
)


def load_coremltools():
    import coremltools as ct

    return ct


def find_default_image() -> Path | None:
    for candidate in DEFAULT_IMAGE_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def first_image_input_name(model: Any) -> str:
    spec = model.get_spec()
    for feature in spec.description.input:
        if feature.type.WhichOneof("Type") == "imageType":
            return feature.name
    raise RuntimeError("Could not find an image input in the Core ML model.")


def image_input_side(model: Any, input_name: str) -> int:
    spec = model.get_spec()
    for feature in spec.description.input:
        if feature.name != input_name:
            continue
        image_type = feature.type.imageType
        if image_type.width > 0:
            return int(image_type.width)
    raise RuntimeError(f"Could not determine image side for input '{input_name}'.")


def stretch_image(source_image: Image.Image, model_side: int) -> Image.Image:
    return source_image.convert("RGB").resize((model_side, model_side), Image.BILINEAR)


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


def describe_outputs(outputs: dict[str, object]) -> dict[str, np.ndarray]:
    converted: dict[str, np.ndarray] = {}
    print("Output tensors:")
    for output_name, output_value in outputs.items():
        output_array = np.asarray(output_value)
        converted[output_name] = output_array
        finite = output_array[np.isfinite(output_array)]
        if finite.size == 0:
            min_value = max_value = float("nan")
        else:
            min_value = float(finite.min())
            max_value = float(finite.max())
        print(
            f"  {output_name}: shape={tuple(output_array.shape)} "
            f"dtype={output_array.dtype} min={min_value:.4f} max={max_value:.4f}"
        )
    return converted


def compact_shape(array: np.ndarray) -> tuple[int, ...]:
    return tuple(int(dim) for dim in array.shape)


def pick_boxes_array(outputs: dict[str, np.ndarray]) -> tuple[str, np.ndarray] | None:
    for name, array in outputs.items():
        shape = compact_shape(array)
        if len(shape) >= 2 and shape[-1] in (5, 6):
            lname = name.lower()
            if "bbox" in lname or "det" in lname:
                return name, array
    for name, array in outputs.items():
        shape = compact_shape(array)
        if len(shape) >= 2 and shape[-1] in (5, 6):
            return name, array
    return None


def pick_labels_array(outputs: dict[str, np.ndarray], target_count: int) -> tuple[str, np.ndarray] | None:
    for name, array in outputs.items():
        shape = compact_shape(array)
        lname = name.lower()
        if len(shape) <= 2 and np.prod(shape, dtype=int) == target_count and ("label" in lname or "class" in lname):
            return name, array
    return None


def pick_mask_array(outputs: dict[str, np.ndarray], target_count: int) -> tuple[str, np.ndarray] | None:
    for name, array in outputs.items():
        shape = compact_shape(array)
        lname = name.lower()
        if len(shape) >= 3 and ("mask" in lname or "seg" in lname):
            count = mask_count(shape)
            if target_count == 0 or count == target_count:
                return name, array
    for name, array in outputs.items():
        shape = compact_shape(array)
        if len(shape) >= 3 and shape[-1] > 1 and shape[-2] > 1:
            return name, array
    return None


def detection_count(shape: tuple[int, ...]) -> int:
    if len(shape) == 2:
        return int(shape[0])
    if len(shape) == 3:
        return int(shape[1] if shape[0] == 1 else shape[0])
    return 0


def mask_count(shape: tuple[int, ...]) -> int:
    if len(shape) == 3:
        return int(shape[0])
    if len(shape) == 4:
        if shape[0] == 1:
            return int(shape[1])
        if shape[1] == 1:
            return int(shape[0])
        return int(shape[-3])
    return 0


def print_box_preview(boxes: np.ndarray, labels: np.ndarray | None, top_k: int) -> None:
    rows = boxes
    if rows.ndim == 3:
        rows = rows[0] if rows.shape[0] == 1 else rows.reshape(-1, rows.shape[-1])
    elif rows.ndim != 2:
        print("Boxes preview skipped: unexpected box tensor rank.")
        return

    if rows.shape[1] not in (5, 6):
        print("Boxes preview skipped: last dimension is not 5 or 6.")
        return

    labels_flat = labels.reshape(-1) if labels is not None else None
    order = np.argsort(rows[:, 4])[::-1]
    print(f"Top {min(top_k, len(order))} detections by score:")
    for rank, row_index in enumerate(order[:top_k], start=1):
        row = rows[row_index]
        x1, y1, x2, y2 = [float(v) for v in row[:4]]
        score = float(row[4])
        if rows.shape[1] >= 6:
            class_id = int(round(float(row[5])))
        elif labels_flat is not None and row_index < labels_flat.shape[0]:
            class_id = int(round(float(labels_flat[row_index])))
        else:
            class_id = -1
        print(
            f"  #{rank:02d} row={row_index:03d} "
            f"box=({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f}) "
            f"wh=({x2 - x1:.1f},{y2 - y1:.1f}) "
            f"score={score:.4f} class_id={class_id}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe RTMDet-Ins Core ML outputs.")
    parser.add_argument("--model", type=Path, required=True, help="Path to RTMDet Core ML package")
    parser.add_argument("--image", type=Path, default=find_default_image(), help="Path to test image")
    parser.add_argument("--top-k", type=int, default=12, help="How many detection rows to preview")
    parser.add_argument(
        "--preprocess",
        choices=("stretch", "letterbox"),
        default="stretch",
        help="Preprocess strategy to use before Core ML inference",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Missing Core ML package: {args.model}")
    if args.image is None or not args.image.exists():
        raise FileNotFoundError("Missing test image. Pass --image explicitly.")

    ct = load_coremltools()
    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_ONLY)
    input_name = first_image_input_name(model)
    side = image_input_side(model, input_name)
    source_image = Image.open(args.image)

    if args.preprocess == "letterbox":
        prepared_image, scale, pad_x, pad_y = letterbox_image(source_image, side)
        prep_meta = f"letterbox side={side} scale={scale:.6f} pad=({pad_x},{pad_y})"
    else:
        prepared_image = stretch_image(source_image, side)
        prep_meta = f"stretch side={side}"

    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"Input feature: {input_name}")
    print(f"Source size: {source_image.width}x{source_image.height}")
    print(f"Preprocess: {prep_meta}")

    outputs = model.predict({input_name: prepared_image})
    output_arrays = describe_outputs(outputs)

    boxes_match = pick_boxes_array(output_arrays)
    labels_match = None
    mask_match = None

    if boxes_match is not None:
        box_name, box_array = boxes_match
        count_hint = detection_count(compact_shape(box_array))
        labels_match = pick_labels_array(output_arrays, count_hint)
        mask_match = pick_mask_array(output_arrays, count_hint)

        print("\nHeuristic picks:")
        print(f"  boxes:  {box_name} {compact_shape(box_array)}")
        if labels_match is not None:
            print(f"  labels: {labels_match[0]} {compact_shape(labels_match[1])}")
        else:
            print("  labels: <none>")
        if mask_match is not None:
            print(f"  masks:  {mask_match[0]} {compact_shape(mask_match[1])}")
        else:
            print("  masks:  <none>")

        print()
        print_box_preview(box_array, labels_match[1] if labels_match is not None else None, args.top_k)
    else:
        print("\nCould not identify a boxes tensor heuristically.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
