#!/usr/bin/env python3
"""Verify the exported RTMDet-Ins Core ML package interface and smoke output."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "Furnit/Models/RTMDet/rtmdet-ins-m.mlpackage"
DEFAULT_IMAGE = REPO_ROOT / "bus.jpg"


def load_coremltools():
    import coremltools as ct

    return ct


def feature_kind(feature: Any) -> str:
    return feature.type.WhichOneof("Type") or "<unset>"


def multiarray_shape(feature: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in feature.type.multiArrayType.shape)


def describe_spec(model: Any) -> None:
    spec = model.get_spec()
    print("Inputs:")
    for feature in spec.description.input:
        kind = feature_kind(feature)
        if kind == "imageType":
            image = feature.type.imageType
            print(f"  {feature.name}: image {image.width}x{image.height} colorSpace={image.colorSpace}")
        elif kind == "multiArrayType":
            print(f"  {feature.name}: multiArray shape={multiarray_shape(feature)}")
        else:
            print(f"  {feature.name}: {kind}")

    print("Outputs:")
    for feature in spec.description.output:
        kind = feature_kind(feature)
        if kind == "multiArrayType":
            print(f"  {feature.name}: multiArray shape={multiarray_shape(feature)}")
        else:
            print(f"  {feature.name}: {kind}")


def model_input(model: Any) -> tuple[str, str, int]:
    spec = model.get_spec()
    if len(spec.description.input) != 1:
        raise AssertionError(f"Expected one model input, found {len(spec.description.input)}")

    feature = spec.description.input[0]
    kind = feature_kind(feature)
    if kind == "imageType":
        image = feature.type.imageType
        if image.width != 640 or image.height != 640:
            raise AssertionError(f"Expected 640x640 image input, found {image.width}x{image.height}")
        return feature.name, kind, 640

    if kind == "multiArrayType":
        shape = multiarray_shape(feature)
        if shape != (1, 3, 640, 640):
            raise AssertionError(f"Expected NCHW input shape (1, 3, 640, 640), found {shape}")
        return feature.name, kind, 640

    raise AssertionError(f"Expected image or multiArray input, found {kind}")


def prepare_input(image_path: Path, input_name: str, kind: str, side: int) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB").resize((side, side), Image.BILINEAR)
    if kind == "imageType":
        return {input_name: image}

    array = np.asarray(image, dtype=np.float32)
    array = array.transpose(2, 0, 1)[None, ...]
    return {input_name: array}


def as_arrays(outputs: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    print("Runtime Outputs:")
    for name, value in outputs.items():
        array = np.asarray(value)
        arrays[name] = array
        finite = array[np.isfinite(array)]
        min_value = float(finite.min()) if finite.size else float("nan")
        max_value = float(finite.max()) if finite.size else float("nan")
        print(f"  {name}: shape={array.shape} dtype={array.dtype} min={min_value:.4f} max={max_value:.4f}")
    return arrays


def pick_boxes(outputs: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    for name, array in outputs.items():
        if array.ndim >= 2 and array.shape[-1] in (5, 6):
            return name, array
    raise AssertionError("Could not find boxes/dets output with last dimension 5 or 6")


def pick_labels(outputs: dict[str, np.ndarray], count: int) -> tuple[str, np.ndarray]:
    for name, array in outputs.items():
        if "label" in name.lower() and array.size == count:
            return name, array
    raise AssertionError("Could not find labels output matching detection count")


def pick_masks(outputs: dict[str, np.ndarray], count: int) -> tuple[str, np.ndarray]:
    for name, array in outputs.items():
        lname = name.lower()
        if ("mask" in lname or "seg" in lname) and array.ndim >= 3:
            if count == 0 or mask_count(array) == count:
                return name, array
    raise AssertionError("Could not find RTMDet-Ins mask output matching detection count")


def flatten_boxes(boxes: np.ndarray) -> np.ndarray:
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        return boxes[0]
    if boxes.ndim == 2:
        return boxes
    return boxes.reshape(-1, boxes.shape[-1])


def mask_count(masks: np.ndarray) -> int:
    if masks.ndim == 3:
        return int(masks.shape[0])
    if masks.ndim == 4 and masks.shape[0] == 1:
        return int(masks.shape[1])
    if masks.ndim == 4:
        return int(masks.shape[0])
    return 0


def assert_smoke_outputs(outputs: dict[str, np.ndarray], min_score: float) -> None:
    box_name, boxes = pick_boxes(outputs)
    rows = flatten_boxes(boxes)
    count = int(rows.shape[0])
    if count <= 0:
        raise AssertionError(f"{box_name} has no detection rows")

    label_name, labels = pick_labels(outputs, count)
    mask_name, masks = pick_masks(outputs, count)

    scores = rows[:, 4]
    positive_scores = scores[np.isfinite(scores)]
    if positive_scores.size == 0 or float(positive_scores.max()) < min_score:
        raise AssertionError(f"No detection score >= {min_score}; max={positive_scores.max() if positive_scores.size else 'nan'}")

    if not np.isfinite(masks).all():
        raise AssertionError(f"{mask_name} contains non-finite values")
    if float(np.max(np.abs(masks))) <= 0.0:
        raise AssertionError(f"{mask_name} is all zero")

    print("Heuristic Picks:")
    print(f"  boxes:  {box_name} {boxes.shape}")
    print(f"  labels: {label_name} {labels.shape}")
    print(f"  masks:  {mask_name} {masks.shape}")
    print(f"  max_score={float(positive_scores.max()):.4f} mask_abs_max={float(np.max(np.abs(masks))):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify RTMDet-Ins Core ML package.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--interface-only", action="store_true", help="Skip Core ML prediction.")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Missing Core ML package: {args.model}")
    if not args.interface_only and not args.image.exists():
        raise FileNotFoundError(f"Missing test image: {args.image}")

    ct = load_coremltools()
    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_ONLY)
    describe_spec(model)
    input_name, kind, side = model_input(model)
    print(f"Verified input: {input_name} ({kind})")

    if not args.interface_only:
        outputs = model.predict(prepare_input(args.image, input_name, kind, side))
        assert_smoke_outputs(as_arrays(outputs), args.min_score)

    print("RTMDet-Ins Core ML verification OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
