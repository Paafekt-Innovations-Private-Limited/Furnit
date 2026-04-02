#!/usr/bin/env python3
"""
Compare raw YOLOE PyTorch outputs against raw Core ML outputs on the same letterboxed image.

This is the correct parity check:
- PT raw detection tensor: unnamed `out[0][0]`
- PT raw proto tensor: unnamed `out[0][1]`
- Core ML detection tensor: usually `var_2346`
- Core ML proto tensor: usually `var_2429`

The Core ML `var_*` names are export-time renames. PyTorch tensors are unnamed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent


def letterbox_rgb(source: Image.Image, side: int) -> Image.Image:
    rgb = source.convert("RGB")
    scale = min(side / rgb.width, side / rgb.height)
    new_width = max(1, int(round(rgb.width * scale)))
    new_height = max(1, int(round(rgb.height * scale)))
    resized = rgb.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (side, side), (114, 114, 114))
    pad_x = (side - new_width) // 2
    pad_y = (side - new_height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas


def coreml_image_input_side(model) -> int | None:
    spec = model.get_spec()
    for feature in spec.description.input:
        if feature.name != "image":
            continue
        width = feature.type.imageType.width
        if width > 0:
            return int(width)
    return None


def pick_coreml_outputs(outputs: dict[str, object]) -> tuple[str, np.ndarray, str, np.ndarray]:
    detection_name = ""
    proto_name = ""
    detection_tensor: np.ndarray | None = None
    proto_tensor: np.ndarray | None = None

    for name, value in outputs.items():
        array = np.asarray(value).astype(np.float32)
        if array.ndim == 3 and array.shape[0] == 1 and array.shape[2] == 38 and detection_tensor is None:
            detection_name = name
            detection_tensor = array
        elif array.ndim == 4 and array.shape[1] == 32 and proto_tensor is None:
            proto_name = name
            proto_tensor = array

    if detection_tensor is None or proto_tensor is None:
        shapes = {name: tuple(np.asarray(value).shape) for name, value in outputs.items()}
        raise RuntimeError(f"Could not resolve Core ML det/proto tensors from outputs: {shapes}")

    return detection_name, detection_tensor, proto_name, proto_tensor


def extract_pt_outputs(raw_output) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(raw_output, tuple) or len(raw_output) < 1:
        raise RuntimeError(f"Unexpected PT output type: {type(raw_output)}")
    first = raw_output[0]
    if not isinstance(first, tuple) or len(first) < 2:
        raise RuntimeError(f"Unexpected PT output[0] structure: {type(first)}")
    det_tensor = first[0]
    proto_tensor = first[1]
    return (
        det_tensor.detach().cpu().numpy().astype(np.float32),
        proto_tensor.detach().cpu().numpy().astype(np.float32),
    )


def print_tensor_diff(name: str, pt_tensor: np.ndarray, coreml_tensor: np.ndarray) -> None:
    diff = np.abs(pt_tensor - coreml_tensor)
    print(f"\n{name}:")
    print(f"  shape={pt_tensor.shape}")
    print(f"  max_abs_diff={float(diff.max()):.6f}")
    print(f"  mean_abs_diff={float(diff.mean()):.6f}")
    print(f"  pt_mean={float(pt_tensor.mean()):.6f} coreml_mean={float(coreml_tensor.mean()):.6f}")
    print(
        f"  pt_minmax=({float(pt_tensor.min()):.6f}, {float(pt_tensor.max()):.6f}) "
        f"coreml_minmax=({float(coreml_tensor.min()):.6f}, {float(coreml_tensor.max()):.6f})"
    )


def print_top_rows(label: str, det_tensor: np.ndarray, class_id: int, top_k: int) -> None:
    rows = det_tensor[0]
    hits: list[tuple[float, int, np.ndarray]] = []
    for row_index, row in enumerate(rows):
        if int(round(float(row[5]))) != class_id:
            continue
        hits.append((float(row[4]), row_index, row))
    hits.sort(key=lambda item: item[0], reverse=True)
    print(f"\n{label} top class_id={class_id} rows:")
    if not hits:
        print("  none")
        return
    for confidence, row_index, row in hits[:top_k]:
        print(
            f"  row={row_index} conf={confidence:.4f} "
            f"xyxy=({row[0]:.2f},{row[1]:.2f},{row[2]:.2f},{row[3]:.2f}) class={int(round(float(row[5])))}"
        )


def print_row_diffs(pt_det: np.ndarray, coreml_det: np.ndarray, row_count: int) -> None:
    print(f"\nFirst {row_count} detection-row diffs (same row index, same raw tensor layout):")
    rows = min(row_count, pt_det.shape[1], coreml_det.shape[1])
    for row_index in range(rows):
        pt_row = pt_det[0, row_index]
        coreml_row = coreml_det[0, row_index]
        row_diff = np.abs(pt_row - coreml_row)
        print(
            f"  row={row_index} "
            f"max_abs_diff={float(row_diff.max()):.6f} "
            f"mean_abs_diff={float(row_diff.mean()):.6f} "
            f"pt_conf={float(pt_row[4]):.4f} cm_conf={float(coreml_row[4]):.4f} "
            f"pt_cls={int(round(float(pt_row[5])))} cm_cls={int(round(float(coreml_row[5])))}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare raw YOLOE PT tensors with Core ML tensors.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--pt", type=Path, required=True, help="YOLOE .pt checkpoint")
    parser.add_argument("--mlpackage", type=Path, required=True, help="Core ML .mlpackage")
    parser.add_argument("--class-id", type=int, default=821, help="Class id to inspect")
    parser.add_argument("--top-k", type=int, default=5, help="How many rows to print for the chosen class")
    parser.add_argument("--row-diffs", type=int, default=10, help="How many raw detection rows to diff")
    args = parser.parse_args()

    import coremltools as ct
    import torch
    from ultralytics import YOLO

    image_path = args.image if args.image.is_file() else REPO_ROOT / args.image
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.pt.is_file():
        raise FileNotFoundError(f"PT checkpoint not found: {args.pt}")
    if not args.mlpackage.is_dir():
        raise FileNotFoundError(f"Core ML package not found: {args.mlpackage}")

    coreml_model = ct.models.MLModel(str(args.mlpackage), compute_units=ct.ComputeUnit.CPU_ONLY)
    side = coreml_image_input_side(coreml_model) or 1280

    source = Image.open(image_path)
    letterboxed = letterbox_rgb(source, side)

    coreml_outputs = coreml_model.predict({"image": letterboxed})
    coreml_det_name, coreml_det, coreml_proto_name, coreml_proto = pick_coreml_outputs(coreml_outputs)

    pt_wrapper = YOLO(str(args.pt))
    head = pt_wrapper.model.model[-1]
    if hasattr(head, "end2end"):
        head.end2end = True
    pt_wrapper.model.eval()
    pt_wrapper.model.export = True

    image_array = np.asarray(letterboxed, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))[None]
    pt_input = torch.from_numpy(image_array)
    with torch.inference_mode():
        pt_raw = pt_wrapper.model(pt_input)
    pt_det, pt_proto = extract_pt_outputs(pt_raw)

    print(f"Image: {image_path}")
    print(f"Letterbox side: {side}")
    print("\nPyTorch raw tensors:")
    print("  det = out[0][0]")
    print("  proto = out[0][1]")
    print(f"  det shape = {pt_det.shape}")
    print(f"  proto shape = {pt_proto.shape}")
    print("\nCore ML raw tensors:")
    print(f"  det = {coreml_det_name}")
    print(f"  proto = {coreml_proto_name}")
    print(f"  det shape = {coreml_det.shape}")
    print(f"  proto shape = {coreml_proto.shape}")

    print_tensor_diff("Detection tensor", pt_det, coreml_det)
    print_tensor_diff("Proto tensor", pt_proto, coreml_proto)
    print_top_rows("PyTorch", pt_det, args.class_id, args.top_k)
    print_top_rows("Core ML", coreml_det, args.class_id, args.top_k)
    print_row_diffs(pt_det, coreml_det, args.row_diffs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
