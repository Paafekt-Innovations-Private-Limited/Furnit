#!/usr/bin/env python3
"""Verify raw RTMDet-Ins Core ML masks on real images.

This is a real integration check, not a mock. It runs the raw Core ML model,
decodes pre-NMS boxes/kernels in Python, executes the CondInst mask MLP, and
fails if the selected chair mask is empty/collapsed.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image


DEFAULT_MODEL = Path("Furnit/Models/RTMDet/rtmdet-ins-m.mlpackage")
DEFAULT_IMAGES = [
    Path("FurnitTests/rtmdet_repeated_chair_frame.jpg"),
]

MODEL_INPUT = 640
MASK_SIZE = 160
PREPROCESS = "stretch"


def load_resized_rgb_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB").resize((640, 640), Image.BILINEAR)


def preprocess_bgr_nchw(image_path: Path) -> np.ndarray:
    image = load_resized_rgb_image(image_path)
    rgb = np.asarray(image).astype(np.float32)
    blob = np.empty((1, 3, 640, 640), dtype=np.float16)
    blob[0, 0] = (rgb[:, :, 2] - 103.53) / 57.375
    blob[0, 1] = (rgb[:, :, 1] - 116.28) / 57.12
    blob[0, 2] = (rgb[:, :, 0] - 123.675) / 58.395
    return blob


def model_input_payload(model: ct.models.MLModel, image_path: Path) -> dict:
    input_description = model.get_spec().description.input[0]
    input_name = input_description.name
    if input_description.type.WhichOneof("Type") == "imageType":
        return {input_name: load_resized_rgb_image(image_path)}
    return {input_name: preprocess_bgr_nchw(image_path)}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def decode_chairs(out: dict[str, np.ndarray], class_id: int, threshold: float) -> list[tuple]:
    candidates: list[tuple] = []
    for side, stride in [(80, 8), (40, 16), (20, 32)]:
        cls = out[f"cls_{side}"][0]
        bbox = out[f"bbox_{side}"][0]
        kernel = out[f"kernel_{side}"][0]
        scores = sigmoid(cls[class_id])
        ys, xs = np.where(scores >= threshold)
        for y, x in zip(ys, xs):
            cx = (float(x) + 0.5) * stride
            cy = (float(y) + 0.5) * stride
            left, top, right, bottom = [float(v) for v in bbox[:, y, x]]
            box = (
                max(0.0, cx - left),
                max(0.0, cy - top),
                min(640.0, cx + right),
                min(640.0, cy + bottom),
            )
            if box[2] <= box[0] + 1 or box[3] <= box[1] + 1:
                continue
            candidates.append((float(scores[y, x]), box, kernel[:, y, x].astype(np.float32), cx, cy, float(stride)))

    kept: list[tuple] = []
    for candidate in sorted(candidates, reverse=True, key=lambda item: item[0])[:500]:
        if all(iou(candidate[1], kept_candidate[1]) <= 0.5 for kept_candidate in kept):
            kept.append(candidate)
        if len(kept) >= 6:
            break
    return kept


def build_mask(candidate: tuple, mask_feat: np.ndarray) -> np.ndarray:
    _, _, kernel, center_x, center_y, stride = candidate
    w1 = kernel[0:80].reshape(8, 10)
    w2 = kernel[80:144].reshape(8, 8)
    w3 = kernel[144:152].reshape(1, 8)
    b1 = kernel[152:160]
    b2 = kernel[160:168]
    b3 = float(kernel[168])

    mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
    mask_stride = MODEL_INPUT / MASK_SIZE
    for y in range(MASK_SIZE):
        for x in range(MASK_SIZE):
            grid_x = (x + 0.5) * mask_stride
            grid_y = (y + 0.5) * mask_stride
            values = np.empty(10, dtype=np.float32)
            values[0] = (center_x - grid_x) / (stride * 8)
            values[1] = (center_y - grid_y) / (stride * 8)
            values[2:] = mask_feat[:, y, x]
            h1 = np.maximum(0, w1 @ values + b1)
            h2 = np.maximum(0, w2 @ h1 + b2)
            logit = float((w3 @ h2)[0] + b3)
            mask[y, x] = 1 / (1 + math.exp(-logit))
    return mask


def mask_bounds(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    positions = np.argwhere(binary)
    if positions.size == 0:
        return None
    y0, x0 = positions.min(axis=0)
    y1, x1 = positions.max(axis=0)
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def map_model_box_to_source(
    box640: tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
    preprocess: str,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box640
    if preprocess == "stretch":
        sx = orig_w / MODEL_INPUT
        sy = orig_h / MODEL_INPUT
        return (
            max(0.0, min(float(orig_w), x1 * sx)),
            max(0.0, min(float(orig_h), y1 * sy)),
            max(0.0, min(float(orig_w), x2 * sx)),
            max(0.0, min(float(orig_h), y2 * sy)),
        )
    if preprocess == "letterbox":
        gain = min(MODEL_INPUT / orig_w, MODEL_INPUT / orig_h)
        pad_x = (MODEL_INPUT - orig_w * gain) * 0.5
        pad_y = (MODEL_INPUT - orig_h * gain) * 0.5
        return (
            max(0.0, min(float(orig_w), (x1 - pad_x) / gain)),
            max(0.0, min(float(orig_h), (y1 - pad_y) / gain)),
            max(0.0, min(float(orig_w), (x2 - pad_x) / gain)),
            max(0.0, min(float(orig_h), (y2 - pad_y) / gain)),
        )
    raise ValueError(f"unknown preprocess mode: {preprocess}")


def swift_mask_sample_coordinate(
    source_x: int,
    source_y: int,
    orig_w: int,
    orig_h: int,
    preprocess: str,
) -> tuple[int, int]:
    """Mirror RTMDetImageInference.maskSampleCoordinate for the 640 raw model."""
    if preprocess == "stretch":
        model_x = (source_x + 0.5) * MODEL_INPUT / orig_w
        model_y = (source_y + 0.5) * MODEL_INPUT / orig_h
    elif preprocess == "letterbox":
        gain = min(MODEL_INPUT / orig_w, MODEL_INPUT / orig_h)
        pad_x = (MODEL_INPUT - orig_w * gain) * 0.5
        pad_y = (MODEL_INPUT - orig_h * gain) * 0.5
        model_x = (source_x + 0.5) * gain + pad_x
        model_y = (source_y + 0.5) * gain + pad_y
    else:
        raise ValueError(f"unknown preprocess mode: {preprocess}")

    sample_x = min(MASK_SIZE - 1, max(0, int(model_x * MASK_SIZE / MODEL_INPUT)))
    sample_y = min(MASK_SIZE - 1, max(0, int(model_y * MASK_SIZE / MODEL_INPUT)))
    return sample_x, sample_y


def mask_to_image_space_swift(
    mask: np.ndarray,
    box640: tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
    mask_threshold: float,
    preprocess: str = PREPROCESS,
) -> np.ndarray:
    """Mirror Swift raw-mask rendering: source bbox loop -> mask grid sample -> threshold."""
    mapped = map_model_box_to_source(box640, orig_w, orig_h, preprocess)
    x_min = max(0, min(orig_w - 1, int(math.floor(mapped[0]))))
    y_min = max(0, min(orig_h - 1, int(math.floor(mapped[1]))))
    x_max = max(0, min(orig_w - 1, int(math.ceil(mapped[2]))))
    y_max = max(0, min(orig_h - 1, int(math.ceil(mapped[3]))))

    out = np.zeros((orig_h, orig_w), dtype=np.uint8)
    if x_max < x_min or y_max < y_min:
        return out

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            sample_x, sample_y = swift_mask_sample_coordinate(x, y, orig_w, orig_h, preprocess)
            value = float(mask[sample_y, sample_x])
            if math.isfinite(value) and value > mask_threshold:
                out[y, x] = 1
    return out


def save_overlay(
    orig_path: Path,
    mask: np.ndarray,
    box640: tuple[float, float, float, float],
    out_path: Path,
    mask_threshold: float,
) -> Path:
    base = Image.open(orig_path).convert("RGB")
    ow, oh = base.size
    binm = mask_to_image_space_swift(mask, box640, ow, oh, mask_threshold)

    rgba = np.array(base.convert("RGBA"))
    green = np.array([0, 255, 0, 255], dtype=np.float32)
    hit = binm == 1
    rgba[hit] = (rgba[hit].astype(np.float32) * 0.5 + green * 0.5).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(out_path)
    cover = int(binm.sum()) / max(1, ow * oh)
    mapped = map_model_box_to_source(box640, ow, oh, PREPROCESS)
    print(
        f">>> overlay saved: {out_path} | coverage={cover:.3%} "
        f"bbox640={[round(v) for v in box640]} "
        f"bbox_source={[round(v) for v in mapped]} preprocess={PREPROCESS}"
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, action="append", dest="images")
    parser.add_argument("--class-id", type=int, default=56, help="COCO class id to verify; 56 is chair")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--overlay-out", type=Path, help="Save a Swift-transform mask overlay PNG for the first valid image")
    parser.add_argument("--overlay-all", action="store_true", help="Save overlays for every valid image")
    parser.add_argument("--require-all", action="store_true", help="Fail if any image has no valid class/mask")
    args = parser.parse_args()

    os.environ.setdefault("TMPDIR", "/private/tmp/coreml-compile-tmp")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    images = args.images or DEFAULT_IMAGES
    missing = [str(path) for path in [args.model, *images] if not path.exists()]
    if missing:
        raise SystemExit("Missing required file(s):\n  " + "\n  ".join(missing))

    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_ONLY)
    failures: list[str] = []
    passed = 0

    for image_path in images:
        out = {name: np.asarray(value) for name, value in model.predict(model_input_payload(model, image_path)).items()}
        chairs = decode_chairs(out, args.class_id, args.threshold)
        print(f"\nimage={image_path}")
        print(f"raw outputs: " + ", ".join(f"{k}={v.shape}" for k, v in sorted(out.items())))
        print(f"chair candidates kept={len(chairs)}")
        if not chairs:
            if args.require_all:
                failures.append(f"{image_path.name}: no decoded chair candidates")
            continue

        mask = build_mask(chairs[0], out["mask_feat"][0].astype(np.float32))
        binary = mask > args.mask_threshold
        bounds = mask_bounds(binary)
        pixels = int(binary.sum())
        score, box, *_ = chairs[0]
        print(
            f"selected score={score:.4f} box=({box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}) "
            f"mask_min={float(mask.min()):.6f} mask_max={float(mask.max()):.6f} "
            f"mask_mean={float(mask.mean()):.6f} pixels={pixels} bounds={bounds}"
        )

        if bounds is None or pixels < 32 or float(mask.max()) < 0.5:
            failures.append(f"{image_path.name}: decoded mask collapsed pixels={pixels} bounds={bounds}")
        else:
            passed += 1
            if args.overlay_out and (args.overlay_all or passed == 1):
                out_path = args.overlay_out
                if args.overlay_all and len(images) > 1:
                    out_path = out_path.with_name(f"{out_path.stem}_{image_path.stem}{out_path.suffix}")
                save_overlay(image_path, mask, box, out_path, args.mask_threshold)

    if passed == 0:
        failures.append("no image produced a valid decoded mask")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
