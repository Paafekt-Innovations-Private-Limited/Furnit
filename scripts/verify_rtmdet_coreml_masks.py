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
    Path("/Users/al/Downloads/WhatsApp Image 2026-06-08 at 15.19.42 (1).jpeg"),
    Path("/Users/al/Downloads/WhatsApp Image 2026-06-08 at 16.08.53.jpeg"),
]


def preprocess_bgr_nchw(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((640, 640), Image.BILINEAR)
    rgb = np.asarray(image).astype(np.float32)
    blob = np.empty((1, 3, 640, 640), dtype=np.float16)
    blob[0, 0] = (rgb[:, :, 2] - 103.53) / 57.375
    blob[0, 1] = (rgb[:, :, 1] - 116.28) / 57.12
    blob[0, 2] = (rgb[:, :, 0] - 123.675) / 58.395
    return blob


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

    mask = np.zeros((80, 80), dtype=np.float32)
    for y in range(80):
        for x in range(80):
            grid_x = (x + 0.5) * 8
            grid_y = (y + 0.5) * 8
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, action="append", dest="images")
    parser.add_argument("--class-id", type=int, default=56, help="COCO class id to verify; 56 is chair")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
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
        out = {name: np.asarray(value) for name, value in model.predict({"input": preprocess_bgr_nchw(image_path)}).items()}
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
