#!/usr/bin/env python3
"""Build a union exclude mask from RTMDet-Ins (Core ML) for room measurement."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "Furnit/Models/RTMDet/rtmdet-ins-m.mlpackage"
CLASSES_JSON = REPO_ROOT / "scripts/rtmdet-coco80-classes.json"
MODEL_INPUT = 640
MASK_SIZE = 160
PREPROCESS = "stretch"

# COCO classes to exclude from structural plane sampling (furniture / people / clutter).
DEFAULT_EXCLUDE_CLASS_IDS = frozenset({
    0,   # person
    24,  # backpack
    26,  # handbag
    28,  # suitcase
    39,  # bottle
    41,  # cup
    56,  # chair
    57,  # couch
    58,  # potted plant
    59,  # bed
    60,  # dining table
    62,  # tv
    63,  # laptop
    67,  # cell phone
    75,  # vase
})


def _load_resized_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize((MODEL_INPUT, MODEL_INPUT), Image.BILINEAR)


def _preprocess_bgr_nchw(rgb: Image.Image) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    blob = np.empty((1, 3, MODEL_INPUT, MODEL_INPUT), dtype=np.float16)
    blob[0, 0] = (arr[:, :, 2] - 103.53) / 57.375
    blob[0, 1] = (arr[:, :, 1] - 116.28) / 57.12
    blob[0, 2] = (arr[:, :, 0] - 123.675) / 58.395
    return blob


def _model_input_payload(model, rgb: Image.Image) -> dict:
    input_description = model.get_spec().description.input[0]
    input_name = input_description.name
    if input_description.type.WhichOneof("Type") == "imageType":
        return {input_name: _load_resized_rgb(rgb)}
    return {input_name: _preprocess_bgr_nchw(_load_resized_rgb(rgb))}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _decode_class_candidates(
    outputs: dict[str, np.ndarray],
    class_id: int,
    score_threshold: float,
) -> list[tuple]:
    candidates: list[tuple] = []
    for side, stride in ((80, 8), (40, 16), (20, 32)):
        cls = outputs[f"cls_{side}"][0]
        bbox = outputs[f"bbox_{side}"][0]
        kernel = outputs[f"kernel_{side}"][0]
        scores = _sigmoid(cls[class_id])
        ys, xs = np.where(scores >= score_threshold)
        for y, x in zip(ys, xs):
            cx = (float(x) + 0.5) * stride
            cy = (float(y) + 0.5) * stride
            left, top, right, bottom = [float(v) for v in bbox[:, y, x]]
            box = (
                max(0.0, cx - left),
                max(0.0, cy - top),
                min(float(MODEL_INPUT), cx + right),
                min(float(MODEL_INPUT), cy + bottom),
            )
            if box[2] <= box[0] + 1 or box[3] <= box[1] + 1:
                continue
            candidates.append(
                (float(scores[y, x]), box, kernel[:, y, x].astype(np.float32), cx, cy, float(stride)),
            )

    kept: list[tuple] = []
    for candidate in sorted(candidates, reverse=True, key=lambda item: item[0])[:300]:
        if all(_iou(candidate[1], kept_candidate[1]) <= 0.5 for kept_candidate in kept):
            kept.append(candidate)
        if len(kept) >= 8:
            break
    return kept


def _build_mask(candidate: tuple, mask_feat: np.ndarray) -> np.ndarray:
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
            h1 = np.maximum(0.0, w1 @ values + b1)
            h2 = np.maximum(0.0, w2 @ h1 + b2)
            logit = float((w3 @ h2)[0] + b3)
            mask[y, x] = 1.0 / (1.0 + math.exp(-logit))
    return mask


def _map_model_box_to_source(
    box640: tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box640
    sx = orig_w / MODEL_INPUT
    sy = orig_h / MODEL_INPUT
    return (
        max(0.0, min(float(orig_w), x1 * sx)),
        max(0.0, min(float(orig_h), y1 * sy)),
        max(0.0, min(float(orig_w), x2 * sx)),
        max(0.0, min(float(orig_h), y2 * sy)),
    )


def _mask_sample_coordinate(source_x: int, source_y: int, orig_w: int, orig_h: int) -> tuple[int, int]:
    model_x = (source_x + 0.5) * MODEL_INPUT / orig_w
    model_y = (source_y + 0.5) * MODEL_INPUT / orig_h
    sample_x = min(MASK_SIZE - 1, max(0, int(model_x * MASK_SIZE / MODEL_INPUT)))
    sample_y = min(MASK_SIZE - 1, max(0, int(model_y * MASK_SIZE / MODEL_INPUT)))
    return sample_x, sample_y


def _mask_to_image_space(
    mask: np.ndarray,
    box640: tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
    mask_threshold: float,
) -> np.ndarray:
    mapped = _map_model_box_to_source(box640, orig_w, orig_h)
    x_min = max(0, min(orig_w - 1, int(math.floor(mapped[0]))))
    y_min = max(0, min(orig_h - 1, int(math.floor(mapped[1]))))
    x_max = max(0, min(orig_w - 1, int(math.ceil(mapped[2]))))
    y_max = max(0, min(orig_h - 1, int(math.ceil(mapped[3]))))

    out = np.zeros((orig_h, orig_w), dtype=bool)
    if x_max < x_min or y_max < y_min:
        return out

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            sample_x, sample_y = _mask_sample_coordinate(x, y, orig_w, orig_h)
            value = float(mask[sample_y, sample_x])
            if math.isfinite(value) and value > mask_threshold:
                out[y, x] = True
    return out


def build_rtmdet_exclude_mask(
    rgb: np.ndarray,
    *,
    model_path: Path = DEFAULT_MODEL,
    class_ids: frozenset[int] = DEFAULT_EXCLUDE_CLASS_IDS,
    score_threshold: float = 0.30,
    mask_threshold: float = 0.50,
) -> tuple[np.ndarray, dict]:
    """Return (exclude_mask, meta). exclude_mask is True where furniture/clutter was detected."""
    height, width = rgb.shape[:2]
    meta: dict = {
        "source": "none",
        "model_path": str(model_path),
        "detections": [],
    }
    exclude = np.zeros((height, width), dtype=bool)
    if not model_path.exists():
        meta["source"] = "model_missing"
        return exclude, meta

    try:
        import coremltools as ct
    except ImportError:
        meta["source"] = "coremltools_missing"
        return exclude, meta

    os.environ.setdefault("TMPDIR", "/private/tmp/coreml-compile-tmp")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    image = Image.fromarray(rgb)
    outputs = {
        name: np.asarray(value)
        for name, value in model.predict(_model_input_payload(model, image)).items()
    }
    mask_feat = outputs.get("mask_feat")
    if mask_feat is None:
        meta["source"] = "mask_feat_missing"
        return exclude, meta

    mask_feat = mask_feat[0].astype(np.float32)
    classes = json.loads(CLASSES_JSON.read_text(encoding="utf-8")) if CLASSES_JSON.is_file() else {}

    for class_id in sorted(class_ids):
        candidates = _decode_class_candidates(outputs, class_id, score_threshold)
        for candidate in candidates:
            score, box, *_ = candidate
            mask = _build_mask(candidate, mask_feat)
            hit = _mask_to_image_space(mask, box, width, height, mask_threshold)
            exclude |= hit
            meta["detections"].append({
                "class_id": class_id,
                "class_name": classes.get(str(class_id), str(class_id)),
                "score": round(score, 3),
                "pixels": int(hit.sum()),
            })

    meta["source"] = "rtmdet_ins_coreml"
    meta["exclude_pixels"] = int(exclude.sum())
    meta["exclude_fraction"] = round(float(exclude.mean()), 4)
    return exclude, meta


def apply_exclude_mask(masks: dict[str, np.ndarray], exclude: np.ndarray) -> dict[str, np.ndarray]:
    """Remove excluded pixels from all region masks."""
    if not exclude.any():
        return masks
    return {name: mask & ~exclude for name, mask in masks.items()}
