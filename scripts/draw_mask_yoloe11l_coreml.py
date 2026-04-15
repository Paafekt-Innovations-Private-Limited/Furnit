#!/usr/bin/env python3
"""
Draw YOLOE-11L segmentation masks from a Core ML model in **legacy anchor format**:
  det:  [1, 4 + num_classes + 32, num_anchors]  e.g. [1, 4621, 33600]
  proto: [1, 32, Hp, Wp]  e.g. [1, 32, 320, 320]

Matches ``YoloEDetectionParser`` / docs (center-x, center-y, w, h per anchor).

Default model path:
  .build/yoloe-11l-seg-pf-from-onnx.mlmodel

Input is MultiArray ``images`` NCHW float32 (not ImageType) — letterbox + /255.

Example:
  python3 scripts/draw_mask_yoloe11l_coreml.py --image chair.jpeg --out .build/yoloe11l_mask.png

**Global red/orange “wash”:** ``union`` mode merges **every** detection into one layer (floor +
walls + chair + …), so the tint covers almost the whole frame. Use **``--viz per_instance``**
(default): each object gets its **own color**. Use ``--single-top`` for only the highest-score
detection (e.g. isolate the main chair).

Note: Some exports use **MLImage** input (``image``, 1280² RGB) instead of MultiArray
``images``; this script probes the spec and passes a letterboxed PIL for Image inputs.

Python ``coremltools`` on some legacy ``.mlmodel`` blobs returns error -1; the script
can fall back to ``android/yoloe-11l-seg-pf.onnx``. Use ``--backend onnx`` to force ONNX.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


def letterbox_pil(source: Image.Image, side: int) -> tuple[Image.Image, float, int, int]:
    rgb = source.convert("RGB")
    w, h = rgb.size
    scale = min(side / w, side / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = rgb.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (side, side), (114, 114, 114))
    pad_x = (side - nw) // 2
    pad_y = (side - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def letterbox_mask_to_source(
    mask_lb: np.ndarray,
    side: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    src_w: int,
    src_h: int,
) -> np.ndarray:
    nh = max(1, int(round(src_h * scale)))
    nw = max(1, int(round(src_w * scale)))
    crop = mask_lb[pad_y : pad_y + nh, pad_x : pad_x + nw].astype(np.float32)
    pil_crop = Image.fromarray(np.asarray(crop, dtype=np.float32))
    return np.array(pil_crop.resize((src_w, src_h), Image.BILINEAR), dtype=np.float32)


def blend_overlay(
    rgb: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.4,
    *,
    binarize: bool = True,
    mask_threshold: float = 0.5,
) -> Image.Image:
    """
    Tint pixels inside the mask. By default **binarizes** the mask at ``mask_threshold`` so
    the overlay is crisp; soft sigmoid values (0.1–0.9) everywhere look like a cloudy red wash.
    """
    base = np.asarray(rgb.convert("RGB"), dtype=np.float32) / 255.0
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if binarize:
        m = (m >= mask_threshold).astype(np.float32)
    col = np.array(color, dtype=np.float32) / 255.0
    for c in range(3):
        base[:, :, c] = base[:, :, c] * (1.0 - alpha * m) + col[c] * (alpha * m)
    return Image.fromarray(np.clip(base * 255.0, 0, 255).astype(np.uint8))


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + ba - inter + 1e-6)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int) -> list[int]:
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        order = rest[ious <= iou_thres]
    return keep


def parse_anchors_xywh(
    det: np.ndarray, conf_thres: float, top_pre_nms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    det: [1, F, A] feature-major (same as Swift).
    Returns boxes_xyxy [N,4], scores [N], cls [N], coeffs [N,32] before NMS candidates.
    """
    d = det[0]
    f, a = d.shape
    nc = f - 4 - 32
    if nc <= 0:
        raise ValueError(f"Bad det shape {d.shape}")
    box = d[0:4, :]
    scores_all = d[4 : 4 + nc, :]
    coeffs_all = d[4 + nc :, :]

    max_scores = scores_all.max(axis=0)
    max_cls = scores_all.argmax(axis=0)

    cand_idx = np.where(max_scores >= conf_thres)[0]
    if cand_idx.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 32), dtype=np.float32),
        )

    order = cand_idx[np.argsort(-max_scores[cand_idx])][:top_pre_nms]

    boxes_xyxy = []
    sc = []
    cl = []
    cf = []
    for anchor in order:
        cx, cy, w, h = float(box[0, anchor]), float(box[1, anchor]), float(box[2, anchor]), float(box[3, anchor])
        if not (np.isfinite(cx) and np.isfinite(cy) and w > 0 and h > 0):
            continue
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        boxes_xyxy.append([x1, y1, x2, y2])
        sc.append(float(max_scores[anchor]))
        cl.append(int(max_cls[anchor]))
        cf.append(coeffs_all[:, anchor].astype(np.float32))

    if not boxes_xyxy:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 32), dtype=np.float32),
        )

    return (
        np.array(boxes_xyxy, dtype=np.float32),
        np.array(sc, dtype=np.float32),
        np.array(cl, dtype=np.int64),
        np.stack(cf, axis=0),
    )


def _masks_logits_per_instance(
    proto: np.ndarray,
    coeffs: np.ndarray,
    boxes_xyxy: np.ndarray,
    letterbox_side: int,
):
    """Return (N, H, H) mask logits in letterbox space, box-cropped per instance."""
    import torch
    import torch.nn.functional as F

    if coeffs.shape[0] == 0:
        return None

    p = torch.from_numpy(proto[0])
    mh, mw = p.shape[1], p.shape[2]
    c = torch.from_numpy(coeffs).float()
    masks = (c @ p.view(32, -1)).view(-1, mh, mw).unsqueeze(1)
    masks = F.interpolate(masks, size=(letterbox_side, letterbox_side), mode="bilinear", align_corners=False).squeeze(1)
    b = torch.from_numpy(boxes_xyxy).float()
    for i in range(masks.shape[0]):
        x1, y1, x2, y2 = b[i].round().int().tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(letterbox_side - 1, x2), min(letterbox_side - 1, y2)
        t = masks[i]
        if y2 > y1 and x2 > x1:
            t[:y1] = 0
            t[y2 + 1 :] = 0
            t[:, :x1] = 0
            t[:, x2 + 1 :] = 0
    return masks


def masks_from_coeffs_union(
    proto: np.ndarray,
    coeffs: np.ndarray,
    boxes_xyxy: np.ndarray,
    letterbox_side: int,
) -> np.ndarray:
    """Union of per-instance masks — **looks like a red wash** when many large objects exist."""
    import torch

    logits = _masks_logits_per_instance(proto, coeffs, boxes_xyxy, letterbox_side)
    if logits is None:
        return np.zeros((letterbox_side, letterbox_side), dtype=np.float32)
    union, _ = logits.max(dim=0)
    return torch.sigmoid(union).numpy().astype(np.float32)


def masks_from_coeffs_per_instance(
    proto: np.ndarray,
    coeffs: np.ndarray,
    boxes_xyxy: np.ndarray,
    letterbox_side: int,
) -> np.ndarray:
    """(N, H, H) sigmoid masks in letterbox space — use for **per-object** overlays."""
    import torch

    logits = _masks_logits_per_instance(proto, coeffs, boxes_xyxy, letterbox_side)
    if logits is None:
        return np.zeros((0, letterbox_side, letterbox_side), dtype=np.float32)
    return torch.sigmoid(logits).numpy().astype(np.float32)


def composite_colored_instance_masks(
    rgb: Image.Image,
    per_instance_masks_src: list[np.ndarray],
    colors: list[tuple[int, int, int]],
    alpha: float,
    *,
    binarize: bool,
    mask_threshold: float,
) -> Image.Image:
    """Stack tints on the original image (later instances paint on top of earlier)."""
    base = np.asarray(rgb.convert("RGB"), dtype=np.float32) / 255.0
    for m, col in zip(per_instance_masks_src, colors):
        mm = np.clip(m.astype(np.float32), 0.0, 1.0)
        if binarize:
            mm = (mm >= mask_threshold).astype(np.float32)
        c = np.array(col, dtype=np.float32) / 255.0
        for ch in range(3):
            base[:, :, ch] = base[:, :, ch] * (1.0 - alpha * mm) + c[ch] * (alpha * mm)
    return Image.fromarray(np.clip(base * 255.0, 0, 255).astype(np.uint8))


def instance_overlay_colors(n: int) -> list[tuple[int, int, int]]:
    """Distinct BGR-friendly RGB colors for up to ``n`` instances."""
    palette = [
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (255, 215, 0),
        (186, 85, 211),
        (0, 191, 255),
        (255, 105, 180),
        (154, 205, 50),
        (255, 140, 0),
        (106, 90, 205),
        (32, 178, 170),
        (220, 20, 60),
    ]
    return [palette[i % len(palette)] for i in range(n)]


def coreml_input_spec(model) -> tuple[str, int]:
    """Return input tensor name and square side (for MultiArray NCHW)."""
    name, side, _kind = probe_coreml_inputs(model)
    return name, side


def probe_coreml_inputs(model) -> tuple[str, int, str]:
    """
    Return (input_name, square_side, kind) where kind is ``multiarray`` (NCHW float blob)
    or ``image`` (RGB PIL / MLImage — use ``predict({name: letterboxed_pil})``).
    """
    spec = model.get_spec()
    for feature in spec.description.input:
        t = feature.type
        w = t.WhichOneof("Type")
        if w == "multiArrayType":
            shape = list(t.multiArrayType.shape)
            if len(shape) == 4 and shape[2] == shape[3]:
                return feature.name, int(shape[2]), "multiarray"
        if w == "imageType":
            it = t.imageType
            iw, ih = int(it.width), int(it.height)
            if iw > 0 and ih > 0 and iw == ih:
                return feature.name, iw, "image"
    raise RuntimeError("Could not find NCHW MultiArray or square Image input in model.")


def infer_onnx_11l(onnx_path: Path, blob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: blob})
    return out[0], out[1]


def infer_coreml_11l(
    model_path: Path,
    input_name: str,
    input_kind: str,
    letterbox_rgb: Image.Image,
    blob_nchw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import coremltools as ct

    ml = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    if input_kind == "image":
        outputs = ml.predict({input_name: letterbox_rgb.convert("RGB")})
    else:
        outputs = ml.predict({input_name: blob_nchw})
    det = proto = None
    for k, v in outputs.items():
        a = np.asarray(v, dtype=np.float32)
        if a.ndim == 3 and a.shape[2] > 1000:
            det = a
        elif a.ndim == 4 and a.shape[1] == 32:
            proto = a
    if det is None or proto is None:
        shapes = {k: tuple(np.asarray(v).shape) for k, v in outputs.items()}
        raise RuntimeError(f"Unexpected Core ML outputs: {shapes}")
    return det, proto


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw masks from YOLOE-11L Core ML (anchor format).")
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / ".build" / "yoloe-11l-seg-pf-from-onnx.mlmodel",
        help="Path to .mlmodel / .mlpackage (Core ML)",
    )
    parser.add_argument(
        "--onnx-fallback",
        type=Path,
        default=REPO_ROOT / "android" / "yoloe-11l-seg-pf.onnx",
        help="Same graph as typical onnx→coreml 11L export; used if Core ML predict fails in Python",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "coreml", "onnx"),
        default="auto",
        help="auto: try Core ML on --model, then ONNX fallback",
    )
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "chair.jpeg")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / ".build" / "yoloe11l_mask_overlay.png")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=16)
    parser.add_argument("--top-pre-nms", type=int, default=400)
    parser.add_argument(
        "--soft-mask",
        action="store_true",
        help="Blend using soft mask probabilities (wider fuzzy red haze). Default is binary mask for crisp fill.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="When not --soft-mask: pixels with mask >= this value are tinted (typical 0.45–0.55).",
    )
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.35,
        help="Overlay tint strength 0…1 (lower = subtler color).",
    )
    parser.add_argument(
        "--viz",
        choices=("per_instance", "union"),
        default="per_instance",
        help="per_instance: each detection its own color (default). union: one combined mask (often looks like a full-frame tint).",
    )
    parser.add_argument(
        "--single-top",
        action="store_true",
        help="Keep only the single highest-confidence detection after NMS (clean single-object mask).",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    model_path = args.model
    has_coreml_path = model_path.is_file() or model_path.is_dir()
    if args.backend != "onnx" and not has_coreml_path:
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1
    if args.backend == "onnx" and not args.onnx_fallback.is_file():
        print(f"ONNX not found: {args.onnx_fallback}", file=sys.stderr)
        return 1

    image_path = args.image if args.image.is_file() else REPO_ROOT / args.image
    if not image_path.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    import coremltools as ct

    src = Image.open(image_path).convert("RGB")
    src_w, src_h = src.size

    # Resolve input side + letterbox once
    side = 1280
    input_name = "images"
    input_kind = "multiarray"
    if args.backend in ("coreml", "auto") and has_coreml_path:
        try:
            ml_probe = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_ONLY)
            input_name, side, input_kind = probe_coreml_inputs(ml_probe)
        except Exception as exc:
            if args.backend == "coreml":
                print(f"Failed to load Core ML model: {exc}", file=sys.stderr)
                return 1

    letterboxed, scale, pad_x, pad_y = letterbox_pil(src, side)
    arr = np.asarray(letterboxed, dtype=np.float32) / 255.0
    blob = np.transpose(arr, (2, 0, 1))[None, ...]

    backend_note = ""
    det: np.ndarray | None = None
    proto: np.ndarray | None = None

    if args.backend == "onnx":
        det, proto = infer_onnx_11l(args.onnx_fallback, blob)
        backend_note = f"ONNX {args.onnx_fallback.name}"
    elif args.backend == "coreml":
        det, proto = infer_coreml_11l(model_path, input_name, input_kind, letterboxed, blob)
        backend_note = f"Core ML {model_path.name}"
    else:
        # auto: prefer Core ML inference; ONNX matches the same onnx→coreml graph
        if not has_coreml_path:
            det, proto = infer_onnx_11l(args.onnx_fallback, blob)
            backend_note = f"ONNX {args.onnx_fallback.name}"
        else:
            try:
                det, proto = infer_coreml_11l(model_path, input_name, input_kind, letterboxed, blob)
                backend_note = f"Core ML {model_path.name}"
            except Exception as exc:
                if not args.onnx_fallback.is_file():
                    print(f"Core ML failed ({exc}) and no ONNX at {args.onnx_fallback}", file=sys.stderr)
                    return 1
                print(
                    f"Core ML predict failed ({exc!r}); using equivalent ONNX: {args.onnx_fallback}",
                    file=sys.stderr,
                )
                det, proto = infer_onnx_11l(args.onnx_fallback, blob)
                backend_note = f"ONNX (fallback) {args.onnx_fallback.name}"

    boxes, scores, classes, coeffs = parse_anchors_xywh(det, args.conf, args.top_pre_nms)
    if boxes.shape[0] == 0:
        print("No anchors above conf threshold.")
        overlay = src.copy()
    else:
        keep = nms_xyxy(boxes, scores, args.iou, args.max_det)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
        coeffs = coeffs[keep]

        if args.single_top and scores.shape[0] > 0:
            j = int(np.argmax(scores))
            boxes = boxes[j : j + 1]
            scores = scores[j : j + 1]
            classes = classes[j : j + 1]
            coeffs = coeffs[j : j + 1]

        if args.viz == "per_instance":
            inst_lb = masks_from_coeffs_per_instance(proto, coeffs, boxes, side)
            masks_src = [
                letterbox_mask_to_source(inst_lb[i], side, scale, pad_x, pad_y, src_w, src_h)
                for i in range(inst_lb.shape[0])
            ]
            cols = instance_overlay_colors(len(masks_src))
            overlay = composite_colored_instance_masks(
                src,
                masks_src,
                cols,
                args.blend_alpha,
                binarize=not args.soft_mask,
                mask_threshold=args.mask_threshold,
            )
        else:
            mask_lb = masks_from_coeffs_union(proto, coeffs, boxes, side)
            mask_src = letterbox_mask_to_source(mask_lb, side, scale, pad_x, pad_y, src_w, src_h)
            overlay = blend_overlay(
                src,
                mask_src,
                color=(255, 140, 60),
                alpha=args.blend_alpha,
                binarize=not args.soft_mask,
                mask_threshold=args.mask_threshold,
            )

        for i in range(len(scores)):
            print(f"  det {i}: class={int(classes[i])} score={scores[i]:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.out, quality=95)

    print()
    print("=" * 60)
    print("YOLOE-11L Core ML mask overlay")
    print("=" * 60)
    print(f"Backend:   {backend_note}")
    print(f"Model:     {model_path}")
    print(f"Input:     {input_name}  letterbox {side}x{side}")
    print(f"Image:     {image_path}  ({src_w}x{src_h})")
    print(f"det:       {tuple(det.shape)}  proto: {tuple(proto.shape)}")
    print(f"Saved:     {args.out}")
    print("=" * 60)

    if not args.no_show:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))
            plt.imshow(overlay)
            plt.axis("off")
            plt.title(f"YOLOE-11L mask ({side}px) — {input_name}")
            plt.tight_layout()
            plt.show()
        except Exception as exc:  # noqa: BLE001
            print(f"(matplotlib show skipped: {exc})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
