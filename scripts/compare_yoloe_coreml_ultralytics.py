#!/usr/bin/env python3
"""
Compare YOLOE **PyTorch** `model.predict()` vs **Core ML** raw detection rows on the **same** scene.

Default parity with the standard export path:
- use the source image
- let Ultralytics letterbox with `imgsz`
- do not manually call `model.fuse()`
- do not manually mutate the head

Use this to verify low on-device confidences (e.g. chair ~0.6) against the reference `.pt`:
- If PT predict shows ~0.9 but Core ML rows show ~0.6 → re-export Core ML with
  `nms=False` (see `export_yoloe26_fixed.py` / `yoloe_export_and_backup.py`).
- If both show ~0.6 → checkpoint/export is consistent; expectations vs yoloe-11l-pf differ.

`room.jpg` is not committed; pass your file explicitly:

  python3 scripts/compare_yoloe_coreml_ultralytics.py \\
    --image /path/to/room.jpg \\
    --pt /path/to/yoloe-26l-seg-pf.pt \\
    --mlpackage /path/to/yoloe-26l-seg-pf.mlpackage

Optional: `--class-id 821` (chair in `classes.json`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSES = REPO_ROOT / "Furnit" / "Views" / "FurnitureFit" / "classes.json"


def letterbox_rgb(source: Image.Image, side: int) -> Image.Image:
    rgb = source.convert("RGB")
    scale = min(side / rgb.width, side / rgb.height)
    nw = max(1, int(round(rgb.width * scale)))
    nh = max(1, int(round(rgb.height * scale)))
    resized = rgb.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (side, side), (114, 114, 114))
    pad_x = (side - nw) // 2
    pad_y = (side - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas


def load_class_name(classes_path: Path, class_id: int) -> str:
    if not classes_path.is_file():
        return f"id{class_id}"
    with classes_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return str(data.get(str(class_id), data.get(class_id, f"id{class_id}")))


def coreml_image_input_side(model) -> int | None:
    spec = model.get_spec()
    for feature in spec.description.input:
        if feature.name != "image":
            continue
        width = feature.type.imageType.width
        if width > 0:
            return int(width)
    return None


def detection_output_from_coreml(outputs: dict[str, object]) -> tuple[str, np.ndarray] | None:
    """Resolve raw YOLOE-seg row tensor (1, N, 38): xyxy, conf, class_id, 32 mask coeffs.

    Some bundled 11L packages expose a different graph (e.g. shape ending in 33600); those are skipped.
    """
    preferred_order = ("var_2346", "var_2413", "var_2374")
    for name in preferred_order:
        if name not in outputs:
            continue
        arr = np.asarray(outputs[name])
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 38:
            return (name, arr)
    for name, value in outputs.items():
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 38:
            return (name, arr)
    return None


def coreml_top_for_class(det: np.ndarray, class_id: int) -> list[tuple[float, int, np.ndarray]]:
    """Return sorted (conf, row_idx, row) for rows whose class column matches."""
    rows = det[0]
    out: list[tuple[float, int, np.ndarray]] = []
    for i in range(rows.shape[0]):
        row = rows[i]
        cid = int(round(float(row[5])))
        if cid != class_id:
            continue
        out.append((float(row[4]), i, row))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare YOLOE PyTorch vs Core ML confidences.")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Test image (e.g. room.jpg — place anywhere and pass full path)",
    )
    parser.add_argument("--pt", type=Path, default=None, help="YOLOE .pt checkpoint")
    parser.add_argument(
        "--mlpackage",
        type=Path,
        default=None,
        help="Core ML .mlpackage (default: repo root yoloe-26l-seg-pf.mlpackage)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Letterbox side (default: Core ML image input width from the .mlpackage)",
    )
    parser.add_argument("--class-id", type=int, default=821, help="COCO-style id (821=chair)")
    parser.add_argument(
        "--classes-json",
        type=Path,
        default=DEFAULT_CLASSES,
        help="classes.json for labels",
    )
    args = parser.parse_args()

    image_path = args.image if args.image.is_file() else REPO_ROOT / args.image
    if not image_path.is_file():
        print(f"Image not found: {args.image} (also tried {image_path})", file=sys.stderr)
        return 1

    mlpackage = args.mlpackage or (REPO_ROOT / "yoloe-26l-seg-pf.mlpackage")
    if not mlpackage.is_dir():
        print(f"Core ML package not found: {mlpackage}", file=sys.stderr)
        return 1

    pt = args.pt
    if pt is None:
        for candidate in (
            REPO_ROOT / "android" / "yoloe-11l-seg-pf.pt",
            REPO_ROOT / "yoloe-26l-seg-pf.pt",
        ):
            if candidate.is_file():
                pt = candidate
                break
    if pt is None or not pt.is_file():
        print("Pass --pt to your yoloe-*-seg-pf.pt (26L or 11L).", file=sys.stderr)
        return 1

    class_id = args.class_id
    label = load_class_name(args.classes_json, class_id)

    # --- Core ML (load first to get input size for letterbox) ---
    import coremltools as ct

    cm = ct.models.MLModel(str(mlpackage), compute_units=ct.ComputeUnit.CPU_ONLY)
    side = args.imgsz if args.imgsz is not None else coreml_image_input_side(cm)
    if side is None:
        side = 1280
        print("Could not read image input width from Core ML; using 1280.", file=sys.stderr)

    src = Image.open(image_path)
    letterboxed = letterbox_rgb(src, side)
    print(f"Image: {image_path} source={src.size} letterbox={side}x{side} (pad RGB 114)")
    print(f"Class filter: {class_id} ({label})")
    print(f"PT: {pt}")
    print(f"Core ML: {mlpackage}\n")
    outputs = cm.predict({"image": letterboxed})
    picked = detection_output_from_coreml(outputs)
    if picked is None:
        detail = ", ".join(f"{k}={tuple(np.asarray(v).shape)}" for k, v in outputs.items())
        print(
            "No raw detection tensor with shape (1, N, 38) in Core ML outputs. "
            f"This .mlpackage may use a different export (not comparable to iOS YoloEDetectionParser). "
            f"Outputs: {detail}",
            file=sys.stderr,
        )
        return 1
    det_name, det = picked
    coreml_hits = coreml_top_for_class(det, class_id)
    print(f"Core ML {det_name} (raw rows, same layout as iOS):")
    if not coreml_hits:
        print(f"  No rows with class_id=={class_id}")
    else:
        for rank, (conf, idx, row) in enumerate(coreml_hits[:8], 1):
            x1, y1, x2, y2 = row[:4]
            print(
                f"  #{rank} row={idx} conf={conf:.4f} box=({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f}) "
                f"wh=({x2-x1:.1f},{y2-y1:.1f})"
            )
        print(f"  best_conf={coreml_hits[0][0]:.4f}")

    # --- Ultralytics ---
    from ultralytics import YOLOE

    model = YOLOE(str(pt))
    head = model.model.model[-1]
    print(f"\nHead end2end before predict: {getattr(head, 'end2end', None)}")

    # Let Ultralytics letterbox the *source* image (same 114 pad, same imgsz) to match training/export.
    results = model.predict(
        source=src,
        imgsz=side,
        conf=0.01,
        verbose=False,
    )
    print("Ultralytics predict (source image, imgsz=letterbox side, post-NMS boxes):")
    pt_best = 0.0
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes
        for j in range(len(boxes)):
            cid = int(boxes.cls[j].item())
            if cid != class_id:
                continue
            cf = float(boxes.conf[j].item())
            pt_best = max(pt_best, cf)
            xyxy = boxes.xyxy[j].cpu().numpy()
            print(f"  conf={cf:.4f} xyxy={xyxy!s}")
    if pt_best > 0:
        print(f"  best_conf={pt_best:.4f}")
    else:
        print(f"  No boxes with class_id=={class_id} at conf>=0.01")

    print("\n--- Interpret ---")
    if coreml_hits and pt_best > 0:
        ratio = coreml_hits[0][0] / pt_best if pt_best > 0 else 0
        print(f"CoreML_best / PT_best = {ratio:.3f}")
        if ratio < 0.75:
            print(
                "Large gap: re-export Core ML with nms=False (see scripts/export_yoloe26_fixed.py) "
                "and replace the bundled mlpackage."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
