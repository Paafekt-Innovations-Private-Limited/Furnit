#!/usr/bin/env python3
"""
Isolate each YOLOE **26L seg-pf** detection as a **BGRA** PNG (transparent background),
following Ultralytics: https://docs.ultralytics.com/guides/isolating-segmentation-objects/

Uses **raw pixel masks** ``results.masks.data`` (per-detection probabilities), resized to
``orig_img`` size so **internal holes/gaps** are preserved (unfilled polygon contours).
``r.orig_img`` is BGR for typical file inputs; checkpoint resolution matches
``scripts/draw_mask_yoloe_pt.py``.

Example:
  python3 scripts/isolate_yoloe26l_pf_objects.py --image chair.jpeg --out-dir .build/isolated
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PT_CANDIDATES = (
    lambda: Path(os.environ["YOLOE_PT"]).expanduser() if os.environ.get("YOLOE_PT") else None,
    lambda: REPO_ROOT / "yoloe-26l-seg-pf.pt",
    lambda: Path.home() / "Downloads" / "yoloe-26l-seg-pf.pt",
)


def resolve_yoloe26l_pf_pt(explicit: Path | None) -> tuple[Path, list[Path]]:
    tried: list[Path] = []
    if explicit is not None:
        p = explicit.expanduser()
        tried.append(p)
        return p, tried
    for factory in DEFAULT_PT_CANDIDATES:
        p = factory()
        if p is None:
            continue
        tried.append(p)
        if p.is_file():
            return p, tried
    return tried[-1] if tried else REPO_ROOT / "yoloe-26l-seg-pf.pt", tried


def safe_label_for_filename(label: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", str(label).strip())
    return s[:80] if len(s) > 80 else s


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Isolate YOLOE 26L seg-pf detections as transparent PNG crops (Ultralytics guide)."
    )
    parser.add_argument(
        "--pt",
        type=Path,
        default=None,
        help="Path to yoloe-26l-seg-pf.pt (default: YOLOE_PT, repo root, or ~/Downloads)",
    )
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "chair.jpeg")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / ".build" / "isolated",
        help="Output directory for BGRA PNGs",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--half",
        action="store_true",
        help="FP16 on CUDA only (ignored on CPU/MPS; same as draw_mask_yoloe_pt.py)",
    )
    parser.add_argument(
        "--no-retina-masks",
        action="store_true",
        help="Disable retina_masks (coarser masks)",
    )
    args = parser.parse_args()

    pt, tried = resolve_yoloe26l_pf_pt(args.pt)
    if not pt.is_file():
        print("Checkpoint not found. Tried:", file=sys.stderr)
        for p in tried:
            print(f"  - {p}", file=sys.stderr)
        return 1

    image_path = args.image if args.image.is_file() else REPO_ROOT / args.image
    if not image_path.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    import torch
    from ultralytics import YOLO

    use_half = bool(args.half and torch.cuda.is_available())
    if args.half and not use_half:
        print(
            "Note: --half ignored (CUDA not available). Using float32.",
            file=sys.stderr,
        )

    model = YOLO(str(pt))
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        retina_masks=not args.no_retina_masks,
        half=use_half,
        verbose=False,
    )

    img_stem = image_path.stem
    out_path = args.out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    total = 0
    for batch_i, r in enumerate(results):
        img = r.orig_img
        if img is None:
            continue
        if r.masks is None or r.boxes is None or len(r.boxes) == 0:
            print("No detections with masks; nothing to save.", file=sys.stderr)
            continue
        if r.masks.data is None:
            print("No masks.data; nothing to save.", file=sys.stderr)
            continue

        h, w = img.shape[:2]
        names = r.names

        for det_i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[det_i].item())
            label = names[cls_id]
            tag = safe_label_for_filename(str(label))

            # Raw pixel-wise mask (preserves holes vs filled contours from masks.xy)
            mask_raw = r.masks.data[det_i].detach().cpu().numpy().astype(np.float32)
            if mask_raw.ndim == 3 and mask_raw.shape[0] == 1:
                mask_raw = mask_raw.squeeze(0)
            mask_raw = (np.clip(mask_raw, 0.0, 1.0) * 255.0).astype(np.uint8)
            mask_resized = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_LINEAR)

            isolated = np.dstack([img, mask_resized])

            xyxy = r.boxes.xyxy[det_i].cpu().numpy()
            x1, y1, x2, y2 = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            iso_crop = isolated[y1:y2, x1:x2]
            suffix = f"_{batch_i}" if len(results) > 1 else ""
            save_name = f"{img_stem}{suffix}_{tag}_{det_i}.png"
            out_file = out_path / save_name
            cv2.imwrite(str(out_file), iso_crop)
            print(f"Saved {label} -> {out_file}")
            total += 1

    print(f"Done. {total} PNG(s) in {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
