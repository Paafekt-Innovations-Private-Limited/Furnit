#!/usr/bin/env python3
"""
**YOLOE 26L seg-pf (``yoloe-26l-seg-pf.pt``)** — Ultralytics-recommended mask visualization:

1. ``YOLO(path).predict(..., retina_masks=True)`` for full-resolution masks where supported.
2. ``Results.plot(pil=True, conf=True, boxes=True)`` for a PIL RGB image (no manual BGR flip).

Checkpoint resolution (when ``--pt`` is omitted): ``$YOLOE_PT``, then repo
``yoloe-26l-seg-pf.pt``, then ``~/Downloads/yoloe-26l-seg-pf.pt`` (same idea as
``scripts/export_yoloe26_onemany_user.py``).

For **26L @ imgsz=1280** on a **Mac**, use **Float32** (default): ``--half`` is **CUDA-only**
here because FP16 on CPU/MPS can trigger ``process_mask_native`` dtype mismatches
(``protos.float()`` vs half tensors). On NVIDIA GPU, ``--half`` may reduce VRAM.

**YOLO11 vs YOLO26 (YOLOE)** — YOLO26 uses ``YOLOESegment26`` with **Proto26** for masks
(``construct_result`` / segmentation path), which tends toward **smoother, more globally
consistent** regions than older heads; small internal gaps (e.g. chair handle openings)
can look more “filled” by design. Mitigations you can try from this script:

- Keep **``retina_masks=True``** (default; use ``--no-retina-masks`` only to compare).
- Raise ``--conf`` slightly if the model over-groups foreground.
- **Lower ``--iou``** if NMS is merging instances you want separated (default matches Ultralytics NMS).

See also:
  - `YOLOESegment26`: https://docs.ultralytics.com/reference/nn/modules/head/#ultralytics.nn.modules.head.YOLOESegment26
  - `SegmentationPredictor`: https://docs.ultralytics.com/reference/models/yolo/segment/predict/

Docs:
  - ``Results.plot``: https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.plot
  - Isolating segmentation objects: https://docs.ultralytics.com/guides/isolating-segmentation-objects/

Example (26L PF, 1280, Float32 — recommended on Mac):
  python3 scripts/draw_mask_yoloe_pt.py --image chair.jpeg --imgsz 1280 --out .build/yoloe26l_pf_mask.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PT_CANDIDATES = (
    lambda: Path(os.environ["YOLOE_PT"]).expanduser() if os.environ.get("YOLOE_PT") else None,
    lambda: REPO_ROOT / "yoloe-26l-seg-pf.pt",
    lambda: Path.home() / "Downloads" / "yoloe-26l-seg-pf.pt",
)


def resolve_yoloe26l_pf_pt(explicit: Path | None) -> tuple[Path, list[Path]]:
    """Return checkpoint path and list of candidates tried (for errors)."""
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw masks from YOLOE 26L seg-pf .pt (Ultralytics predict + plot(pil=True))."
    )
    parser.add_argument(
        "--pt",
        type=Path,
        default=None,
        help="Path to yoloe-26l-seg-pf.pt (default: YOLOE_PT, repo root, or ~/Downloads)",
    )
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "chair.jpeg")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / ".build" / "yoloe_pt_mask_plot.png")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference square size (e.g. 1280 for fine protos)")
    parser.add_argument("--conf", type=float, default=0.25, help="Min detection confidence (NMS)")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        metavar="N",
        help="NMS IoU threshold (lower can split merged YOLO26 masks; Ultralytics default 0.7)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="FP16 inference on CUDA only (ignored on CPU/MPS to avoid mask dtype errors)",
    )
    parser.add_argument(
        "--no-retina-masks",
        action="store_true",
        help="Disable retina_masks (faster, coarser mask upsampling in some paths)",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    pt, tried = resolve_yoloe26l_pf_pt(args.pt)
    if not pt.is_file():
        print("Checkpoint not found. Tried:", file=sys.stderr)
        for p in tried:
            print(f"  - {p}", file=sys.stderr)
        print("Set --pt or YOLOE_PT to your yoloe-26l-seg-pf.pt path.", file=sys.stderr)
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
            "Note: --half ignored (CUDA not available). Using float32 for masks — "
            "avoids Ultralytics process_mask_native dtype mismatch on CPU/MPS.",
            file=sys.stderr,
        )

    model = YOLO(str(pt))
    predict_kw: dict = {
        "source": str(image_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
        "retina_masks": not args.no_retina_masks,
        "half": use_half,
    }

    results = model.predict(**predict_kw)
    r = results[0]

    plotted = r.plot(pil=True, conf=True, boxes=True)
    if isinstance(plotted, Image.Image):
        vis = plotted
    else:
        import numpy as np

        vis = Image.fromarray(np.ascontiguousarray(plotted[..., ::-1]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vis.save(args.out, quality=95)

    n_mask = 0 if r.masks is None else len(r.masks)
    n_box = 0 if r.boxes is None else len(r.boxes)
    print()
    print("=" * 60)
    print("YOLOE 26L seg-pf — Ultralytics predict + plot(pil=True)")
    print("=" * 60)
    print(f"PT:      {pt}")
    print(f"Image:   {image_path}")
    print(
        f"imgsz:   {args.imgsz}  conf: {args.conf}  iou: {args.iou}  "
        f"retina_masks: {not args.no_retina_masks}  half: {use_half}"
    )
    print(f"masks:   {n_mask}  boxes: {n_box}")
    print(f"Saved:   {args.out}")
    print("=" * 60)

    if not args.no_show:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 10))
            plt.imshow(vis)
            plt.axis("off")
            plt.title(f"{pt.name}  ({n_mask} masks)")
            plt.tight_layout()
            plt.show()
        except Exception as exc:  # noqa: BLE001
            print(f"(matplotlib show skipped: {exc})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
