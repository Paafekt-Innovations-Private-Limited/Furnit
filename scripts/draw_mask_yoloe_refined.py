#!/usr/bin/env python3
"""
**YOLOE 26L seg-pf — Enhanced mask refinement pipeline.**

Adds three post-processing refinement stages on top of Ultralytics predict:

1. **Morphological refinement** (``--refine morph``): erode→dilate to clean jagged edges,
   then Gaussian blur the mask boundary for sub-pixel smoothness.
2. **Guided filter** (``--refine guided``): uses the original RGB image as a guide to
   snap mask edges onto real luminance boundaries (chair legs, spindles).
   Requires ``opencv-contrib-python`` for ``cv2.ximgproc.guidedFilter``.
3. **CRF densification** (``--refine crf``): fully-connected CRF that respects pixel
   color similarity — the gold standard for mask cleanup. Requires ``pydensecrf``.

All three can be combined: ``--refine morph guided crf`` applies them in sequence.

Morphological params:
  --morph-kernel     Erosion/dilation kernel size (default: 3)
  --morph-iter       Erosion/dilation iterations (default: 1)
  --boundary-blur    Gaussian blur sigma on mask boundary band (default: 1.5)

Guided filter params:
  --guide-radius     Guided filter radius (default: 8)
  --guide-eps        Guided filter regularization (default: 0.01; lower = sharper edges)

CRF params:
  --crf-iter         CRF inference iterations (default: 5)
  --crf-sxy          CRF spatial sigma (default: 3)
  --crf-srgb         CRF color sigma (default: 10)
  --crf-compat       CRF compatibility (default: 10)

Example (recommended for chair masks on Mac):
  python3 scripts/draw_mask_yoloe_refined.py \
      --image chair.jpeg --imgsz 1280 \
      --refine morph guided \
      --out .build/yoloe26l_refined_mask.png

Example (maximum quality — needs pydensecrf):
  pip install pydensecrf
  python3 scripts/draw_mask_yoloe_refined.py \
      --image chair.jpeg --imgsz 1280 \
      --refine morph guided crf \
      --out .build/yoloe26l_crf_mask.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
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


def refine_mask_morphological(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    boundary_blur_sigma: float = 1.5,
) -> np.ndarray:
    """
    Erode → dilate to remove jagged single-pixel spurs, then Gaussian-blur
    a narrow band around the mask boundary for sub-pixel edge smoothness.

    Args:
        mask: float32 [H, W] in [0, 1].
        kernel_size: structuring element size.
        iterations: morphological op iterations.
        boundary_blur_sigma: Gaussian sigma applied to boundary band (0 = skip).

    Returns:
        Refined float32 mask in [0, 1].
    """
    import cv2

    binary = (mask > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)

    refined = opened.astype(np.float32) / 255.0

    if boundary_blur_sigma > 0:
        blurred = cv2.GaussianBlur(refined, (0, 0), boundary_blur_sigma)
        dilated = cv2.dilate(opened, kernel, iterations=2)
        eroded = cv2.erode(opened, kernel, iterations=2)
        band = ((dilated - eroded) > 0).astype(np.float32)
        refined = refined * (1.0 - band) + blurred * band

    return np.clip(refined, 0.0, 1.0)


def refine_mask_guided_filter(
    mask: np.ndarray,
    image_rgb: np.ndarray,
    radius: int = 8,
    eps: float = 0.01,
) -> np.ndarray:
    """
    Guided filter: use original image luminance as guide so mask edges
    snap to real photometric boundaries (chair legs, spindle edges).

    Args:
        mask: float32 [H, W] in [0, 1].
        image_rgb: uint8 [H, W, 3] original image at mask resolution.
        radius: filter radius (larger = smoother, but respects edges).
        eps: regularization — lower means sharper edge adherence.

    Returns:
        Refined float32 mask in [0, 1].
    """
    import cv2

    try:
        guided = cv2.ximgproc.guidedFilter(
            guide=image_rgb,
            src=mask,
            radius=radius,
            eps=eps,
        )
    except AttributeError:
        print(
            "WARNING: cv2.ximgproc.guidedFilter not available. "
            "Install opencv-contrib-python:\n"
            "  pip install opencv-contrib-python",
            file=sys.stderr,
        )
        return mask

    return np.clip(guided, 0.0, 1.0)


def refine_mask_crf(
    mask: np.ndarray,
    image_rgb: np.ndarray,
    n_iter: int = 5,
    sxy: float = 3.0,
    srgb: float = 10.0,
    compat: float = 10.0,
) -> np.ndarray:
    """
    Dense CRF: fully-connected pairwise potentials that respect pixel color
    similarity. Gold standard for snapping masks to real object boundaries.

    Args:
        mask: float32 [H, W] in [0, 1] — treated as foreground probability.
        image_rgb: uint8 [H, W, 3].
        n_iter: CRF inference iterations.
        sxy: bilateral spatial sigma.
        srgb: bilateral color sigma.
        compat: compatibility weight.

    Returns:
        Refined float32 mask in [0, 1].
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        print(
            "WARNING: pydensecrf not installed. Skipping CRF refinement.\n"
            "  pip install pydensecrf",
            file=sys.stderr,
        )
        return mask

    h, w = mask.shape

    fg = np.clip(mask, 1e-6, 1.0 - 1e-6)
    probs = np.stack([1.0 - fg, fg], axis=0).astype(np.float32)
    unary = unary_from_softmax(probs)

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseBilateral(
        sxy=sxy,
        srgb=srgb,
        rgbim=np.ascontiguousarray(image_rgb),
        compat=compat,
    )
    d.addPairwiseGaussian(sxy=sxy, compat=compat / 2.0)

    q = d.inference(n_iter)
    refined = np.array(q)[1].reshape(h, w)
    return refined.astype(np.float32)


def build_refined_overlay(
    image_rgb: np.ndarray,
    masks: list[np.ndarray],
    boxes: np.ndarray | None,
    class_names: list[str],
    confs: list[float],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Composite refined masks onto the original image with semi-transparent
    color fill, contour outlines, and label text.

    Args:
        image_rgb: uint8 [H, W, 3].
        masks: list of float32 [H, W] masks in [0, 1].
        boxes: [N, 4] xyxy or None.
        class_names: list of class name strings.
        confs: list of confidence scores.
        alpha: mask overlay transparency.

    Returns:
        uint8 [H, W, 3] composited image.
    """
    import cv2

    overlay = image_rgb.copy()
    contour_img = image_rgb.copy()

    rng = np.random.RandomState(42)
    colors = rng.randint(60, 255, size=(len(masks), 3)).tolist()

    for i, current_mask in enumerate(masks):
        color = colors[i]
        colored = np.zeros_like(overlay)
        colored[:] = color
        mask_3c = np.stack([current_mask, current_mask, current_mask], axis=-1)
        overlay = (overlay * (1 - alpha * mask_3c) + colored * alpha * mask_3c).astype(np.uint8)

        binary = (current_mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, color, 2)

    result = cv2.addWeighted(overlay, 0.85, contour_img, 0.15, 0)

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            color = colors[i] if i < len(colors) else (255, 255, 255)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names[i]} {confs[i]:.2f}" if i < len(class_names) else ""
            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
                cv2.putText(result, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YOLOE 26L seg-pf with post-inference mask refinement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pt", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "chair.jpeg")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / ".build" / "yoloe_refined_mask.png")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--no-retina-masks", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--refine",
        nargs="+",
        choices=["morph", "guided", "crf", "none"],
        default=["morph", "guided"],
        help="Refinement stages to apply in order (default: morph guided)",
    )
    parser.add_argument("--morph-kernel", type=int, default=3)
    parser.add_argument("--morph-iter", type=int, default=1)
    parser.add_argument("--boundary-blur", type=float, default=1.5)
    parser.add_argument("--guide-radius", type=int, default=8)
    parser.add_argument("--guide-eps", type=float, default=0.01)
    parser.add_argument("--crf-iter", type=int, default=5)
    parser.add_argument("--crf-sxy", type=float, default=3.0)
    parser.add_argument("--crf-srgb", type=float, default=10.0)
    parser.add_argument("--crf-compat", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay alpha")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Save side-by-side: original YOLO plot vs refined (doubled width)",
    )

    args = parser.parse_args()
    refine_stages = [stage for stage in args.refine if stage != "none"]

    pt, tried = resolve_yoloe26l_pf_pt(args.pt)
    if not pt.is_file():
        print("Checkpoint not found. Tried:", file=sys.stderr)
        for candidate in tried:
            print(f"  - {candidate}", file=sys.stderr)
        return 1

    image_path = args.image if args.image.is_file() else REPO_ROOT / args.image
    if not image_path.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    import cv2
    import torch
    from ultralytics import YOLO

    use_half = bool(args.half and torch.cuda.is_available())
    if args.half and not use_half:
        print("Note: --half ignored (CUDA not available). Using float32.", file=sys.stderr)

    model = YOLO(str(pt))
    predict_kw = {
        "source": str(image_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
        "retina_masks": not args.no_retina_masks,
        "half": use_half,
    }

    results = model.predict(**predict_kw)
    result = results[0]

    n_mask = 0 if result.masks is None else len(result.masks)
    n_box = 0 if result.boxes is None else len(result.boxes)

    orig_rgb = np.array(Image.open(image_path).convert("RGB"))
    oh, ow = orig_rgb.shape[:2]

    orig_plot = result.plot(pil=True, conf=True, boxes=True)
    if not isinstance(orig_plot, Image.Image):
        orig_plot = Image.fromarray(np.ascontiguousarray(orig_plot[..., ::-1]))

    if result.masks is not None and len(refine_stages) > 0:
        raw_masks = result.masks.data.cpu().numpy()
        refined_masks: list[np.ndarray] = []

        for i in range(raw_masks.shape[0]):
            current_mask = raw_masks[i].astype(np.float32)

            if current_mask.shape[0] != oh or current_mask.shape[1] != ow:
                current_mask = cv2.resize(current_mask, (ow, oh), interpolation=cv2.INTER_LINEAR)

            for stage in refine_stages:
                if stage == "morph":
                    current_mask = refine_mask_morphological(
                        current_mask,
                        kernel_size=args.morph_kernel,
                        iterations=args.morph_iter,
                        boundary_blur_sigma=args.boundary_blur,
                    )
                elif stage == "guided":
                    current_mask = refine_mask_guided_filter(
                        current_mask,
                        orig_rgb,
                        radius=args.guide_radius,
                        eps=args.guide_eps,
                    )
                elif stage == "crf":
                    current_mask = refine_mask_crf(
                        current_mask,
                        orig_rgb,
                        n_iter=args.crf_iter,
                        sxy=args.crf_sxy,
                        srgb=args.crf_srgb,
                        compat=args.crf_compat,
                    )

            refined_masks.append(current_mask)

        boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
        class_names = [result.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()] if result.boxes is not None else []
        confs = result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None else []

        vis_refined = build_refined_overlay(
            orig_rgb, refined_masks, boxes_xyxy, class_names, confs, alpha=args.alpha
        )
    else:
        vis_refined = np.array(orig_plot)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.compare:
        orig_arr = np.array(orig_plot.resize((ow, oh), Image.LANCZOS))
        combined = np.concatenate([orig_arr, vis_refined], axis=1)
        vis_out = Image.fromarray(combined)
    else:
        vis_out = Image.fromarray(vis_refined)

    vis_out.save(args.out, quality=95)

    print()
    print("=" * 68)
    print("YOLOE 26L seg-pf — Refined mask pipeline")
    print("=" * 68)
    print(f"PT:         {pt}")
    print(f"Image:      {image_path}  ({ow}x{oh})")
    print(
        f"imgsz:      {args.imgsz}  conf: {args.conf}  iou: {args.iou}  "
        f"retina: {not args.no_retina_masks}  half: {use_half}"
    )
    print(f"Refine:     {' -> '.join(refine_stages) if refine_stages else '(none)'}")
    if "morph" in refine_stages:
        print(f"  morph:    kernel={args.morph_kernel} iter={args.morph_iter} blur={args.boundary_blur}")
    if "guided" in refine_stages:
        print(f"  guided:   radius={args.guide_radius} eps={args.guide_eps}")
    if "crf" in refine_stages:
        print(f"  crf:      iter={args.crf_iter} sxy={args.crf_sxy} srgb={args.crf_srgb} compat={args.crf_compat}")
    print(f"Masks:      {n_mask}  Boxes: {n_box}")
    print(f"Compare:    {args.compare}")
    print(f"Saved:      {args.out}")
    print("=" * 68)

    if not args.no_show:
        try:
            import matplotlib.pyplot as plt

            if args.compare:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                ax1.imshow(np.array(orig_plot.resize((ow, oh), Image.LANCZOS)))
                ax1.set_title(f"Original YOLO plot ({n_mask} masks)")
                ax1.axis("off")
                ax2.imshow(vis_refined)
                ax2.set_title(f"Refined: {' -> '.join(refine_stages)}")
                ax2.axis("off")
            else:
                plt.figure(figsize=(12, 10))
                plt.imshow(vis_refined)
                plt.title(f"{pt.name} — refined: {' -> '.join(refine_stages)} ({n_mask} masks)")
                plt.axis("off")
            plt.tight_layout()
            plt.show()
        except Exception as exc:  # noqa: BLE001
            print(f"(matplotlib show skipped: {exc})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
