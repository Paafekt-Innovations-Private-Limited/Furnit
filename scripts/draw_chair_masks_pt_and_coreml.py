#!/usr/bin/env python3
"""
Draw YOLOE segmentation masks from the same image using:
  (1) Ultralytics PyTorch .pt     — model.predict()
  (2) ONNX Runtime seg_o2m.onnx   — det_output + proto_output (same mask math as Core ML)
  (3) Core ML .mlpackage          — raw tensors + mask head (end2end or one-to-many)

Saves PNGs under --out-dir and shows a matplotlib figure at the end (if display works).

Examples:
  python3 scripts/draw_chair_masks_pt_and_coreml.py \\
    --image chair.jpeg \\
    --pt ~/Downloads/yoloe-26l-seg-pf.pt \\
    --onnx .build/yoloe-26l-seg-pf_seg_o2m.onnx \\
    --mlpackage yoloe-26l-seg-pf_seg_o2m.mlpackage

  python3 scripts/draw_chair_masks_pt_and_coreml.py --image chair.jpeg
    (uses default search paths; ONNX skipped if .build/*.onnx missing)
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
    """Map a H=W=side float/bool mask to original image size (nearest crop+resize)."""
    w, h = src_w, src_h
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    crop = mask_lb[pad_y : pad_y + nh, pad_x : pad_x + nw].astype(np.float32)
    pil_crop = Image.fromarray(np.asarray(crop, dtype=np.float32))
    out = np.array(pil_crop.resize((src_w, src_h), Image.BILINEAR), dtype=np.float32)
    return out


def blend_overlay(rgb: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> Image.Image:
    """mask: HxW, values in [0,1] or bool."""
    base = np.asarray(rgb.convert("RGB"), dtype=np.float32) / 255.0
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if m.max() <= 1.0 and m.max() > 0 and m.max() < 0.05:
        m = m / (m.max() + 1e-6)
    col = np.array(color, dtype=np.float32) / 255.0
    for c in range(3):
        base[:, :, c] = base[:, :, c] * (1.0 - alpha * m) + col[c] * (alpha * m)
    return Image.fromarray(np.clip(base * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def find_default_pt() -> Path | None:
    for p in (
        REPO_ROOT / "yoloe-26l-seg-pf.pt",
        Path.home() / "Downloads" / "yoloe-26l-seg-pf.pt",
        REPO_ROOT / "android" / "yoloe-11l-seg-pf.pt",
    ):
        if p.is_file():
            return p
    return None


def find_default_mlpackage() -> Path | None:
    for p in (
        REPO_ROOT / "yoloe-26l-seg-pf_seg_o2m.mlpackage",
        REPO_ROOT / ".build" / "yoloe-26l-seg-pf_seg_o2m.mlpackage",
        REPO_ROOT / "yoloe-26l-seg-pf.mlpackage",
    ):
        if p.is_dir():
            return p
    return None


def find_default_onnx() -> Path | None:
    for p in (
        REPO_ROOT / ".build" / "yoloe-26l-seg-pf_seg_o2m.onnx",
        REPO_ROOT / "yoloe-26l-seg-pf_seg_o2m.onnx",
    ):
        if p.is_file():
            return p
    return None


def onnx_input_side(session) -> int:
    """Infer square input side from ONNX graph (NCHW). Dynamic dims fall back to 640."""
    for inp in session.get_inputs():
        shape = inp.shape
        if len(shape) != 4:
            continue
        for dim in (shape[2], shape[3]):
            if isinstance(dim, int) and dim > 0:
                return dim
    return 640


def infer_onnx_seg_o2m(onnx_path: Path, src: Image.Image) -> tuple[str, np.ndarray, str, np.ndarray, int, float, int, int]:
    """Letterbox to ONNX input side, run, return det/proto tensors + letterbox params."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    side = onnx_input_side(sess)
    letterboxed, scale, pad_x, pad_y = letterbox_pil(src, side)
    inp = sess.get_inputs()[0]
    input_name = inp.name
    arr = np.asarray(letterboxed, dtype=np.float32) / 255.0
    blob = np.transpose(arr, (2, 0, 1))[None, ...]
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, {input_name: blob})
    outputs = dict(zip(out_names, outs))
    det_name, det, proto_name, proto = pick_coreml_det_proto(outputs)
    return det_name, det, proto_name, proto, side, scale, pad_x, pad_y


def coreml_input_side(model) -> int | None:
    import coremltools as ct  # noqa: F401

    spec = model.get_spec()
    for feature in spec.description.input:
        if feature.name != "image":
            continue
        w = feature.type.imageType.width
        if w > 0:
            return int(w)
    return None


def pick_coreml_det_proto(outputs: dict) -> tuple[str, np.ndarray, str, np.ndarray]:
    det_name = proto_name = ""
    det_arr: np.ndarray | None = None
    proto_arr: np.ndarray | None = None
    for name, value in outputs.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 4 and arr.shape[1] == 32:
            proto_name, proto_arr = name, arr
        elif arr.ndim == 3 and arr.shape[0] == 1:
            if arr.shape[2] == 38:
                det_name, det_arr = name, arr
            elif arr.shape[1] >= 4 + 32 and det_arr is None:
                # [1, C, A] one-to-many
                det_name, det_arr = name, arr
    if det_arr is None or proto_arr is None:
        shapes = {k: tuple(np.asarray(v).shape) for k, v in outputs.items()}
        raise RuntimeError(f"Could not find det+proto tensors. Outputs: {shapes}")
    return det_name, det_arr, proto_name, proto_arr


def torch_process_mask_union(
    proto: np.ndarray,
    det: np.ndarray,
    letterbox_side: int,
    conf_thres: float,
    max_det: int,
) -> np.ndarray:
    """Return float mask in letterbox space [side, side] in [0,1]."""
    import torch
    import torch.nn.functional as F

    # proto: [1, 32, mh, mw]
    p = torch.from_numpy(proto[0])
    mh, mw = p.shape[1], p.shape[2]

    # --- End-to-end [1, N, 38] ---
    if det.ndim == 3 and det.shape[2] == 38 and det.shape[0] == 1:
        rows = torch.from_numpy(det[0])
        conf = rows[:, 4]
        order = torch.argsort(conf, descending=True)
        keep = []
        for idx in order[:300]:
            if float(conf[idx]) < conf_thres:
                break
            keep.append(int(idx))
            if len(keep) >= max_det:
                break
        if not keep:
            return np.zeros((letterbox_side, letterbox_side), dtype=np.float32)
        boxes = rows[keep, :4]
        coeffs = rows[keep, 6:38]
        masks = (coeffs @ p.view(32, -1)).view(-1, mh, mw)
        masks = masks.unsqueeze(1)
        masks = F.interpolate(masks, size=(letterbox_side, letterbox_side), mode="bilinear", align_corners=False)
        masks = masks.squeeze(1)
        # crop to boxes (xyxy in letterbox space)
        for i in range(masks.shape[0]):
            x1, y1, x2, y2 = boxes[i].round().int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(letterbox_side - 1, x2), min(letterbox_side - 1, y2)
            t = masks[i]
            if y2 > y1 and x2 > x1:
                t[:y1] = 0
                t[y2 + 1 :] = 0
                t[:, :x1] = 0
                t[:, x2 + 1 :] = 0
        union, _ = masks.max(dim=0)
        return torch.sigmoid(union).numpy().astype(np.float32)

    # --- One-to-many [1, C, A] ---
    if det.ndim == 3 and det.shape[1] > 40:
        d = det[0]  # [C, A]
        num_channels = d.shape[0]
        num_anchors = d.shape[1]
        num_classes = num_channels - 4 - 32
        if num_classes <= 0:
            raise RuntimeError(f"Unexpected det channels layout: {d.shape}")
        box_block = d[0:4, :]
        scores = d[4 : 4 + num_classes, :]
        coeff_block = d[4 + num_classes :, :]
        max_scores = scores.max(axis=0)
        max_cls = scores.argmax(axis=0)
        cand = []
        for a in range(num_anchors):
            conf = float(max_scores[a])
            if conf < conf_thres:
                continue
            cx, cy, bw, bh = (float(box_block[r, a]) for r in range(4))
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            coeffs = coeff_block[:, a].copy()
            cand.append((conf, (x1, y1, x2, y2), coeffs, int(max_cls[a])))
        cand.sort(key=lambda t: -t[0])
        cand = cand[:max_det]
        if not cand:
            return np.zeros((letterbox_side, letterbox_side), dtype=np.float32)
        coeffs_t = torch.from_numpy(np.stack([c[2] for c in cand], axis=0)).float()
        boxes_t = torch.tensor([c[1] for c in cand], dtype=torch.float32)
        masks = (coeffs_t @ p.view(32, -1)).view(-1, mh, mw).unsqueeze(1)
        masks = F.interpolate(masks, size=(letterbox_side, letterbox_side), mode="bilinear", align_corners=False).squeeze(
            1
        )
        for i in range(masks.shape[0]):
            x1, y1, x2, y2 = boxes_t[i].round().int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(letterbox_side - 1, x2), min(letterbox_side - 1, y2)
            t = masks[i]
            if y2 > y1 and x2 > x1:
                t[:y1] = 0
                t[y2 + 1 :] = 0
                t[:, :x1] = 0
                t[:, x2 + 1 :] = 0
        union, _ = masks.max(dim=0)
        return torch.sigmoid(union).numpy().astype(np.float32)

    raise RuntimeError(f"Unsupported det shape: {det.shape}")


def run_pt(
    image_path: Path,
    pt_path: Path,
    imgsz: int,
    conf: float,
) -> tuple[Image.Image, str]:
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        verbose=False,
    )
    r = results[0]
    # BGR uint8 annotated image
    plotted = r.plot()
    rgb = np.ascontiguousarray(plotted[..., ::-1])
    pil = Image.fromarray(rgb, mode="RGB")
    n = 0 if r.masks is None else len(r.masks)
    return pil, f"masks={n} boxes={len(r.boxes) if r.boxes is not None else 0}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw PT / ONNX / Core ML YOLOE masks on one image.")
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "chair.jpeg")
    parser.add_argument("--pt", type=Path, default=None, help="YOLOE .pt (default: search repo/Downloads)")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="seg_o2m ONNX (default: .build/yoloe-26l-seg-pf_seg_o2m.onnx if present)",
    )
    parser.add_argument("--mlpackage", type=Path, default=None, help="Core ML package (default: search repo)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / ".build" / "chair_mask_compare")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=None, help="PT imgsz; CM/ONNX use each model's input side if unset")
    parser.add_argument("--no-show", action="store_true", help="Skip matplotlib display")
    args = parser.parse_args()

    image_path = args.image if args.image.is_file() else REPO_ROOT / args.image
    if not image_path.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    pt_path = args.pt or find_default_pt()
    if pt_path is None or not pt_path.is_file():
        print("Pass --pt to yoloe-*-seg-pf.pt (not found in default locations).", file=sys.stderr)
        return 1

    mlpackage = args.mlpackage or find_default_mlpackage()
    if mlpackage is None or not mlpackage.is_dir():
        print("Pass --mlpackage to a .mlpackage (not found in default locations).", file=sys.stderr)
        return 1

    onnx_path = args.onnx if args.onnx is not None else find_default_onnx()
    run_onnx = onnx_path is not None and onnx_path.is_file()

    import coremltools as ct

    cm = ct.models.MLModel(str(mlpackage), compute_units=ct.ComputeUnit.CPU_ONLY)
    side_cm = coreml_input_side(cm) or 640
    imgsz = args.imgsz if args.imgsz is not None else side_cm

    src = Image.open(image_path).convert("RGB")
    src_w, src_h = src.size

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = args.out_dir / "mask_pt_predict_plot.png"
    out_onnx = args.out_dir / "mask_onnx_overlay.png"
    out_cm = args.out_dir / "mask_coreml_overlay.png"
    out_triptych = args.out_dir / "pt_onnx_coreml_masks.png"

    # --- PyTorch ---
    pt_vis, pt_meta = run_pt(image_path, pt_path, imgsz=imgsz, conf=args.conf)
    pt_vis.save(out_pt)

    # --- ONNX (same mask pipeline as Core ML: coeffs @ proto → bilinear → crop → union) ---
    onnx_overlay: Image.Image | None = None
    onnx_meta = ""
    if run_onnx:
        odet_name, odet, oproto_name, oproto, side_onnx, scale_o, padxo, padyo = infer_onnx_seg_o2m(onnx_path, src)
        mask_lb_o = torch_process_mask_union(
            oproto, odet, letterbox_side=side_onnx, conf_thres=args.conf, max_det=args.max_det
        )
        mask_src_o = letterbox_mask_to_source(mask_lb_o, side_onnx, scale_o, padxo, padyo, src_w, src_h)
        onnx_overlay = blend_overlay(src, mask_src_o, color=(120, 160, 255), alpha=0.5)
        onnx_overlay.save(out_onnx)
        onnx_meta = f"{odet_name} {tuple(odet.shape)} + {oproto_name} {tuple(oproto.shape)} @ {side_onnx}px"
    else:
        print(f"ONNX skipped (file not found): {args.onnx or REPO_ROOT / '.build' / 'yoloe-26l-seg-pf_seg_o2m.onnx'}", file=sys.stderr)

    # --- Core ML ---
    letterboxed, scale, pad_x, pad_y = letterbox_pil(src, side_cm)
    outputs = cm.predict({"image": letterboxed})
    det_name, det, proto_name, proto = pick_coreml_det_proto(outputs)
    mask_lb = torch_process_mask_union(proto, det, letterbox_side=side_cm, conf_thres=args.conf, max_det=args.max_det)
    mask_src = letterbox_mask_to_source(mask_lb, side_cm, scale, pad_x, pad_y, src_w, src_h)
    cm_overlay = blend_overlay(src, mask_src, color=(80, 200, 120), alpha=0.5)
    cm_overlay.save(out_cm)

    # Triptych: PT | ONNX | Core ML
    ncols = 3 if onnx_overlay is not None else 2
    wcell = max(pt_vis.width, cm_overlay.width, onnx_overlay.width if onnx_overlay else 0)
    hcell = max(pt_vis.height, cm_overlay.height, onnx_overlay.height if onnx_overlay else 0)
    combined = Image.new("RGB", (wcell * ncols, hcell), (40, 40, 40))
    combined.paste(pt_vis.resize((wcell, hcell), Image.Resampling.BILINEAR), (0, 0))
    col = 1
    if onnx_overlay is not None:
        combined.paste(onnx_overlay.resize((wcell, hcell), Image.Resampling.BILINEAR), (wcell * col, 0))
        col += 1
    combined.paste(cm_overlay.resize((wcell, hcell), Image.Resampling.BILINEAR), (wcell * col, 0))
    combined.save(out_triptych)

    print()
    print("=" * 60)
    print("YOLOE mask visualization: PT + ONNX + Core ML")
    print("=" * 60)
    print(f"Image:     {image_path} ({src_w}x{src_h})")
    print(f"PT:        {pt_path}")
    print(f"PT imgsz:  {imgsz}  →  {pt_meta}")
    if run_onnx:
        print(f"ONNX:      {onnx_path}")
        print(f"ONNX:      {onnx_meta}")
    print(f"Core ML:   {mlpackage}")
    print(f"CM input:  {side_cm}x{side_cm}  tensors: {det_name} {tuple(det.shape)}, {proto_name} {tuple(proto.shape)}")
    print()
    print("Saved:")
    print(f"  {out_pt}")
    if run_onnx:
        print(f"  {out_onnx}")
    print(f"  {out_cm}")
    print(f"  {out_triptych}")
    print("=" * 60)

    if not args.no_show:
        try:
            import matplotlib.pyplot as plt

            if onnx_overlay is not None:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(pt_vis)
                axes[0].set_title(f"PyTorch ({imgsz}px)\n{pt_meta}")
                axes[0].axis("off")
                axes[1].imshow(onnx_overlay)
                axes[1].set_title(f"ONNX seg_o2m\n{onnx_meta}")
                axes[1].axis("off")
                axes[2].imshow(cm_overlay)
                axes[2].set_title(f"Core ML ({side_cm}px)\n{det_name} + {proto_name}")
                axes[2].axis("off")
            else:
                fig, axes = plt.subplots(1, 2, figsize=(14, 7))
                axes[0].imshow(pt_vis)
                axes[0].set_title(f"Ultralytics PT ({imgsz}px)\n{pt_meta}")
                axes[0].axis("off")
                axes[1].imshow(cm_overlay)
                axes[1].set_title(f"Core ML ({side_cm}px)\n{det_name} + {proto_name}")
                axes[1].axis("off")
            plt.tight_layout()
            plt.show()
        except Exception as exc:  # noqa: BLE001
            print(f"(matplotlib show skipped: {exc})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
