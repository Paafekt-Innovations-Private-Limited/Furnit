#!/usr/bin/env python3
"""
Evaluate YOLOE-26L seg PF (one-to-many ONNX) on room vs bathroom images for **corner** class (id 1124).

Uses the same letterbox + det tensor decode as ``visualize_yoloe_onemany_onnx.py``.
Writes per-image overlays (only **corner** boxes in magenta) and ``summary.json``.

HEIC: include ``.heic`` / ``.HEIC`` in folders. Install ``pillow-heif`` so Pillow can decode them
(``pip install pillow-heif``).

Examples:
  # Two category folders (jpg/jpeg/png/heic):
  python3 scripts/eval_yoloe26_pf_corners.py \\
    --room-dir /path/to/room_photos \\
    --bathroom-dir /path/to/bathroom_photos \\
    --out-dir .build/yoloe_corner_eval

  # Single flat list of images:
  python3 scripts/eval_yoloe26_pf_corners.py --images a.jpg b.jpg --label-set demo --out-dir .build/yoloe_corner_eval
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent


def _register_heif_opener_if_available() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass


_register_heif_opener_if_available()


def open_image_rgb(path: Path) -> Image.Image:
    """Open JPEG/PNG/HEIC/etc. as RGB (HEIC needs optional ``pillow-heif``)."""
    try:
        return Image.open(path).convert("RGB")
    except OSError as exc:
        suffix = path.suffix.lower()
        if suffix in (".heic", ".heif"):
            raise SystemExit(
                f"Could not read {path} (HEIC). Install: pip install pillow-heif\n{exc}"
            ) from exc
        raise


IMAGE_GLOBS = (
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.heic",
    "*.heif",
    "*.JPG",
    "*.JPEG",
    "*.PNG",
    "*.HEIC",
    "*.HEIF",
)
# Prefer local export paths; iOS bundle no longer ships this ONNX (Core ML + ODR only).
DEFAULT_ONNX_CANDIDATES = (
    REPO_ROOT / "Furnit" / "Resources" / "yoloe-26l-seg-pf_seg_o2m.onnx",
    REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "yoloe-26l-seg-pf_seg_o2m.onnx",
    REPO_ROOT / ".build" / "yoloe-26l-seg-pf_seg_o2m.onnx",
)
DEFAULT_CLASSES = REPO_ROOT / "Furnit" / "Views" / "FurnitureFit" / "classes.json"


def resolve_default_onnx_path() -> Path:
    for candidate in DEFAULT_ONNX_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_ONNX_CANDIDATES[0]

# PF vocabulary: exact label "corner" (not "street corner")
CORNER_CLASS_ID = 1124
CORNER_LABEL = "corner"


def load_class_names(classes_path: Path) -> dict[int, str]:
    with classes_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(key): str(value) for key, value in raw.items()}


def letterbox_image(
    source_image: Image.Image, model_side: int
) -> tuple[Image.Image, float, int, int]:
    rgb_image = source_image.convert("RGB")
    scale = min(model_side / rgb_image.width, model_side / rgb_image.height)
    resized_width = max(1, int(round(rgb_image.width * scale)))
    resized_height = max(1, int(round(rgb_image.height * scale)))
    resized_image = rgb_image.resize((resized_width, resized_height), Image.BILINEAR)
    pad_x = (model_side - resized_width) // 2
    pad_y = (model_side - resized_height) // 2
    canvas = Image.new("RGB", (model_side, model_side), (114, 114, 114))
    canvas.paste(resized_image, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def letterbox_xywh_to_source(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    source_width: int,
    source_height: int,
) -> tuple[float, float, float, float]:
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    x1 = (x1 - pad_x) / scale
    x2 = (x2 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    y2 = (y2 - pad_y) / scale
    x1 = max(0, min(source_width - 1, x1))
    x2 = max(0, min(source_width - 1, x2))
    y1 = max(0, min(source_height - 1, y1))
    y2 = max(0, min(source_height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter + 1e-6
    return inter / union


def nms_corner_boxes(
    items: list[tuple[float, tuple[float, float, float, float]]],
    iou_threshold: float,
    max_boxes: int,
) -> list[tuple[float, tuple[float, float, float, float]]]:
    """Greedy NMS on corner-only boxes (by score desc)."""
    items = sorted(items, key=lambda t: -t[0])
    kept: list[tuple[float, tuple[float, float, float, float]]] = []
    for score, box in items:
        if any(iou_xyxy(box, prior[1]) > iou_threshold for prior in kept):
            continue
        kept.append((score, box))
        if len(kept) >= max_boxes:
            break
    return kept


def collect_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for pattern in IMAGE_GLOBS:
        out.extend(directory.glob(pattern))
    return sorted(set(out), key=lambda p: p.as_posix().lower())


def safe_stem(path: Path) -> str:
    s = path.stem
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:120] if len(s) > 120 else s


@dataclass
class ImageResult:
    category: str
    source_path: str
    corner_count: int
    corner_scores: list[float]
    max_corner_score: float | None
    mean_corner_score: float | None
    # Highest max-class score among anchors whose argmax class is "corner" (diagnostic).
    max_winning_corner_argmax_score: float | None
    overlay_relative: str


def infer_corners_for_image(
    sess,
    input_name: str,
    image_path: Path,
    side: int,
    conf_floor: float,
    iou_nms: float,
    max_corners: int,
) -> tuple[
    list[tuple[float, tuple[float, float, float, float]]],
    float | None,
]:
    source = open_image_rgb(image_path)
    source_width, source_height = source.size
    canvas, scale, pad_x, pad_y = letterbox_image(source, side)
    tensor = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]

    det, _ = sess.run(None, {input_name: tensor})
    num_features, num_anchors = det.shape[1], det.shape[2]
    num_classes = num_features - 4 - 32
    if CORNER_CLASS_ID >= num_classes:
        raise SystemExit(
            f"corner id {CORNER_CLASS_ID} out of range for nc={num_classes} "
            f"(check classes.json vs ONNX export)."
        )

    scores_block = det[0, 4 : 4 + num_classes, :]
    box_block = det[0, 0:4, :]

    # Same as visualize_yoloe_onemany_onnx: each anchor picks one class via argmax.
    max_scores = np.max(scores_block, axis=0)
    max_idx = np.argmax(scores_block, axis=0)
    winning_corner = max_idx == CORNER_CLASS_ID
    max_winning_corner_argmax_score: float | None = None
    if np.any(winning_corner):
        max_winning_corner_argmax_score = float(np.max(max_scores[winning_corner]))

    mask = winning_corner & (max_scores >= conf_floor)
    indices = np.where(mask)[0]
    indices = indices[np.argsort(-max_scores[indices])]

    raw: list[tuple[float, tuple[float, float, float, float]]] = []
    for anchor_index in indices:
        confidence = float(max_scores[anchor_index])
        cx, cy, bw, bh = (float(box_block[row, anchor_index]) for row in range(4))
        xyxy = letterbox_xywh_to_source(
            cx, cy, bw, bh, scale, pad_x, pad_y, source_width, source_height
        )
        if xyxy[2] - xyxy[0] < 2 or xyxy[3] - xyxy[1] < 2:
            continue
        raw.append((confidence, xyxy))

    kept = nms_corner_boxes(raw, iou_threshold=iou_nms, max_boxes=max_corners)
    return kept, max_winning_corner_argmax_score


def draw_overlay(
    image_path: Path,
    corners: list[tuple[float, tuple[float, float, float, float]]],
    out_path: Path,
    *,
    max_winning_corner_diag: float | None,
    conf_threshold: float,
) -> None:
    source = open_image_rgb(image_path)
    vis = source.copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    magenta = (255, 0, 200)
    for score, (x1, y1, x2, y2) in corners:
        draw.rectangle([x1, y1, x2, y2], outline=magenta, width=4)
        label = f"{CORNER_LABEL} {score * 100:.1f}%"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(0.0, y1 - th - 6)
        draw.rectangle([x1, ly, x1 + tw + 8, ly + th + 4], fill=magenta)
        draw.text((x1 + 4, ly + 2), label, fill=(0, 0, 0), font=font)

    title = f"{len(corners)} corner box(es)  |  conf≥{conf_threshold:.2f}"
    sub = (
        f"max P(winning class=corner): {max_winning_corner_diag:.5f}"
        if max_winning_corner_diag is not None
        else "max P(winning class=corner): —"
    )
    draw.rectangle([8, 8, min(source.width - 8, 520), 52], fill=(0, 0, 0))
    draw.text((12, 8), title, fill=(255, 255, 255), font=font)
    draw.text((12, 28), sub, fill=(200, 200, 200), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path, quality=95)


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOE-26L PF corner detection eval + overlays.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="YOLOE seg_o2m ONNX. Default: first existing among Furnit/Resources, android assets, .build.",
    )
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--room-dir", type=Path, default=None, help="Directory of room images.")
    parser.add_argument(
        "--bathroom-dir",
        type=Path,
        default=None,
        help="Directory of bathroom images.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="*",
        default=(),
        help="Optional explicit image paths (use with --label-set).",
    )
    parser.add_argument(
        "--label-set",
        type=str,
        default="custom",
        help="Category label for --images (default: custom).",
    )
    parser.add_argument("--side", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Min score for corner class.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU among corner boxes.")
    parser.add_argument("--max-corners", type=int, default=30)
    args = parser.parse_args()

    onnx_path = args.onnx if args.onnx is not None else resolve_default_onnx_path()
    if not onnx_path.is_file():
        raise SystemExit(
            f"Missing ONNX: {onnx_path}\n"
            "Place yoloe-26l-seg-pf_seg_o2m.onnx under .build/, android assets, or Furnit/Resources, "
            "or pass --onnx explicitly."
        )
    if not args.classes.is_file():
        raise SystemExit(f"Missing classes.json: {args.classes}")

    work_items: list[tuple[str, Path]] = []

    if args.images:
        label = args.label_set.strip() or "custom"
        for p in args.images:
            if p.is_file():
                work_items.append((label, p.resolve()))
            else:
                print(f"Skip missing: {p}", file=sys.stderr)
    else:
        room_dir = args.room_dir
        bath_dir = args.bathroom_dir
        demo_note = None
        if room_dir is None and bath_dir is None:
            demo = REPO_ROOT / ".build" / "yoloe_corner_eval_inputs"
            r, b = demo / "room", demo / "bathroom"
            if r.is_dir() and collect_images(r):
                room_dir = r
            if b.is_dir() and collect_images(b):
                bath_dir = b
        if room_dir is None and bath_dir is None:
            # Demo fallback: bundled tests (user should pass real room/bathroom dirs).
            tr = REPO_ROOT / "FurnitTests"
            if (tr / "room.jpeg").is_file():
                work_items.append(("room", tr / "room.jpeg"))
            if (tr / "landscape.jpeg").is_file():
                work_items.append(("room", tr / "landscape.jpeg"))
            if (tr / "bus.jpg").is_file():
                work_items.append(("bathroom", tr / "bus.jpg"))
            demo_note = (
                "No --room-dir/--bathroom-dir and no .build/yoloe_corner_eval_inputs/{room,bathroom}. "
                "Using FurnitTests placeholders (room.jpeg, landscape → room; bus → bathroom)."
            )
            print(demo_note, file=sys.stderr)
        else:
            if room_dir and collect_images(room_dir):
                for p in collect_images(room_dir):
                    work_items.append(("room", p.resolve()))
            elif room_dir:
                print(f"Warning: no images in {room_dir}", file=sys.stderr)
            if bath_dir and collect_images(bath_dir):
                for p in collect_images(bath_dir):
                    work_items.append(("bathroom", p.resolve()))
            elif bath_dir:
                print(f"Warning: no images in {bath_dir}", file=sys.stderr)

    if not work_items:
        raise SystemExit(
            "No images to process. Add photos (jpg/png/heic) under --room-dir / --bathroom-dir, "
            "or pass --images, or place files in .build/yoloe_corner_eval_inputs/room|bathroom/. "
            "For HEIC: pip install pillow-heif."
        )

    import onnxruntime as ort

    class_names = load_class_names(args.classes)
    if class_names.get(CORNER_CLASS_ID) != CORNER_LABEL:
        print(
            f"Warning: class {CORNER_CLASS_ID} is {class_names.get(CORNER_CLASS_ID)!r}, "
            f"expected {CORNER_LABEL!r}.",
            file=sys.stderr,
        )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    side = args.side

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    results: list[ImageResult] = []
    by_cat: dict[str, list[int]] = {}

    for category, img_path in work_items:
        corners, max_winning_corner_diag = infer_corners_for_image(
            sess,
            input_name,
            img_path,
            side,
            args.conf,
            args.iou,
            args.max_corners,
        )
        scores = [c[0] for c in corners]
        stem = safe_stem(img_path)
        rel = f"overlays/{category}__{stem}.png"
        out_png = out_dir / rel
        draw_overlay(
            img_path,
            corners,
            out_png,
            max_winning_corner_diag=max_winning_corner_diag,
            conf_threshold=args.conf,
        )

        mx = max(scores) if scores else None
        mean = float(sum(scores) / len(scores)) if scores else None

        results.append(
            ImageResult(
                category=category,
                source_path=str(img_path),
                corner_count=len(corners),
                corner_scores=[round(s, 5) for s in scores],
                max_corner_score=round(mx, 5) if mx is not None else None,
                mean_corner_score=round(mean, 5) if mean is not None else None,
                max_winning_corner_argmax_score=round(max_winning_corner_diag, 6)
                if max_winning_corner_diag is not None
                else None,
                overlay_relative=rel,
            )
        )
        by_cat.setdefault(category, []).append(len(corners))

    summary = {
        "model": str(onnx_path.resolve()),
        "corner_class_id": CORNER_CLASS_ID,
        "corner_label": CORNER_LABEL,
        "decode_note": (
            "Boxes use the same rule as visualize_yoloe_onemany_onnx: per-anchor argmax over "
            "4585 PF classes; a detection counts as corner only if that argmax is class "
            f"{CORNER_CLASS_ID} ({CORNER_LABEL!r}) and max class score ≥ conf_threshold."
        ),
        "conf_threshold": args.conf,
        "iou_nms": args.iou,
        "max_corners": args.max_corners,
        "input_side": side,
        "per_image": [asdict(r) for r in results],
        "consistency_by_category": {},
    }

    for cat, counts in by_cat.items():
        summary["consistency_by_category"][cat] = {
            "image_count": len(counts),
            "total_corner_boxes": int(sum(counts)),
            "mean_corners_per_image": round(sum(counts) / len(counts), 4),
            "min_corners_per_image": min(counts),
            "max_corners_per_image": max(counts),
            "fraction_images_with_at_least_one_corner": round(
                sum(1 for c in counts if c >= 1) / len(counts), 4
            ),
        }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary["consistency_by_category"], indent=2))
    print(f"Wrote {summary_path}")
    print(f"Overlays under {overlays_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
