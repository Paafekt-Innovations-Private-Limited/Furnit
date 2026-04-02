#!/usr/bin/env python3
"""
Quick ONNX Runtime sanity check for one-to-many YOLOE seg export (det_output + proto_output).

Expects shapes like det [1, 4+nc+32, 8400], proto [1, 32, 160, 160].
For a clear photo, max class score often lands ~0.85–0.95 (not identical to Ultralytics predict()).

Default preprocessing: 640 letterbox + /255 (matches app-style visualize script).
Use --stretch for naive 640×640 resize (matches a minimal cv2.resize smoke test).

Example:
  python3 scripts/sanity_check_yoloe_seg_o2m_onnx.py \\
    --onnx .build/yoloe-26l-seg-pf_seg_o2m.onnx \\
    --image FurnitTests/room.jpeg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


def letterbox_rgb(image: Image.Image, side: int) -> np.ndarray:
    """RGB float32 NCHW [1,3,H,W] in [0,1], letterbox gray 114."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    scale = min(side / w, side / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = rgb.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (side, side), (114, 114, 114))
    pad_x = (side - nw) // 2
    pad_y = (side - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...]


def stretch_rgb(image: Image.Image, side: int) -> np.ndarray:
    """RGB float32 NCHW [1,3,H,W], naive resize."""
    rgb = image.convert("RGB").resize((side, side), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check YOLOE seg_o2m ONNX via ORT.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=REPO_ROOT / ".build" / "yoloe-26l-seg-pf_seg_o2m.onnx",
        help="Path to exported ONNX.",
    )
    parser.add_argument("--image", type=Path, required=True, help="Test image path.")
    parser.add_argument(
        "--stretch",
        action="store_true",
        help="Use 640×640 stretch instead of letterbox (faster smoke test, different scores).",
    )
    parser.add_argument("--side", type=int, default=640, help="Input square side.")
    args = parser.parse_args()

    import onnxruntime as ort

    if not args.onnx.is_file():
        raise SystemExit(f"Missing ONNX: {args.onnx}")
    if not args.image.is_file():
        raise SystemExit(f"Missing image: {args.image}")

    pil = Image.open(args.image)
    blob = stretch_rgb(pil, args.side) if args.stretch else letterbox_rgb(pil, args.side)

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    det, proto = sess.run(None, {input_name: blob})

    print(f"input: {input_name}  blob: {blob.shape}  preprocess: {'stretch' if args.stretch else 'letterbox'}")
    print(f"det:   {det.shape}")
    print(f"proto: {proto.shape}")

    # det [1, C, A] -> (A, C)
    pred = det[0].T
    num_channels = pred.shape[1]
    num_classes = num_channels - 4 - 32
    if num_classes <= 0:
        raise SystemExit(f"Unexpected det channels: {num_channels}")

    scores = pred[:, 4 : 4 + num_classes]
    max_per_anchor = scores.max(axis=1)
    global_max = float(max_per_anchor.max())
    count_above = int((max_per_anchor > 0.5).sum())

    print(f"class channels: {num_classes}  anchors: {scores.shape[0]}")
    print(f"max class score (any anchor): {global_max:.4f}")
    print(f"anchors with max(class) > 0.5: {count_above}")

    if global_max >= 0.75:
        print("OK — scores look like a healthy one-to-many head on this image.")
    elif global_max >= 0.35:
        print("Marginal — try letterbox if you used --stretch, or a busier/cluttered scene.")
    else:
        print("Low — wrong model, wrong input layout, or broken export if this image is an easy scene.")


if __name__ == "__main__":
    main()
