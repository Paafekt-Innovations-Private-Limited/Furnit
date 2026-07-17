#!/usr/bin/env python3
"""
Run GeoCalib on an image and print calibration outputs.

Also useful to inspect preprocessing tensors for Core ML integration:
  resize_scales = [resized_width / original_width, resized_height / original_height]
  crop_pad = [cropped_width - resized_width, cropped_height - resized_height]

Example:
  python3 scripts/run_geocalib.py --image /path/to/room.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
GEOCALIB_ROOT = REPO_ROOT / "third_party" / "GeoCalib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GeoCalib on a single image.")
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image path")
    parser.add_argument("--weights", default="pinhole", help='GeoCalib weights: "pinhole", "distorted", or path')
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(args.image)

    geocalib_path = str(GEOCALIB_ROOT)
    if geocalib_path not in sys.path:
        sys.path.insert(0, geocalib_path)

    from geocalib.extractor import GeoCalib
    from geocalib.utils import ImagePreprocessor, get_device, load_image, rad2deg

    device = get_device()
    model = GeoCalib(weights=args.weights).to(device)
    image = load_image(args.image).to(device)
    if image.ndim == 3:
        image = image[None]

    preprocessor = ImagePreprocessor({"resize": 320, "edge_divisible_by": 32})
    img_data = preprocessor(image)

    with torch.no_grad():
        result = model.calibrate(image)

    camera = result["camera"]
    gravity = result["gravity"]
    roll_deg, pitch_deg = rad2deg(gravity.rp).unbind(-1)
    vfov_deg = rad2deg(camera.vfov)
    fx, fy = camera.f[0, 0].item(), camera.f[0, 1].item()

    payload = {
        "image": str(args.image.resolve()),
        "width_px": int(camera.size[0, 0].round().item()),
        "height_px": int(camera.size[0, 1].round().item()),
        "focal_x_px": float(fx),
        "focal_y_px": float(fy),
        "vertical_fov_deg": float(vfov_deg.item()),
        "roll_deg": float(roll_deg.item()),
        "pitch_deg": float(pitch_deg.item()),
        "gravity": gravity.vec3d[0].tolist(),
        "focal_uncertainty_px": float(result.get("focal_uncertainty", torch.tensor(0.0))[0].item()),
        "roll_uncertainty_deg": float(rad2deg(result.get("roll_uncertainty", torch.tensor(0.0)))[0].item()),
        "pitch_uncertainty_deg": float(rad2deg(result.get("pitch_uncertainty", torch.tensor(0.0)))[0].item()),
        "preprocess": {
            "model_input_size": [int(img_data["image"].shape[-1]), int(img_data["image"].shape[-2])],
            "resize_scales": img_data["scales"].tolist(),
            "crop_pad": img_data.get("crop_pad", torch.zeros(2)).tolist(),
            "original_image_size": img_data["original_image_size"].tolist(),
        },
    }

    print(json.dumps(payload, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
