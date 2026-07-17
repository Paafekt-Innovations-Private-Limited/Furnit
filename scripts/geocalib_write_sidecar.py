#!/usr/bin/env python3
"""
Run GeoCalib on an image and write a JSON sidecar for Furnit camera metadata.

The sidecar keys match DepthAnythingRoomReconstructor focal lookup:
  geoCalibFocalLengthPx, geocalibFocalLengthPx, focalLengthPx

Example:
  python3 scripts/geocalib_write_sidecar.py --image room.jpg
  python3 scripts/geocalib_write_sidecar.py --image room.jpg --sidecar room.jpg.meta.json
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
    parser = argparse.ArgumentParser(description="Write GeoCalib focal/gravity sidecar JSON.")
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image path")
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="Output JSON path (default: <image>.geocalib.json)",
    )
    parser.add_argument("--weights", default="pinhole", help='GeoCalib weights: "pinhole", "distorted", or path')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(args.image)

    sidecar_path = args.sidecar or args.image.with_suffix(args.image.suffix + ".geocalib.json")

    geocalib_path = str(GEOCALIB_ROOT)
    if geocalib_path not in sys.path:
        sys.path.insert(0, geocalib_path)

    from geocalib.extractor import GeoCalib
    from geocalib.utils import get_device, rad2deg

    device = get_device()
    model = GeoCalib(weights=args.weights).to(device)
    image = model.load_image(args.image).to(device)

    with torch.no_grad():
        result = model.calibrate(image)

    camera = result["camera"]
    gravity = result["gravity"]
    roll_deg, pitch_deg = rad2deg(gravity.rp).unbind(-1)
    fx, fy = camera.f[0, 0].item(), camera.f[0, 1].item()

    payload = {
        "source": "GeoCalib",
        "weights": args.weights,
        "image": str(args.image.resolve()),
        "width_px": int(camera.size[0, 0].round().item()),
        "height_px": int(camera.size[0, 1].round().item()),
        "focalLengthPx": fx,
        "focalLengthYPx": fy,
        "geoCalibFocalLengthPx": fx,
        "geocalibFocalLengthPx": fx,
        "geoCalibFocalLengthYPx": fy,
        "geocalibFocalLengthYPx": fy,
        "vertical_fov_deg": float(rad2deg(camera.vfov).item()),
        "roll_deg": float(roll_deg.item()),
        "pitch_deg": float(pitch_deg.item()),
        "gravity": gravity.vec3d[0].tolist(),
    }

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {sidecar_path}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
