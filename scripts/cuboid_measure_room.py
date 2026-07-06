#!/usr/bin/env python3
"""Manhattan room cuboid: M-LSD RG quad + GeoCalib + Depth Anything + RTMDet masks.

Pipeline:
  Image → RTMDet exclude mask (chair, curtain clutter, etc.)
       → M-LSD lines → red×green intersection quad (yellow)
       → Depth Anything + camera-height scale
       → fit floor / ceiling / left / right / back wall planes (RANSAC)
       → Manhattan snap + plane separations → Width × Height × Depth

Example:
  python3 scripts/run_geocalib.py --image room.jpg --json-out /tmp/room_geocalib.json
  python3 scripts/cuboid_measure_room.py \\
    --image room.jpg \\
    --geocalib-json /tmp/room_geocalib.json \\
    --tape-height 2.85 --tape-width 3.15 --tape-depth 3.37
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from depthanything_measure_room import DEFAULT_ONNX, focal_from_exif, read_rgb, run_depth_onnx
from rtmdet_exclude_mask import DEFAULT_MODEL, build_rtmdet_exclude_mask
from structure_box_measure_room import (
    DEFAULT_OUT_DIR,
    draw_box_connected_overlay,
    draw_rg_intersection_quad_on_canvas,
    measure_room_cuboid,
    tape_error,
)

DEFAULT_RTMDET = DEFAULT_MODEL


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Manhattan cuboid room measurement (Python).")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--geocalib-json", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rtmdet-model", type=Path, default=DEFAULT_RTMDET)
    parser.add_argument("--no-rtmdet", action="store_true", help="Skip RTMDet furniture masking.")
    parser.add_argument("--camera-height-prior", type=float, default=1.40)
    parser.add_argument("--region-inset-px", type=int, default=12)
    parser.add_argument("--max-samples-per-region", type=int, default=2500)
    parser.add_argument("--score-thr", type=float, default=0.10)
    parser.add_argument("--dist-thr", type=float, default=20.0)
    parser.add_argument("--tape-width", type=float, default=None)
    parser.add_argument("--tape-height", type=float, default=None)
    parser.add_argument("--tape-depth", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    geocalib = json.loads(args.geocalib_json.expanduser().read_text(encoding="utf-8"))
    gravity = geocalib.get("gravity")
    if not gravity:
        raise SystemExit("GeoCalib JSON missing gravity vector")

    image = read_rgb(image_path)
    width, height = image.size
    exif_fx, exif_fy, focal_source = focal_from_exif(image_path, width, height)
    fx = float(geocalib.get("focal_x_px") or geocalib.get("focalLengthPx") or exif_fx)
    fy = float(geocalib.get("focal_y_px") or geocalib.get("focalLengthYPx") or fx)

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    exclude_mask = None
    rtmdet_meta = {"source": "disabled"}
    if not args.no_rtmdet:
        print("Running RTMDet exclude mask…")
        exclude_mask, rtmdet_meta = build_rtmdet_exclude_mask(rgb, model_path=args.rtmdet_model.expanduser().resolve())
        print(f"  RTMDet: {rtmdet_meta.get('source')} exclude={rtmdet_meta.get('exclude_fraction', 0):.1%}")

    print("Running Depth Anything…")
    depth = run_depth_onnx(image, args.onnx, args.input_size)

    rng = np.random.default_rng(42)
    result = measure_room_cuboid(
        image=image,
        depth=depth,
        gravity=gravity,
        fx=fx,
        fy=fy,
        camera_height_prior=args.camera_height_prior,
        region_inset_px=args.region_inset_px,
        max_samples=args.max_samples_per_region,
        score_thr=args.score_thr,
        dist_thr=args.dist_thr,
        rng=rng,
        exclude_mask=exclude_mask,
        rtmdet_meta=rtmdet_meta,
    )

    tape_compare = {}
    for key, tape_val in (
        ("width", args.tape_width),
        ("height", args.tape_height),
        ("depth", args.tape_depth),
    ):
        err = tape_error(result.get(f"{key}_m"), tape_val)
        if err:
            tape_compare[key] = err
    if tape_compare:
        result["tape_comparison"] = tape_compare

    segments = result.pop("_segments", [])
    rg_box = result.pop("_rg_box", None)

    connected_path = out_dir / f"{stem}_cuboid_lines.png"
    filter_meta = draw_box_connected_overlay(rgb, segments, connected_path)

    if rg_box is not None:
        canvas = np.array(Image.open(connected_path).convert("RGB"), dtype=np.uint8)
        draw_rg_intersection_quad_on_canvas(canvas, rg_box, color=(255, 255, 0), thickness=5)
        Image.fromarray(canvas).save(connected_path)

    json_path = out_dir / f"{stem}_cuboid.json"
    payload = {
        "image": str(image_path),
        "geocalib_json": str(args.geocalib_json.expanduser().resolve()),
        "focal_px": round(fx, 2),
        "focal_source": focal_source,
        "cuboid_measurement": result,
        "box_connected_filter": filter_meta,
        "overlay_png": str(connected_path),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\noverlay: {connected_path}")
    print(f"json:    {json_path}")
    return 0 if result.get("w_h_d") else 1


if __name__ == "__main__":
    raise SystemExit(main())
