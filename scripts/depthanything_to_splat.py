#!/usr/bin/env python3
"""Convert Depth Anything V2 metric depth into a simple 3DGS PLY.

This is a legal-clean Tier 1 visual prototype:
RGB image -> Depth Anything ONNX metric depth -> deterministic Gaussians -> PLY.

It intentionally uses Depth Anything output directly rather than a learned Gaussian head. The PLY
layout follows INRIA-style degree-0 spherical-harmonic conventions.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageOps
from plyfile import PlyData, PlyElement


DEFAULT_IMAGE = Path("/Users/al/Downloads/USRoom.jpeg")
DEFAULT_ONNX = Path(
    "/Volumes/LaCie/apr8th2026depth/android/depthanything_metric_handoff/"
    "DepthAnythingV2MetricIndoorSmall.onnx"
)
DEFAULT_OUT = Path("/tmp/depthanything_splat/USRoom_depthanything_splat.ply")
DEFAULT_INPUT_SIZE = 518
C0 = 0.28209479177387814
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class DepthStats(NamedTuple):
    minimum: float
    p02: float
    median: float
    p98: float
    maximum: float


class FocalInfo(NamedTuple):
    fx_px: float
    fy_px: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Depth Anything metric depth -> deterministic 3DGS PLY."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--opacity", type=float, default=0.99)
    parser.add_argument("--scale-mult", type=float, default=1.0)
    parser.add_argument("--depth-min", type=float, default=None)
    parser.add_argument("--depth-max", type=float, default=None)
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--flip-z", action="store_true")
    parser.add_argument("--depth-png", type=Path, default=None)
    parser.add_argument("--preview-png", type=Path, default=None)
    return parser.parse_args()


def read_rgb(path: Path) -> Image.Image:
    image = Image.open(path.expanduser().resolve())
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def focal_from_exif(path: Path, image_width: int) -> FocalInfo:
    """Return focal length in pixels using 35mm equivalent when available.

    Full-frame sensor width is 36mm:
        fx_px = (focal_length_35mm_equiv / 36.0) * image_width

    If EXIF is missing, use a 28mm equivalent fallback for visual testing only.
    """
    fallback_35mm = 28.0
    try:
        original = Image.open(path.expanduser().resolve())
        exif = original.getexif()
    except Exception:
        exif = {}

    tag_names = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
    focal_35mm = tag_names.get("FocalLengthIn35mmFilm") or tag_names.get("FocalLenIn35mmFilm")
    if focal_35mm and float(focal_35mm) > 1:
        focal_equiv = float(focal_35mm)
        fx_px = (focal_equiv / 36.0) * float(image_width)
        return FocalInfo(fx_px, fx_px, f"exif_35mm_equiv_{focal_equiv:.2f}mm")

    fx_px = (fallback_35mm / 36.0) * float(image_width)
    return FocalInfo(fx_px, fx_px, f"fallback_35mm_equiv_{fallback_35mm:.2f}mm")


def preprocess_depthanything(image: Image.Image, input_size: int) -> np.ndarray:
    # Known-minor Tier 1 behavior: square resize distorts aspect. That is acceptable
    # for this first visual test because the depth map is resized back to W x H.
    resized = image.resize((input_size, input_size), Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def run_depth_onnx(image: Image.Image, onnx_path: Path, input_size: int) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path.expanduser().resolve()),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: preprocess_depthanything(image, input_size)})[0]
    depth = np.asarray(output, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D predicted_depth, got shape {output.shape}")

    width, height = image.size
    depth_image = Image.fromarray(depth.astype(np.float32), mode="F")
    depth_full = depth_image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(depth_full, dtype=np.float32)


def depth_stats(depth: np.ndarray) -> DepthStats:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        raise ValueError("Depth output has no positive finite samples")
    return DepthStats(
        minimum=float(valid.min()),
        p02=float(np.percentile(valid, 2)),
        median=float(np.median(valid)),
        p98=float(np.percentile(valid, 98)),
        maximum=float(valid.max()),
    )


def assert_metric_depth(stats: DepthStats) -> None:
    if stats.maximum <= 1.5 and stats.median <= 1.0:
        raise ValueError(
            "Depth output looks normalized rather than metric meters "
            f"(median={stats.median:.4f}, max={stats.maximum:.4f}); refusing to write PLY."
        )
    if stats.median <= 0.0 or stats.median > 100.0:
        raise ValueError(
            "Depth output does not look like plausible room-scale meters "
            f"(median={stats.median:.4f}); refusing to write PLY."
        )


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / C0


def inv_sigmoid(value: float) -> float:
    value = min(max(value, 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


def unproject_depth(
    depth: np.ndarray,
    focal: FocalInfo,
    flip_y: bool,
    flip_z: bool,
) -> np.ndarray:
    height, width = depth.shape
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    z = depth.astype(np.float32)
    x = (xs - float(width) * 0.5) * z / focal.fx_px
    y = (ys - float(height) * 0.5) * z / focal.fy_px
    if flip_y:
        y = -y
    if flip_z:
        z = -z
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def points_to_gaussians(
    rgb: np.ndarray,
    points: np.ndarray,
    depth: np.ndarray,
    focal: FocalInfo,
    stride: int,
    opacity: float,
    scale_mult: float,
    depth_min: float | None,
    depth_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    ys = np.arange(0, height, max(1, stride), dtype=np.int32)
    xs = np.arange(0, width, max(1, stride), dtype=np.int32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")

    sampled_depth = depth[gy, gx]
    keep = np.isfinite(sampled_depth) & (sampled_depth > 0)
    if depth_min is not None:
        keep &= sampled_depth >= depth_min
    if depth_max is not None:
        keep &= sampled_depth <= depth_max

    xyz = points[gy, gx][keep]
    colors = rgb[gy, gx][keep]
    if xyz.size == 0:
        raise ValueError("No sampled points survived depth filtering")

    z_abs = np.abs(xyz[:, 2]) + 1e-6
    world_size = np.clip(z_abs * float(stride) / focal.fx_px * scale_mult, 1e-4, 0.5)
    scale_log = np.log(world_size)[:, None].repeat(3, axis=1).astype(np.float32)
    f_dc = rgb_to_sh_dc(colors).astype(np.float32)
    opa = np.full((xyz.shape[0], 1), inv_sigmoid(opacity), dtype=np.float32)
    rot = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (xyz.shape[0], 1))
    return xyz.astype(np.float32), f_dc, opa, scale_log, rot


def save_ply(path: Path, xyz: np.ndarray, f_dc: np.ndarray, opa: np.ndarray, scale: np.ndarray, rot: np.ndarray) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    vertices = np.empty(xyz.shape[0], dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = f_dc.T
    vertices["opacity"] = opa[:, 0]
    vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = scale.T
    vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"] = rot.T
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


def save_depth_png(depth: np.ndarray, path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    lo, hi = np.percentile(valid, [2, 98])
    scaled = np.clip((depth - lo) / max(float(hi - lo), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(scaled, mode="L").save(path)


def save_preview_png(rgb: np.ndarray, depth: np.ndarray, path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    lo, hi = np.percentile(valid, [2, 98])
    depth_vis = np.clip((depth - lo) / max(float(hi - lo), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")
    depth_image = Image.fromarray(depth_vis, mode="L").convert("RGB")
    width, height = image.size
    canvas = Image.new("RGB", (width * 2 + 24, height + 28), (255, 255, 255))
    canvas.paste(image, (0, 28))
    canvas.paste(depth_image, (width + 24, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "RGB", fill=(0, 0, 0))
    draw.text((width + 32, 8), "Depth Anything metric depth (visualized 2-98%)", fill=(0, 0, 0))
    canvas.save(path)


def main() -> int:
    args = parse_args()
    image = read_rgb(args.image)
    width, height = image.size
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    focal = focal_from_exif(args.image, width)

    depth = run_depth_onnx(image, args.onnx, args.input_size)
    stats = depth_stats(depth)
    assert_metric_depth(stats)
    points = unproject_depth(depth, focal, args.flip_y, args.flip_z)
    xyz, f_dc, opa, scale, rot = points_to_gaussians(
        rgb=rgb,
        points=points,
        depth=depth,
        focal=focal,
        stride=args.stride,
        opacity=args.opacity,
        scale_mult=args.scale_mult,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
    save_ply(args.out, xyz, f_dc, opa, scale, rot)

    if args.depth_png:
        save_depth_png(depth, args.depth_png)
    if args.preview_png:
        save_preview_png(rgb, depth, args.preview_png)

    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    print(f"image_size={width}x{height}")
    print(f"focal_source={focal.source} fx_px={focal.fx_px:.3f} fy_px={focal.fy_px:.3f}")
    print(
        "depth_m "
        f"min={stats.minimum:.4f} p02={stats.p02:.4f} median={stats.median:.4f} "
        f"p98={stats.p98:.4f} max={stats.maximum:.4f}"
    )
    print(f"sample_stride={args.stride}")
    print(f"gaussian_count={len(xyz):,}")
    print(f"bbox_min=[{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}]")
    print(f"bbox_max=[{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}]")
    print(f"z_range={xyz[:, 2].min():.4f}..{xyz[:, 2].max():.4f}m")
    print(f"scale_log_range={scale.min():.4f}..{scale.max():.4f}")
    print(f"output_path={args.out.expanduser().resolve()}")
    if args.depth_png:
        print(f"depth_png={args.depth_png.expanduser().resolve()}")
    if args.preview_png:
        print(f"preview_png={args.preview_png.expanduser().resolve()}")
    print(
        "note=visual prototype only; fallback intrinsics make X/Y approximate, not measurement-grade"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
