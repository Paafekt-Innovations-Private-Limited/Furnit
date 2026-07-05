#!/usr/bin/env python3
"""Predict room dimensions from a single photo using Depth Anything V2 metric depth.

No manual measurements required. Auto-writes:
  - measurements JSON
  - depth preview PNG
  - textured room mesh GLB

Example:
  python3 scripts/depthanything_measure_room.py --image /Users/al/Downloads/USRoom.jpeg
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image, ImageOps

DEFAULT_IMAGE = Path("/Users/al/Downloads/USRoom.jpeg")
DEFAULT_ONNX = Path(
    "/Volumes/LaCie/apr8th2026depth/android/depthanything_metric_handoff/"
    "DepthAnythingV2MetricIndoorSmall.onnx"
)
DEFAULT_OUT_DIR = Path("/tmp/depthanything_splat")
DEFAULT_INPUT_SIZE = 518
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class RoomMeasurements:
    width_m: float
    height_m: float
    depth_m: float
    image_width: int
    image_height: int
    intrinsics_source: str
    focal_px: float
    depth_median_m: float
    depth_min_m: float
    depth_max_m: float
    wall_rect_norm: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single photo -> Depth Anything metric room dimensions (meters)."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument(
        "--wall-margin",
        type=float,
        default=0.05,
        help="Fraction inset from each image edge when estimating visible wall (default 5%%).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Folder for auto-written outputs (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Override measurements JSON path (default: <out-dir>/<image>_measurements.json).",
    )
    parser.add_argument(
        "--depth-png",
        type=Path,
        default=None,
        help="Override depth PNG path (default: <out-dir>/<image>_depth.png).",
    )
    parser.add_argument("--no-depth-png", action="store_true", help="Skip writing depth preview PNG.")
    parser.add_argument("--no-glb", action="store_true", help="Skip writing textured room GLB.")
    parser.add_argument(
        "--mesh-step",
        type=int,
        default=2,
        help="Pixel stride when building GLB mesh grid (default 2).",
    )
    parser.add_argument(
        "--max-depth-jump",
        type=float,
        default=0.15,
        help="Skip mesh triangles across depth edges larger than this many meters.",
    )
    parser.add_argument(
        "--flat-mesh",
        action="store_true",
        help="Put the photo on a single flat plane (no depth relief). Cleanest texture, no drag.",
    )
    parser.add_argument(
        "--depth-smooth",
        type=int,
        default=7,
        help="Median-filter kernel for depth before meshing (0=off). Reduces wall rippling.",
    )
    return parser.parse_args()


def default_output_paths(image_path: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    stem = image_path.expanduser().resolve().stem
    folder = out_dir.expanduser().resolve()
    return (
        folder / f"{stem}_measurements.json",
        folder / f"{stem}_depth.png",
        folder / f"{stem}_room.glb",
    )


def read_rgb(path: Path) -> Image.Image:
    image = Image.open(path.expanduser().resolve())
    return ImageOps.exif_transpose(image).convert("RGB")


def focal_from_exif(path: Path, image_width: int, image_height: int) -> tuple[float, float, str]:
    fallback_35mm = 28.0
    try:
        exif = Image.open(path.expanduser().resolve()).getexif()
    except Exception:
        exif = {}
    tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
    focal_35mm = tags.get("FocalLengthIn35mmFilm") or tags.get("FocalLenIn35mmFilm")
    if focal_35mm and float(focal_35mm) > 1:
        fx = (float(focal_35mm) / 36.0) * image_width
        fy = fx * image_height / image_width
        return fx, fy, f"exif_35mm_equiv_{float(focal_35mm):.1f}mm"
    fx = (fallback_35mm / 36.0) * image_width
    fy = fx * image_height / image_width
    return fx, fy, f"fallback_35mm_equiv_{fallback_35mm:.1f}mm"


def preprocess(image: Image.Image, input_size: int) -> np.ndarray:
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
    inp = session.get_inputs()[0].name
    out = session.get_outputs()[0].name
    raw = session.run([out], {inp: preprocess(image, input_size)})[0]
    depth = np.asarray(raw, dtype=np.float32).squeeze()
    w, h = image.size
    depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
    depth_img = depth_img.resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(depth_img, dtype=np.float32)


def wall_rect_px(image_width: int, image_height: int, margin: float) -> tuple[float, float, float, float]:
    m = max(0.0, min(margin, 0.45))
    x = m * image_width
    y = m * image_height
    w = (1.0 - 2.0 * m) * image_width
    h = (1.0 - 2.0 * m) * image_height
    return x, y, w, h


def median_at(depth: np.ndarray, px: int, py: int, radius: int = 5) -> float | None:
    h, w = depth.shape
    patch = depth[max(0, py - radius): min(h, py + radius + 1), max(0, px - radius): min(w, px + radius + 1)]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(valid)) if valid.size else None


def measure_wall(
    depth: np.ndarray,
    rect: tuple[float, float, float, float],
    fx: float,
    fy: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float]:
    x, y, rw, rh = rect
    cx = intrinsics_cx = (image_width - 1) * 0.5
    cy = intrinsics_cy = (image_height - 1) * 0.5
    left_x = int(np.clip(round(x), 0, image_width - 1))
    right_x = int(np.clip(round(x + rw - 1), 0, image_width - 1))
    top_y = int(np.clip(round(y), 0, image_height - 1))
    bottom_y = int(np.clip(round(y + rh - 1), 0, image_height - 1))
    center_x = int(np.clip(round(x + rw * 0.5), 0, image_width - 1))
    center_y = int(np.clip(round(y + rh * 0.5), 0, image_height - 1))

    center_depth = median_at(depth, center_x, center_y)
    if center_depth is None:
        raise ValueError("Could not sample depth at image center — depth map may be empty.")

    left_plane = (left_x - cx) * center_depth / fx
    right_plane = (right_x - cx) * center_depth / fx
    top_plane = (top_y - cy) * center_depth / fy
    bottom_plane = (bottom_y - cy) * center_depth / fy
    return (
        abs(right_plane - left_plane),
        abs(bottom_plane - top_plane),
        center_depth,
    )


def depth_summary(depth: np.ndarray) -> tuple[float, float, float]:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return math.nan, math.nan, math.nan
    return float(valid.min()), float(np.median(valid)), float(valid.max())


def save_depth_png(depth: np.ndarray, path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    lo, hi = np.percentile(valid, [2, 98])
    scaled = np.clip((depth - lo) / max(float(hi - lo), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(scaled).save(path)


def smooth_depth_for_mesh(depth: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return depth
    from scipy.ndimage import median_filter

    valid = np.isfinite(depth) & (depth > 0)
    smoothed = median_filter(np.where(valid, depth, 0.0), size=kernel)
    counts = median_filter(valid.astype(np.float32), size=kernel)
    out = depth.copy()
    mask = counts > 0.5
    out[mask] = smoothed[mask]
    out[~valid] = np.nan
    return out


def flatten_depth_for_mesh(depth: np.ndarray) -> np.ndarray:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return depth
    plane_depth = float(np.median(valid))
    out = depth.copy()
    out[np.isfinite(out) & (out > 0)] = plane_depth
    return out


def build_textured_mesh_glb(
    rgb_image: Image.Image,
    depth: np.ndarray,
    room_width_m: float,
    glb_path: Path,
    mesh_step: int,
    max_depth_jump: float,
) -> tuple[int, int]:
    """Build a flat/relief mesh aligned with DepthAnythingRoomReconstructor on iOS.

    XY is proportional to pixel position. Z is depth relief only (or flat if depth is constant).
    Uses UV texture mapping so colors stay pinned to pixels instead of smearing via vertex colors.
    """
    import trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    img_np = np.asarray(rgb_image, dtype=np.uint8)
    img_h, img_w = depth.shape
    step = max(1, mesh_step)

    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    depth_max = float(np.max(valid_depth))
    if not math.isfinite(depth_max) or depth_max <= 0:
        raise ValueError("Depth map has no valid samples for mesh export.")

    pixel_scale = room_width_m / float(img_w)
    center_x = img_w * 0.5
    center_y = img_h * 0.5

    rows = np.arange(0, img_h, step, dtype=np.int32)
    cols = np.arange(0, img_w, step, dtype=np.int32)
    grid_h, grid_w = len(rows), len(cols)

    vertex_indices = np.full(grid_h * grid_w, -1, dtype=np.int32)
    vertices: list[list[float]] = []
    uvs: list[list[float]] = []

    for ri, row in enumerate(rows):
        for ci, col in enumerate(cols):
            d = float(depth[row, col])
            if not math.isfinite(d) or d <= 0:
                continue
            vertex_indices[ri * grid_w + ci] = len(vertices)
            vertices.append([
                -(float(col) - center_x) * pixel_scale,
                (float(row) - center_y) * pixel_scale,
                -(depth_max - d),
            ])
            uvs.append([
                float(col) / max(img_w - 1, 1),
                1.0 - float(row) / max(img_h - 1, 1),
            ])

    if not vertices:
        raise ValueError("No valid depth vertices for mesh export.")

    faces: list[list[int]] = []
    for ri in range(grid_h - 1):
        for ci in range(grid_w - 1):
            i00 = ri * grid_w + ci
            i10 = ri * grid_w + (ci + 1)
            i01 = (ri + 1) * grid_w + ci
            i11 = (ri + 1) * grid_w + (ci + 1)
            v00 = int(vertex_indices[i00])
            v10 = int(vertex_indices[i10])
            v01 = int(vertex_indices[i01])
            v11 = int(vertex_indices[i11])
            if min(v00, v10, v01, v11) < 0:
                continue

            d00 = float(depth[rows[ri], cols[ci]])
            d10 = float(depth[rows[ri], cols[ci + 1]])
            d01 = float(depth[rows[ri + 1], cols[ci]])
            d11 = float(depth[rows[ri + 1], cols[ci + 1]])
            depths = (d00, d10, d01, d11)
            if not all(math.isfinite(v) and v > 0 for v in depths):
                continue
            if max(abs(a - b) for i, a in enumerate(depths) for b in depths[i + 1:]) > max_depth_jump:
                continue

            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    if not faces:
        raise ValueError("Could not build any mesh faces from depth map.")

    material = PBRMaterial(baseColorTexture=rgb_image, metallicFactor=0.0, roughnessFactor=1.0)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        process=False,
    )
    mesh.visual = TextureVisuals(uv=np.asarray(uvs, dtype=np.float32), material=material)
    glb_path = glb_path.expanduser().resolve()
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="room")
    scene.export(glb_path)
    return len(vertices), len(faces)


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    json_path, depth_png_path, glb_path = default_output_paths(image_path, args.out_dir)
    if args.out_json:
        json_path = args.out_json.expanduser().resolve()
    if args.depth_png:
        depth_png_path = args.depth_png.expanduser().resolve()

    image = read_rgb(image_path)
    width, height = image.size
    fx, fy, focal_source = focal_from_exif(image_path, width, height)
    depth = run_depth_onnx(image, args.onnx, args.input_size)
    d_min, d_median, d_max = depth_summary(depth)

    margin = args.wall_margin
    rect = wall_rect_px(width, height, margin)
    width_m, height_m, depth_m = measure_wall(depth, rect, fx, fy, width, height)
    mesh_room_width_m = max(width_m, 2.0)
    mesh_room_height_m = mesh_room_width_m * height / width

    result = RoomMeasurements(
        width_m=round(mesh_room_width_m, 3),
        height_m=round(mesh_room_height_m, 3),
        depth_m=round(depth_m, 3),
        image_width=width,
        image_height=height,
        intrinsics_source=focal_source,
        focal_px=round(fx, 2),
        depth_median_m=round(d_median, 3),
        depth_min_m=round(d_min, 3),
        depth_max_m=round(d_max, 3),
        wall_rect_norm=[margin, margin, 1.0 - 2 * margin, 1.0 - 2 * margin],
    )

    payload = asdict(result)
    payload["image_path"] = str(image_path)
    payload["onnx_path"] = str(args.onnx.expanduser().resolve())
    outputs: dict[str, str] = {"measurements_json": str(json_path)}
    payload["note"] = (
        "Model-predicted dimensions only. Width/height assume the visible frame is one wall surface; "
        "depth is camera-to-wall at center. EXIF missing -> 28mm-equiv fallback affects X/Y scale."
    )

    json_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_depth_png:
        save_depth_png(depth, depth_png_path)
        outputs["depth_png"] = str(depth_png_path)

    mesh_vertices = 0
    mesh_faces = 0
    if not args.no_glb:
        mesh_depth = depth.copy()
        if args.flat_mesh:
            mesh_depth = flatten_depth_for_mesh(mesh_depth)
        elif args.depth_smooth > 1:
            mesh_depth = smooth_depth_for_mesh(mesh_depth, args.depth_smooth)
        mesh_vertices, mesh_faces = build_textured_mesh_glb(
            rgb_image=image,
            depth=mesh_depth,
            room_width_m=mesh_room_width_m,
            glb_path=glb_path,
            mesh_step=args.mesh_step,
            max_depth_jump=args.max_depth_jump,
        )
        outputs["room_glb"] = str(glb_path)
        payload["mesh_vertices"] = mesh_vertices
        payload["mesh_faces"] = mesh_faces
        payload["mesh_mode"] = "flat" if args.flat_mesh else "relief"

    payload["outputs"] = outputs

    text = json.dumps(payload, indent=2)
    json_path.write_text(text + "\n", encoding="utf-8")

    print(text)
    print()
    print("=== OUTPUT FILES ===")
    print(f"measurements_json: {json_path}")
    if not args.no_depth_png:
        print(f"depth_png:         {depth_png_path}")
    if not args.no_glb:
        print(f"room_glb:          {glb_path}  ({mesh_vertices:,} verts, {mesh_faces:,} faces)")
    print(f"open output folder: open {json_path.parent}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
