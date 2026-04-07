#!/usr/bin/env python3
"""
RUN THE OFFICIAL APPLE DEPTH PRO PYTORCH MODEL ON A SHARP ROOM THUMBNAIL AND WRITE
`depthpro_metric_depth.bin` IN THE BINARY FORMAT CONSUMED BY `WallMeasurementEstimator`.

EXAMPLE:
  python3 scripts/run_depthpro_reference.py \
    --room-folder /path/to/Room_20260406_123456
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import types
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPTH_PRO_SRC = REPO_ROOT / "android" / "third_party" / "ml-depth-pro" / "src"
DEPTH_PRO_CHECKPOINT = (
    REPO_ROOT / "android" / "third_party" / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"
)

if str(DEPTH_PRO_SRC) not in sys.path:
    sys.path.insert(0, str(DEPTH_PRO_SRC))

if "pillow_heif" not in sys.modules:
    pillow_heif_stub = types.ModuleType("pillow_heif")
    pillow_heif_stub.register_heif_opener = lambda: None

    def _open_heif_unavailable(*_args, **_kwargs):
        raise RuntimeError("HEIF INPUT REQUIRES pillow_heif")

    pillow_heif_stub.open_heif = _open_heif_unavailable
    sys.modules["pillow_heif"] = pillow_heif_stub

_original_timm_create_model = timm.create_model


def _patched_timm_create_model(*args, **kwargs):
    try:
        return _original_timm_create_model(*args, **kwargs)
    except TypeError as exc:
        if "dynamic_img_size" in kwargs and "dynamic_img_size" in str(exc):
            kwargs = dict(kwargs)
            kwargs.pop("dynamic_img_size", None)
            return _original_timm_create_model(*args, **kwargs)
        raise


timm.create_model = _patched_timm_create_model

import depth_pro  # noqa: E402
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT  # noqa: E402


def log(message: str) -> None:
    print(f"DEPTH_PRO {message.upper()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Apple Depth Pro reference inference.")
    parser.add_argument("--room-folder", type=Path, help="Room folder containing thumbnail + camera_exif.json")
    parser.add_argument("--image", type=Path, help="Explicit image path; overrides room thumbnail discovery")
    parser.add_argument("--out", type=Path, help="Output depth binary path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Torch device")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEPTH_PRO_CHECKPOINT,
        help="Depth Pro checkpoint path",
    )
    return parser.parse_args()


def discover_thumbnail(room_folder: Path) -> Path:
    candidates = sorted(room_folder.glob("*_thumbnail.jpg")) + sorted(room_folder.glob("*_thumbnail.png"))
    if not candidates:
        raise FileNotFoundError(f"NO ROOM THUMBNAIL FOUND IN {room_folder}")
    return candidates[0]


def load_sidecar(room_folder: Path | None) -> dict[str, float]:
    if room_folder is None:
        return {}
    path = room_folder / "camera_exif.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def write_sidecar(room_folder: Path | None, additions: dict[str, float]) -> None:
    if room_folder is None or not additions:
        return
    path = room_folder / "camera_exif.json"
    merged = load_sidecar(room_folder)
    merged.update(additions)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, sort_keys=True)
    log(f"UPDATED SIDECAR {path}")


def exact_focal_px(sidecar: dict[str, float], image_width: int) -> tuple[float | None, str]:
    focal_px = sidecar.get("focalLengthPx")
    if focal_px and focal_px > 0:
        return focal_px, "SIDECAR_FOCAL_PX"

    focal_mm = sidecar.get("focalLengthMm")
    if focal_mm and focal_mm > 0:
        sensor_width_mm = sidecar.get("sensorWidthMm")
        if not sensor_width_mm or sensor_width_mm <= 0:
            focal35 = sidecar.get("focalLength35mmEquivMm")
            if focal35 and focal35 > 0:
                sensor_width_mm = 36.0 * focal_mm / focal35
        if sensor_width_mm and sensor_width_mm > 0:
            return (focal_mm / sensor_width_mm) * image_width, "EXIF_FOCAL_MM"
    return None, "DEPTH_PRO_FOV"


def write_depth_bin(path: Path, depth_m: np.ndarray) -> None:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if depth_m.ndim != 2:
        raise ValueError(f"EXPECTED HxW DEPTH ARRAY, GOT {depth_m.shape}")
    h, w = depth_m.shape
    header = struct.pack("<iii", w, h, 1)
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(depth_m.astype("<f4", copy=False).tobytes(order="C"))
    log(f"WROTE DEPTH BIN {path} WIDTH={w} HEIGHT={h}")


def main() -> int:
    args = parse_args()

    if args.room_folder is None and args.image is None:
        raise SystemExit("PASS --ROOM-FOLDER OR --IMAGE")

    room_folder = args.room_folder.resolve() if args.room_folder else None
    image_path = args.image.resolve() if args.image else discover_thumbnail(room_folder)
    out_path = args.out.resolve() if args.out else ((room_folder or image_path.parent) / "depthpro_metric_depth.bin")

    sidecar = load_sidecar(room_folder)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_height, image_width = image_np.shape[:2]

    log(f"IMAGE {image_path} SIZE={image_width}X{image_height}")
    log(f"CHECKPOINT {args.checkpoint}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"CHECKPOINT NOT FOUND: {args.checkpoint}")

    device = torch.device(args.device)
    config = DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = str(args.checkpoint)

    log(f"LOADING MODEL DEVICE={device}")
    model, transform = depth_pro.create_model_and_transforms(config=config, device=device, precision=torch.float32)
    model.eval()

    with torch.no_grad():
        tensor = transform(image_np)
        focal_px_exact, focal_source = exact_focal_px(sidecar, image_width)
        log(f"FOCAL_INPUT SOURCE={focal_source} VALUE={focal_px_exact if focal_px_exact else 'NONE'}")
        prediction = model.infer(tensor, f_px=focal_px_exact)

    depth_m = prediction["depth"].detach().cpu().numpy().astype(np.float32)
    predicted_focal_px = float(prediction["focallength_px"].detach().cpu().item())
    if focal_px_exact is None:
        focal_px_final = predicted_focal_px
        focal_source = "DEPTH_PRO_FOV"
    else:
        focal_px_final = float(focal_px_exact)

    # Approximate FOV from the final focal in the same pixel space as the thumbnail.
    fov_deg = math.degrees(2.0 * math.atan((image_width * 0.5) / max(focal_px_final, 1e-6)))

    write_depth_bin(out_path, depth_m)
    write_sidecar(
        room_folder,
        {
            "focalLengthPx": float(focal_px_final),
            "depthProEstimatedFocalLengthPx": float(predicted_focal_px),
            "depthProFovDegrees": float(fov_deg),
            "depthProMetricDepthWidthPx": float(depth_m.shape[1]),
            "depthProMetricDepthHeightPx": float(depth_m.shape[0]),
        },
    )

    center_depth = float(depth_m[depth_m.shape[0] // 2, depth_m.shape[1] // 2])
    log(
        "RESULT "
        f"DEPTH_SIZE={depth_m.shape[1]}X{depth_m.shape[0]} "
        f"CENTER_DEPTH_M={center_depth:.4f} "
        f"FINAL_FOCAL_PX={focal_px_final:.2f} "
        f"PREDICTED_FOCAL_PX={predicted_focal_px:.2f} "
        f"FOV_DEG={fov_deg:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
