#!/usr/bin/env python3
"""
CONVERT APPLE'S DEPTH PRO PYTORCH CHECKPOINT INTO A CORE ML PACKAGE THAT OUTPUTS:
  1. CANONICAL INVERSE DEPTH
  2. FOV DEGREES

SWIFT THEN APPLIES THE SAME POST-PROCESSING AS THE REFERENCE `infer()` PATH:
  - USE EXACT EXIF FOCAL WHEN AVAILABLE
  - OTHERWISE DERIVE FOCAL PX FROM FOV
  - SCALE CANONICAL INVERSE DEPTH -> METRIC DEPTH
  - WRITE `depthpro_metric_depth.bin`
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch
import timm
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPTH_PRO_SRC = REPO_ROOT / "android" / "third_party" / "ml-depth-pro" / "src"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "android" / "third_party" / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"
)
DEFAULT_OUT = REPO_ROOT / "Furnit" / "Resources" / "Models" / "DepthProCanonical.mlpackage"

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


class DepthProCoreMLWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        canonical_inverse_depth, fov_deg = self.model(image)
        if fov_deg is None:
            fov_deg = torch.zeros((image.shape[0], 1), dtype=image.dtype, device=image.device)
        return canonical_inverse_depth, fov_deg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Depth Pro to Core ML.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Depth Pro checkpoint")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output .mlpackage path")
    parser.add_argument("--input-side", type=int, default=1536, help="Square model input size")
    parser.add_argument(
        "--deployment-target",
        default="iOS18",
        choices=["iOS17", "iOS18"],
        help="Core ML minimum deployment target",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"CHECKPOINT NOT FOUND: {args.checkpoint}")
    if args.out.suffix != ".mlpackage":
        raise ValueError("OUTPUT PATH MUST END WITH .MLPACKAGE")

    log(f"CHECKPOINT {args.checkpoint}")
    log(f"OUTPUT {args.out}")

    import coremltools as ct

    config = DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = str(args.checkpoint)

    log("LOADING PYTORCH MODEL")
    model, _ = depth_pro.create_model_and_transforms(
        config=config,
        device=torch.device("cpu"),
        precision=torch.float32,
    )
    model.eval()

    wrapper = DepthProCoreMLWrapper(model).eval()
    example = torch.zeros((1, 3, args.input_side, args.input_side), dtype=torch.float32)

    log("TRACING MODEL")
    traced = torch.jit.trace(wrapper, example, strict=False)

    minimum_target = ct.target.iOS18 if args.deployment_target == "iOS18" else ct.target.iOS17

    log("CONVERTING TO COREML")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.ImageType(
                name="image",
                shape=example.shape,
                scale=1.0 / 127.5,
                bias=[-1.0, -1.0, -1.0],
                color_layout=ct.colorlayout.RGB,
            ),
        ],
        outputs=[
            ct.TensorType(name="canonical_inverse_depth"),
            ct.TensorType(name="fov_degrees"),
        ],
        minimum_deployment_target=minimum_target,
        compute_precision=ct.precision.FLOAT16,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.out))
    log(f"COREML PACKAGE WRITTEN {args.out}")
    log("ADD THIS PACKAGE TO THE IOS APP TARGET IF XCODE DOES NOT PICK IT UP AUTOMATICALLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
