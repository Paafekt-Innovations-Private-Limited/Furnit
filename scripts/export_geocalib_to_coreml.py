#!/usr/bin/env python3
"""
Export GeoCalib (pinhole) for Furnit.

GeoCalib's LM optimizer cannot be Torch-traced, so this script exports:
  1. GeoCalibPinholeCNN.mlpackage — Core ML backbone + perspective-field heads
  2. geocalib-pinhole.onnx — same CNN graph in ONNX
  3. geocalib-pinhole.tar — pretrained weights (downloaded if missing)

Full on-device focal/gravity calibration uses this CNN plus the Swift LM solver in
Furnit/Services/RoomReconstruction/GeoCalibCalibrationService.swift.

Example:
  python3 scripts/export_geocalib_to_coreml.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import coremltools as ct
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
GEOCALIB_ROOT = REPO_ROOT / "third_party" / "GeoCalib"
MODEL_DIR = REPO_ROOT / "Furnit" / "Models" / "GeoCalib"
DEFAULT_CNN_OUT = MODEL_DIR / "GeoCalibPinholeCNN.mlpackage"
DEFAULT_ONNX = MODEL_DIR / "geocalib-pinhole-cnn.onnx"
DEFAULT_WEIGHTS = MODEL_DIR / "weights" / "geocalib-pinhole.tar"
DEFAULT_INPUT_SIDE = 320

CNN_OUTPUT_NAMES = [
    "up_field",
    "latitude_field",
    "up_confidence",
    "latitude_confidence",
]


def _ensure_geocalib_import() -> None:
    geocalib_path = str(GEOCALIB_ROOT)
    if geocalib_path not in sys.path:
        sys.path.insert(0, geocalib_path)


class GeoCalibCNNWrapper(nn.Module):
    """Traceable GeoCalib CNN (MSCAN + perspective-field decoders, no LM optimizer)."""

    def __init__(self, geocalib_model: nn.Module) -> None:
        super().__init__()
        self.backbone = geocalib_model.backbone
        self.ll_enc = geocalib_model.ll_enc
        self.perspective_decoder = geocalib_model.perspective_decoder

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = {
            "hl": self.backbone({"image": image})["features"],
            "ll": self.ll_enc({"image": image})["features"],
        }
        fields = self.perspective_decoder({"features": features})
        return (
            fields["up_field"],
            fields["latitude_field"],
            fields["up_confidence"],
            fields["latitude_confidence"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GeoCalib CNN to Core ML / ONNX.")
    parser.add_argument("--weights", default="pinhole", help='GeoCalib weights: "pinhole", "distorted", or path')
    parser.add_argument("--out", type=Path, default=DEFAULT_CNN_OUT, help="Output .mlpackage path")
    parser.add_argument("--onnx-out", type=Path, default=DEFAULT_ONNX, help="Output .onnx path")
    parser.add_argument("--weights-out", type=Path, default=DEFAULT_WEIGHTS, help="Copy downloaded weights here")
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_SIDE, help="Trace input height")
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_SIDE, help="Trace input width")
    parser.add_argument("--deployment-target", default="iOS17", choices=["iOS17", "iOS18"])
    parser.add_argument("--skip-coreml", action="store_true", help="Only export ONNX / weights")
    parser.add_argument("--export-onnx", action="store_true", help="Also export ONNX (experimental)")
    return parser.parse_args()


def load_geocalib(weights: str) -> nn.Module:
    _ensure_geocalib_import()
    from geocalib.extractor import GeoCalib

    model = GeoCalib(weights=weights)
    model.eval()
    return model.model


def copy_cached_weights(weights: str, destination: Path) -> None:
    import torch

    if destination.exists():
        print(f"GEOCALIB WEIGHTS ALREADY AT {destination}")
        return

    if weights not in {"pinhole", "distorted"}:
        source = Path(weights)
        if not source.exists():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"GEOCALIB COPIED WEIGHTS {source} -> {destination}")
        return

    cached = Path(torch.hub.get_dir()) / "geocalib" / f"{weights}.tar"
    if not cached.exists():
        load_geocalib(weights)
        cached = Path(torch.hub.get_dir()) / "geocalib" / f"{weights}.tar"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, destination)
    print(f"GEOCALIB COPIED WEIGHTS {cached} -> {destination}")


def main() -> int:
    args = parse_args()

    print(f"GEOCALIB WEIGHTS={args.weights}")
    copy_cached_weights(args.weights, args.weights_out)

    print("GEOCALIB LOADING MODEL")
    core_model = load_geocalib(args.weights)
    wrapper = GeoCalibCNNWrapper(core_model).eval()

    dummy_image = torch.zeros(1, 3, args.input_height, args.input_width, dtype=torch.float32)
    print(f"GEOCALIB TRACING CNN INPUT={args.input_width}x{args.input_height}")
    with torch.no_grad():
        sample = wrapper(dummy_image)
        print(
            "GEOCALIB CNN OUTPUT SHAPES",
            {name: tuple(value.shape) for name, value in zip(CNN_OUTPUT_NAMES, sample)},
        )
        traced = torch.jit.trace(wrapper, dummy_image, strict=False)

    if args.export_onnx:
        args.onnx_out.parent.mkdir(parents=True, exist_ok=True)
        print(f"GEOCALIB EXPORTING ONNX -> {args.onnx_out}")
        torch.onnx.export(
            wrapper,
            dummy_image,
            str(args.onnx_out),
            input_names=["image"],
            output_names=CNN_OUTPUT_NAMES,
            opset_version=18,
            dynamo=False,
        )

    if not args.skip_coreml:
        target = ct.target.iOS17 if args.deployment_target == "iOS17" else ct.target.iOS18
        args.out.parent.mkdir(parents=True, exist_ok=True)
        print("GEOCALIB CONVERTING CNN TO COREML")
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(
                    name="image",
                    shape=dummy_image.shape,
                    dtype=float,
                )
            ],
            outputs=[ct.TensorType(name=name) for name in CNN_OUTPUT_NAMES],
            minimum_deployment_target=target,
            compute_precision=ct.precision.FLOAT16,
        )
        mlmodel.short_description = (
            "GeoCalib CNN perspective fields (up/latitude + confidence). "
            "Furnit runs focal/gravity LM natively in Swift after this model."
        )
        mlmodel.save(str(args.out.resolve()))
        print(f"GEOCALIB SAVED {args.out.resolve()}")

    print("GEOCALIB NOTE: full LM calibration runs natively in Swift in the iOS app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
