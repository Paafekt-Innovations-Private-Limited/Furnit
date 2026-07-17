#!/usr/bin/env python3
"""
Convert the Hugging Face Depth Anything V2 metric indoor small model to Core ML.

Example:
  python3 scripts/convert_depthanything_metric_indoor_small_to_coreml.py \
    --out Furnit/Resources/DepthAnythingV2MetricIndoorSmall.mlpackage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import torch
from transformers import AutoModelForDepthEstimation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEFAULT_OUT = REPO_ROOT / "Furnit" / "Resources" / "DepthAnythingV2MetricIndoorSmall.mlpackage"
DEFAULT_INPUT_SIDE = 518


class MetricIndoorWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pixel_values = image / 255.0
        pixel_values = (pixel_values - self.mean) / self.std
        outputs = self.model(pixel_values=pixel_values)
        return outputs.predicted_depth.unsqueeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert metric indoor Depth Anything V2 small to Core ML.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model identifier")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output .mlpackage path")
    parser.add_argument("--input-side", type=int, default=DEFAULT_INPUT_SIDE, help="Square image input size")
    parser.add_argument("--deployment-target", default="iOS17", choices=["iOS17", "iOS18"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"DEPTH_ANYTHING_METRIC MODEL_ID={args.model_id}")
    print("DEPTH_ANYTHING_METRIC LOADING MODEL")
    base_model = AutoModelForDepthEstimation.from_pretrained(args.model_id)
    base_model.eval()

    wrapper = MetricIndoorWrapper(base_model).eval()
    dummy = torch.zeros(1, 3, args.input_side, args.input_side, dtype=torch.float32)

    print(f"DEPTH_ANYTHING_METRIC TRACING INPUT={args.input_side}X{args.input_side}")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy, strict=False)

    target = ct.target.iOS17 if args.deployment_target == "iOS17" else ct.target.iOS18
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("DEPTH_ANYTHING_METRIC CONVERTING TO COREML")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.ImageType(
                name="image",
                shape=dummy.shape,
                color_layout=ct.colorlayout.RGB,
                scale=1.0,
                bias=[0.0, 0.0, 0.0],
            )
        ],
        outputs=[ct.TensorType(name="metric_depth")],
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
    )

    mlmodel.save(str(out_path))
    print(f"DEPTH_ANYTHING_METRIC SAVED {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
