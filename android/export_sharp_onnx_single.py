#!/usr/bin/env python3
"""
Export full SHARP model as a single ONNX file, then quantize to INT8.

Produces:
  sharp_single_fp32.onnx + .data  (~2.4 GB)
  sharp_single_int8.onnx + .data  (~600-700 MB)

Usage:
  cd android
  python export_sharp_onnx_single.py [--weights /path/to/sharp.pt] [--sharp-src /path/to/ml-sharp/src]

Then push INT8 model to device:
  adb push sharp_single_int8.onnx /storage/emulated/0/Android/data/com.furnit.android/files/models/
  adb push sharp_single_int8.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_SHARP_SRC = Path("/tmp/ml-sharp/src")
DEFAULT_WEIGHTS = Path("/Users/al/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path(__file__).resolve().parent  # android/


class SharpForONNX(nn.Module):
    """Wrapper that runs the full SHARP predictor and returns 5 output tensors.

    Output tensors (matching OnnxSharp.kt expectations):
      positions:  [1, N, 3]
      scales:     [1, N, 3]
      rotations:  [1, N, 4]
      colors:     [1, N, 3]
      opacity:    [1, N]
    """

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.register_buffer("disparity_factor", torch.tensor([1.0]))

    def forward(self, image: torch.Tensor):
        result = self.predictor(image, self.disparity_factor)
        positions = result.mean_vectors       # [1, N, 3]
        scales = result.singular_values       # [1, N, 3]
        rotations = result.quaternions        # [1, N, 4]
        colors = result.colors                # [1, N, 3]
        opacities = result.opacities          # [1, N] or [1, N, 1]
        if opacities.dim() == 3:
            opacities = opacities.squeeze(-1)  # -> [1, N]
        return positions, scales, rotations, colors, opacities


def parse_args():
    parser = argparse.ArgumentParser(description="Export SHARP as single ONNX + INT8")
    parser.add_argument("--sharp-src", default=str(DEFAULT_SHARP_SRC),
                        help=f"ml-sharp src directory (default: {DEFAULT_SHARP_SRC})")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help=f"SHARP .pt checkpoint (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--skip-fp32", action="store_true",
                        help="Skip FP32 export (use existing sharp_single_fp32.onnx)")
    parser.add_argument("--skip-int8", action="store_true",
                        help="Skip INT8 quantization")
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset version (default: 18)")
    return parser.parse_args()


def export_fp32(model, output_path, opset_version):
    """Export the full SHARP model to a single FP32 ONNX with external data."""
    print(f"\nExporting FP32 ONNX to {output_path} ...")
    dummy_input = torch.randn(1, 3, 1536, 1536)

    t0 = time.time()
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["image"],
        output_names=["positions", "scales", "rotations", "colors", "opacity"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    elapsed = time.time() - t0

    graph_size = output_path.stat().st_size / 1e6
    data_path = Path(str(output_path) + ".data")
    data_size = data_path.stat().st_size / 1e6 if data_path.exists() else 0
    print(f"  FP32 export done in {elapsed:.1f}s")
    print(f"  Graph: {graph_size:.1f} MB, Weights: {data_size:.1f} MB, Total: {graph_size + data_size:.1f} MB")
    return output_path


def quantize_int8(fp32_path, int8_path):
    """Quantize FP32 ONNX to INT8 using onnxruntime dynamic quantization."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"\nQuantizing to INT8: {fp32_path} -> {int8_path} ...")
    t0 = time.time()
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    elapsed = time.time() - t0

    int8_size = int8_path.stat().st_size / 1e6
    int8_data = Path(str(int8_path) + ".data")
    int8_data_size = int8_data.stat().st_size / 1e6 if int8_data.exists() else 0
    total = int8_size + int8_data_size
    print(f"  INT8 quantization done in {elapsed:.1f}s")
    print(f"  Graph: {int8_size:.1f} MB, Weights: {int8_data_size:.1f} MB, Total: {total:.1f} MB")
    return int8_path


def main():
    args = parse_args()

    sharp_src = Path(args.sharp_src)
    if not sharp_src.exists():
        print(f"ERROR: SHARP source not found at {sharp_src}")
        print("Clone ml-sharp repo: git clone ... /tmp/ml-sharp")
        return 1

    sys.path.insert(0, str(sharp_src))
    from sharp.models import PredictorParams, create_predictor

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: Weights not found at {weights_path}")
        return 1

    fp32_path = OUTPUT_DIR / "sharp_single_fp32.onnx"
    int8_path = OUTPUT_DIR / "sharp_single_int8.onnx"

    if not args.skip_fp32:
        print("=" * 60)
        print("Step 1: Export SHARP -> single FP32 ONNX")
        print("=" * 60)

        print(f"Loading SHARP model from {weights_path} ...")
        state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()

        model = SharpForONNX(predictor)
        model.eval()

        print("Running test forward pass ...")
        with torch.no_grad():
            dummy = torch.randn(1, 3, 1536, 1536)
            out = model(dummy)
            print(f"  Output shapes: positions={out[0].shape}, scales={out[1].shape}, "
                  f"rotations={out[2].shape}, colors={out[3].shape}, opacity={out[4].shape}")

        with torch.no_grad():
            export_fp32(model, fp32_path, args.opset)

        del model, predictor, state_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        if not fp32_path.exists():
            print(f"ERROR: --skip-fp32 but {fp32_path} does not exist")
            return 1
        print(f"Skipping FP32 export (using existing {fp32_path})")

    if not args.skip_int8:
        print("\n" + "=" * 60)
        print("Step 2: Quantize FP32 -> INT8")
        print("=" * 60)
        quantize_int8(fp32_path, int8_path)
    else:
        print("Skipping INT8 quantization")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    dest = "/storage/emulated/0/Android/data/com.furnit.android/files/models"
    if int8_path.exists():
        print(f"\nPush to device:")
        print(f"  adb push {int8_path} {dest}/")
        int8_data = Path(str(int8_path) + ".data")
        if int8_data.exists():
            print(f"  adb push {int8_data} {dest}/")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
