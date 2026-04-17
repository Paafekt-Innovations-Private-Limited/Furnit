#!/usr/bin/env python3
"""
Export the Gaussian-specific FPN decoder to NCNN.

The Gaussian decoder (feature_model.decoder) has:
- dims_encoder: [256, 256, 512, 1024, 1024]
- dims_decoder: [128, 128, 128, 128, 128]

This is different from the monodepth decoder which has dims_decoder=[256,256,256,256,256].
"""

import sys
from pathlib import Path
import subprocess
import os

SHARP_SRC = Path("/tmp/ml-sharp/src")
MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/ml_experiments/models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")
PNNX_PATH = "/opt/miniconda3/bin/pnnx"

sys.path.insert(0, str(SHARP_SRC))

import torch
import torch.nn as nn


class GaussianDecoderForExport(nn.Module):
    """Wrapper for Gaussian FPN decoder."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, f0, f1, f2, f3, f4):
        features = [f0, f1, f2, f3, f4]
        return self.decoder(features)


def export_gaussian_decoder():
    """Export the Gaussian-specific FPN decoder."""
    from sharp.models import PredictorParams, create_predictor

    print("Loading SHARP model...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    # Get the Gaussian decoder (feature_model.decoder)
    decoder = predictor.feature_model.decoder
    print(f"Gaussian decoder: {type(decoder)}")
    print(f"  dims_encoder: {decoder.dims_encoder}")
    print(f"  dims_decoder: {decoder.dims_decoder}")

    wrapped = GaussianDecoderForExport(decoder)
    wrapped.eval()

    # Create test inputs matching expected full-size dimensions
    # Full scale: 768, 384, 192, 96, 48
    f0 = torch.randn(1, 256, 768, 768)
    f1 = torch.randn(1, 256, 384, 384)
    f2 = torch.randn(1, 512, 192, 192)
    f3 = torch.randn(1, 1024, 96, 96)
    f4 = torch.randn(1, 1024, 48, 48)

    print(f"\nTest inputs:")
    print(f"  f0: {f0.shape}")
    print(f"  f1: {f1.shape}")
    print(f"  f2: {f2.shape}")
    print(f"  f3: {f3.shape}")
    print(f"  f4: {f4.shape}")

    with torch.no_grad():
        output = wrapped(f0, f1, f2, f3, f4)
        print(f"  output: {output.shape}")

    # Export to ONNX
    onnx_path = OUTPUT_DIR / "gaussian_fpn_decoder.onnx"
    print(f"\nExporting to {onnx_path}...")

    torch.onnx.export(
        wrapped,
        (f0, f1, f2, f3, f4),
        onnx_path,
        input_names=['f0', 'f1', 'f2', 'f3', 'f4'],
        output_names=['decoder_out'],
        opset_version=17,
    )
    print(f"Exported to {onnx_path}")

    # Convert to NCNN using pnnx with explicit input shapes
    print("\nConverting to NCNN...")
    # Full scale shapes: 768, 384, 192, 96, 48
    input_shapes = "[1,256,768,768],[1,256,384,384],[1,512,192,192],[1,1024,96,96],[1,1024,48,48]"

    cmd = [PNNX_PATH, str(onnx_path), f"inputshape={input_shapes}"]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(OUTPUT_DIR), capture_output=True, text=True)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    if result.returncode != 0:
        print(f"PNNX stderr: {result.stderr[-1000:]}")

    # Check output files
    ncnn_param = OUTPUT_DIR / "gaussian_fpn_decoder.ncnn.param"
    ncnn_bin = OUTPUT_DIR / "gaussian_fpn_decoder.ncnn.bin"

    if ncnn_param.exists() and ncnn_bin.exists():
        print(f"\nSuccess! Created:")
        print(f"  {ncnn_param} ({ncnn_param.stat().st_size} bytes)")
        print(f"  {ncnn_bin} ({ncnn_bin.stat().st_size} bytes)")

        # Rename to fpn_decoder for consistency
        (OUTPUT_DIR / "fpn_decoder.ncnn.param").unlink(missing_ok=True)
        (OUTPUT_DIR / "fpn_decoder.ncnn.bin").unlink(missing_ok=True)
        ncnn_param.rename(OUTPUT_DIR / "fpn_decoder.ncnn.param")
        ncnn_bin.rename(OUTPUT_DIR / "fpn_decoder.ncnn.bin")
        print("\nRenamed to fpn_decoder.ncnn.param/bin")
    else:
        print("\nNCNN conversion may have failed. Check output files.")


def main():
    print("=" * 60)
    print("Exporting Gaussian FPN Decoder")
    print("=" * 60)
    export_gaussian_decoder()


if __name__ == "__main__":
    main()
