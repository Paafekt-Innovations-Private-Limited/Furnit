#!/usr/bin/env python3
"""
Export the SHARP FPN decoder to NCNN via PNNX.

The decoder takes 5 feature maps from the encoder and produces
a single feature map suitable for the Gaussian heads.
"""

import sys
import os
from pathlib import Path
import subprocess
import tempfile

SHARP_SRC = Path("/tmp/ml-sharp/src")
MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/ml_experiments/models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")
PNNX_PATH = "/opt/miniconda3/bin/pnnx"

sys.path.insert(0, str(SHARP_SRC))

import torch
import torch.nn as nn


class DecoderForExport(nn.Module):
    """Wrapper for decoder that accepts 5 separate inputs."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, f0, f1, f2, f3, f4):
        features = [f0, f1, f2, f3, f4]
        return self.decoder(features)


def analyze_decoder():
    """Analyze the decoder architecture and dimensions."""
    from sharp.models import PredictorParams, create_predictor

    print("Loading SHARP model...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    # Explore the predictor structure
    print("\nPredictor structure:")
    for name, child in predictor.named_children():
        print(f"  {name}: {type(child)}")

    # Get the monodepth predictor which contains encoder/decoder
    mono = predictor.monodepth_model.monodepth_predictor
    print(f"\nMonodepth predictor: {type(mono)}")
    for name, child in mono.named_children():
        print(f"  {name}: {type(child)}")

    # Get the decoder
    decoder = mono.decoder
    print(f"\nDecoder: {type(decoder)}")
    if hasattr(decoder, 'dims_encoder'):
        print(f"  dims_encoder: {decoder.dims_encoder}")
    if hasattr(decoder, 'dims_decoder'):
        print(f"  dims_decoder: {decoder.dims_decoder}")

    # Get Gaussian decoder
    gaussian_decoder = predictor.gaussian_composer
    print(f"\nGaussian composer: {type(gaussian_decoder)}")

    # Print decoder structure
    print("\n=== Decoder Layer Structure ===")
    for name, module in decoder.named_modules():
        if name and hasattr(module, 'weight'):
            print(f"  {name}: {module.__class__.__name__}, weight: {module.weight.shape}")

    # Check Gaussian composer
    for name, child in gaussian_decoder.named_children():
        print(f"  {name}: {type(child)}")

    return decoder


def export_decoder_to_onnx():
    """Export the decoder to ONNX format."""
    from sharp.models import PredictorParams, create_predictor

    print("Loading SHARP model...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    # Get the monodepth decoder (FPN decoder)
    decoder = predictor.monodepth_model.monodepth_predictor.decoder

    # Check what we have
    if decoder is not None:
        print(f"Found decoder: dims_encoder={decoder.dims_encoder}, dims_decoder={decoder.dims_decoder}")

        # Wrap for export
        wrapped_decoder = DecoderForExport(decoder)
        wrapped_decoder.eval()

        # Create dummy inputs matching encoder output dimensions
        # Use smaller scale but maintain proper ratio
        # Full: 768, 384, 192, 96, 48 (each level is 2x smaller)
        # We use base=48 to keep memory reasonable
        base = 48
        f0 = torch.randn(1, 256, base * 16, base * 16)  # 768 equivalent
        f1 = torch.randn(1, 256, base * 8, base * 8)    # 384 equivalent
        f2 = torch.randn(1, 512, base * 4, base * 4)    # 192 equivalent
        f3 = torch.randn(1, 1024, base * 2, base * 2)   # 96 equivalent
        f4 = torch.randn(1, 1024, base, base)           # 48 equivalent

        print(f"\nTest inputs:")
        print(f"  f0: {f0.shape}")
        print(f"  f1: {f1.shape}")
        print(f"  f2: {f2.shape}")
        print(f"  f3: {f3.shape}")
        print(f"  f4: {f4.shape}")

        with torch.no_grad():
            output = wrapped_decoder(f0, f1, f2, f3, f4)
            print(f"  output: {output.shape}")

        # Export to ONNX
        onnx_path = OUTPUT_DIR / "fpn_decoder.onnx"
        print(f"\nExporting to {onnx_path}...")

        torch.onnx.export(
            wrapped_decoder,
            (f0, f1, f2, f3, f4),
            onnx_path,
            input_names=['f0', 'f1', 'f2', 'f3', 'f4'],
            output_names=['decoder_out'],
            dynamic_axes={
                'f0': {2: 'h0', 3: 'w0'},
                'f1': {2: 'h1', 3: 'w1'},
                'f2': {2: 'h2', 3: 'w2'},
                'f3': {2: 'h3', 3: 'w3'},
                'f4': {2: 'h4', 3: 'w4'},
                'decoder_out': {2: 'out_h', 3: 'out_w'},
            },
            opset_version=17,
        )
        print(f"  Exported to {onnx_path}")
        return onnx_path

    else:
        print("No separate decoder found in Gaussian predictor")
        return None


def convert_onnx_to_ncnn(onnx_path):
    """Convert ONNX model to NCNN using pnnx."""
    if not os.path.exists(PNNX_PATH):
        print(f"PNNX not found at {PNNX_PATH}")
        return False

    print(f"\nConverting {onnx_path} to NCNN...")

    # Run pnnx
    cmd = [PNNX_PATH, str(onnx_path)]
    result = subprocess.run(cmd, cwd=str(OUTPUT_DIR), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"PNNX failed: {result.stderr}")
        return False

    print("  PNNX conversion successful")
    return True


def main():
    print("=" * 60)
    print("SHARP FPN Decoder Export")
    print("=" * 60)

    # First analyze the decoder
    print("\n--- Analyzing Decoder ---")
    analyze_decoder()

    # Try to export
    print("\n--- Exporting Decoder ---")
    onnx_path = export_decoder_to_onnx()

    if onnx_path:
        print("\n--- Converting to NCNN ---")
        if convert_onnx_to_ncnn(onnx_path):
            print("\nDecoder exported successfully!")
        else:
            print("\nNCNN conversion failed")
    else:
        print("\nDecoder export failed")


if __name__ == "__main__":
    main()
