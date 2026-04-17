#!/usr/bin/env python3
"""
Export GaussianHead from NCNN weights to ExecuTorch .pte format.

The GaussianHead is a lightweight decoder (~3.6MB) that takes
merged encoder features [1, 1024, 96, 96] and outputs Gaussian
parameters [1, 14, 384, 384] with 4x spatial upsampling.

Architecture (from gaussian_head.ncnn.param):
  Conv 1024→256, 1x1, ReLU
  Conv 256→256, 3x3, pad=1, ReLU
  Upsample 2x (bilinear)
  Conv 256→256, 3x3, pad=1, ReLU
  Upsample 2x (bilinear)
  Conv 256→128, 3x3, pad=1, ReLU
  Conv 128→64, 3x3, pad=1, ReLU
  Conv 64→14, 1x1

Input:  [1, 1024, 96, 96]  (merged 1x scale features)
Output: [1, 14, 384, 384]  (Gaussian parameters)
"""

import struct
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


NCNN_BIN = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models/gaussian_head.ncnn.bin")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")


class GaussianHead(nn.Module):
    """
    Lightweight Gaussian prediction head matching NCNN architecture.
    """
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1024, 256, 1, bias=True)
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(256, 128, 3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(64, 14, 1, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.relu(self.conv2(x))
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


def read_ncnn_blob(f, num_elements):
    """
    Read a weight blob from NCNN bin file.
    Returns float32 numpy array.
    """
    # Read flag (4 bytes)
    flag_bytes = f.read(4)
    if len(flag_bytes) < 4:
        raise ValueError("Unexpected end of file")
    flag = struct.unpack('<I', flag_bytes)[0]

    if flag == 0:
        # float32
        data = f.read(num_elements * 4)
        return np.frombuffer(data, dtype=np.float32).copy()
    elif flag == 0x01306B47:
        # float16
        data = f.read(num_elements * 2)
        return np.frombuffer(data, dtype=np.float16).astype(np.float32).copy()
    elif flag == 0x000D4B38:
        # quantized int8 - read quantization params then data
        # This shouldn't happen for our model but handle gracefully
        raise ValueError(f"int8 quantized weights not supported (flag=0x{flag:08X})")
    else:
        # Flag might be part of data (NCNN sometimes stores raw float32 without flag)
        # Try treating the flag as first 4 bytes of float32 data
        remaining = f.read((num_elements - 1) * 4)
        return np.frombuffer(flag_bytes + remaining, dtype=np.float32).copy()


def load_weights_from_ncnn(model, bin_path):
    """
    Load weights from NCNN bin file into PyTorch model.

    NCNN layer order (from param):
    convrelu_0: Conv 1024→256, 1x1 (weight + bias)
    convrelu_1: Conv 256→256, 3x3 (weight + bias)
    upsample_11: Interp (no weights)
    convrelu_2: Conv 256→256, 3x3 (weight + bias)
    upsample_12: Interp (no weights)
    convrelu_3: Conv 256→128, 3x3 (weight + bias)
    convrelu_4: Conv 128→64, 3x3 (weight + bias)
    conv_5: Conv 64→14, 1x1 (weight + bias)
    """

    conv_specs = [
        # (pytorch_layer, out_ch, in_ch, kernel, name)
        (model.conv0, 256, 1024, 1, "convrelu_0"),
        (model.conv1, 256, 256, 3, "convrelu_1"),
        (model.conv2, 256, 256, 3, "convrelu_2"),
        (model.conv3, 128, 256, 3, "convrelu_3"),
        (model.conv4, 64, 128, 3, "convrelu_4"),
        (model.conv5, 14, 64, 1, "conv_5"),
    ]

    with open(bin_path, 'rb') as f:
        for layer, out_ch, in_ch, kernel, name in conv_specs:
            weight_size = out_ch * in_ch * kernel * kernel
            print(f"  Loading {name}: weight[{out_ch}x{in_ch}x{kernel}x{kernel}] = {weight_size} floats")

            weight = read_ncnn_blob(f, weight_size)
            bias = read_ncnn_blob(f, out_ch)

            # NCNN stores weights as [out_ch, in_ch, kH, kW] (same as PyTorch)
            weight_tensor = torch.from_numpy(weight.reshape(out_ch, in_ch, kernel, kernel))
            bias_tensor = torch.from_numpy(bias)

            layer.weight.data = weight_tensor
            layer.bias.data = bias_tensor

            print(f"    weight range: [{weight.min():.4f}, {weight.max():.4f}]")
            print(f"    bias range: [{bias.min():.4f}, {bias.max():.4f}]")

        # Check if we've consumed all data
        remaining = f.read()
        if len(remaining) > 0:
            print(f"  Warning: {len(remaining)} bytes remaining in bin file")
        else:
            print("  All weights loaded successfully!")


def export_to_pte(model, example_input, output_path):
    """Export model to ExecuTorch .pte format."""
    from executorch.exir import EdgeCompileConfig, to_edge

    print("\nExporting to ExecuTorch...")

    # Test inference
    with torch.no_grad():
        output = model(example_input)
        print(f"  Input: {example_input.shape}")
        print(f"  Output: {output.shape}")

    # Export
    print("  torch.export...")
    exported = torch.export.export(model, (example_input,), strict=False)

    print("  to_edge...")
    edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))

    print("  to_executorch...")
    et = edge.to_executorch()

    print(f"  Saving {output_path.name}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Done: {size_mb:.1f} MB")
    return output_path


def main():
    print("=" * 60)
    print("GaussianHead Export: NCNN → ExecuTorch .pte")
    print("=" * 60)

    if not NCNN_BIN.exists():
        print(f"ERROR: NCNN bin not found at {NCNN_BIN}")
        return 1

    print(f"\nNCNN bin: {NCNN_BIN} ({NCNN_BIN.stat().st_size / 1024:.1f} KB)")

    # Create model
    model = GaussianHead()
    model.eval()

    # Load weights
    print("\nLoading weights from NCNN bin...")
    load_weights_from_ncnn(model, NCNN_BIN)

    # Verify with test input
    print("\nVerifying model...")
    dummy_input = torch.randn(1, 1024, 96, 96)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Input: {dummy_input.shape} → Output: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Export to .pte
    output_path = OUTPUT_DIR / "sharp_gaussian_head.pte"
    export_to_pte(model, dummy_input, output_path)

    print(f"\nPush to device:")
    print(f"  adb push {output_path} /sdcard/Android/data/com.furnit.android/files/models/")
    print(f"  adb push {output_path} /data/local/tmp/furnit/")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
