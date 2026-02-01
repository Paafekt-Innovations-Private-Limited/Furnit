#!/usr/bin/env python3
"""
Export GaussianHead to ONNX and convert to NCNN for mobile deployment.

This creates an untrained model with random weights - for testing the pipeline.
For production, train the model first using train_gaussian_head.py.

Usage:
    python export_gaussian_head.py --output_dir ../app/src/main/assets/models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import subprocess
from pathlib import Path


class GaussianHeadNCNN(nn.Module):
    """
    Lightweight head for NCNN deployment.

    Input: [B, 1024, 96, 96] merged encoder features
    Output: [B, 14, 384, 384] Gaussian parameters per pixel
        - channels 0-2: xyz positions (tanh scaled)
        - channel 3: opacity (sigmoid)
        - channels 4-6: scales (softplus)
        - channels 7-10: rotation quaternion (normalized)
        - channels 11-13: RGB color (sigmoid)
    """

    def __init__(self, in_channels: int = 1024, hidden_channels: int = 256):
        super().__init__()

        # Channel reduction (1024 -> 256)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),  # Use ReLU instead of SiLU for better NCNN support
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Upsample path: 96 -> 192 -> 384
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Output head: 14 channels for full Gaussian parameters
        # xyz(3) + opacity(1) + scale(3) + rotation(4) + color(3) = 14
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, hidden_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, 14, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, 1024, 96, 96] merged encoder features

        Returns:
            [B, 14, 384, 384] raw Gaussian parameters (activations applied in C++)
        """
        x = self.reduce(features)  # [B, 256, 96, 96]
        x = self.up1(x)            # [B, 256, 192, 192]
        x = self.up2(x)            # [B, 128, 384, 384]
        out = self.head(x)         # [B, 14, 384, 384]
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def export_to_onnx(model: GaussianHeadNCNN, output_path: Path):
    """Export to ONNX format."""
    model.eval()

    dummy_input = torch.randn(1, 1024, 96, 96)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['encoder_features'],
        output_names=['gaussian_params'],
        opset_version=11,  # Use 11 for better NCNN compatibility
        do_constant_folding=True,
    )
    print(f"Exported ONNX to: {output_path}")


def convert_to_ncnn(onnx_path: Path, output_dir: Path):
    """Convert ONNX to NCNN using onnx2ncnn."""
    param_path = output_dir / "gaussian_head.ncnn.param"
    bin_path = output_dir / "gaussian_head.ncnn.bin"

    # Try to find onnx2ncnn
    onnx2ncnn_paths = [
        "onnx2ncnn",  # In PATH
        "/usr/local/bin/onnx2ncnn",
        "/opt/homebrew/bin/onnx2ncnn",
        str(Path.home() / "ncnn/build/tools/onnx/onnx2ncnn"),
    ]

    onnx2ncnn = None
    for path in onnx2ncnn_paths:
        try:
            result = subprocess.run([path, "--help"], capture_output=True)
            if result.returncode == 0 or "onnx2ncnn" in result.stderr.decode():
                onnx2ncnn = path
                break
        except FileNotFoundError:
            continue

    if onnx2ncnn is None:
        print("WARNING: onnx2ncnn not found. Please convert manually:")
        print(f"  onnx2ncnn {onnx_path} {param_path} {bin_path}")
        return None, None

    try:
        result = subprocess.run(
            [onnx2ncnn, str(onnx_path), str(param_path), str(bin_path)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Converted to NCNN:")
            print(f"  Param: {param_path}")
            print(f"  Bin: {bin_path}")
            return param_path, bin_path
        else:
            print(f"onnx2ncnn failed: {result.stderr}")
    except Exception as e:
        print(f"Conversion error: {e}")

    return None, None


def main():
    parser = argparse.ArgumentParser(description='Export GaussianHead to NCNN')
    parser.add_argument('--output_dir', type=str, default='./gaussian_head_ncnn',
                        help='Output directory for NCNN models')
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='Hidden channel dimension')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model = GaussianHeadNCNN(
        in_channels=1024,
        hidden_channels=args.hidden_channels,
    )

    # Print model info
    params = count_parameters(model)
    size_mb = params * 4 / 1024 / 1024
    print(f"GaussianHead model:")
    print(f"  Parameters: {params:,}")
    print(f"  Size: {size_mb:.2f} MB")

    # Test forward pass
    with torch.no_grad():
        x = torch.randn(1, 1024, 96, 96)
        out = model(x)
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")

    # Export to ONNX
    onnx_path = output_dir / "gaussian_head.onnx"
    export_to_onnx(model, onnx_path)

    # Convert to NCNN
    param_path, bin_path = convert_to_ncnn(onnx_path, output_dir)

    if param_path and bin_path:
        print(f"\nNCNN model ready at: {output_dir}")
        print("\nTo deploy to Android, copy:")
        print(f"  {param_path}")
        print(f"  {bin_path}")
        print("to app/src/main/assets/models/")
    else:
        print("\nONNX export complete. Manual NCNN conversion needed.")


if __name__ == "__main__":
    main()
