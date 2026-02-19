#!/usr/bin/env python3
"""
Export SHARP model with correct structure for NCNN conversion.

The issue with PNNX:
- Traces the Sliding Pyramid loop as channel concatenation
- Creates conv_106 with 105-channel input but 3-channel weights

Solution:
- Export each component separately
- Use batch dimension for patches instead of channel concatenation
- Create a combined model that processes patches correctly
"""

import sys
import os
from pathlib import Path
import argparse

# Configuration
SHARP_SRC = Path("/tmp/ml-sharp/src")
MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/ml_experiments/models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")

sys.path.insert(0, str(SHARP_SRC))

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedExport(nn.Module):
    """
    Standalone patch embedding for NCNN.
    Takes a single patch [1, 3, 384, 384] and outputs [1, 1024, 24, 24].
    """
    def __init__(self, patch_embed):
        super().__init__()
        self.proj = patch_embed.proj

    def forward(self, patch):
        return self.proj(patch)


class BatchedPatchEmbed(nn.Module):
    """
    Patch embedding that processes 35 patches as a batch.
    Input: [35, 3, 384, 384]
    Output: [35, 1024, 576] (576 = 24*24 flattened spatial)
    """
    def __init__(self, patch_embed):
        super().__init__()
        self.proj = patch_embed.proj

    def forward(self, patches):
        # patches: [35, 3, 384, 384]
        x = self.proj(patches)  # [35, 1024, 24, 24]
        x = x.flatten(2)  # [35, 1024, 576]
        x = x.transpose(1, 2)  # [35, 576, 1024]
        return x


class SlidingPyramidExport(nn.Module):
    """
    Sliding Pyramid that creates patches and processes them as a batch.

    This version uses batch dimension for patches, which NCNN handles correctly.
    """
    def __init__(self, patch_conv):
        super().__init__()
        self.patch_conv = patch_conv
        self.patch_size = 384

    def forward(self, image):
        """
        Args:
            image: [1, 3, 1536, 1536]

        Returns:
            features: [35, 576, 1024] - patches as batch, spatial flattened
        """
        # Scale factors and grids
        # 1.0x scale: 5x5 grid, stride 288 = (1536-384)/(5-1)
        # 0.5x scale: 3x3 grid, stride 192 = (768-384)/(3-1)
        # 0.25x scale: 1 patch at 384x384

        patches = []

        # 1.0x scale: 25 patches from 1536x1536
        stride_1x = (image.shape[2] - self.patch_size) // 4  # 288
        for i in range(5):
            for j in range(5):
                y = i * stride_1x
                x = j * stride_1x
                patch = image[:, :, y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)

        # 0.5x scale: 9 patches from 768x768
        img_05 = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False)
        stride_05 = (img_05.shape[2] - self.patch_size) // 2  # 192
        for i in range(3):
            for j in range(3):
                y = i * stride_05
                x = j * stride_05
                patch = img_05[:, :, y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)

        # 0.25x scale: 1 patch from 384x384
        img_025 = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=False)
        patches.append(img_025)

        # Stack as batch: [35, 3, 384, 384]
        batch = torch.cat(patches, dim=0)

        # Apply patch conv: [35, 1024, 24, 24]
        features = self.patch_conv(batch)

        # Flatten spatial: [35, 1024, 576]
        features = features.flatten(2)

        # Transpose for transformer: [35, 576, 1024]
        features = features.transpose(1, 2)

        return features


def load_sharp_model():
    """Load the full SHARP model."""
    from sharp.models import PredictorParams, create_predictor

    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    return predictor


def export_patch_embed():
    """Export the patch embedding layer separately."""
    print("="*60)
    print("Exporting Patch Embedding Layer")
    print("="*60)

    predictor = load_sharp_model()
    patch_embed = predictor.monodepth_model.monodepth_predictor.encoder.patch_encoder.patch_embed

    # Create export wrapper
    model = PatchEmbedExport(patch_embed)
    model.eval()

    # Test
    dummy = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        output = model(dummy)
        print(f"Input: {dummy.shape} → Output: {output.shape}")

    # Export to ONNX
    onnx_path = OUTPUT_DIR / "sharp_patch_embed.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["patch"],
        output_names=["features"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported to {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Export to TorchScript
    traced = torch.jit.trace(model, dummy)
    ts_path = OUTPUT_DIR / "sharp_patch_embed.pt"
    traced.save(str(ts_path))
    print(f"Exported TorchScript to {ts_path}")

    return onnx_path


def export_batched_sliding_pyramid():
    """Export the sliding pyramid with batched patch processing."""
    print("\n" + "="*60)
    print("Exporting Batched Sliding Pyramid")
    print("="*60)

    predictor = load_sharp_model()
    patch_conv = predictor.monodepth_model.monodepth_predictor.encoder.patch_encoder.patch_embed.proj

    # Create export wrapper
    model = SlidingPyramidExport(patch_conv)
    model.eval()

    # Test
    dummy = torch.randn(1, 3, 1536, 1536)
    with torch.no_grad():
        output = model(dummy)
        print(f"Input: {dummy.shape} → Output: {output.shape}")
        # Expected: [35, 576, 1024]

    # Export to TorchScript
    print("\nTracing model...")
    traced = torch.jit.trace(model, dummy)
    ts_path = OUTPUT_DIR / "sharp_sliding_pyramid.pt"
    traced.save(str(ts_path))
    print(f"Exported TorchScript to {ts_path}")

    # Convert with PNNX
    print("\nConverting with PNNX...")
    import subprocess
    pnnx_path = "/opt/miniconda3/bin/pnnx"
    if os.path.exists(pnnx_path):
        result = subprocess.run(
            [pnnx_path, str(ts_path), "inputshape=[1,3,1536,1536]"],
            cwd=str(OUTPUT_DIR),
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print("PNNX conversion successful!")
            # Check output files
            for f in OUTPUT_DIR.glob("sharp_sliding_pyramid*.ncnn.*"):
                print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")
        else:
            print(f"PNNX error: {result.stderr[:500]}")
    else:
        print(f"PNNX not found at {pnnx_path}")

    return ts_path


def verify_ncnn_conversion():
    """Verify the NCNN param file has correct structure."""
    param_path = OUTPUT_DIR / "sharp_sliding_pyramid.ncnn.param"
    if not param_path.exists():
        print("No NCNN param file found")
        return

    print("\n" + "="*60)
    print("Verifying NCNN Param File")
    print("="*60)

    with open(param_path, 'r') as f:
        lines = f.readlines()

    # Check for Convolution layers and their channel configurations
    for i, line in enumerate(lines):
        if 'Convolution' in line:
            parts = line.split()
            if len(parts) >= 4:
                layer_name = parts[1]
                # Parse weight_data_size (param 6)
                for p in parts:
                    if p.startswith('6='):
                        weight_size = int(p[2:])
                        # For 16x16 kernel, 1024 output: input_channels = weight_size / (1024 * 256)
                        input_ch = weight_size // (1024 * 16 * 16)
                        print(f"  {layer_name}: weight_data_size={weight_size}, inferred_input_channels={input_ch}")


def main():
    print("\n" + "="*60)
    print("SHARP Model Export for NCNN")
    print("="*60)

    # Check dependencies
    if not SHARP_SRC.exists():
        print(f"ERROR: ml-sharp not found at {SHARP_SRC}")
        print("Clone it: git clone <ml-sharp-repo> /tmp/ml-sharp")
        return 1

    if not MODEL_WEIGHTS.exists():
        print(f"ERROR: Model weights not found at {MODEL_WEIGHTS}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Export components
    export_patch_embed()
    export_batched_sliding_pyramid()
    verify_ncnn_conversion()

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("""
Exported model components:
1. sharp_patch_embed.onnx - Single patch embedding (for testing)
2. sharp_sliding_pyramid.pt - Full pyramid with batched processing
3. sharp_sliding_pyramid.ncnn.* - NCNN version (if PNNX succeeded)

Next steps:
1. If NCNN conversion shows correct input channels (3, not 105):
   - Push the new .param and .bin to device
   - Update sharp_ncnn.cpp to use the new model

2. If still showing 105 channels:
   - The model needs to be restructured further
   - Consider splitting into smaller components

3. For full SHARP inference:
   - The Sliding Pyramid output feeds into the Patch Encoder (ViT)
   - Then Image Encoder processes the full image
   - Finally, the Gaussian Decoder produces output
   - Each component should be exported separately
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
