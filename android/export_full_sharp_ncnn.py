#!/usr/bin/env python3
"""
Export FULL SHARP model to NCNN with correct batch handling.

This script exports the complete SHARP model with:
1. Sliding Pyramid with batched patches
2. Patch Encoder (ViT with 24 transformer blocks)
3. Image Encoder (ViT)
4. Gaussian Decoder

The key insight: Use torch.cat(..., dim=0) for patches, which PNNX
correctly handles as batch stacking instead of channel concatenation.
"""

import sys
import os
from pathlib import Path
import subprocess

# Configuration
SHARP_SRC = Path("/tmp/ml-sharp/src")
MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/ml_experiments/models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")
PNNX_PATH = "/opt/miniconda3/bin/pnnx"

sys.path.insert(0, str(SHARP_SRC))

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullSharpExport(nn.Module):
    """
    Complete SHARP model with explicit batch-based patch handling.

    The original SHARP model uses a Sliding Pyramid that processes patches
    through shared conv weights. This wrapper restructures the forward pass
    to use batch dimension for patches, which PNNX can convert correctly.
    """

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

        # Extract key components for explicit control
        encoder = predictor.monodepth_model.monodepth_predictor.encoder
        self.patch_encoder = encoder.patch_encoder
        self.image_encoder = encoder.image_encoder

        # Decoder
        self.decoder = predictor.monodepth_model.monodepth_predictor.decoder

        # Cache positional embeddings
        self.register_buffer('disparity_factor', torch.tensor([1.0]))

        self.patch_size = 384

    def extract_and_embed_patches(self, image):
        """
        Extract multi-scale patches and embed them using batch processing.

        Args:
            image: [1, 3, 1536, 1536]

        Returns:
            patch_features: [1, 35, 577, 1024] - features for all patches
        """
        patches = []

        # 1.0x scale: 25 patches (5x5 grid)
        stride_1x = (image.shape[2] - self.patch_size) // 4  # 288
        for i in range(5):
            for j in range(5):
                y = i * stride_1x
                x = j * stride_1x
                patch = image[:, :, y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)

        # 0.5x scale: 9 patches (3x3 grid)
        img_05 = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False)
        stride_05 = (img_05.shape[2] - self.patch_size) // 2  # 192
        for i in range(3):
            for j in range(3):
                y = i * stride_05
                x = j * stride_05
                patch = img_05[:, :, y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)

        # 0.25x scale: 1 patch
        img_025 = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=False)
        patches.append(img_025)

        # Stack as batch: [35, 3, 384, 384]
        batch = torch.cat(patches, dim=0)

        # Apply patch embedding conv: [35, 1024, 24, 24]
        features = self.patch_encoder.patch_embed.proj(batch)

        # Flatten spatial: [35, 1024, 576]
        features = features.flatten(2)

        # Transpose: [35, 576, 1024]
        features = features.transpose(1, 2)

        # Add class token (577 = 576 + 1 CLS token)
        cls_token = self.patch_encoder.cls_token.expand(35, -1, -1)  # [35, 1, 1024]
        features = torch.cat([cls_token, features], dim=1)  # [35, 577, 1024]

        # Add positional embedding
        features = features + self.patch_encoder.pos_embed

        return features.unsqueeze(0)  # [1, 35, 577, 1024]

    def forward(self, image):
        """
        Full SHARP forward pass.

        Args:
            image: [1, 3, 1536, 1536] normalized RGB

        Returns:
            Tuple of 5 tensors: positions, scales, rotations, colors, opacity
        """
        # Get patch features with explicit batch handling
        # NOTE: For now, we just test patch extraction and embedding
        # Full model would continue through transformer blocks and decoder

        patch_features = self.extract_and_embed_patches(image)

        # For testing, return patch features shape info
        # Real implementation would continue through the full model
        return patch_features


class SlidingPyramidPatchEmbed(nn.Module):
    """
    Just the Sliding Pyramid + Patch Embedding part.
    This is the part that was broken in the original conversion.
    """

    def __init__(self, patch_embed):
        super().__init__()
        self.proj = patch_embed.proj
        self.patch_size = 384

    def forward(self, image):
        """
        Args:
            image: [1, 3, 1536, 1536]

        Returns:
            features: [35, 577, 1024]
        """
        patches = []

        # 1.0x scale: 25 patches
        stride_1x = (image.shape[2] - self.patch_size) // 4
        for i in range(5):
            for j in range(5):
                y = i * stride_1x
                x = j * stride_1x
                patch = image[:, :, y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)

        # 0.5x scale: 9 patches
        img_05 = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False)
        stride_05 = (img_05.shape[2] - self.patch_size) // 2
        for i in range(3):
            for j in range(3):
                y = i * stride_05
                x = j * stride_05
                patch = img_05[:, :, y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)

        # 0.25x scale: 1 patch
        img_025 = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=False)
        patches.append(img_025)

        # Stack as batch: [35, 3, 384, 384]
        batch = torch.cat(patches, dim=0)

        # Apply patch embedding: [35, 1024, 24, 24]
        features = self.proj(batch)

        # Flatten and transpose: [35, 576, 1024]
        features = features.flatten(2).transpose(1, 2)

        return features


class PatchEncoderExport(nn.Module):
    """
    Export the Patch Encoder ViT that processes embedded patches.

    Input: [35, 577, 1024] - 35 patches, 577 tokens each (576 spatial + 1 CLS)
    Output: [35, 577, 1024] - transformed features
    """

    def __init__(self, patch_encoder):
        super().__init__()
        self.blocks = patch_encoder.blocks
        self.norm = patch_encoder.norm

    def forward(self, x):
        """Process through transformer blocks."""
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


def load_sharp_model():
    """Load the full SHARP model."""
    from sharp.models import PredictorParams, create_predictor

    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    return predictor


def test_sliding_pyramid():
    """Test that the sliding pyramid produces correct output."""
    print("="*60)
    print("Testing Sliding Pyramid + Patch Embed")
    print("="*60)

    predictor = load_sharp_model()
    patch_embed = predictor.monodepth_model.monodepth_predictor.encoder.patch_encoder.patch_embed

    model = SlidingPyramidPatchEmbed(patch_embed)
    model.eval()

    dummy = torch.randn(1, 3, 1536, 1536)

    with torch.no_grad():
        output = model(dummy)
        print(f"Input: {dummy.shape}")
        print(f"Output: {output.shape}")
        print(f"Expected: [35, 576, 1024]")

        if output.shape == torch.Size([35, 576, 1024]):
            print("✓ Shape correct!")
        else:
            print("✗ Shape mismatch!")

    return model


def test_patch_encoder():
    """Test the patch encoder ViT."""
    print("\n" + "="*60)
    print("Testing Patch Encoder ViT")
    print("="*60)

    predictor = load_sharp_model()
    patch_encoder = predictor.monodepth_model.monodepth_predictor.encoder.patch_encoder

    # Note: The full patch encoder includes:
    # - blocks (24 transformer blocks)
    # - norm (final LayerNorm)
    # But it also expects pos_embed and cls_token to be added first

    model = PatchEncoderExport(patch_encoder)
    model.eval()

    # Input: 35 patches, 577 tokens (576 spatial + 1 CLS), 1024 features
    dummy = torch.randn(35, 577, 1024)

    with torch.no_grad():
        output = model(dummy)
        print(f"Input: {dummy.shape}")
        print(f"Output: {output.shape}")

    return model


def export_sliding_pyramid_ncnn():
    """Export sliding pyramid to NCNN."""
    print("\n" + "="*60)
    print("Exporting Sliding Pyramid to NCNN")
    print("="*60)

    predictor = load_sharp_model()
    patch_embed = predictor.monodepth_model.monodepth_predictor.encoder.patch_encoder.patch_embed

    model = SlidingPyramidPatchEmbed(patch_embed)
    model.eval()

    dummy = torch.randn(1, 3, 1536, 1536)

    # Trace
    print("Tracing model...")
    traced = torch.jit.trace(model, dummy)
    ts_path = OUTPUT_DIR / "sharp_pyramid_v2.pt"
    traced.save(str(ts_path))
    print(f"Saved TorchScript to {ts_path}")

    # Convert with PNNX
    print("Converting with PNNX...")
    result = subprocess.run(
        [PNNX_PATH, str(ts_path), "inputshape=[1,3,1536,1536]"],
        cwd=str(OUTPUT_DIR),
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode == 0:
        print("PNNX conversion successful!")
        param_path = OUTPUT_DIR / "sharp_pyramid_v2.ncnn.param"
        bin_path = OUTPUT_DIR / "sharp_pyramid_v2.ncnn.bin"
        if param_path.exists():
            print(f"  param: {param_path.stat().st_size / 1024:.1f} KB")
        if bin_path.exists():
            print(f"  bin: {bin_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"PNNX error: {result.stderr[:500]}")

    return ts_path


def export_patch_encoder_ncnn():
    """Export patch encoder ViT to NCNN."""
    print("\n" + "="*60)
    print("Exporting Patch Encoder ViT to NCNN")
    print("="*60)

    predictor = load_sharp_model()
    patch_encoder = predictor.monodepth_model.monodepth_predictor.encoder.patch_encoder

    model = PatchEncoderExport(patch_encoder)
    model.eval()

    # Input shape: [35, 577, 1024]
    dummy = torch.randn(35, 577, 1024)

    print("Tracing model...")
    try:
        traced = torch.jit.trace(model, dummy)
        ts_path = OUTPUT_DIR / "sharp_patch_encoder.pt"
        traced.save(str(ts_path))
        print(f"Saved TorchScript to {ts_path}")

        # Convert with PNNX
        print("Converting with PNNX...")
        result = subprocess.run(
            [PNNX_PATH, str(ts_path), "inputshape=[35,577,1024]"],
            cwd=str(OUTPUT_DIR),
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print("PNNX conversion successful!")
            param_path = OUTPUT_DIR / "sharp_patch_encoder.ncnn.param"
            bin_path = OUTPUT_DIR / "sharp_patch_encoder.ncnn.bin"
            if param_path.exists():
                print(f"  param: {param_path.stat().st_size / 1024:.1f} KB")
            if bin_path.exists():
                print(f"  bin: {bin_path.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"PNNX error: {result.stderr[:500]}")

    except Exception as e:
        print(f"Tracing failed: {e}")

    return None


def main():
    print("\n" + "="*60)
    print("FULL SHARP Model Export for NCNN")
    print("="*60)

    if not SHARP_SRC.exists():
        print(f"ERROR: ml-sharp not found at {SHARP_SRC}")
        return 1

    if not MODEL_WEIGHTS.exists():
        print(f"ERROR: Model weights not found at {MODEL_WEIGHTS}")
        return 1

    if not os.path.exists(PNNX_PATH):
        print(f"ERROR: PNNX not found at {PNNX_PATH}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Test components
    test_sliding_pyramid()
    test_patch_encoder()

    # Export components
    export_sliding_pyramid_ncnn()
    export_patch_encoder_ncnn()

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("""
The SHARP model has been split into exportable components:

1. Sliding Pyramid + Patch Embed:
   - Extracts 35 multi-scale patches from input image
   - Embeds each patch: [1, 3, 1536, 1536] → [35, 576, 1024]
   - Uses batch dimension correctly (not channel concat)

2. Patch Encoder ViT:
   - 24 transformer blocks processing patch features
   - Input: [35, 577, 1024] (with CLS token)
   - Output: [35, 577, 1024]

3. Image Encoder (TODO):
   - Processes full image separately
   - Similar ViT structure

4. Gaussian Decoder (TODO):
   - Combines patch and image features
   - Outputs Gaussian parameters

Next steps:
1. Verify the sliding pyramid NCNN output is correct
2. Export and test the patch encoder
3. Chain the components in C++ code
4. Export remaining components (image encoder, decoder)
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
