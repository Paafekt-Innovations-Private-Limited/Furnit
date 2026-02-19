#!/usr/bin/env python3
"""
Export COMPLETE SHARP model to NCNN.

This exports the full model as a single NCNN file, fixing the
patch processing issue by using batch dimension correctly.
"""

import sys
import os
from pathlib import Path
import subprocess

SHARP_SRC = Path("/tmp/ml-sharp/src")
MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/ml_experiments/models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")
PNNX_PATH = "/opt/miniconda3/bin/pnnx"

sys.path.insert(0, str(SHARP_SRC))

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompleteSharpForNCNN(nn.Module):
    """
    Complete SHARP model restructured for correct NCNN conversion.

    Key change: Use batch dimension for patches instead of channel concat.
    """

    def __init__(self, predictor):
        super().__init__()
        enc = predictor.monodepth_model.monodepth_predictor.encoder

        # Patch Encoder components
        self.patch_embed_proj = enc.patch_encoder.patch_embed.proj
        self.patch_cls_token = enc.patch_encoder.cls_token
        self.patch_pos_embed = enc.patch_encoder.pos_embed
        self.patch_blocks = enc.patch_encoder.blocks
        self.patch_norm = enc.patch_encoder.norm

        # Image Encoder components
        self.image_embed_proj = enc.image_encoder.patch_embed.proj
        self.image_cls_token = enc.image_encoder.cls_token
        self.image_pos_embed = enc.image_encoder.pos_embed
        self.image_blocks = enc.image_encoder.blocks
        self.image_norm = enc.image_encoder.norm

        # Decoder
        self.decoder = predictor.monodepth_model.monodepth_predictor.decoder

        # Fixed parameters
        self.register_buffer('disparity_factor', torch.tensor([1.0]))
        self.patch_size = 384

    def forward(self, image):
        """
        Full SHARP forward pass.

        Args:
            image: [1, 3, 1536, 1536]

        Returns:
            Tuple of 5 tensors: positions, scales, rotations, colors, opacity
        """
        # ===== PATCH ENCODER =====
        # Extract patches using batch dimension (KEY FIX)
        patches = self._extract_patches(image)  # List of [1, 3, 384, 384]
        patch_batch = torch.cat(patches, dim=0)  # [35, 3, 384, 384]

        # Embed patches
        patch_feat = self.patch_embed_proj(patch_batch)  # [35, 1024, 24, 24]
        patch_feat = patch_feat.flatten(2).transpose(1, 2)  # [35, 576, 1024]

        # Add CLS token
        cls = self.patch_cls_token.expand(35, -1, -1)  # [35, 1, 1024]
        patch_feat = torch.cat([cls, patch_feat], dim=1)  # [35, 577, 1024]

        # Add positional embedding
        patch_feat = patch_feat + self.patch_pos_embed

        # Transform through blocks
        for block in self.patch_blocks:
            patch_feat = block(patch_feat)
        patch_feat = self.patch_norm(patch_feat)  # [35, 577, 1024]

        # ===== IMAGE ENCODER =====
        # Process full image
        image_feat = self.image_embed_proj(image)  # [1, 1024, 96, 96]
        image_feat = image_feat.flatten(2).transpose(1, 2)  # [1, 9216, 1024]

        # Add CLS token
        img_cls = self.image_cls_token  # [1, 1, 1024]
        image_feat = torch.cat([img_cls, image_feat], dim=1)  # [1, 9217, 1024]

        # Add positional embedding
        image_feat = image_feat + self.image_pos_embed

        # Transform through blocks
        for block in self.image_blocks:
            image_feat = block(image_feat)
        image_feat = self.image_norm(image_feat)  # [1, 9217, 1024]

        # ===== DECODER =====
        # Reshape patch features for decoder
        # The decoder expects specific tensor shapes based on the original model
        # patch_feat: [35, 577, 1024] → needs to be [1, 35, 577, 1024]
        patch_feat = patch_feat.unsqueeze(0)

        # Run decoder
        # Note: The decoder's exact interface depends on the SHARP implementation
        # This is a simplified version; full implementation may need adjustment
        output = self.decoder(patch_feat, image_feat, self.disparity_factor, depth=None)

        # Extract Gaussian parameters
        positions = output.mean_vectors.flatten(1, -2)  # [1, N, 3]
        scales = output.singular_values.flatten(1, -2)  # [1, N, 3]
        rotations = output.quaternions.flatten(1, -2)  # [1, N, 4]
        colors = output.colors.flatten(1, -2)  # [1, N, 3]
        opacity = output.opacities.flatten(1, -1)  # [1, N]

        return positions, scales, rotations, colors, opacity

    def _extract_patches(self, image):
        """Extract 35 patches at 3 scales."""
        patches = []
        h, w = image.shape[2], image.shape[3]

        # 1.0x: 25 patches (5x5)
        stride = (h - self.patch_size) // 4
        for i in range(5):
            for j in range(5):
                y, x = i * stride, j * stride
                patches.append(image[:, :, y:y+self.patch_size, x:x+self.patch_size])

        # 0.5x: 9 patches (3x3)
        img_half = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False)
        stride_half = (img_half.shape[2] - self.patch_size) // 2
        for i in range(3):
            for j in range(3):
                y, x = i * stride_half, j * stride_half
                patches.append(img_half[:, :, y:y+self.patch_size, x:x+self.patch_size])

        # 0.25x: 1 patch
        img_quarter = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=False)
        patches.append(img_quarter)

        return patches


def load_sharp_model():
    """Load SHARP model."""
    from sharp.models import PredictorParams, create_predictor

    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    return predictor


def main():
    print("="*60)
    print("Complete SHARP Model Export for NCNN")
    print("="*60)

    if not SHARP_SRC.exists():
        print(f"ERROR: ml-sharp not found at {SHARP_SRC}")
        return 1

    if not MODEL_WEIGHTS.exists():
        print(f"ERROR: Model weights not found at {MODEL_WEIGHTS}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading SHARP model...")
    predictor = load_sharp_model()

    # Create export wrapper
    print("Creating export wrapper...")
    model = CompleteSharpForNCNN(predictor)
    model.eval()

    # Test inference
    print("\nTesting inference...")
    dummy = torch.randn(1, 3, 1536, 1536)

    try:
        with torch.no_grad():
            outputs = model(dummy)
            print(f"Positions: {outputs[0].shape}")
            print(f"Scales: {outputs[1].shape}")
            print(f"Rotations: {outputs[2].shape}")
            print(f"Colors: {outputs[3].shape}")
            print(f"Opacity: {outputs[4].shape}")

            num_gaussians = outputs[0].shape[1]
            print(f"\nTotal Gaussians: {num_gaussians:,}")

    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: The decoder interface may need adjustment.")
        print("Proceeding with component-by-component export instead.")
        return 1

    # Trace model
    print("\nTracing model (this may take several minutes)...")
    try:
        traced = torch.jit.trace(model, dummy)
        ts_path = OUTPUT_DIR / "sharp_complete.pt"
        traced.save(str(ts_path))
        print(f"Saved TorchScript to {ts_path} ({ts_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB)")
    except Exception as e:
        print(f"Tracing failed: {e}")
        return 1

    # Convert with PNNX
    print("\nConverting with PNNX (this will take 15-20 minutes)...")
    result = subprocess.run(
        [PNNX_PATH, str(ts_path), "inputshape=[1,3,1536,1536]", "fp16=0"],
        cwd=str(OUTPUT_DIR),
        capture_output=True,
        text=True,
        timeout=1800
    )

    if result.returncode == 0:
        print("PNNX conversion successful!")
        param_path = OUTPUT_DIR / "sharp_complete.ncnn.param"
        bin_path = OUTPUT_DIR / "sharp_complete.ncnn.bin"
        if param_path.exists():
            print(f"  param: {param_path.stat().st_size / 1024:.1f} KB")
        if bin_path.exists():
            print(f"  bin: {bin_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")

        # Verify conversion
        print("\nVerifying conversion...")
        with open(param_path, 'r') as f:
            lines = f.readlines()
            # Find first Convolution and check channels
            for line in lines:
                if 'Convolution' in line and '6=' in line:
                    parts = line.split()
                    for p in parts:
                        if p.startswith('6='):
                            wsize = int(p[2:])
                            in_ch = wsize // (1024 * 16 * 16)
                            print(f"  First conv weight size: {wsize}, inferred input channels: {in_ch}")
                            if in_ch == 3:
                                print("  ✓ Channel handling is CORRECT!")
                            else:
                                print("  ✗ Channel mismatch still present")
                    break
    else:
        print(f"PNNX failed: {result.stderr[:1000]}")
        return 1

    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)
    print(f"""
Files created:
- {OUTPUT_DIR / 'sharp_complete.pt'}
- {OUTPUT_DIR / 'sharp_complete.ncnn.param'}
- {OUTPUT_DIR / 'sharp_complete.ncnn.bin'}

To deploy:
1. Push files to device:
   adb push sharp_complete.ncnn.param /storage/emulated/0/Android/data/com.furnit.android/files/models/sharp.ncnn.param
   adb push sharp_complete.ncnn.bin /storage/emulated/0/Android/data/com.furnit.android/files/models/sharp.ncnn.bin

2. The existing sharp_ncnn.cpp should work with the new model.
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
