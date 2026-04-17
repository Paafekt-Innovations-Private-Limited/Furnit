#!/usr/bin/env python3
"""
Export SHARP model components to NCNN.

The model structure:
1. SlidingPyramidNetwork (encoder):
   - Creates multi-scale patches
   - Processes through patch_encoder (ViT)
   - Processes through image_encoder (ViT)
   - Fuses and upsamples
   - Output: List of 5 feature maps

2. MultiresConvDecoder:
   - Takes 5 feature maps
   - Output: [1, 256, 768, 768]

3. Gaussian Head (in predictor):
   - Converts decoder output to Gaussians

Export strategy:
- Export the FULL encoder as one NCNN model
- Export decoder + gaussian head as another NCNN model
- Chain them in C++
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


def load_sharp_model():
    """Load the full SHARP model."""
    from sharp.models import PredictorParams, create_predictor

    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    return predictor


class SharpEncoderForNCNN(nn.Module):
    """
    Wrapper for the full SlidingPyramidNetwork encoder.

    The encoder internally uses torch.cat(..., dim=0) for batch stacking,
    which PNNX should handle correctly.

    Input: [1, 3, 1536, 1536]
    Output: 5 feature maps (we'll output them concatenated for NCNN)
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        """
        Forward pass returning stacked features.

        Since NCNN doesn't handle list outputs well, we return a dict-like
        structure by outputting multiple named tensors.
        """
        features = self.encoder(x)

        # features is a list of 5 tensors:
        # [1, 256, 768, 768], [1, 256, 384, 384], [1, 512, 192, 192],
        # [1, 1024, 96, 96], [1, 1024, 48, 48]

        # Return as tuple for PNNX
        return features[0], features[1], features[2], features[3], features[4]


class SharpDecoderForNCNN(nn.Module):
    """
    Wrapper for decoder + gaussian head.

    Input: 5 feature maps from encoder
    Output: 5 gaussian parameter tensors
    """

    def __init__(self, decoder, gaussian_head):
        super().__init__()
        self.decoder = decoder
        self.gaussian_head = gaussian_head

    def forward(self, f0, f1, f2, f3, f4):
        """
        Process encoder features to Gaussian parameters.
        """
        # Reconstruct feature list
        features = [f0, f1, f2, f3, f4]

        # Run decoder
        decoded = self.decoder(features)  # [1, 256, 768, 768]

        # Run gaussian head
        gaussians = self.gaussian_head(decoded)

        # Return tuple of tensors
        return (
            gaussians.mean_vectors,      # positions
            gaussians.singular_values,   # scales
            gaussians.quaternions,       # rotations
            gaussians.colors,            # colors
            gaussians.opacities          # opacity
        )


class FullSharpForNCNN(nn.Module):
    """
    Complete SHARP model as a single NCNN export.

    This wraps the entire predictor, ensuring batch-based patch processing.
    """

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.register_buffer('disparity_factor', torch.tensor([1.0]))

    def forward(self, image):
        """
        Full SHARP forward pass.

        Args:
            image: [1, 3, 1536, 1536]

        Returns:
            Tuple of 5 tensors: positions, scales, rotations, colors, opacity
        """
        gaussians = self.predictor(image, self.disparity_factor, depth=None)

        # Flatten spatial dimensions
        positions = gaussians.mean_vectors.flatten(1, -2)      # [1, N, 3]
        scales = gaussians.singular_values.flatten(1, -2)      # [1, N, 3]
        rotations = gaussians.quaternions.flatten(1, -2)       # [1, N, 4]
        colors = gaussians.colors.flatten(1, -2)               # [1, N, 3]
        opacity = gaussians.opacities.flatten(1, -1)           # [1, N]

        return positions, scales, rotations, colors, opacity


def test_full_model():
    """Test the full model wrapper."""
    print("="*60)
    print("Testing Full SHARP Model")
    print("="*60)

    predictor = load_sharp_model()
    model = FullSharpForNCNN(predictor)
    model.eval()

    dummy = torch.randn(1, 3, 1536, 1536)

    print("Running inference...")
    with torch.no_grad():
        outputs = model(dummy)
        print(f"Positions: {outputs[0].shape}")
        print(f"Scales: {outputs[1].shape}")
        print(f"Rotations: {outputs[2].shape}")
        print(f"Colors: {outputs[3].shape}")
        print(f"Opacity: {outputs[4].shape}")

        n = outputs[0].shape[1]
        print(f"Total Gaussians: {n:,}")

    return model


def export_full_model():
    """Export the full model to NCNN."""
    print("\n" + "="*60)
    print("Exporting Full SHARP Model to NCNN")
    print("="*60)

    predictor = load_sharp_model()
    model = FullSharpForNCNN(predictor)
    model.eval()

    dummy = torch.randn(1, 3, 1536, 1536)

    # Test first
    print("Testing model...")
    with torch.no_grad():
        outputs = model(dummy)
        print(f"Output shapes: {[o.shape for o in outputs]}")

    # Trace
    print("\nTracing model (this takes a few minutes)...")
    traced = torch.jit.trace(model, dummy)

    ts_path = OUTPUT_DIR / "sharp_full.pt"
    traced.save(str(ts_path))
    print(f"Saved TorchScript to {ts_path}")
    print(f"File size: {ts_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")

    # Convert with PNNX
    print("\nConverting with PNNX (this takes 15-20 minutes)...")
    print("Command: pnnx sharp_full.pt inputshape=[1,3,1536,1536]")

    result = subprocess.run(
        [PNNX_PATH, str(ts_path), "inputshape=[1,3,1536,1536]", "fp16=0"],
        cwd=str(OUTPUT_DIR),
        capture_output=True,
        text=True,
        timeout=1800
    )

    if result.returncode == 0:
        print("PNNX conversion successful!")

        param_path = OUTPUT_DIR / "sharp_full.ncnn.param"
        bin_path = OUTPUT_DIR / "sharp_full.ncnn.bin"

        if param_path.exists():
            print(f"  param: {param_path.stat().st_size / 1024:.1f} KB")
        if bin_path.exists():
            print(f"  bin: {bin_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")

        # Verify the conversion
        verify_ncnn_conversion(param_path)

        return param_path, bin_path
    else:
        print(f"PNNX failed!")
        print(f"stdout: {result.stdout[:2000]}")
        print(f"stderr: {result.stderr[:2000]}")
        return None, None


def verify_ncnn_conversion(param_path):
    """Verify the NCNN param file has correct channel configuration."""
    print("\n--- Verifying NCNN Conversion ---")

    if not param_path.exists():
        print("No param file to verify")
        return

    with open(param_path, 'r') as f:
        lines = f.readlines()

    # Check header
    layer_count, blob_count = map(int, lines[1].strip().split())
    print(f"Layers: {layer_count}, Blobs: {blob_count}")

    # Find first few Convolution layers and check their weight sizes
    conv_count = 0
    issues = []

    for line in lines[2:]:
        if 'Convolution' in line and '6=' in line:
            conv_count += 1
            parts = line.split()
            layer_name = parts[1]

            # Parse weight_data_size
            for p in parts:
                if p.startswith('6='):
                    wsize = int(p[2:])
                    # For patch embed (16x16 kernel, 1024 output):
                    # expected: 1024 * 3 * 16 * 16 = 786432
                    if wsize == 786432:
                        in_ch = wsize // (1024 * 16 * 16)
                        if in_ch != 3:
                            issues.append(f"{layer_name}: weight expects {in_ch} input channels (should be 3)")
                        else:
                            print(f"  ✓ {layer_name}: correct 3-channel patch embedding")

            if conv_count >= 5:  # Only check first few
                break

    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No obvious channel mismatch issues found")


def main():
    print("\n" + "="*60)
    print("SHARP Model Export for NCNN")
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

    # Test the model first
    test_full_model()

    # Export
    param_path, bin_path = export_full_model()

    if param_path and bin_path:
        print("\n" + "="*60)
        print("Export Complete!")
        print("="*60)
        print(f"""
Files created:
- {param_path}
- {bin_path}

To deploy to device:
  adb push {param_path} /storage/emulated/0/Android/data/com.furnit.android/files/models/sharp.ncnn.param
  adb push {bin_path} /storage/emulated/0/Android/data/com.furnit.android/files/models/sharp.ncnn.bin

The model should now work with the existing sharp_ncnn.cpp code.
""")
    else:
        print("\n⚠️  Export failed. Check the errors above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
