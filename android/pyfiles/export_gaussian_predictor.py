#!/usr/bin/env python3
"""
Export the full Gaussian prediction pipeline to NCNN.

This exports the GaussianDensePredictionTransformer which includes:
- Decoder (multi-res conv)
- Image encoder (skip conv backbone)
- Fusion block
- Geometry and texture heads

Input: 5 multi-scale features from SPN encoder
Output: Gaussian parameters (positions, scales, rotations, colors, opacity)
"""

import sys
from pathlib import Path
import subprocess

SHARP_SRC = Path("/tmp/ml-sharp/src")
MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/ml_experiments/models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")
PNNX_PATH = "/opt/miniconda3/bin/pnnx"

sys.path.insert(0, str(SHARP_SRC))

import torch
import torch.nn as nn


class GaussianPredictorForExport(nn.Module):
    """
    Wrapper for the Gaussian prediction pipeline.

    Takes 5 encoder features + input image, outputs Gaussian parameters.
    """

    def __init__(self, feature_model, prediction_head):
        super().__init__()
        self.feature_model = feature_model
        self.prediction_head = prediction_head

    def forward(self, image, f0, f1, f2, f3, f4):
        """
        Args:
            image: [1, 3, H, W] - original input image
            f0-f4: encoder features at different scales

        Returns:
            positions, scales, rotations, colors, opacities
        """
        # Run feature model (decoder + image encoder + fusion + heads)
        encoder_features = [f0, f1, f2, f3, f4]

        # The feature model expects encoder_encodings
        head_features = self.feature_model(image, encoder_features)

        # Run prediction head
        gaussians = self.prediction_head(head_features)

        # Extract Gaussian parameters
        return (
            gaussians.mean_vectors,
            gaussians.singular_values,
            gaussians.quaternions,
            gaussians.colors,
            gaussians.opacities
        )


def analyze_feature_model():
    """Analyze the feature model structure in detail."""
    from sharp.models import PredictorParams, create_predictor

    print("Loading SHARP model...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    feature_model = predictor.feature_model

    print("\n=== GaussianDensePredictionTransformer ===")
    print(f"Type: {type(feature_model)}")

    # Check forward method signature
    import inspect
    sig = inspect.signature(feature_model.forward)
    print(f"\nForward signature: {sig}")

    # Print all children
    print("\nChildren:")
    for name, child in feature_model.named_children():
        print(f"  {name}: {type(child).__name__}")
        if hasattr(child, 'weight'):
            print(f"    weight: {child.weight.shape}")

    # Check geometry_head structure
    print("\n=== geometry_head structure ===")
    for name, module in feature_model.geometry_head.named_modules():
        if hasattr(module, 'weight'):
            print(f"  {name}: {module.__class__.__name__}, weight: {module.weight.shape}")

    print("\n=== texture_head structure ===")
    for name, module in feature_model.texture_head.named_modules():
        if hasattr(module, 'weight'):
            print(f"  {name}: {module.__class__.__name__}, weight: {module.weight.shape}")

    # Check prediction_head
    prediction_head = predictor.prediction_head
    print("\n=== prediction_head structure ===")
    print(f"Type: {type(prediction_head)}")
    for name, module in prediction_head.named_modules():
        if hasattr(module, 'weight'):
            print(f"  {name}: {module.__class__.__name__}, weight: {module.weight.shape}")

    return feature_model, prediction_head


def test_small_forward():
    """Test forward pass with small inputs."""
    from sharp.models import PredictorParams, create_predictor

    print("\nLoading model for test...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    feature_model = predictor.feature_model

    # Create small test inputs
    # Use 1/4 scale: 192 instead of 768
    scale = 4
    h0 = 768 // scale  # 192

    image = torch.randn(1, 3, h0, h0)  # Small image
    f0 = torch.randn(1, 256, h0, h0)        # 192
    f1 = torch.randn(1, 256, h0//2, h0//2)  # 96
    f2 = torch.randn(1, 512, h0//4, h0//4)  # 48
    f3 = torch.randn(1, 1024, h0//8, h0//8) # 24
    f4 = torch.randn(1, 1024, h0//16, h0//16) # 12

    print(f"\nTest inputs (1/{scale} scale):")
    print(f"  image: {image.shape}")
    print(f"  f0: {f0.shape}")
    print(f"  f1: {f1.shape}")
    print(f"  f2: {f2.shape}")
    print(f"  f3: {f3.shape}")
    print(f"  f4: {f4.shape}")

    encoder_features = [f0, f1, f2, f3, f4]

    print("\nRunning forward pass...")
    with torch.no_grad():
        try:
            head_features = feature_model(image, encoder_features)
            print(f"  head_features: geometry_features={head_features.geometry_features.shape}, texture_features={head_features.texture_features.shape}")

            # Run prediction head
            prediction_head = predictor.prediction_head
            gaussians = prediction_head(head_features)
            print(f"  positions: {gaussians.mean_vectors.shape}")
            print(f"  scales: {gaussians.singular_values.shape}")
            print(f"  rotations: {gaussians.quaternions.shape}")
            print(f"  colors: {gaussians.colors.shape}")
            print(f"  opacities: {gaussians.opacities.shape}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    print("=" * 60)
    print("SHARP Gaussian Predictor Analysis")
    print("=" * 60)

    analyze_feature_model()
    test_small_forward()


if __name__ == "__main__":
    main()
