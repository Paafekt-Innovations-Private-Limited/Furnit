#!/usr/bin/env python3
"""
Convert SHARP PyTorch model to CoreML with FP32 for iOS.
Run from ml-sharp directory: python convert_to_coreml_fp32.py
"""

import torch
import coremltools as ct
import sys
import os

# Add ml-sharp/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-sharp", "src"))

from sharp.models import create_predictor, PredictorParams

# Model config
INPUT_SIZE = 1536  # SHARP uses 1536x1536 internally
MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


def main():
    print("=" * 50)
    print("SHARP to CoreML Converter (FP32)")
    print("=" * 50)

    # 1. Create model
    print("\n1. Creating SHARP model...")
    predictor = create_predictor(PredictorParams())

    # 2. Load weights
    print("\n2. Loading weights from Apple CDN...")
    state_dict = torch.hub.load_state_dict_from_url(MODEL_URL, progress=True)
    predictor.load_state_dict(state_dict)
    predictor.eval()
    print("   ✅ Weights loaded")

    # 3. Create wrapper for CoreML
    print("\n3. Creating CoreML-compatible wrapper...")

    class SHARPWrapper(torch.nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor
            # Pre-define disparity_factor as buffer to avoid tracer warnings
            self.register_buffer('disparity_factor', torch.tensor([1.0]))

        def forward(self, image):
            # image: [1, 3, 1536, 1536]
            # CoreML (ImageType) will normalize from 0–255 to 0–1 via scale=1/255
            gaussians = self.predictor(image, self.disparity_factor)
            return (
                gaussians.mean_vectors,      # positions
                gaussians.singular_values,   # scales
                gaussians.quaternions,       # rotations
                gaussians.colors,            # RGB in [0,1]
                gaussians.opacities          # alpha
            )

    wrapper = SHARPWrapper(predictor).eval()

    # 4. Trace the model
    print("\n4. Tracing model (this may take a few minutes)...")
    example_input = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_input)
    traced.eval()
    print("   ✅ Model traced")

    # 5. Convert to CoreML (let coremltools choose precision for BNNS compatibility)
    print("\n5. Converting to CoreML (auto precision)...")
    ml_model = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
                scale=1.0 / 255.0,  # 0–255 → 0–1
                bias=[0.0, 0.0, 0.0],
                color_layout=ct.colorlayout.RGB,
            )
        ],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )
    print("   ✅ Converted to CoreML")

    # 6. Save
    output_path = "/tmp/SHARP_fp32_1536.mlpackage"
    print(f"\n6. Saving to {output_path}...")
    ml_model.save(output_path)
    print(f"   ✅ Saved: {output_path}")

    # Info
    print("\n" + "=" * 50)
    print("Model Info:")
    print(f"  Input: image (RGB, {INPUT_SIZE}x{INPUT_SIZE})")
    print("  Preprocessing: scale=1/255 (built-in, 0–255 → 0–1)")
    print("  Format: mlprogram (iOS17+)")
    print("  Outputs:")
    print("    - mean_vectors: 3D positions")
    print("    - singular_values: scale factors")
    print("    - quaternions: rotation")
    print("    - colors: RGB")
    print("    - opacities: alpha")
    print("=" * 50)
    print("\nCopy SHARP_f32.mlpackage to your Xcode project!")

if __name__ == "__main__":
    main()
