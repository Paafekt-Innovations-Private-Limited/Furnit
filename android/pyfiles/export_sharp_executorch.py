#!/usr/bin/env python3
"""
Export SHARP model to ExecuTorch .pte format for Android deployment.
"""

import sys
from pathlib import Path

SHARP_SRC = Path("/tmp/ml-sharp/src")
sys.path.insert(0, str(SHARP_SRC))

import torch
import torch.nn as nn

MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/ml_experiments/models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")


class SharpForExport(nn.Module):
    """Wrapper for SHARP model export to ExecuTorch."""

    def __init__(self, predictor, default_disparity_factor: float = 1.0):
        super().__init__()
        self.predictor = predictor
        self.register_buffer('disparity_factor', torch.tensor([default_disparity_factor]))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [1, 3, 1536, 1536] normalized RGB image
        Returns:
            Gaussian parameters [N, 14]: pos(3), scale(3), rot(4), opacity(1), color(3)
        """
        result = self.predictor(image, self.disparity_factor)

        # Gaussians3D has: mean_vectors, singular_values, quaternions, colors, opacities
        means = result.mean_vectors        # [B, N, 3]
        scales = result.singular_values    # [B, N, 3]
        rotations = result.quaternions     # [B, N, 4]
        colors = result.colors             # [B, N, 3]
        opacities = result.opacities       # [B, N] or [B, N, 1]

        # Ensure opacities has 3 dims
        if opacities.dim() == 2:
            opacities = opacities.unsqueeze(-1)

        # Concatenate: [B, N, 14]
        params = torch.cat([means, scales, rotations, opacities, colors], dim=-1)
        return params.squeeze(0)  # [N, 14]


def main():
    from executorch.exir import EdgeCompileConfig, to_edge
    from sharp.models import PredictorParams, create_predictor

    print("=" * 60)
    print("Exporting SHARP to ExecuTorch (.pte)")
    print("=" * 60)

    print(f"\nLoading SHARP model from {MODEL_WEIGHTS}...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    model = SharpForExport(predictor)
    model.eval()

    example_input = torch.randn(1, 3, 1536, 1536)

    print("Running test inference...")
    with torch.no_grad():
        test_output = model(example_input)
        print(f"Output shape: {test_output.shape}")

    print("\nExporting with torch.export...")
    exported_program = torch.export.export(model, (example_input,), strict=False)

    print("Converting to edge format...")
    edge_program = to_edge(exported_program, compile_config=EdgeCompileConfig(_check_ir_validity=False))

    print("Converting to ExecuTorch format...")
    et_program = edge_program.to_executorch()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sharp.pte"

    print(f"Saving to {output_path}...")
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSuccess! Created {output_path}")
    print(f"File size: {file_size:.1f} MB")
    print(f"\nadb push {output_path} /sdcard/Android/data/com.furnit.android/files/models/")


if __name__ == "__main__":
    main()
