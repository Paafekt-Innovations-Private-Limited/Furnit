#!/usr/bin/env python3
"""
Export full SHARP model to TFLite (LiteRT) using ai-edge-torch.

DIRECT EXPORT of pre-trained weights — no training, no approximation.
The original SHARP predictor (303M params) is converted directly to TFLite
with optional FP16 quantization for GPU delegate acceleration on Android.

Prerequisites:
    pip install ai-edge-torch torch

Usage:
    python export_sharp_litert.py

Output:
    sharp_ncnn_models/vit_gaussian_fp16.tflite  (~550MB with FP16)
    sharp_ncnn_models/vit_gaussian_fp32.tflite  (~1.1GB fallback)

Push to Android device:
    adb push sharp_ncnn_models/vit_gaussian_fp16.tflite /data/local/tmp/furnit/
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full SHARP to single LiteRT TFLite.")
    parser.add_argument(
        "--sharp-src",
        default=str(Path("/tmp/ml-sharp/src")),
        help="Path to ml-sharp 'src' directory (added to PYTHONPATH). Default: /tmp/ml-sharp/src",
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Path to SHARP PyTorch weights (.pt). Required.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "sharp_litert_models"),
        help="Output directory for .tflite. Default: android/sharp_litert_models",
    )
    return parser.parse_args()


class SharpForExport(nn.Module):
    """Wrapper that runs the full SHARP predictor and packs output as [1, N, 14].

    Preserves ALL original model weights — zero training.

    Output channel layout (per Gaussian, interleaved [N, 14]):
        [0-2]   position xyz
        [3]     opacity (raw logit, apply sigmoid on device)
        [4-6]   scale (singular values, positive)
        [7-10]  rotation quaternion wxyz
        [11-13] color RGB [0-1]
    """

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [1, 3, 1536, 1536] normalized to [0, 1]
        Returns:
            packed: [1, N, 14] Gaussian parameters
                    N = 2 * 768 * 768 = 1,179,648 for 1536x1536 input
        """
        disp_f = torch.ones(1, device=image.device)
        gaussians = self.predictor(image, disp_f)

        # gaussians attributes: mean_vectors, singular_values, quaternions,
        #                        colors, opacities
        positions = gaussians.mean_vectors               # [1, N, 3]
        opacities = gaussians.opacities.unsqueeze(-1)    # [1, N, 1]
        scales = gaussians.singular_values               # [1, N, 3]
        quaternions = gaussians.quaternions               # [1, N, 4]
        colors = gaussians.colors                         # [1, N, 3]

        # Pack into single tensor: [1, N, 14]
        packed = torch.cat(
            [positions, opacities, scales, quaternions, colors], dim=-1
        )
        return packed


def main():
    overall_start = time.time()
    print("=" * 60)
    print("Export SHARP to TFLite — direct, no training")
    print("=" * 60)

    args = parse_args()
    sharp_src = Path(args.sharp_src)
    weights_path = Path(args.weights) if args.weights else None
    output_dir = Path(args.output_dir)

    # ---- Validate prerequisites ----
    if not sharp_src.exists():
        print(f"ERROR: SHARP source not found at {sharp_src}")
        print("  Clone it: git clone <sharp-repo> /tmp/ml-sharp")
        return 1
    if weights_path is None:
        print("ERROR: --weights is required (path to SHARP .pt checkpoint)")
        return 1
    if not weights_path.exists():
        print(f"ERROR: Model weights not found at {weights_path}")
        return 1

    sys.path.insert(0, str(sharp_src))

    try:
        import litert_torch
    except ImportError:
        print("ERROR: litert-torch not installed")
        print("  pip install litert-torch")
        return 1

    # ---- Monkey-patch pow → exp(a*log(x)) for TFLite compatibility ----
    # TFLite doesn't support tfl.pow with float exponents.
    # sRGB2linearRGB uses ((x+0.055)/1.055)**2.4 which triggers this.
    # Replace with mathematically equivalent exp(2.4 * log(x)).
    import sharp.utils.color_space as _cs
    from sharp.utils.robust import robust_where

    def _srgb_to_linear_tflite(srgb_color: torch.Tensor) -> torch.Tensor:
        """TFLite-compatible sRGB→linearRGB (no pow op)."""
        threshold = 0.04045
        def branch_true(x):
            return x / 12.92
        def branch_false(x):
            t = (x + 0.055) / 1.055
            return torch.exp(2.4 * torch.log(t.clamp(min=1e-7)))
        return robust_where(
            srgb_color <= threshold, srgb_color,
            branch_true, branch_false,
            branch_false_safe_value=threshold,
        )

    _cs.sRGB2linearRGB = _srgb_to_linear_tflite
    # Also patch in composer module which imported the name directly
    import sharp.models.composer as _composer
    _composer.sRGB2linearRGB = _srgb_to_linear_tflite
    print("Patched sRGB2linearRGB for TFLite (pow → exp*log)")

    # ---- Load full SHARP predictor (pre-trained, 303M params) ----
    from sharp.models import PredictorParams, create_predictor

    print("\nLoading full SHARP predictor...")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    del state_dict

    param_count = sum(p.numel() for p in predictor.parameters())
    print(f"  Loaded: {param_count / 1e6:.0f}M parameters")

    # ---- Wrap for export ----
    wrapper = SharpForExport(predictor)
    wrapper.eval()

    # Fixed input size — SHARP always uses 1536x1536
    sample_input = torch.rand(1, 3, 1536, 1536)

    # ---- PyTorch FP32 reference inference ----
    print("\nRunning PyTorch FP32 reference inference...")
    with torch.no_grad():
        reference_output = wrapper(sample_input)

    gaussian_count = reference_output.shape[1]
    print(f"  Output shape: {reference_output.shape}")
    print(f"  Gaussians:    {gaussian_count:,}")
    print(f"  Position:     [{reference_output[0, :, :3].min():.3f}, "
          f"{reference_output[0, :, :3].max():.3f}]")
    print(f"  Color:        [{reference_output[0, :, 11:14].min():.3f}, "
          f"{reference_output[0, :, 11:14].max():.3f}]")

    # ---- Convert FP32 model to TFLite with FP16 weight quantization ----
    # FP32 model is ~2.5GB which exceeds TFLite's 2GB flatbuffer limit.
    # Use TFLiteConverter's built-in FP16 quantization to halve weight sizes.
    # The converter stores weights in FP16 but keeps compute in FP32.
    # GPU delegate will use FP16 compute anyway.
    import tensorflow as tf

    print("\nConverting to TFLite with FP16 quantization via litert-torch...")
    conversion_start = time.time()
    edge_model = litert_torch.convert(
        wrapper, (sample_input,),
        _ai_edge_converter_flags={
            "optimizations": [tf.lite.Optimize.DEFAULT],
            "target_spec": {
                "supported_types": [tf.float16],
            },
        },
    )
    conversion_time = time.time() - conversion_start
    print(f"  Conversion completed in {conversion_time:.0f}s")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export FP16
    fp16_path = output_dir / "vit_gaussian_fp16.tflite"
    edge_model.export(str(fp16_path))
    fp16_size_mb = fp16_path.stat().st_size / 1024 / 1024
    print(f"  FP16: {fp16_path.name} ({fp16_size_mb:.0f} MB)")

    # ---- Validate conversion quality ----
    print("\nValidating TFLite conversion...")
    tflite_output = edge_model(sample_input)
    if isinstance(tflite_output, (list, tuple)):
        tflite_output = tflite_output[0]

    max_diff = (reference_output - tflite_output).abs().max().item()
    mean_diff = (reference_output - tflite_output).abs().mean().item()
    color_max_diff = (
        reference_output[0, :, 11:14] - tflite_output[0, :, 11:14]
    ).abs().max().item()

    print(f"  Max diff overall: {max_diff:.6f}")
    print(f"  Mean diff:        {mean_diff:.6f}")
    print(f"  Max color diff:   {color_max_diff:.6f}", end="")
    if color_max_diff < 0.01:
        print("  (OK — quality preserved)")
    else:
        print("  (WARNING: > 0.01 threshold)")

    # ---- Summary ----
    elapsed = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"Export complete in {elapsed:.0f}s")
    print(f"{'=' * 60}")

    print(f"\nModel: {fp16_path.name} ({fp16_size_mb:.0f} MB)")
    print(f"Gaussians per image: {gaussian_count:,}")
    print(f"\nPush to Android:")
    print(f"  adb push {fp16_path} /data/local/tmp/furnit/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
