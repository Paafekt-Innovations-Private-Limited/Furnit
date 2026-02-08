#!/usr/bin/env python3
"""
Build FPN bridge layers for NCNN.

This creates lateral projection layers to reconstruct the 5-level
feature hierarchy from component mode's merged features.

Component mode produces:
  - merged1x: [1024, 96, 96]
  - merged05x: [1024, 48, 48]

FPN decoder expects:
  - f0: [256, 768, 768]
  - f1: [256, 384, 384]
  - f2: [512, 192, 192]
  - f3: [1024, 96, 96]
  - f4: [1024, 48, 48]

Reconstruction:
  f4 = merged05x (already 1024ch, 48x48)
  f3 = merged1x (already 1024ch, 96x96)
  f2 = upsample(f3, 2x) → project 1024→512 → [512, 192, 192]
  f1 = upsample(f2, 2x) → project 512→256 → [256, 384, 384]
  f0 = upsample(f1, 2x) → project 256→256 → [256, 768, 768]
"""

import numpy as np
from pathlib import Path
import sys

SHARP_SRC = Path("/tmp/ml-sharp/src")
MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/ml_experiments/models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")

sys.path.insert(0, str(SHARP_SRC))


def create_lateral_conv_param(name: str, in_ch: int, out_ch: int) -> str:
    """Create NCNN param for a 1x1 lateral projection conv."""
    weight_size = in_ch * out_ch
    lines = [
        "7767517",  # NCNN magic
        "2 2",      # 2 layers, 2 blobs
        "Input                    in0                      0 1 in0",
        f"Convolution              conv                     1 1 in0 out0 0={out_ch} 1=1 2=1 3=1 4=0 5=1 6={weight_size} 9=1",
    ]
    return "\n".join(lines) + "\n"


def create_lateral_conv_bin(in_ch: int, out_ch: int, init_scale: float = 0.02) -> bytes:
    """Create NCNN bin with random weights for lateral conv."""
    # Conv weights [out_ch, in_ch, 1, 1]
    weights = np.random.randn(out_ch, in_ch, 1, 1).astype(np.float32) * init_scale
    # Bias
    bias = np.zeros(out_ch, dtype=np.float32)

    return np.concatenate([weights.flatten(), bias]).tobytes()


def extract_real_decoder_weights():
    """
    Extract the real FPN decoder weights and understand channel projections.
    """
    import torch

    print("Loading SHARP model to extract decoder info...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)

    # Find decoder-related weights
    decoder_weights = {k: v for k, v in state_dict.items() if 'decoder' in k.lower() or 'convs' in k.lower()}

    print("\nDecoder-related weights found:")
    for k, v in sorted(decoder_weights.items()):
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")

    return state_dict


def build_fpn_bridge_models():
    """
    Build the lateral projection models for FPN bridge.

    We need:
    - lateral_f2: 1024 → 512 (for f2 level)
    - lateral_f1: 512 → 256 (for f1 level)
    - lateral_f0: 256 → 256 (for f0 level, identity but with refinement)
    """

    print("\n=== Building FPN Bridge Models ===")

    # Lateral f2: 1024 → 512
    print("\nBuilding lateral_f2 (1024→512)...")
    param = create_lateral_conv_param("lateral_f2", 1024, 512)
    param_path = OUTPUT_DIR / "lateral_f2.ncnn.param"
    with open(param_path, 'w') as f:
        f.write(param)

    bin_data = create_lateral_conv_bin(1024, 512)
    bin_path = OUTPUT_DIR / "lateral_f2.ncnn.bin"
    with open(bin_path, 'wb') as f:
        f.write(bin_data)
    print(f"  Written: {param_path} ({len(bin_data)} bytes)")

    # Lateral f1: 512 → 256
    print("\nBuilding lateral_f1 (512→256)...")
    param = create_lateral_conv_param("lateral_f1", 512, 256)
    param_path = OUTPUT_DIR / "lateral_f1.ncnn.param"
    with open(param_path, 'w') as f:
        f.write(param)

    bin_data = create_lateral_conv_bin(512, 256)
    bin_path = OUTPUT_DIR / "lateral_f1.ncnn.bin"
    with open(bin_path, 'wb') as f:
        f.write(bin_data)
    print(f"  Written: {param_path} ({len(bin_data)} bytes)")

    # Lateral f0: 256 → 256 (refine)
    print("\nBuilding lateral_f0 (256→256)...")
    param = create_lateral_conv_param("lateral_f0", 256, 256)
    param_path = OUTPUT_DIR / "lateral_f0.ncnn.param"
    with open(param_path, 'w') as f:
        f.write(param)

    bin_data = create_lateral_conv_bin(256, 256)
    bin_path = OUTPUT_DIR / "lateral_f0.ncnn.bin"
    with open(bin_path, 'wb') as f:
        f.write(bin_data)
    print(f"  Written: {param_path} ({len(bin_data)} bytes)")

    print("\n=== FPN Bridge Models Complete ===")
    print("\nReconstruction pipeline:")
    print("  f4 = merged05x                    [1024, 48, 48]")
    print("  f3 = merged1x                     [1024, 96, 96]")
    print("  f2 = lateral_f2(upsample(f3, 2x)) [512, 192, 192]")
    print("  f1 = lateral_f1(upsample(f2, 2x)) [256, 384, 384]")
    print("  f0 = lateral_f0(upsample(f1, 2x)) [256, 768, 768]")
    print("\nThen feed [f0, f1, f2, f3, f4] to fpn_decoder.ncnn")


def main():
    print("=" * 60)
    print("Building FPN Bridge Layers")
    print("=" * 60)

    # Check if we have real weights to extract
    try:
        state_dict = extract_real_decoder_weights()
    except Exception as e:
        print(f"Could not load model: {e}")
        state_dict = None

    # Build bridge models (with random init for now)
    build_fpn_bridge_models()

    print("\n" + "=" * 60)
    print("WARNING: Lateral projections use random weights!")
    print("For best results, extract weights from SHARP decoder")
    print("or train these projection layers.")
    print("=" * 60)


if __name__ == "__main__":
    main()
