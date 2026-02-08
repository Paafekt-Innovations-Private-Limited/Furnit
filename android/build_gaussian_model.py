#!/usr/bin/env python3
"""
Build a combined NCNN model for Gaussian prediction.

Pipeline:
  Input (1024 channels) → Projection (1024→128) →
    → geometry_model → 6 channels (position deltas)
    → texture_model → 22 channels (texture params)
  → Combined output: 28 channels

For C++, we'll create three separate models:
1. encoder_projection.ncnn - projects 1024→128
2. geometry_model.ncnn - predicts position (128→6)
3. texture_model.ncnn - predicts texture (128→22)
"""

import numpy as np
from pathlib import Path

WEIGHTS_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")
OUTPUT_DIR = WEIGHTS_DIR


def read_weight_file(path: Path) -> np.ndarray:
    with open(path, 'rb') as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.float32)


def create_projection_param():
    """Create NCNN param for encoder projection (1024→128, 3x3 conv)."""
    # Weight size: 128 * 1024 * 3 * 3 = 1179648
    weight_size = 128 * 1024 * 9

    lines = [
        "7767517",
        "2 2",
        "Input                    in0                      0 1 in0",
        f"Convolution              conv_proj                1 1 in0 out0 0=128 1=3 2=1 3=1 4=1 5=1 6={weight_size}",
    ]
    return "\n".join(lines) + "\n"


def build_projection_bin():
    """Build bin file for projection model."""
    print("Building encoder_projection...")

    w = read_weight_file(WEIGHTS_DIR / "encoder_projection.weight")
    b = read_weight_file(WEIGHTS_DIR / "encoder_projection.bias")

    print(f"  Weight: {len(w)} floats")
    print(f"  Bias: {len(b)} floats")

    bin_path = WEIGHTS_DIR / "encoder_projection.ncnn.bin"
    with open(bin_path, 'wb') as f:
        f.write(w.astype(np.float32).tobytes())
        f.write(b.astype(np.float32).tobytes())

    print(f"  Written to {bin_path} ({bin_path.stat().st_size} bytes)")
    return True


def main():
    print("=" * 60)
    print("Building Encoder Projection Model")
    print("=" * 60)

    # Write projection param
    param = create_projection_param()
    param_path = WEIGHTS_DIR / "encoder_projection.ncnn.param"
    with open(param_path, 'w') as f:
        f.write(param)
    print(f"Wrote {param_path}")

    # Build projection bin
    build_projection_bin()

    print("\n" + "=" * 60)
    print("Done! Model files:")
    print("  - encoder_projection.ncnn (1024→128)")
    print("  - geometry_model.ncnn (128→6)")
    print("  - texture_model.ncnn (128→22)")
    print("\nTotal pipeline: 1024 → 128 → [6 + 22] = 28 channels")
    print("=" * 60)


if __name__ == "__main__":
    main()
