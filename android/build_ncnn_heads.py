#!/usr/bin/env python3
"""
Build proper NCNN gaussian prediction models from weight files.

Architecture (from SHARP checkpoint analysis):
- Decoder output: 128 channels
- geometry_head: 128 → 32 channels (via 2 ResBlocks + final conv)
- texture_head: 128 → 32 channels (same structure)
- geometry_prediction_head: 32 → 6 channels (3 pos × 2 layers)
- texture_prediction_head: 32 → 22 channels (11 params × 2 layers)

ResidualBlock structure:
  Sequential:
    [0] GroupNorm(128, num_groups=8)
    [1] ReLU
    [2] Conv2d(128, 64, 3x3)
    [3] GroupNorm(64, num_groups=8)
    [4] ReLU
    [5] Conv2d(64, 128, 3x3)
  + skip connection (identity since in_dim == out_dim)

Combined model for C++:
  geometry_model: 128 → 6 (geometry_head + geometry_prediction_head)
  texture_model: 128 → 22 (texture_head + texture_prediction_head)
"""

import struct
import numpy as np
from pathlib import Path

WEIGHTS_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")
OUTPUT_DIR = WEIGHTS_DIR


def read_weight_file(path: Path) -> np.ndarray:
    """Read a raw weight file as float32 numpy array."""
    with open(path, 'rb') as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.float32)


def create_combined_head_param(head_name: str, final_out_ch: int) -> str:
    """
    Create NCNN param for combined head + prediction_head.

    Input: 128 channels
    Output: final_out_ch channels (6 for geometry, 22 for texture)
    """

    # Weight sizes
    # ResBlock convs: 128→64 and 64→128, kernel 3x3
    conv_128_64_size = 128 * 64 * 9  # 73728
    conv_64_128_size = 64 * 128 * 9  # 73728

    # Final conv in head: 128→32, kernel 1x1
    conv_128_32_size = 128 * 32 * 1  # 4096

    # Prediction head: 32→final_out_ch, kernel 1x1
    conv_32_out_size = 32 * final_out_ch * 1

    lines = [
        "7767517",  # NCNN magic
        "21 23",    # layer_count blob_count (21 layers, 23 blobs)

        # Input: 128 channels
        "Input                    in0                      0 1 in0",

        # Split for first residual skip connection
        "Split                    split0                   1 2 in0 s0_skip s0_main",

        # === ResBlock 0 ===
        # GroupNorm(128, 8 groups) + weights
        "GroupNorm                gn0_0                    1 1 s0_main gn0_0 0=8 1=128 2=1e-6 3=1",
        "ReLU                     relu0_0                  1 1 gn0_0 relu0_0",
        # Conv 128→64, 3x3
        f"Convolution              conv0_0                  1 1 relu0_0 conv0_0 0=64 1=3 2=1 3=1 4=1 5=1 6={conv_128_64_size}",
        # GroupNorm(64, 8 groups)
        "GroupNorm                gn0_1                    1 1 conv0_0 gn0_1 0=8 1=64 2=1e-6 3=1",
        "ReLU                     relu0_1                  1 1 gn0_1 relu0_1",
        # Conv 64→128, 3x3
        f"Convolution              conv0_1                  1 1 relu0_1 conv0_1 0=128 1=3 2=1 3=1 4=1 5=1 6={conv_64_128_size}",
        # Skip connection
        "BinaryOp                 add0                     2 1 s0_skip conv0_1 res0 0=0",

        # Split for second residual
        "Split                    split1                   1 2 res0 s1_skip s1_main",

        # === ResBlock 1 ===
        "GroupNorm                gn1_0                    1 1 s1_main gn1_0 0=8 1=128 2=1e-6 3=1",
        "ReLU                     relu1_0                  1 1 gn1_0 relu1_0",
        f"Convolution              conv1_0                  1 1 relu1_0 conv1_0 0=64 1=3 2=1 3=1 4=1 5=1 6={conv_128_64_size}",
        "GroupNorm                gn1_1                    1 1 conv1_0 gn1_1 0=8 1=64 2=1e-6 3=1",
        "ReLU                     relu1_1                  1 1 gn1_1 relu1_1",
        f"Convolution              conv1_1                  1 1 relu1_1 conv1_1 0=128 1=3 2=1 3=1 4=1 5=1 6={conv_64_128_size}",
        "BinaryOp                 add1                     2 1 s1_skip conv1_1 res1 0=0",

        # Head final layers
        "ReLU                     relu_h0                  1 1 res1 relu_h0",
        f"Convolution              conv_head                1 1 relu_h0 head_out 0=32 1=1 2=1 3=1 4=0 5=1 6={conv_128_32_size}",
        "ReLU                     relu_h1                  1 1 head_out relu_h1",

        # Prediction head: 32 → final_out_ch
        f"Convolution              conv_pred                1 1 relu_h1 out0 0={final_out_ch} 1=1 2=1 3=1 4=0 5=1 6={conv_32_out_size}",
    ]

    return "\n".join(lines) + "\n"


def build_combined_head_bin(head_name: str, pred_name: str, output_path: Path):
    """
    Build NCNN bin for combined head + prediction_head.

    Layer order must match param file:
    1. gn0_0: GroupNorm weights for 128ch
    2. conv0_0: Conv 128→64
    3. gn0_1: GroupNorm weights for 64ch
    4. conv0_1: Conv 64→128
    5. gn1_0: GroupNorm weights for 128ch
    6. conv1_0: Conv 128→64
    7. gn1_1: GroupNorm weights for 64ch
    8. conv1_1: Conv 64→128
    9. conv_head: Conv 128→32
    10. conv_pred: Conv 32→out
    """

    all_weights = []

    # Helper to load weight and bias
    def load_w(name):
        path = WEIGHTS_DIR / f"{name}.weight"
        if not path.exists():
            print(f"  ERROR: Missing {path}")
            return None
        return read_weight_file(path)

    def load_b(name):
        path = WEIGHTS_DIR / f"{name}.bias"
        if not path.exists():
            print(f"  Warning: Missing bias {path}, using zeros")
            return None
        return read_weight_file(path)

    # ResBlock 0
    # gn0_0: GroupNorm 128
    gn0_0_w = load_w(f"{head_name}.0.residual.0")
    gn0_0_b = load_b(f"{head_name}.0.residual.0")
    if gn0_0_w is None:
        return False
    all_weights.append(np.concatenate([gn0_0_w, gn0_0_b]))
    print(f"  gn0_0: {len(gn0_0_w)} + {len(gn0_0_b)} = {len(gn0_0_w) + len(gn0_0_b)}")

    # conv0_0: Conv 128→64
    conv0_0_w = load_w(f"{head_name}.0.residual.2")
    conv0_0_b = load_b(f"{head_name}.0.residual.2")
    if conv0_0_w is None:
        return False
    all_weights.append(conv0_0_w)
    all_weights.append(conv0_0_b)
    print(f"  conv0_0: {len(conv0_0_w)} weights, {len(conv0_0_b)} bias")

    # gn0_1: GroupNorm 64
    gn0_1_w = load_w(f"{head_name}.0.residual.3")
    gn0_1_b = load_b(f"{head_name}.0.residual.3")
    if gn0_1_w is None:
        return False
    all_weights.append(np.concatenate([gn0_1_w, gn0_1_b]))
    print(f"  gn0_1: {len(gn0_1_w)} + {len(gn0_1_b)}")

    # conv0_1: Conv 64→128
    conv0_1_w = load_w(f"{head_name}.0.residual.5")
    conv0_1_b = load_b(f"{head_name}.0.residual.5")
    if conv0_1_w is None:
        return False
    all_weights.append(conv0_1_w)
    all_weights.append(conv0_1_b)
    print(f"  conv0_1: {len(conv0_1_w)} weights, {len(conv0_1_b)} bias")

    # ResBlock 1
    # gn1_0: GroupNorm 128
    gn1_0_w = load_w(f"{head_name}.1.residual.0")
    gn1_0_b = load_b(f"{head_name}.1.residual.0")
    if gn1_0_w is None:
        return False
    all_weights.append(np.concatenate([gn1_0_w, gn1_0_b]))
    print(f"  gn1_0: {len(gn1_0_w)} + {len(gn1_0_b)}")

    # conv1_0: Conv 128→64
    conv1_0_w = load_w(f"{head_name}.1.residual.2")
    conv1_0_b = load_b(f"{head_name}.1.residual.2")
    if conv1_0_w is None:
        return False
    all_weights.append(conv1_0_w)
    all_weights.append(conv1_0_b)
    print(f"  conv1_0: {len(conv1_0_w)} weights, {len(conv1_0_b)} bias")

    # gn1_1: GroupNorm 64
    gn1_1_w = load_w(f"{head_name}.1.residual.3")
    gn1_1_b = load_b(f"{head_name}.1.residual.3")
    if gn1_1_w is None:
        return False
    all_weights.append(np.concatenate([gn1_1_w, gn1_1_b]))
    print(f"  gn1_1: {len(gn1_1_w)} + {len(gn1_1_b)}")

    # conv1_1: Conv 64→128
    conv1_1_w = load_w(f"{head_name}.1.residual.5")
    conv1_1_b = load_b(f"{head_name}.1.residual.5")
    if conv1_1_w is None:
        return False
    all_weights.append(conv1_1_w)
    all_weights.append(conv1_1_b)
    print(f"  conv1_1: {len(conv1_1_w)} weights, {len(conv1_1_b)} bias")

    # Head final conv: 128→32
    conv_head_w = load_w(f"{head_name}.3")
    conv_head_b = load_b(f"{head_name}.3")
    if conv_head_w is None:
        return False
    all_weights.append(conv_head_w)
    all_weights.append(conv_head_b)
    print(f"  conv_head: {len(conv_head_w)} weights, {len(conv_head_b)} bias")

    # Prediction head: 32→out
    pred_w = load_w(pred_name)
    pred_b = load_b(pred_name)
    if pred_w is None:
        return False
    all_weights.append(pred_w)
    all_weights.append(pred_b)
    print(f"  conv_pred: {len(pred_w)} weights, {len(pred_b)} bias")

    # Write bin file
    print(f"\nWriting {output_path}...")
    with open(output_path, 'wb') as f:
        for w in all_weights:
            f.write(w.astype(np.float32).tobytes())

    total_size = output_path.stat().st_size
    print(f"  Total: {total_size} bytes ({total_size/1024:.1f} KB)")

    return True


def main():
    print("=" * 60)
    print("Building Combined NCNN Gaussian Prediction Models")
    print("=" * 60)
    print("\nArchitecture:")
    print("  Input: 128 channels (from decoder)")
    print("  geometry_model: 128 → 6 (3 pos × 2 layers)")
    print("  texture_model: 128 → 22 (11 params × 2 layers)")

    # Build geometry model: 128 → 6
    print("\n--- Building geometry_model ---")
    param = create_combined_head_param("geometry", 6)
    param_path = OUTPUT_DIR / "geometry_model.ncnn.param"
    bin_path = OUTPUT_DIR / "geometry_model.ncnn.bin"

    with open(param_path, 'w') as f:
        f.write(param)
    print(f"Wrote {param_path}")

    if build_combined_head_bin("geometry_head", "geometry_prediction_head", bin_path):
        print("geometry_model built successfully!")
    else:
        print("ERROR building geometry_model")
        return

    # Build texture model: 128 → 22
    print("\n--- Building texture_model ---")
    param = create_combined_head_param("texture", 22)
    param_path = OUTPUT_DIR / "texture_model.ncnn.param"
    bin_path = OUTPUT_DIR / "texture_model.ncnn.bin"

    with open(param_path, 'w') as f:
        f.write(param)
    print(f"Wrote {param_path}")

    if build_combined_head_bin("texture_head", "texture_prediction_head", bin_path):
        print("texture_model built successfully!")
    else:
        print("ERROR building texture_model")
        return

    print("\n" + "=" * 60)
    print("Done! Created:")
    print("  - geometry_model.ncnn.param/bin (128 → 6 channels)")
    print("    Output: [6, H, W] = 2 layers × (pos_x, pos_y, pos_z)")
    print("  - texture_model.ncnn.param/bin (128 → 22 channels)")
    print("    Output: [22, H, W] = 2 layers × (scale×3, rot×4, color×3, opacity)")
    print("\nTotal output: 28 channels per pixel (14 params × 2 layers)")
    print("=" * 60)


if __name__ == "__main__":
    main()
