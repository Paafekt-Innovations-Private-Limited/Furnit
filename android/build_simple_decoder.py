#!/usr/bin/env python3
"""
Build a simplified upsampling decoder for NCNN.

The full SHARP FPN decoder needs 5 multi-scale features which the component
approach doesn't produce correctly. This creates a simpler decoder that:
1. Takes merged encoder features (1024 channels)
2. Projects to 128 channels
3. Upsamples with residual blocks to 768x768
4. Outputs features suitable for the Gaussian heads

Architecture:
Input: [1, 1024, H, W] (merged encoder output, H/W varies by scale)
Output: [1, 128, H*8, W*8] (upsampled decoder features)
"""

import struct
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")


class SimpleUpsampleDecoder(nn.Module):
    """
    Simplified decoder that upsamples encoder features.

    Input: [1, 1024, H, W]
    Output: [1, 128, H*8, W*8]
    """

    def __init__(self, in_channels=1024, out_channels=128):
        super().__init__()

        # Project 1024 -> 256
        self.proj1 = nn.Conv2d(in_channels, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)

        # Upsample 2x: 256 -> 256
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)

        # Project 256 -> 128
        self.proj2 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        # Upsample 2x: 128 -> 128
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)

        # Upsample 2x: 128 -> 128
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)

        # Final conv to refine
        self.final = nn.Conv2d(128, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # Project
        x = self.relu1(self.bn1(self.proj1(x)))

        # Upsample 2x
        x = self.relu2(self.bn2(self.up1(x)))

        # Project
        x = self.relu3(self.bn3(self.proj2(x)))

        # Upsample 2x
        x = self.relu4(self.bn4(self.up2(x)))

        # Upsample 2x (total 8x)
        x = self.relu5(self.bn5(self.up3(x)))

        # Final refinement
        x = self.final(x)

        return x


def create_ncnn_param(input_channels=1024, output_channels=128):
    """Create NCNN param file for simple decoder."""

    lines = [
        "7767517",  # NCNN magic
        "19 20",    # layer_count blob_count

        # Input
        "Input                    in0                      0 1 in0",

        # proj1: Conv 1024->256, 1x1
        f"Convolution              proj1                    1 1 in0 proj1 0=256 1=1 2=1 3=1 4=0 5=1 6={1024*256}",
        "BatchNorm                bn1                      1 1 proj1 bn1 0=256",
        "ReLU                     relu1                    1 1 bn1 relu1",

        # up1: Deconv 256->256, 4x4, stride 2
        f"Deconvolution            up1                      1 1 relu1 up1 0=256 1=4 2=1 3=2 4=1 5=1 6={256*256*16}",
        "BatchNorm                bn2                      1 1 up1 bn2 0=256",
        "ReLU                     relu2                    1 1 bn2 relu2",

        # proj2: Conv 256->128, 1x1
        f"Convolution              proj2                    1 1 relu2 proj2 0=128 1=1 2=1 3=1 4=0 5=1 6={256*128}",
        "BatchNorm                bn3                      1 1 proj2 bn3 0=128",
        "ReLU                     relu3                    1 1 bn3 relu3",

        # up2: Deconv 128->128, 4x4, stride 2
        f"Deconvolution            up2                      1 1 relu3 up2 0=128 1=4 2=1 3=2 4=1 5=1 6={128*128*16}",
        "BatchNorm                bn4                      1 1 up2 bn4 0=128",
        "ReLU                     relu4                    1 1 bn4 relu4",

        # up3: Deconv 128->128, 4x4, stride 2
        f"Deconvolution            up3                      1 1 relu4 up3 0=128 1=4 2=1 3=2 4=1 5=1 6={128*128*16}",
        "BatchNorm                bn5                      1 1 up3 bn5 0=128",
        "ReLU                     relu5                    1 1 bn5 relu5",

        # final: Conv 128->128, 3x3
        f"Convolution              final                    1 1 relu5 out0 0={output_channels} 1=3 2=1 3=1 4=1 5=1 6={128*output_channels*9} 9=1",
    ]

    return "\n".join(lines) + "\n"


def create_ncnn_bin():
    """Create NCNN bin file with random initialized weights."""

    weights = []

    # proj1: Conv 1024->256, 1x1
    weights.append(np.random.randn(256, 1024, 1, 1).astype(np.float32) * 0.02)

    # bn1: BatchNorm 256 (gamma, beta, mean, var)
    weights.append(np.ones(256, dtype=np.float32))  # gamma
    weights.append(np.zeros(256, dtype=np.float32))  # beta
    weights.append(np.zeros(256, dtype=np.float32))  # mean
    weights.append(np.ones(256, dtype=np.float32))  # var

    # up1: Deconv 256->256, 4x4
    weights.append(np.random.randn(256, 256, 4, 4).astype(np.float32) * 0.02)

    # bn2: BatchNorm 256
    weights.append(np.ones(256, dtype=np.float32))
    weights.append(np.zeros(256, dtype=np.float32))
    weights.append(np.zeros(256, dtype=np.float32))
    weights.append(np.ones(256, dtype=np.float32))

    # proj2: Conv 256->128, 1x1
    weights.append(np.random.randn(128, 256, 1, 1).astype(np.float32) * 0.02)

    # bn3: BatchNorm 128
    weights.append(np.ones(128, dtype=np.float32))
    weights.append(np.zeros(128, dtype=np.float32))
    weights.append(np.zeros(128, dtype=np.float32))
    weights.append(np.ones(128, dtype=np.float32))

    # up2: Deconv 128->128, 4x4
    weights.append(np.random.randn(128, 128, 4, 4).astype(np.float32) * 0.02)

    # bn4: BatchNorm 128
    weights.append(np.ones(128, dtype=np.float32))
    weights.append(np.zeros(128, dtype=np.float32))
    weights.append(np.zeros(128, dtype=np.float32))
    weights.append(np.ones(128, dtype=np.float32))

    # up3: Deconv 128->128, 4x4
    weights.append(np.random.randn(128, 128, 4, 4).astype(np.float32) * 0.02)

    # bn5: BatchNorm 128
    weights.append(np.ones(128, dtype=np.float32))
    weights.append(np.zeros(128, dtype=np.float32))
    weights.append(np.zeros(128, dtype=np.float32))
    weights.append(np.ones(128, dtype=np.float32))

    # final: Conv 128->128, 3x3 (with bias)
    weights.append(np.random.randn(128, 128, 3, 3).astype(np.float32) * 0.02)
    weights.append(np.zeros(128, dtype=np.float32))  # bias

    # Concatenate all weights
    all_weights = np.concatenate([w.flatten() for w in weights])

    return all_weights.tobytes()


def extract_real_decoder_weights():
    """
    Try to extract real decoder weights from SHARP checkpoint.
    This requires the full model to be loaded.
    """
    import sys
    SHARP_SRC = Path("/tmp/ml-sharp/src")
    MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/ml_experiments/models/sharp_2572gikvuh.pt")

    if not MODEL_WEIGHTS.exists():
        print(f"Model weights not found at {MODEL_WEIGHTS}")
        return None

    sys.path.insert(0, str(SHARP_SRC))

    try:
        from sharp.models import PredictorParams, create_predictor

        print("Loading SHARP model...")
        state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()

        # Get the decoder
        decoder = predictor.monodepth_model.monodepth_predictor.decoder
        print(f"Decoder type: {type(decoder)}")

        # Print decoder structure
        print("\nDecoder structure:")
        for name, module in decoder.named_modules():
            if hasattr(module, 'weight'):
                print(f"  {name}: {module.__class__.__name__}, weight shape: {module.weight.shape}")

        return decoder

    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("Building Simple Upsampling Decoder for NCNN")
    print("=" * 60)

    # Try to extract real weights
    print("\nAttempting to extract real decoder weights...")
    decoder = extract_real_decoder_weights()

    if decoder is None:
        print("\nUsing random initialization (decoder needs training)")
        use_random = True
    else:
        print("\nReal decoder found, but architecture differs from our simple decoder")
        print("The full decoder is too complex - using simplified architecture with random init")
        use_random = True

    # Create param file
    print("\nCreating simple_decoder.ncnn.param...")
    param = create_ncnn_param()
    param_path = OUTPUT_DIR / "simple_decoder.ncnn.param"
    with open(param_path, 'w') as f:
        f.write(param)
    print(f"  Written to {param_path}")

    # Create bin file
    print("\nCreating simple_decoder.ncnn.bin...")
    bin_data = create_ncnn_bin()
    bin_path = OUTPUT_DIR / "simple_decoder.ncnn.bin"
    with open(bin_path, 'wb') as f:
        f.write(bin_data)
    print(f"  Written to {bin_path} ({len(bin_data)} bytes)")

    print("\n" + "=" * 60)
    print("WARNING: This decoder uses random weights!")
    print("It needs to be trained on SHARP encoder-decoder pairs")
    print("to produce meaningful outputs.")
    print("=" * 60)

    print("\nFiles created:")
    print(f"  - {param_path}")
    print(f"  - {bin_path}")
    print("\nArchitecture:")
    print("  Input: [1, 1024, H, W]")
    print("  Output: [1, 128, H*8, W*8]")
    print("  Total 8x upsampling with channel projection")


if __name__ == "__main__":
    main()
