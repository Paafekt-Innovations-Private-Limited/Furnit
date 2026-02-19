
"""
Export ALL SHARP components for NCNN.

Components:
1. Pyramid + Patch Embed: [1, 3, 1536, 1536] → [35, 576, 1024] (DONE)
2. Patch Encoder ViT: [35, 577, 1024] → [35, 577, 1024] (DONE)
3. Image Encoder ViT: [1, 3, 384, 384] → [1, 577, 1024]
4. Feature Merger: combines patch/image features → 5 feature maps
5. Decoder: 5 feature maps → Gaussian parameters

C++ will chain these together, handling CLS token and pos_embed in between.
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


class ImageEncoderExport(nn.Module):
    """
    Export the Image Encoder ViT that processes the low-res image.

    Input: [1, 3, 384, 384] - lowest resolution from pyramid
    Output: [1, 577, 1024] - encoded features (576 spatial + 1 CLS)
    """

    def __init__(self, image_encoder):
        super().__init__()
        self.patch_embed = image_encoder.patch_embed
        self.cls_token = image_encoder.cls_token
        self.pos_embed = image_encoder.pos_embed
        self.blocks = image_encoder.blocks
        self.norm = image_encoder.norm

    def forward(self, x):
        """
        Process 384x384 image through image encoder ViT.
        """
        # Patch embed: [1, 3, 384, 384] → [1, 1024, 24, 24]
        x = self.patch_embed.proj(x)
        # Flatten: [1, 1024, 576]
        x = x.flatten(2)
        # Transpose: [1, 576, 1024]
        x = x.transpose(1, 2)

        # Add CLS token: [1, 577, 1024]
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transform through blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        return x


class GaussianDecoderExport(nn.Module):
    """
    Export the Gaussian Decoder.

    Takes features from encoder and produces Gaussian parameters.
    This is complex because it needs intermediate features from both encoders.

    For simplicity, we'll export a simplified version that takes:
    - Input: decoder features [1, 256, 768, 768]
    - Output: 5 Gaussian tensors
    """

    def __init__(self, gaussian_head):
        super().__init__()
        self.head = gaussian_head

    def forward(self, decoder_features):
        """
        Convert decoder output to Gaussian parameters.

        Args:
            decoder_features: [1, 256, 768, 768]

        Returns:
            Tuple of 5 tensors: positions, scales, rotations, colors, opacity
        """
        gaussians = self.head(decoder_features)

        positions = gaussians.mean_vectors.flatten(1, -2)  # [1, N, 3]
        scales = gaussians.singular_values.flatten(1, -2)  # [1, N, 3]
        rotations = gaussians.quaternions.flatten(1, -2)  # [1, N, 4]
        colors = gaussians.colors.flatten(1, -2)  # [1, N, 3]
        opacity = gaussians.opacities.flatten(1, -1)  # [1, N]

        return positions, scales, rotations, colors, opacity


def export_image_encoder():
    """Export image encoder to NCNN."""
    print("=" * 60)
    print("Exporting Image Encoder ViT to NCNN")
    print("=" * 60)

    predictor = load_sharp_model()
    image_encoder = predictor.monodepth_model.monodepth_predictor.encoder.image_encoder

    model = ImageEncoderExport(image_encoder)
    model.eval()

    # Input: 384x384 image (lowest resolution from pyramid)
    dummy = torch.randn(1, 3, 384, 384)

    print("Testing inference...")
    with torch.no_grad():
        output = model(dummy)
        print(f"Input: {dummy.shape}")
        print(f"Output: {output.shape}")

    print("\nTracing model...")
    traced = torch.jit.trace(model, dummy)
    ts_path = OUTPUT_DIR / "sharp_image_encoder.pt"
    traced.save(str(ts_path))
    print(f"Saved TorchScript to {ts_path}")
    print(f"Size: {ts_path.stat().st_size / 1024 / 1024:.1f} MB")

    print("\nConverting with PNNX...")
    result = subprocess.run(
        [PNNX_PATH, str(ts_path), "inputshape=[1,3,384,384]"],
        cwd=str(OUTPUT_DIR),
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("PNNX conversion successful!")
        param_path = OUTPUT_DIR / "sharp_image_encoder.ncnn.param"
        bin_path = OUTPUT_DIR / "sharp_image_encoder.ncnn.bin"
        if param_path.exists():
            print(f"  param: {param_path.stat().st_size / 1024:.1f} KB")
        if bin_path.exists():
            print(f"  bin: {bin_path.stat().st_size / 1024 / 1024:.1f} MB")
        return True
    else:
        print(f"PNNX error: {result.stderr[:1000]}")
        return False


def export_cls_and_pos_embed():
    """
    Export CLS token and positional embeddings as raw tensors.
    These will be loaded in C++ and added between components.
    """
    print("\n" + "=" * 60)
    print("Exporting CLS tokens and Positional Embeddings")
    print("=" * 60)

    predictor = load_sharp_model()
    enc = predictor.monodepth_model.monodepth_predictor.encoder

    # Patch encoder embeddings
    patch_cls = enc.patch_encoder.cls_token.detach().cpu()  # [1, 1, 1024]
    patch_pos = enc.patch_encoder.pos_embed.detach().cpu()  # [1, 577, 1024]

    # Save as numpy for easy loading in C++
    import numpy as np
    np.save(OUTPUT_DIR / "patch_cls_token.npy", patch_cls.numpy())
    np.save(OUTPUT_DIR / "patch_pos_embed.npy", patch_pos.numpy())

    print(f"patch_cls_token: {patch_cls.shape}")
    print(f"patch_pos_embed: {patch_pos.shape}")

    # Note: Image encoder embeddings are already included in ImageEncoderExport


def analyze_full_encoder_output():
    """
    Analyze what the full encoder produces to understand decoder input.
    """
    print("\n" + "=" * 60)
    print("Analyzing Encoder Output Shapes")
    print("=" * 60)

    predictor = load_sharp_model()
    enc = predictor.monodepth_model.monodepth_predictor.encoder
    dec = predictor.monodepth_model.monodepth_predictor.decoder

    dummy = torch.randn(1, 3, 1536, 1536)

    print("Running encoder...")
    with torch.no_grad():
        outputs = enc(dummy)

    print("Encoder outputs (5 feature maps):")
    for i, out in enumerate(outputs):
        print(f"  {i}: {out.shape}")

    print("\nRunning decoder...")
    with torch.no_grad():
        decoded = dec(outputs)
        print(f"Decoder output: {decoded.shape}")


def export_full_encoder_as_one():
    """
    Try exporting the full encoder (pyramid + both ViTs + merge) as one model.
    The decoder is separate.
    """
    print("\n" + "=" * 60)
    print("Exporting Full Encoder as Single Component")
    print("=" * 60)

    predictor = load_sharp_model()
    enc = predictor.monodepth_model.monodepth_predictor.encoder

    # The encoder is the SlidingPyramidNetwork
    enc.eval()

    dummy = torch.randn(1, 3, 1536, 1536)

    print("Testing encoder...")
    with torch.no_grad():
        outputs = enc(dummy)
        print("Encoder outputs:")
        for i, out in enumerate(outputs):
            print(f"  {i}: {out.shape}")

    print("\nTracing encoder...")
    # The encoder returns a list, but PNNX needs tuple
    class EncoderWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, x):
            outputs = self.encoder(x)
            return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]

    wrapper = EncoderWrapper(enc)
    wrapper.eval()

    traced = torch.jit.trace(wrapper, dummy)
    ts_path = OUTPUT_DIR / "sharp_encoder_full.pt"
    traced.save(str(ts_path))
    print(f"Saved TorchScript to {ts_path}")
    print(f"Size: {ts_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")

    print("\nConverting with PNNX (this will be slow)...")
    result = subprocess.run(
        [PNNX_PATH, str(ts_path), "inputshape=[1,3,1536,1536]", "fp16=0"],
        cwd=str(OUTPUT_DIR),
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("PNNX conversion successful!")
        param_path = OUTPUT_DIR / "sharp_encoder_full.ncnn.param"
        bin_path = OUTPUT_DIR / "sharp_encoder_full.ncnn.bin"
        if param_path.exists():
            print(f"  param: {param_path.stat().st_size / 1024:.1f} KB")
        if bin_path.exists():
            print(f"  bin: {bin_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")

        # Check for channel mismatch
        verify_ncnn_param(param_path)
        return True
    else:
        print(f"PNNX error: {result.stderr[:1000]}")
        return False


def verify_ncnn_param(param_path):
    """Verify NCNN param file for channel issues."""
    print("\n--- Verifying NCNN Conversion ---")

    if not param_path.exists():
        print("No param file to verify")
        return

    with open(param_path, 'r') as f:
        lines = f.readlines()

    layer_count, blob_count = map(int, lines[1].strip().split())
    print(f"Layers: {layer_count}, Blobs: {blob_count}")

    # Find Concat followed by Convolution to check for channel mismatch
    for i, line in enumerate(lines[2:], 2):
        if 'Concat' in line:
            parts = line.split()
            num_inputs = int(parts[2])
            if num_inputs > 10:  # Likely the patch concatenation
                concat_output = parts[4 + num_inputs - 1]
                print(f"  Large Concat at line {i}: {num_inputs} inputs → blob {concat_output}")

                # Find next conv using this blob
                for j, line2 in enumerate(lines[i:], i):
                    if 'Convolution' in line2 and concat_output in line2:
                        conv_parts = line2.split()
                        for p in conv_parts:
                            if p.startswith('6='):
                                wsize = int(p[2:])
                                # For 16x16 kernel, 1024 output:
                                # wsize = out_ch * in_ch * k * k
                                # 786432 = 1024 * 3 * 16 * 16
                                in_ch = wsize // (1024 * 16 * 16)
                                if in_ch == 3:
                                    print(f"  ✓ Conv expects 3 channels (weight size {wsize})")
                                else:
                                    print(f"  ✗ Conv expects {in_ch} channels (weight size {wsize})")
                        break


def main():
    print("\n" + "=" * 60)
    print("SHARP Component Export for NCNN")
    print("=" * 60)

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

    # Analyze encoder output
    analyze_full_encoder_output()

    # Export image encoder (needed for component approach)
    export_image_encoder()

    # Export CLS tokens and positional embeddings
    export_cls_and_pos_embed()

    # Try exporting full encoder
    print("\n" + "=" * 60)
    print("Attempting Full Encoder Export")
    print("=" * 60)
    print("This may have channel issues but let's check...")
    export_full_encoder_as_one()

    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    print("""
Exported Components:
1. sharp_pyramid_v2.ncnn.* (already exists) - Pyramid + Patch Embed
   Input: [1, 3, 1536, 1536]
   Output: [35, 576, 1024]

2. sharp_patch_encoder.ncnn.* (already exists) - Patch Encoder ViT
   Input: [35, 577, 1024]  (add CLS + pos_embed first)
   Output: [35, 577, 1024]

3. sharp_image_encoder.ncnn.* (NEW) - Image Encoder ViT
   Input: [1, 3, 384, 384]
   Output: [1, 577, 1024]

4. patch_cls_token.npy, patch_pos_embed.npy (NEW) - Embeddings for C++

5. sharp_encoder_full.ncnn.* (NEW, may have issues) - Full encoder

Next Steps:
1. Test individual components work in NCNN
2. If full encoder has channel issues, use component chaining in C++
3. Update sharp_ncnn.cpp to chain components
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
