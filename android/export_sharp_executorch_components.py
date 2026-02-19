#!/usr/bin/env python3
"""
Export SHARP model components for memory-efficient Android inference.

Strategy: Export individual components that can be loaded/unloaded sequentially.
The sliding pyramid logic is handled on Android side (similar to NCNN component mode).

Components:
1. sharp_single_patch.pte - Process single 384x384 patch through ViT (~1.2GB each)
2. sharp_fpn_merge.pte - Merge patch features into pyramid (~100MB)
3. sharp_decoder.pte - Decode to Gaussian features (~100MB)
4. sharp_gaussian.pte - Final Gaussian prediction (~400MB)
"""

import sys
from pathlib import Path

SHARP_SRC = Path("/tmp/ml-sharp/src")
sys.path.insert(0, str(SHARP_SRC))

import torch
import torch.nn as nn

MODEL_WEIGHTS = Path("/Users/al/Library/Mobile Documents/com~apple~CloudDocs/ml_experiments/models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")


class SinglePatchEncoder(nn.Module):
    """
    Process a single 384x384 patch through the ViT encoder.

    Input: [1, 3, 384, 384] - Single RGB patch
    Output: [1, 1024, 24, 24] - Encoded features

    This is ~1.2GB but processes one patch at a time on device.
    On Android, we loop through 35 patches, loading/running/unloading each time.
    """
    def __init__(self, vit):
        super().__init__()
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.blocks = vit.blocks
        self.norm = vit.norm

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed.proj(patch)  # [1, 1024, 24, 24]
        x = x.flatten(2).transpose(1, 2)  # [1, 576, 1024]

        # Add CLS token
        cls = self.cls_token.expand(1, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [1, 577, 1024]

        # Add position embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Remove CLS, reshape back to spatial
        x = x[:, 1:, :]  # [1, 576, 1024]
        x = x.transpose(1, 2).reshape(1, 1024, 24, 24)  # [1, 1024, 24, 24]

        return x


def load_sharp_model():
    """Load the full SHARP model."""
    from sharp.models import PredictorParams, create_predictor

    print(f"Loading SHARP model from {MODEL_WEIGHTS}...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    return predictor


def export_to_pte(model, example_input, output_path: Path, name: str):
    """Export a model to ExecuTorch .pte format."""
    from executorch.exir import EdgeCompileConfig, to_edge

    print(f"\n{'='*50}")
    print(f"Exporting {name}")
    print(f"{'='*50}")
    print(f"Input: {example_input.shape}")

    # Test inference
    with torch.no_grad():
        output = model(example_input)
        print(f"Output: {output.shape}")

    # Export
    print("torch.export...")
    exported = torch.export.export(model, (example_input,), strict=False)

    print("to_edge...")
    edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))

    print("to_executorch...")
    et = edge.to_executorch()

    print(f"Saving {output_path.name}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ {name}: {size_mb:.1f} MB")

    return output_path


class ImageEncoderSinglePatch(nn.Module):
    """
    Process a single 384x384 patch through the image encoder ViT.
    Same as patch encoder but separate weights for global context.
    """
    def __init__(self, vit):
        super().__init__()
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.blocks = vit.blocks
        self.norm = vit.norm

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed.proj(patch)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(1, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 1:, :]
        x = x.transpose(1, 2).reshape(1, 1024, 24, 24)
        return x


class DecoderHead(nn.Module):
    """
    Decoder + Gaussian prediction head.
    Takes merged feature pyramid and outputs Gaussian parameters.
    """
    def __init__(self, decoder, head, feature_model, prediction_head, gaussian_composer):
        super().__init__()
        self.decoder = decoder
        self.head = head
        self.feature_model = feature_model
        self.prediction_head = prediction_head
        self.gaussian_composer = gaussian_composer
        self.register_buffer('disparity_factor', torch.tensor([1.0]))

    def forward(self, f0: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor,
                f3: torch.Tensor, f4: torch.Tensor) -> torch.Tensor:
        # Decoder expects list of features at different scales
        features = [f0, f1, f2, f3, f4]
        decoded = self.decoder(features)
        depth_head_out = self.head(decoded)

        # Feature model for Gaussian prediction
        gaussian_features = self.feature_model(depth_head_out)

        # Final predictions
        geometry = self.prediction_head.geometry_prediction_head(gaussian_features)
        texture = self.prediction_head.texture_prediction_head(gaussian_features)

        # Compose Gaussians
        gaussians = self.gaussian_composer(geometry, texture, self.disparity_factor)

        means = gaussians.mean_vectors
        scales = gaussians.singular_values
        rotations = gaussians.quaternions
        colors = gaussians.colors
        opacities = gaussians.opacities
        if opacities.dim() == 2:
            opacities = opacities.unsqueeze(-1)

        params = torch.cat([means, scales, rotations, opacities, colors], dim=-1)
        return params.squeeze(0)


def main():
    print("=" * 60)
    print("SHARP Component Export for ExecuTorch")
    print("=" * 60)

    if not SHARP_SRC.exists():
        print(f"ERROR: ml-sharp not found at {SHARP_SRC}")
        return 1

    if not MODEL_WEIGHTS.exists():
        print(f"ERROR: Model weights not found at {MODEL_WEIGHTS}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    predictor = load_sharp_model()

    # Get components
    encoder = predictor.monodepth_model.monodepth_predictor.encoder
    decoder = predictor.monodepth_model.monodepth_predictor.decoder
    head = predictor.monodepth_model.monodepth_predictor.head
    patch_vit = encoder.patch_encoder
    image_vit = encoder.image_encoder

    print(f"\nComponent sizes:")
    print(f"  Patch encoder: {sum(p.numel() for p in patch_vit.parameters()) / 1e6:.1f}M")
    print(f"  Image encoder: {sum(p.numel() for p in image_vit.parameters()) / 1e6:.1f}M")
    print(f"  Decoder: {sum(p.numel() for p in decoder.parameters()) / 1e6:.1f}M")
    print(f"  Head: {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M")
    print(f"  Feature model: {sum(p.numel() for p in predictor.feature_model.parameters()) / 1e6:.1f}M")

    # === Part 1: Single Patch Encoder ===
    single_patch = SinglePatchEncoder(patch_vit)
    single_patch.eval()
    dummy_patch = torch.randn(1, 3, 384, 384)

    pte_path = OUTPUT_DIR / "sharp_single_patch.pte"
    if not pte_path.exists():
        try:
            export_to_pte(single_patch, dummy_patch, pte_path, "Single Patch Encoder")
        except Exception as e:
            print(f"✗ Patch encoder export failed: {e}")
    else:
        print(f"\n✓ sharp_single_patch.pte already exists ({pte_path.stat().st_size / 1e6:.1f} MB)")

    # === Part 2: Image Encoder (same architecture, different weights) ===
    image_encoder = ImageEncoderSinglePatch(image_vit)
    image_encoder.eval()

    pte_path = OUTPUT_DIR / "sharp_image_encoder.pte"
    if not pte_path.exists():
        try:
            export_to_pte(image_encoder, dummy_patch, pte_path, "Image Encoder")
        except Exception as e:
            print(f"✗ Image encoder export failed: {e}")
    else:
        print(f"\n✓ sharp_image_encoder.pte already exists ({pte_path.stat().st_size / 1e6:.1f} MB)")

    # === Part 3: Decoder + Gaussian Head ===
    decoder_head = DecoderHead(
        decoder=decoder,
        head=head,
        feature_model=predictor.feature_model,
        prediction_head=predictor.prediction_head,
        gaussian_composer=predictor.gaussian_composer,
    )
    decoder_head.eval()

    # Decoder takes 5 feature maps at different scales
    # Typical SHARP pyramid: scales from the encoder merging
    # Need to figure out exact shapes by running a test
    try:
        print("\nDetermining decoder input shapes...")
        with torch.no_grad():
            # Run patch encoder to get feature shapes
            test_patch = torch.randn(1, 3, 384, 384)
            patch_feat = single_patch(test_patch)
            print(f"  Patch feature shape: {patch_feat.shape}")

            # The decoder expects a 5-level feature pyramid
            # From NCNN component code, these are merged features at different scales
            # For now, create dummy inputs matching expected shapes
            f0 = torch.randn(1, 1024, 96, 96)   # Highest res
            f1 = torch.randn(1, 1024, 48, 48)
            f2 = torch.randn(1, 1024, 24, 24)
            f3 = torch.randn(1, 1024, 12, 12)
            f4 = torch.randn(1, 1024, 6, 6)      # Lowest res

            test_out = decoder_head(f0, f1, f2, f3, f4)
            print(f"  Decoder output shape: {test_out.shape}")

        pte_path = OUTPUT_DIR / "sharp_decoder_head.pte"
        if not pte_path.exists():
            export_to_pte_multi_input(
                decoder_head, (f0, f1, f2, f3, f4),
                pte_path, "Decoder + Gaussian Head"
            )
        else:
            print(f"\n✓ sharp_decoder_head.pte already exists ({pte_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"✗ Decoder+Head export failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    exported = list(OUTPUT_DIR.glob("sharp_*.pte"))
    if exported:
        print("\nExported files:")
        for f in sorted(exported):
            size = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size:.1f} MB")

    print("""
On Android, the inference flow would be:
1. Load sharp_single_patch.pte (~1.2GB)
2. Create 35 patches from 1536x1536 image (sliding pyramid)
3. For each patch:
   - Run single_patch.pte
   - Save features [1, 1024, 24, 24] to memory
4. Unload sharp_single_patch.pte
5. Merge features on CPU (no ML model needed)
6. Load decoder/gaussian head for final prediction

This keeps peak memory under control while using ExecuTorch.

Push to device:
  adb push sharp_ncnn_models/sharp_single_patch.pte \\
    /sdcard/Android/data/com.furnit.android/files/models/
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
