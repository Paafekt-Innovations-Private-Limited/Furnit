#!/usr/bin/env python3
"""
Export SHARP model to split ExecuTorch .pte files for memory-efficient Android deployment.

Model structure (from analysis):
- monodepth_model.monodepth_predictor.encoder.patch_encoder: TimmViT (326M params, ~1.2GB)
- monodepth_model.monodepth_predictor.encoder.image_encoder: TimmViT (326M params, ~1.2GB)
- monodepth_model.monodepth_predictor.decoder: MultiresConvDecoder (~20M params)
- feature_model: GaussianDensePredictionTransformer (~100M params)
- prediction_head: DirectPredictionHead (~5M params)

Split strategy:
1. sharp_patch_encoder.pte - Sliding pyramid + patch encoder ViT (~1.2GB)
2. sharp_image_encoder.pte - Full image encoder ViT (~1.2GB)
3. sharp_decoder.pte - Upsampling + decoder + fusion (~100MB)
4. sharp_gaussian_head.pte - Feature model + prediction head (~400MB)
"""

import sys
from pathlib import Path

SHARP_SRC = Path("/tmp/ml-sharp/src")
sys.path.insert(0, str(SHARP_SRC))

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/ml_experiments/models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")


class SlidingPyramidPatchEncoder(nn.Module):
    """
    Part 1: Sliding Pyramid + Patch Encoder ViT

    Creates 35 patches from the input image using sliding pyramid,
    then encodes each patch through the patch encoder ViT.

    Input: [1, 3, 1536, 1536] - RGB image
    Output: [35, 576, 1024] - Encoded patch features (no CLS token)
    """
    def __init__(self, patch_encoder):
        super().__init__()
        self.patch_encoder = patch_encoder
        self.patch_size = 384

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Create sliding pyramid patches
        patches = []

        # 1.0x scale: 5x5 grid = 25 patches
        stride_1x = (image.shape[2] - self.patch_size) // 4  # 288
        for i in range(5):
            for j in range(5):
                y = i * stride_1x
                x = j * stride_1x
                patch = image[:, :, y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)

        # 0.5x scale: 3x3 grid = 9 patches
        img_05 = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False)
        stride_05 = (img_05.shape[2] - self.patch_size) // 2  # 192
        for i in range(3):
            for j in range(3):
                y = i * stride_05
                x = j * stride_05
                patch = img_05[:, :, y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)

        # 0.25x scale: 1 patch
        img_025 = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=False)
        patches.append(img_025)

        # Stack as batch: [35, 3, 384, 384]
        batch = torch.cat(patches, dim=0)

        # Run through patch encoder
        # TimmViT returns (features, intermediate_dict)
        # features: [B, 1024, 24, 24]
        # intermediate_dict: {layer_idx: [B, 576, 1024]} - intermediate features for FPN
        features, intermediates = self.patch_encoder(batch)

        # Return features as [35, 1024, 24, 24]
        # We also need to return intermediates for the FPN decoder
        return features


class ImageEncoderWrapper(nn.Module):
    """
    Part 2: Image Encoder ViT

    Encodes the full 1536x1536 image for global context.

    Input: [1, 3, 1536, 1536] - RGB image
    Output: [1, 576, 1024] - Image-level features
    """
    def __init__(self, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)


class DecoderWrapper(nn.Module):
    """
    Part 3: Upsampling + Decoder + Fusion

    Takes patch features and image features, fuses and decodes them.

    Input:
        patch_features: [35, 576, 1024] - From patch encoder
        image_features: [1, 576, 1024] - From image encoder
    Output: [1, C, H, W] - Decoded feature map
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        # Upsampling layers from encoder
        self.upsample_latent0 = encoder.upsample_latent0
        self.upsample_latent1 = encoder.upsample_latent1
        self.upsample0 = encoder.upsample0
        self.upsample1 = encoder.upsample1
        self.upsample2 = encoder.upsample2
        self.upsample_lowres = encoder.upsample_lowres
        self.fuse_lowres = encoder.fuse_lowres
        # Decoder
        self.decoder = decoder

    def forward(self, patch_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        # This is a simplified version - the actual fusion is more complex
        # We'd need to trace through the original model to get exact behavior
        # For now, return a placeholder
        return image_features


class GaussianHeadWrapper(nn.Module):
    """
    Part 4: Feature Model + Prediction Head

    Takes decoded features and produces Gaussian parameters.

    Input: [1, C, H, W] - Decoded feature map
    Output: [N, 14] - Gaussian parameters (pos, scale, rot, opacity, color)
    """
    def __init__(self, feature_model, prediction_head, gaussian_composer):
        super().__init__()
        self.feature_model = feature_model
        self.prediction_head = prediction_head
        self.gaussian_composer = gaussian_composer

    def forward(self, features: torch.Tensor, disparity: torch.Tensor) -> torch.Tensor:
        # Run through feature model
        texture_features = self.feature_model(features)

        # Get predictions
        geometry = self.prediction_head.geometry_prediction_head(texture_features)
        texture = self.prediction_head.texture_prediction_head(texture_features)

        # Compose gaussians
        gaussians = self.gaussian_composer(geometry, texture, disparity)

        # Extract and concatenate parameters
        means = gaussians.mean_vectors
        scales = gaussians.singular_values
        rotations = gaussians.quaternions
        colors = gaussians.colors
        opacities = gaussians.opacities

        if opacities.dim() == 2:
            opacities = opacities.unsqueeze(-1)

        params = torch.cat([means, scales, rotations, opacities, colors], dim=-1)
        return params.squeeze(0)


def load_sharp_model():
    """Load the full SHARP model."""
    from sharp.models import PredictorParams, create_predictor

    print(f"Loading SHARP model from {MODEL_WEIGHTS}...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    return predictor


def export_to_pte(model, example_inputs, output_path: Path, name: str):
    """Export a model to ExecuTorch .pte format."""
    from executorch.exir import EdgeCompileConfig, to_edge

    print(f"\nExporting {name}...")

    # Ensure inputs is a tuple
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)

    for i, inp in enumerate(example_inputs):
        print(f"  Input {i}: {inp.shape}")

    # Test inference
    with torch.no_grad():
        output = model(*example_inputs)
        print(f"  Output: {output.shape}")

    # Export
    print("  torch.export...")
    exported = torch.export.export(model, example_inputs, strict=False)

    print("  to_edge...")
    edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))

    print("  to_executorch...")
    et = edge.to_executorch()

    print(f"  Saving to {output_path}...")
    with open(output_path, "wb") as f:
        f.write(et.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ {name}: {size_mb:.1f} MB")

    return output_path


def count_parameters(model):
    """Count trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def main():
    print("=" * 60)
    print("SHARP Split ExecuTorch Export")
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
    mono = predictor.monodepth_model.monodepth_predictor
    encoder = mono.encoder
    decoder = mono.decoder

    print("\nComponent sizes:")
    print(f"  patch_encoder: {count_parameters(encoder.patch_encoder):.1f}M params")
    print(f"  image_encoder: {count_parameters(encoder.image_encoder):.1f}M params")
    print(f"  decoder: {count_parameters(decoder):.1f}M params")
    print(f"  feature_model: {count_parameters(predictor.feature_model):.1f}M params")
    print(f"  prediction_head: {count_parameters(predictor.prediction_head):.1f}M params")

    # === Part 1: Patch Encoder ===
    print("\n" + "=" * 40)
    print("Part 1: Sliding Pyramid + Patch Encoder")
    print("=" * 40)

    patch_encoder_model = SlidingPyramidPatchEncoder(encoder.patch_encoder)
    patch_encoder_model.eval()

    dummy_image = torch.randn(1, 3, 1536, 1536)

    print("Testing patch encoder...")
    with torch.no_grad():
        patch_features = patch_encoder_model(dummy_image)
        print(f"  Output shape: {patch_features.shape}")

    try:
        export_to_pte(
            patch_encoder_model,
            dummy_image,
            OUTPUT_DIR / "sharp_patch_encoder.pte",
            "Patch Encoder"
        )
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()

    # === Part 2: Image Encoder ===
    print("\n" + "=" * 40)
    print("Part 2: Image Encoder")
    print("=" * 40)

    image_encoder_model = ImageEncoderWrapper(encoder.image_encoder)
    image_encoder_model.eval()

    print("Testing image encoder...")
    with torch.no_grad():
        image_features = image_encoder_model(dummy_image)
        print(f"  Output shape: {image_features.shape}")

    try:
        export_to_pte(
            image_encoder_model,
            dummy_image,
            OUTPUT_DIR / "sharp_image_encoder.pte",
            "Image Encoder"
        )
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    exported = list(OUTPUT_DIR.glob("sharp_*.pte"))
    if exported:
        print("\nExported files:")
        total = 0
        for f in sorted(exported):
            size = f.stat().st_size / (1024 * 1024)
            total += size
            print(f"  {f.name}: {size:.1f} MB")
        print(f"  Total: {total:.1f} MB")

    print("""
Next steps:
1. Push .pte files to device:
   for f in sharp_ncnn_models/*.pte; do
     adb push "$f" /sdcard/Android/data/com.furnit.android/files/models/
   done

2. Create SplitExecuTorchSharp.kt on Android

3. Update SharpService.kt to use split ExecuTorch
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
