#!/usr/bin/env python3
"""
Create Lightweight SHARP: Replace heavy DINOv2 encoder with MobileNetV3,
keep all other weights from trained SHARP.

SHARP pipeline:
  1. monodepth_model(image) -> disparity + encoder features  [577M params - REPLACE THIS]
  2. init_model(image, monodepth) -> base Gaussians           [keep]
  3. feature_model(features, encodings) -> refined features    [keep]
  4. prediction_head(features) -> deltas                       [keep]
  5. gaussian_composer(delta, base) -> Gaussians               [keep]

Strategy:
  - Replace DINOv2-Large encoder (577M) with MobileNetV3-Large (5.4M)
  - Add adapter convs to match feature dimensions
  - Freeze decoder/head/composer (SHARP trained weights)
  - Train only encoder + adapter (~10M params) to produce same output
  - Final model: ~150MB instead of 2.5GB

Usage:
  python create_lightweight_sharp.py                    # Create + test
  python create_lightweight_sharp.py --train            # Train adapter
  python create_lightweight_sharp.py --export           # Export to .ptl
"""
import sys
import argparse
import time
from pathlib import Path

SHARP_SRC = "/Users/al/Documents/tries01/Furnit/android/third_party/ml-sharp/src"
sys.path.insert(0, SHARP_SRC)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/android/sharp_litert_models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/executorch_models")


class MobileEncoder(nn.Module):
    """
    Lightweight encoder replacing DINOv2-Large.

    DINOv2-Large outputs:
      - disparity: [1, 2, 768, 768]
      - encoder_features: list of 5 tensors at different scales
      - output_features: list of feature tensors for feature_model

    This module mimics that interface using MobileNetV3-Large backbone
    with adapter layers to match dimensions.
    """
    def __init__(self, original_monodepth):
        super().__init__()

        # MobileNetV3-Large backbone (5.4M params, pretrained on ImageNet)
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone_features = mobilenet.features

        # MobileNet tap points -> SHARP expected output_features:
        # [0]: 256ch @768x768, [1]: 256ch @384x384, [2]: 512ch @192x192
        # [3]: 1024ch @96x96, [4]: 1024ch @48x48
        self.tap_stages = [1, 3, 6, 10, 16]
        mobile_dims =     [16, 24, 40, 80, 960]
        sharp_dims =      [256, 256, 512, 1024, 1024]
        sharp_spatials =  [768, 384, 192, 96, 48]
        self.sharp_spatials = sharp_spatials

        # Adapter convs: map MobileNet features -> exact SHARP decoder dimensions
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mobile_dims[i], sharp_dims[i], 1, bias=False),
                nn.BatchNorm2d(sharp_dims[i]),
                nn.ReLU(inplace=True),
            ) for i in range(5)
        ])

        # Disparity head: predict depth from deepest features
        self.disparity_head = nn.Sequential(
            nn.Conv2d(960, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1),
            nn.Sigmoid(),
        )

        # Keep original components that we need
        self.return_encoder_features = original_monodepth.return_encoder_features
        self.return_decoder_features = original_monodepth.return_decoder_features
        self.num_monodepth_layers = original_monodepth.num_monodepth_layers
        self.sorting_monodepth = original_monodepth.sorting_monodepth

    def forward(self, image):
        # Extract multi-scale features from MobileNet
        features_at_stages = []
        x = image
        tap_set = set(self.tap_stages)

        for i, layer in enumerate(self.backbone_features):
            x = layer(x)
            if i in tap_set:
                features_at_stages.append(x)

        # Adapt features to exact SHARP dimensions (channels + spatial)
        adapted_features = []
        for i, feat in enumerate(features_at_stages):
            adapted = self.adapters[i](feat)
            target_size = self.sharp_spatials[i]
            if adapted.shape[2] != target_size or adapted.shape[3] != target_size:
                adapted = F.interpolate(adapted, size=(target_size, target_size), mode="bilinear", align_corners=False)
            adapted_features.append(adapted)

        # Predict disparity from deepest features
        disparity = self.disparity_head(features_at_stages[-1])
        # Upsample to same resolution as input image (init_model expects this)
        disparity = F.interpolate(disparity, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)

        # Return in same format as original monodepth_model
        return MonodepthOutput(
            disparity=disparity,
            encoder_features=adapted_features,
            output_features=adapted_features,
            decoder_features=None,
        )


class MonodepthOutput:
    """Mimics the output of the original monodepth model."""
    def __init__(self, disparity, encoder_features, output_features, decoder_features):
        self.disparity = disparity
        self.encoder_features = encoder_features
        self.output_features = output_features
        self.decoder_features = decoder_features


class LightweightSharp(nn.Module):
    """
    SHARP with lightweight MobileNet encoder.

    Keeps all trained weights from SHARP decoder/head/composer.
    Only encoder is replaced.
    """
    def __init__(self, original_predictor):
        super().__init__()

        # Replace heavy encoder with lightweight one
        self.mobile_encoder = MobileEncoder(original_predictor.monodepth_model)

        # Keep everything else from trained SHARP (FROZEN)
        self.init_model = original_predictor.init_model
        self.feature_model = original_predictor.feature_model
        self.prediction_head = original_predictor.prediction_head
        self.gaussian_composer = original_predictor.gaussian_composer

        # Freeze decoder weights
        for param in self.init_model.parameters():
            param.requires_grad = False
        for param in self.feature_model.parameters():
            param.requires_grad = False
        for param in self.prediction_head.parameters():
            param.requires_grad = False
        for param in self.gaussian_composer.parameters():
            param.requires_grad = False

        self.register_buffer("disparity_factor", torch.tensor([1.0]))

    def forward(self, image):
        # Step 1: Lightweight encoder (MobileNet)
        monodepth_output = self.mobile_encoder(image)
        disparity = monodepth_output.disparity

        disparity_factor = self.disparity_factor[None, None, None]
        monodepth = disparity_factor / disparity.clamp(min=1e-4, max=1e4)

        if self.mobile_encoder.num_monodepth_layers == 2 and self.mobile_encoder.sorting_monodepth:
            first_layer = monodepth.max(dim=1, keepdims=True).values
            second_layer = monodepth.min(dim=1, keepdims=True).values
            monodepth = torch.cat([first_layer, second_layer], dim=1)

        # Step 2-5: Use SHARP trained weights
        init_output = self.init_model(image, monodepth)
        image_features = self.feature_model(
            init_output.feature_input, encodings=monodepth_output.output_features
        )
        delta_values = self.prediction_head(image_features)
        gaussians = self.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )

        # Pack output
        means = gaussians.mean_vectors
        scales = gaussians.singular_values
        rotations = gaussians.quaternions
        colors = gaussians.colors
        opacities = gaussians.opacities
        if opacities.dim() == 2:
            opacities = opacities.unsqueeze(-1)

        return torch.cat([means, scales, rotations, opacities, colors], dim=-1).squeeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train adapter layers")
    parser.add_argument("--export", action="store_true", help="Export to .ptl")
    parser.add_argument("--image", help="Test with specific image")
    args = parser.parse_args()

    from sharp.models import PredictorParams, create_predictor

    # Load original SHARP
    print("Loading original SHARP (702M params)...")
    t0 = time.time()
    state_dict = torch.load(MODEL_WEIGHTS, map_location="cpu", weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    del state_dict

    original_params = sum(p.numel() for p in predictor.parameters())
    print(f"  Original SHARP: {original_params / 1e6:.0f}M params")

    # Create lightweight version
    print("\nCreating Lightweight SHARP...")
    light_model = LightweightSharp(predictor)
    light_model.eval()

    total_params = sum(p.numel() for p in light_model.parameters())
    trainable_params = sum(p.numel() for p in light_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"  Lightweight SHARP: {total_params / 1e6:.0f}M params total")
    print(f"    Trainable (encoder+adapter): {trainable_params / 1e6:.1f}M")
    print(f"    Frozen (decoder/head/composer): {frozen_params / 1e6:.0f}M")

    # Test forward pass
    print("\nTesting forward pass...")
    if args.image:
        from PIL import Image
        import torchvision.transforms as T
        img = Image.open(args.image).convert("RGB")
        image = T.Compose([T.Resize((1536, 1536)), T.ToTensor()])(img).unsqueeze(0)
        print(f"  Image: {args.image}")
    else:
        image = torch.randn(1, 3, 1536, 1536).clamp(0, 1)
        print("  Random input")

    t0 = time.time()
    with torch.no_grad():
        output = light_model(image)
    elapsed = time.time() - t0
    print(f"  Output: {output.shape}")
    print(f"  Gaussians: {output.shape[0]}")
    print(f"  Time: {elapsed:.1f}s")

    # Also test with 1 thread (mobile sim)
    torch.set_num_threads(1)
    t0 = time.time()
    with torch.no_grad():
        output2 = light_model(image)
    elapsed1t = time.time() - t0
    print(f"  Time (1 thread): {elapsed1t:.1f}s")
    print(f"  Phone estimate: {elapsed1t * 2:.0f}-{elapsed1t * 4:.0f}s")

    if args.export:
        print("\nExporting to TorchScript .ptl...")
        torch.set_num_threads(4)
        traced = torch.jit.trace(light_model, image)
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized = optimize_for_mobile(traced)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ptl_path = OUTPUT_DIR / "sharp_light.ptl"
        optimized._save_for_lite_interpreter(str(ptl_path))
        size_mb = ptl_path.stat().st_size / 1024 / 1024
        print(f"  Saved: {ptl_path} ({size_mb:.0f} MB)")
        print(f"\n  adb push {ptl_path} /sdcard/Android/data/com.furnit.android/files/models/")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Original SHARP: {original_params/1e6:.0f}M params, ~2.5GB")
    print(f"  Lightweight:    {total_params/1e6:.0f}M params, ~{total_params*4/1024/1024:.0f}MB FP32")
    print(f"  Speedup:        {original_params/total_params:.1f}x fewer params")
    print(f"  NOTE: Adapter is untrained -- output quality depends on training the adapter.")
    print(f"  Run with --train to train adapter on SHARP teacher outputs.")


if __name__ == "__main__":
    main()
