#!/usr/bin/env python3
"""
Split LiteRT Part 4 into Part 4a (ViT blocks 12-23) and Part 4b (decoder + Gaussians).

The original Part 4 (~387MB) crashes at 68% on Android due to peak memory during
interpreter creation + decoder activations. Splitting at the ViT/decoder boundary
produces two smaller models that can be loaded/unloaded sequentially.

  Part 4a — ViT blocks 12-23 + norm  (~150MB):
    Input:  image_tokens [1, 577, 1024]
    Output: tokens_after_norm [1, 577, 1024]

  Part 4b — Reshape + Upsamplers + Decoder + Gaussians  (~237MB):
    Input:  tokens_after_norm [1,577,1024], image [1,3,1536,1536],
            latent0 [1,1024,96,96], latent1 [1,1024,96,96],
            x0_feat [1,1024,96,96], x1_feat [1,1024,48,48],
            x2_feat [1,1024,24,24]
    Output: packed [1, N, 14] Gaussian parameters

Prerequisites:
    pip install litert-torch torch tensorflow

Usage:
    python export_sharp_litert_part4_chunks.py --weights /path/to/sharp.pt

Push to device:
    adb push sharp_litert_models/sharp_part4a_fp16.tflite \\
        /storage/emulated/0/Android/data/com.furnit.android/files/models/
    adb push sharp_litert_models/sharp_part4b_fp16.tflite \\
        /storage/emulated/0/Android/data/com.furnit.android/files/models/
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_SHARP_SRC = Path("/tmp/ml-sharp/src")
DEFAULT_WEIGHTS = Path("/Users/al/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt")

VIT_SPLIT_BLOCK = 12


def patch_srgb():
    """Monkey-patch sRGB->linear to avoid tfl.pow with float exponents."""
    import sharp.utils.color_space as _cs
    from sharp.utils.robust import robust_where

    def _srgb_to_linear_tflite(srgb_color: torch.Tensor) -> torch.Tensor:
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
    import sharp.models.composer as _composer
    _composer.sRGB2linearRGB = _srgb_to_linear_tflite
    print("Patched sRGB2linearRGB for TFLite (pow -> exp*log)")


class ImageEncoderPartB_ViTOnly(nn.Module):
    """Part 4a: ViT blocks 12-23 + LayerNorm.

    Input:  image_tokens [1, 577, 1024]
    Output: tokens_after_norm [1, 577, 1024]
    """

    def __init__(self, predictor):
        super().__init__()
        spn = predictor.monodepth_model.monodepth_predictor.encoder
        ie = spn.image_encoder

        self.blocks = nn.ModuleList(list(ie.blocks[VIT_SPLIT_BLOCK:]))
        self.norm = ie.norm

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        x = image_tokens
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class ImageEncoderPartB_DecoderOnly(nn.Module):
    """Part 4b: Reshape + upsamplers + fusion + decoder + head + Gaussians.

    Input:  tokens_after_norm [1,577,1024], image [1,3,1536,1536],
            latent0 [1,1024,96,96], latent1 [1,1024,96,96],
            x0_feat [1,1024,96,96], x1_feat [1,1024,48,48],
            x2_feat [1,1024,24,24]
    Output: packed [1, N, 14] Gaussian parameters
    """

    def __init__(self, predictor):
        super().__init__()
        spn = predictor.monodepth_model.monodepth_predictor.encoder
        ie = spn.image_encoder
        mono = predictor.monodepth_model

        self.num_prefix_tokens = ie.num_prefix_tokens
        self.grid_size = ie.patch_embed.grid_size

        self.upsample_latent0 = spn.upsample_latent0
        self.upsample_latent1 = spn.upsample_latent1
        self.upsample0 = spn.upsample0
        self.upsample1 = spn.upsample1
        self.upsample2 = spn.upsample2
        self.upsample_lowres = spn.upsample_lowres
        self.fuse_lowres = spn.fuse_lowres

        self.decoder = mono.monodepth_predictor.decoder
        self.head = mono.monodepth_predictor.head

        self.return_encoder_features = mono.return_encoder_features
        self.return_decoder_features = mono.return_decoder_features
        self.num_monodepth_layers = mono.num_monodepth_layers
        self.sorting_monodepth = mono.sorting_monodepth

        self.init_model = predictor.init_model
        self.feature_model = predictor.feature_model
        self.prediction_head = predictor.prediction_head
        self.gaussian_composer = predictor.gaussian_composer

    def _reshape_feature(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channel = embeddings.shape
        h, w = self.grid_size
        if self.num_prefix_tokens:
            embeddings = embeddings[:, self.num_prefix_tokens:, :]
        return embeddings.reshape(batch, h, w, channel).permute(0, 3, 1, 2)

    def forward(
        self,
        tokens_after_norm: torch.Tensor,
        image: torch.Tensor,
        latent0: torch.Tensor,
        latent1: torch.Tensor,
        x0_feat: torch.Tensor,
        x1_feat: torch.Tensor,
        x2_feat: torch.Tensor,
    ) -> torch.Tensor:
        x_lowres = self._reshape_feature(tokens_after_norm)

        latent0_up = self.upsample_latent0(latent0)
        latent1_up = self.upsample_latent1(latent1)
        x0_up = self.upsample0(x0_feat)
        x1_up = self.upsample1(x1_feat)
        x2_up = self.upsample2(x2_feat)

        x_lowres_up = self.upsample_lowres(x_lowres)
        x_fused = self.fuse_lowres(torch.cat((x2_up, x_lowres_up), dim=1))

        encoder_features = [latent0_up, latent1_up, x0_up, x1_up, x_fused]
        decoder_features = self.decoder(encoder_features)
        disparity = self.head(decoder_features)

        if self.num_monodepth_layers == 2 and self.sorting_monodepth:
            first_layer = disparity.max(dim=1, keepdims=True).values
            second_layer = disparity.min(dim=1, keepdims=True).values
            disparity = torch.cat([first_layer, second_layer], dim=1)

        output_features = []
        if self.return_encoder_features:
            output_features.extend(encoder_features)
        if self.return_decoder_features:
            output_features.append(decoder_features)

        disparity_factor = torch.ones(1, 1, 1, 1, device=image.device)
        monodepth = disparity_factor / disparity.clamp(min=1e-4, max=1e4)

        init_output = self.init_model(image, monodepth)
        image_features = self.feature_model(
            init_output.feature_input, encodings=output_features
        )
        delta_values = self.prediction_head(image_features)
        gaussians = self.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )

        positions = gaussians.mean_vectors
        opacities = gaussians.opacities.unsqueeze(-1)
        scales = gaussians.singular_values
        quaternions = gaussians.quaternions
        colors = gaussians.colors
        return torch.cat(
            [positions, opacities, scales, quaternions, colors], dim=-1
        )


def export_part(name, wrapper, sample_inputs, converter_flags, output_path):
    """Export a single part to TFLite."""
    import litert_torch

    print(f"\n{'=' * 60}")
    print(f"Exporting {name}")
    print(f"{'=' * 60}")

    start = time.time()
    edge_model = litert_torch.convert(
        wrapper, sample_inputs,
        _ai_edge_converter_flags=converter_flags,
    )
    elapsed = time.time() - start
    print(f"  Conversion: {elapsed:.0f}s")

    edge_model.export(str(output_path))
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {output_path.name} ({size_mb:.0f} MB)")

    return size_mb


def parse_args():
    parser = argparse.ArgumentParser(description="Export LiteRT Part 4a/4b chunks")
    parser.add_argument("--sharp-src", default=str(DEFAULT_SHARP_SRC),
                        help=f"ml-sharp src directory (default: {DEFAULT_SHARP_SRC})")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help=f"SHARP .pt checkpoint (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--output-dir",
                        default=str(Path(__file__).resolve().parent / "sharp_litert_models"),
                        help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    sharp_src = Path(args.sharp_src)
    if not sharp_src.exists():
        print(f"ERROR: SHARP source not found at {sharp_src}")
        return 1

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: Weights not found at {weights_path}")
        return 1

    sys.path.insert(0, str(sharp_src))

    try:
        import litert_torch
    except ImportError:
        print("ERROR: litert-torch not installed. pip install litert-torch")
        return 1

    import tensorflow as tf

    patch_srgb()

    from sharp.models import PredictorParams, create_predictor

    print("=" * 60)
    print("Export LiteRT Part 4 chunks (4a: ViT, 4b: Decoder)")
    print("=" * 60)

    print(f"\nLoading SHARP model from {weights_path} ...")
    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    del state_dict

    part4a = ImageEncoderPartB_ViTOnly(predictor).eval()
    part4b = ImageEncoderPartB_DecoderOnly(predictor).eval()

    part4a_params = sum(p.numel() for p in part4a.parameters())
    part4b_params = sum(p.numel() for p in part4b.parameters())
    print(f"  Part 4a (ViT 12-23 + norm): {part4a_params / 1e6:.0f}M params")
    print(f"  Part 4b (Decoder + Gauss):   {part4b_params / 1e6:.0f}M params")

    # Validate: run Part 4a then Part 4b and compare to original Part 4
    from export_sharp_litert_split import ImageEncoderPartB_Full
    part4_full = ImageEncoderPartB_Full(predictor).eval()

    print("\nValidating chunked vs original Part 4...")
    with torch.no_grad():
        sample_image = torch.rand(1, 3, 1536, 1536)
        sample_image_tokens = torch.rand(1, 577, 1024)
        sample_latent0 = torch.rand(1, 1024, 96, 96)
        sample_latent1 = torch.rand(1, 1024, 96, 96)
        sample_x0 = torch.rand(1, 1024, 96, 96)
        sample_x1 = torch.rand(1, 1024, 48, 48)
        sample_x2 = torch.rand(1, 1024, 24, 24)

        ref_output = part4_full(sample_image, sample_image_tokens,
                                sample_latent0, sample_latent1,
                                sample_x0, sample_x1, sample_x2)

        tokens_norm = part4a(sample_image_tokens)
        chunked_output = part4b(tokens_norm, sample_image,
                                sample_latent0, sample_latent1,
                                sample_x0, sample_x1, sample_x2)

        max_diff = (ref_output - chunked_output).abs().max().item()
        mean_diff = (ref_output - chunked_output).abs().mean().item()
        print(f"  Max diff:  {max_diff:.8f}")
        print(f"  Mean diff: {mean_diff:.8f}")
        if max_diff > 0.01:
            print("  WARNING: diff > 0.01")
        else:
            print("  OK — chunked matches original Part 4")

    del part4_full

    # Export
    converter_flags = {
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "target_spec": {"supported_types": [tf.float16]},
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Part 4a: ViT blocks 12-23 + norm
    size_4a = export_part(
        "Part 4a: ViT blocks 12-23 + norm",
        part4a, (sample_image_tokens,), converter_flags,
        output_dir / "sharp_part4a_fp16.tflite",
    )

    # Part 4b: Decoder + Gaussians
    size_4b = export_part(
        "Part 4b: Decoder + Gaussians",
        part4b, (tokens_norm, sample_image,
                 sample_latent0, sample_latent1,
                 sample_x0, sample_x1, sample_x2),
        converter_flags,
        output_dir / "sharp_part4b_fp16.tflite",
    )

    print(f"\n{'=' * 60}")
    print(f"Done!")
    print(f"{'=' * 60}")
    print(f"  Part 4a: {size_4a:.0f} MB (ViT blocks 12-23)")
    print(f"  Part 4b: {size_4b:.0f} MB (Decoder + Gaussians)")
    print(f"  Total:   {size_4a + size_4b:.0f} MB (was ~387 MB)")

    dest = "/storage/emulated/0/Android/data/com.furnit.android/files/models"
    print(f"\nPush to device:")
    print(f"  adb push {output_dir}/sharp_part4a_fp16.tflite {dest}/")
    print(f"  adb push {output_dir}/sharp_part4b_fp16.tflite {dest}/")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
