#!/usr/bin/env python3
"""
Export SHARP model as 4 split TFLite (LiteRT) parts for memory-constrained Android.

The full SHARP model (702M params, ~1258MB FP16) OOMs on Android.
This script splits it into 4 parts, each under 500MB FP16.

Parts 1 & 2 operate on SINGLE patches [1,3,384,384] — Android runs them
in a loop over the 35 sliding-pyramid patches. This avoids the ~750MB
attention tensor that batched [35,...] processing would require.

  Part 1 — Single-Patch Encoder A  (sharp_part1_fp16.tflite, ~290MB):
    Normalizer + patch_embed + ViT blocks 0-11
    Input:  patch [1, 3, 384, 384]
    Output: tokens [1, 577, 1024], block5 [1, 577, 1024]

  Part 2 — Single-Patch Encoder B  (sharp_part2_fp16.tflite, ~288MB):
    ViT blocks 12-23 + norm + reshape (remove CLS, reshape to spatial)
    Input:  tokens [1, 577, 1024]
    Output: features [1, 1024, 24, 24]

  Part 3 — Image Encoder A  (sharp_part3_fp16.tflite, ~291MB):
    Image encoder patch_embed + ViT blocks 0-11
    Input:  image [1, 3, 1536, 1536]
    Output: image_tokens [1, 577, 1024]

  Part 4 — Image Encoder B + Full Decoder + Gaussians  (~387MB):
    Image encoder blocks 12-23 + all upsamplers + fusion + monodepth decoder
    + head + initializer + feature_model + prediction_head + gaussian_composer
    Input:  image [1,3,1536,1536], image_tokens [1,577,1024], latent0-x2 (5 tensors)
    Output: packed [1, N, 14] Gaussian parameters

Android pipeline:
  1. Create pyramid: x0 (1536²), x1 (768²), x2 (384²)
  2. Split into 35 overlapping 384x384 patches (25 + 9 + 1)
  3. Run Part 1 on each patch → tokens[i], block5[i]  (35 runs)
  4. Run Part 2 on each tokens[i] → features[i]  (35 runs)
  5. Reshape block5/tokens: remove CLS, reshape [576,1024] → [1024,24,24]
  6. Merge patches → latent0, latent1, x0_feat, x1_feat, x2_feat
  7. Run Part 3 on full image → image_tokens
  8. Run Part 4 → packed [1, N, 14] Gaussians

Prerequisites:
    pip install litert-torch torch tensorflow

Usage:
    python export_sharp_litert_split.py

Push to device:
    for i in 1 2 3 4; do
      adb push sharp_ncnn_models/sharp_part${i}_fp16.tflite /data/local/tmp/furnit/
    done
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_conv_bn(model):
    """Recursively fuse Conv2d+BatchNorm2d pairs in eval mode."""
    for name, module in model.named_children():
        fuse_conv_bn(module)
        children = list(module.named_children())
        pairs = []
        i = 0
        while i < len(children) - 1:
            cname, cmod = children[i]
            nname, nmod = children[i + 1]
            if isinstance(cmod, nn.Conv2d) and isinstance(nmod, nn.BatchNorm2d):
                pairs.append([cname, nname])
                i += 2
            else:
                i += 1
        if pairs:
            try:
                torch.ao.quantization.fuse_modules(module, pairs, inplace=True)
            except Exception as e:
                print(f"  [warn] Could not fuse {pairs} in {name}: {e}")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SHARP as 4 split LiteRT TFLite parts.")
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
        help="Output directory for .tflite parts. Default: android/sharp_litert_models",
    )
    return parser.parse_args()
IMAGE_SIZE = 1536
PATCH_SIZE = 384
VIT_SPLIT_BLOCK = 12  # Split ViT at this block index


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


# ---------------------------------------------------------------------------
# Utility: split / merge from SPN (for PyTorch validation only)
# ---------------------------------------------------------------------------

def split_patches_list(image, overlap_ratio, patch_size):
    """Split image into overlapping patches, returning a list."""
    patch_stride = int(patch_size * (1 - overlap_ratio))
    image_size = image.shape[-1]
    steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1
    patches = []
    for j in range(steps):
        j0 = j * patch_stride
        j1 = j0 + patch_size
        for i in range(steps):
            i0 = i * patch_stride
            i1 = i0 + patch_size
            patches.append(image[..., j0:j1, i0:i1])
    return patches


def merge_patches_from_list(patches, padding):
    """Merge a list of [1,C,H,W] spatial patches back into a single tensor."""
    steps = int(math.sqrt(len(patches)))
    output_list = []
    idx = 0
    for j in range(steps):
        row_list = []
        for i in range(steps):
            out = patches[idx]
            if padding != 0:
                if j != 0:
                    out = out[..., padding:, :]
                if i != 0:
                    out = out[..., :, padding:]
                if j != steps - 1:
                    out = out[..., :-padding, :]
                if i != steps - 1:
                    out = out[..., :, :-padding]
            row_list.append(out)
            idx += 1
        output_list.append(torch.cat(row_list, dim=-1))
    return torch.cat(output_list, dim=-2)


# ---------------------------------------------------------------------------
# Part 1: Single-Patch Encoder A (blocks 0-11)
# ---------------------------------------------------------------------------

class SinglePatchEncoderA(nn.Module):
    """Normalizer + patch_embed + blocks[0:11] for a SINGLE patch.

    Input:  patch [1, 3, 384, 384] in [0, 1] range
    Output: (tokens [1, 577, 1024], block5 [1, 577, 1024])
    """

    def __init__(self, predictor):
        super().__init__()
        spn = predictor.monodepth_model.monodepth_predictor.encoder
        pe = spn.patch_encoder

        self.normalizer = predictor.monodepth_model.monodepth_predictor.normalizer
        self.patch_embed = pe.patch_embed
        self.cls_token = pe.cls_token
        self.pos_embed = pe.pos_embed
        self.pos_drop = pe.pos_drop
        self.norm_pre = pe.norm_pre
        self.patch_drop = pe.patch_drop
        self.blocks = nn.ModuleList(list(pe.blocks[:VIT_SPLIT_BLOCK]))

    def forward(self, patch: torch.Tensor):
        x = self.normalizer(patch)
        x = self.patch_embed(x)

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        block5_feat = torch.zeros_like(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx == 5:
                block5_feat = x

        return x, block5_feat


# ---------------------------------------------------------------------------
# Part 2: Single-Patch Encoder B (blocks 12-23 + norm + reshape)
# ---------------------------------------------------------------------------

class SinglePatchEncoderB(nn.Module):
    """Blocks[12:23] + norm + reshape for a SINGLE patch token set.

    Input:  tokens [1, 577, 1024]
    Output: features [1, 1024, 24, 24]
    """

    def __init__(self, predictor):
        super().__init__()
        spn = predictor.monodepth_model.monodepth_predictor.encoder
        pe = spn.patch_encoder

        self.blocks = nn.ModuleList(list(pe.blocks[VIT_SPLIT_BLOCK:]))
        self.norm = pe.norm
        self.num_prefix_tokens = pe.num_prefix_tokens
        self.grid_size = pe.patch_embed.grid_size  # (24, 24)

    def forward(self, tokens: torch.Tensor):
        x = tokens
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Reshape: remove CLS token, reshape to spatial grid
        if self.num_prefix_tokens:
            x = x[:, self.num_prefix_tokens:, :]
        B, N, C = x.shape
        h, w = self.grid_size
        x = x.reshape(B, h, w, C).permute(0, 3, 1, 2)
        return x


# ---------------------------------------------------------------------------
# Part 3: Image Encoder A (blocks 0-11) — unchanged, already batch=1
# ---------------------------------------------------------------------------

class ImageEncoderPartA(nn.Module):
    """Extract x2 from image + image_encoder patch_embed + blocks[0:11].

    Input:  image [1, 3, 1536, 1536]
    Output: image_tokens [1, 577, 1024]
    """

    def __init__(self, predictor):
        super().__init__()
        spn = predictor.monodepth_model.monodepth_predictor.encoder
        ie = spn.image_encoder

        self.normalizer = predictor.monodepth_model.monodepth_predictor.normalizer
        self.patch_embed = ie.patch_embed
        self.cls_token = ie.cls_token
        self.pos_embed = ie.pos_embed
        self.pos_drop = ie.pos_drop
        self.norm_pre = ie.norm_pre
        self.patch_drop = ie.patch_drop
        self.blocks = nn.ModuleList(list(ie.blocks[:VIT_SPLIT_BLOCK]))

    def forward(self, image: torch.Tensor):
        x = self.normalizer(image)
        x2 = F.interpolate(x, size=None, scale_factor=0.25, mode="bilinear", align_corners=False)

        x = self.patch_embed(x2)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Part 4: Image Encoder B + Full Decoder + Gaussian output — unchanged
# ---------------------------------------------------------------------------

class ImageEncoderPartB_Full(nn.Module):
    """Image encoder blocks 12-23 + upsamplers + fusion + monodepth decoder +
    head + initializer + feature_model + prediction_head + gaussian_composer.

    Input:  image [1,3,1536,1536],
            image_tokens [1,577,1024],
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

        self.blocks = nn.ModuleList(list(ie.blocks[VIT_SPLIT_BLOCK:]))
        self.norm = ie.norm
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
        image: torch.Tensor,
        image_tokens: torch.Tensor,
        latent0: torch.Tensor,
        latent1: torch.Tensor,
        x0_feat: torch.Tensor,
        x1_feat: torch.Tensor,
        x2_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = image_tokens
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x_lowres = self._reshape_feature(x)

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


# ---------------------------------------------------------------------------
# Full model reference (for validation)
# ---------------------------------------------------------------------------

class FullModelWrapper(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        disp_f = torch.ones(1, device=image.device)
        gaussians = self.predictor(image, disp_f)
        positions = gaussians.mean_vectors
        opacities = gaussians.opacities.unsqueeze(-1)
        scales = gaussians.singular_values
        quaternions = gaussians.quaternions
        colors = gaussians.colors
        return torch.cat(
            [positions, opacities, scales, quaternions, colors], dim=-1
        )


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

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

    return edge_model, size_mb


def reshape_feature(embeddings, num_prefix_tokens=1, grid_size=(24, 24)):
    """Remove CLS token and reshape to spatial grid (for validation)."""
    if num_prefix_tokens:
        embeddings = embeddings[:, num_prefix_tokens:, :]
    B, N, C = embeddings.shape
    h, w = grid_size
    return embeddings.reshape(B, h, w, C).permute(0, 3, 1, 2)


def main():
    overall_start = time.time()
    print("=" * 60)
    print("Export 4-Part Split SHARP to TFLite (single-patch mode)")
    print("Parts 1 & 2 process one patch at a time — Android loops 35x")
    print("=" * 60)

    args = parse_args()
    sharp_src = Path(args.sharp_src)
    weights_path = Path(args.weights) if args.weights else None
    output_dir = Path(args.output_dir)

    if not sharp_src.exists():
        print(f"ERROR: SHARP source not found at {sharp_src}")
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
        print("ERROR: litert-torch not installed. pip install litert-torch")
        return 1

    import tensorflow as tf

    patch_srgb()

    # ---- Load model ----
    from sharp.models import PredictorParams, create_predictor

    print("\nLoading full SHARP predictor...")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    del state_dict

    print("  Fusing Conv+BN layers...")
    fuse_conv_bn(predictor)

    param_count = sum(p.numel() for p in predictor.parameters())
    print(f"  Loaded: {param_count / 1e6:.0f}M parameters")

    # ---- Create wrappers ----
    part1 = SinglePatchEncoderA(predictor).eval()
    part2 = SinglePatchEncoderB(predictor).eval()
    part3 = ImageEncoderPartA(predictor).eval()
    part4 = ImageEncoderPartB_Full(predictor).eval()
    full = FullModelWrapper(predictor).eval()

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    print(f"\n  Part 1 (SinglePatchEnc A):   {count_params(part1)/1e6:.0f}M params")
    print(f"  Part 2 (SinglePatchEnc B):   {count_params(part2)/1e6:.0f}M params")
    print(f"  Part 3 (ImageEnc A):         {count_params(part3)/1e6:.0f}M params")
    print(f"  Part 4 (ImageEnc B+Full):    {count_params(part4)/1e6:.0f}M params")

    # ---- PyTorch reference: simulate Android single-patch loop ----
    sample_image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print("\nRunning PyTorch single-patch pipeline (simulating Android)...")
    with torch.no_grad():
        # Create pyramid and split patches (done on Android side)
        normalizer = predictor.monodepth_model.monodepth_predictor.normalizer
        x_norm = normalizer(sample_image)
        x0 = x_norm
        x1 = F.interpolate(x_norm, scale_factor=0.5, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x_norm, scale_factor=0.25, mode="bilinear", align_corners=False)

        # But Part 1 includes normalizer, so pass un-normalized patches
        x0_raw = sample_image
        x1_raw = F.interpolate(sample_image, scale_factor=0.5, mode="bilinear", align_corners=False)
        x2_raw = F.interpolate(sample_image, scale_factor=0.25, mode="bilinear", align_corners=False)

        x0_patches = split_patches_list(x0_raw, overlap_ratio=0.25, patch_size=PATCH_SIZE)
        x1_patches = split_patches_list(x1_raw, overlap_ratio=0.5, patch_size=PATCH_SIZE)
        x2_patches = [x2_raw]
        all_patches = x0_patches + x1_patches + x2_patches
        print(f"  Patches: {len(x0_patches)} (x0) + {len(x1_patches)} (x1) + 1 (x2) = {len(all_patches)}")

        # Run Part 1 on each patch
        all_tokens = []
        all_block5 = []
        for i, patch in enumerate(all_patches):
            tokens_i, block5_i = part1(patch)
            all_tokens.append(tokens_i)
            all_block5.append(block5_i)
        print(f"  Part 1: {len(all_tokens)} x tokens {all_tokens[0].shape}")

        # Run Part 2 on each token set
        all_features = []
        for i, tokens_i in enumerate(all_tokens):
            feat_i = part2(tokens_i)
            all_features.append(feat_i)
        print(f"  Part 2: {len(all_features)} x features {all_features[0].shape}")

        # Reshape block5 and block11 (=tokens after Part 1) for merge
        all_block5_spatial = [reshape_feature(b5) for b5 in all_block5]
        all_block11_spatial = [reshape_feature(tk) for tk in all_tokens]

        # Merge patches (x0=first 25, x1=next 9, x2=last 1)
        latent0 = merge_patches_from_list(all_block5_spatial[:25], padding=3)
        latent1 = merge_patches_from_list(all_block11_spatial[:25], padding=3)
        x0_feat = merge_patches_from_list(all_features[:25], padding=3)
        x1_feat = merge_patches_from_list(all_features[25:34], padding=6)
        x2_feat = all_features[34]
        print(f"  Merge: latent0 {latent0.shape}, x0 {x0_feat.shape}, x1 {x1_feat.shape}, x2 {x2_feat.shape}")

        # Part 3
        image_tokens = part3(sample_image)
        print(f"  Part 3: image_tokens {image_tokens.shape}")

        # Part 4
        packed = part4(sample_image, image_tokens, latent0, latent1, x0_feat, x1_feat, x2_feat)
        gaussian_count = packed.shape[1]
        print(f"  Part 4: packed {packed.shape} ({gaussian_count:,} Gaussians)")

        # Full model reference
        reference = full(sample_image)

    # Validate
    max_diff = (reference - packed).abs().max().item()
    mean_diff = (reference - packed).abs().mean().item()
    print(f"\n  Single-patch pipeline vs Full model:")
    print(f"    Max diff:  {max_diff:.8f}")
    print(f"    Mean diff: {mean_diff:.8f}")
    if max_diff > 0.01:
        print("    WARNING: differs > 0.01")
    else:
        print("    OK — matches full model")

    # ---- Export all parts ----
    converter_flags = {
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "target_spec": {"supported_types": [tf.float16]},
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = {}

    # Part 1: single patch input
    sample_patch = torch.rand(1, 3, PATCH_SIZE, PATCH_SIZE)
    _, sizes["part1"] = export_part(
        "Part 1: Single-Patch Encoder A (blocks 0-11)",
        part1, (sample_patch,), converter_flags,
        output_dir / "sharp_part1_fp16.tflite",
    )

    # Part 2: single token set
    sample_tokens = torch.rand(1, 577, 1024)
    _, sizes["part2"] = export_part(
        "Part 2: Single-Patch Encoder B (blocks 12-23 + reshape)",
        part2, (sample_tokens,), converter_flags,
        output_dir / "sharp_part2_fp16.tflite",
    )

    # Part 3: full image
    _, sizes["part3"] = export_part(
        "Part 3: Image Encoder A (blocks 0-11)",
        part3, (sample_image,), converter_flags,
        output_dir / "sharp_part3_fp16.tflite",
    )

    # Part 4: full decoder
    _, sizes["part4"] = export_part(
        "Part 4: Image Encoder B + Full Decoder + Gaussians",
        part4, (sample_image, image_tokens, latent0, latent1, x0_feat, x1_feat, x2_feat),
        converter_flags,
        output_dir / "sharp_part4_fp16.tflite",
    )

    # ---- Summary ----
    total_mb = sum(sizes.values())
    elapsed = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"Export complete in {elapsed:.0f}s")
    print(f"{'=' * 60}")
    for name, size in sizes.items():
        status = "OK" if size < 500 else "OVER 500MB!"
        print(f"  {name}: {size:.0f} MB  [{status}]")
    print(f"  Total: {total_mb:.0f} MB")
    print(f"  Gaussians per image: {gaussian_count:,}")
    print(f"\nPush to Android:")
    print(f"  for i in 1 2 3 4; do")
    print(f"    adb push {output_dir}/sharp_part${{i}}_fp16.tflite /data/local/tmp/furnit/")
    print(f"  done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
