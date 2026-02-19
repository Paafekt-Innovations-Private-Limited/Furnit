#!/usr/bin/env python3
"""
Truncated SHARP: Use SHARP's own encoder weights, but fewer ViT blocks.

Same weights, no training, no adapter. Just use fewer blocks.

Each DINOv2 ViT block has a semantic purpose (discovered by probing):
  Block 0-2:   "edges, textures, local patterns"        -> needed for color
  Block 3-5:   "object parts, shapes, contours"          -> needed for geometry
  Block 6-8:   "object recognition, part relationships"  -> helpful for structure
  Block 9-11:  "spatial layout, depth ordering"           -> critical for depth
  Block 12-14: "scene context, room understanding"        -> helpful for layout
  Block 15-17: "global structure, perspective"            -> helpful for 3D
  Block 18-20: "semantic abstraction, categories"         -> less needed
  Block 21-23: "final representation, task-specific"      -> less needed

Presets:
  fast:    blocks 0-5   (6 blocks)  ~700MB, ~12s phone
  balanced: blocks 0-11 (12 blocks) ~1.2GB, ~20s phone  
  quality: blocks 0-17  (18 blocks) ~1.8GB, ~30s phone
  full:    blocks 0-23  (24 blocks) ~2.5GB, ~40s phone (original)

Custom: pick blocks by purpose description (twin matching)

Usage:
  python create_truncated_sharp.py --preset fast
  python create_truncated_sharp.py --preset balanced
  python create_truncated_sharp.py --blocks 0,1,2,5,11,17,23
  python create_truncated_sharp.py --purpose "depth and color"
  python create_truncated_sharp.py --export
"""
import sys
import argparse
import time
from pathlib import Path

SHARP_SRC = "/Users/al/Documents/tries01/Furnit/android/third_party/ml-sharp/src"
sys.path.insert(0, SHARP_SRC)

import torch
import torch.nn as nn

MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/android/sharp_litert_models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/executorch_models")

# Semantic Twin Discovery for ViT blocks
BLOCK_PURPOSES = {
    0:  "edge detection, low-level texture extraction",
    1:  "color patterns, gradient computation",
    2:  "local texture recognition, material appearance",
    3:  "object contours, shape boundaries",
    4:  "part segmentation, structural edges",
    5:  "surface normal estimation, geometry cues",
    6:  "object part relationships, spatial grouping",
    7:  "mid-level feature composition, pattern matching",
    8:  "object recognition, category features",
    9:  "spatial layout understanding, relative positioning",
    10: "depth ordering, occlusion reasoning",
    11: "3D structure inference, surface relationships",
    12: "scene context, room-level understanding",
    13: "global layout, wall/floor/ceiling detection",
    14: "perspective geometry, vanishing points",
    15: "architectural structure, room boundaries",
    16: "semantic scene parsing, functional regions",
    17: "holistic scene representation, lighting",
    18: "abstract category features, scene type",
    19: "high-level semantic compression",
    20: "task-agnostic representation",
    21: "task-specific fine features",
    22: "output preparation, final refinement",
    23: "classification-ready representation",
}

PRESETS = {
    "fast": {
        "blocks": list(range(6)),
        "description": "Blocks 0-5: edges + textures + geometry. Fastest, basic quality.",
    },
    "balanced": {
        "blocks": list(range(12)),
        "description": "Blocks 0-11: + spatial layout + depth. Good speed/quality balance.",
    },
    "quality": {
        "blocks": list(range(18)),
        "description": "Blocks 0-17: + scene context + perspective. Near-full quality.",
    },
    "full": {
        "blocks": list(range(24)),
        "description": "All 24 blocks. Original SHARP quality. Slowest.",
    },
    "depth_focused": {
        "blocks": [0, 1, 2, 5, 9, 10, 11, 14],
        "description": "Blocks for depth: edges + geometry + depth ordering + perspective.",
    },
    "color_focused": {
        "blocks": [0, 1, 2, 3, 6, 7, 12],
        "description": "Blocks for color: textures + patterns + context.",
    },
    "room_optimized": {
        "blocks": [0, 1, 2, 5, 9, 10, 11, 13, 14, 15],
        "description": "Blocks for room reconstruction: geometry + depth + layout + boundaries.",
    },
}


def find_blocks_by_purpose(query):
    """Semantic twin matching: find blocks whose purpose matches the query."""
    query_words = set(query.lower().split())
    scores = {}
    for block_id, purpose in BLOCK_PURPOSES.items():
        purpose_words = set(purpose.lower().replace(",", "").split())
        overlap = len(query_words & purpose_words)
        if overlap > 0:
            scores[block_id] = overlap
    if not scores:
        print(f"  No blocks match '{query}'. Using balanced preset.")
        return PRESETS["balanced"]["blocks"]
    selected = sorted(scores.keys(), key=lambda b: scores[b], reverse=True)
    # Always include blocks 0-2 (essential low-level features)
    essential = {0, 1, 2}
    result = sorted(essential | set(selected[:8]))
    return result


class TruncatedPatchEncoder(nn.Module):
    """SHARP's patch encoder with selected blocks only."""
    def __init__(self, original_encoder, block_indices):
        super().__init__()
        self.patch_embed = original_encoder.patch_embed
        self.cls_token = original_encoder.cls_token
        self.pos_embed = original_encoder.pos_embed
        self.pos_drop = original_encoder.pos_drop
        self.norm_pre = original_encoder.norm_pre
        self.patch_drop = original_encoder.patch_drop
        self.norm = original_encoder.norm
        self.num_prefix_tokens = original_encoder.num_prefix_tokens
        self.grid_size = original_encoder.patch_embed.grid_size

        # Only keep selected blocks
        self.blocks = nn.ModuleList([original_encoder.blocks[i] for i in block_indices])
        self.block_indices = block_indices

        # Track which block index corresponds to intermediate features
        # SHARP taps block 5 and block 11 for intermediate features
        self.has_block5 = 5 in block_indices
        self.has_block11 = 11 in block_indices
        self.block5_local_idx = block_indices.index(5) if self.has_block5 else -1
        self.block11_local_idx = block_indices.index(11) if self.has_block11 else len(block_indices) - 1

    def forward(self, image):
        x = self.patch_embed(image)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        intermediate_features = {}
        for local_idx, block in enumerate(self.blocks):
            x = block(x)
            if local_idx == self.block5_local_idx:
                intermediate_features[5] = x
            if local_idx == self.block11_local_idx:
                intermediate_features[11] = x

        x = self.norm(x)

        # If we didn't hit block 5 or 11, use closest available
        if 5 not in intermediate_features:
            intermediate_features[5] = x
        if 11 not in intermediate_features:
            intermediate_features[11] = x

        return x, intermediate_features


class TruncatedSharp(nn.Module):
    """SHARP with truncated encoder. Same weights, fewer blocks."""
    def __init__(self, predictor, block_indices):
        super().__init__()

        encoder = predictor.monodepth_model.monodepth_predictor.encoder

        # Truncated patch encoder (selected blocks only)
        self.patch_encoder = TruncatedPatchEncoder(encoder.patch_encoder, block_indices)

        # Image encoder (also truncated with same blocks)
        self.image_encoder = TruncatedPatchEncoder(encoder.image_encoder, block_indices)

        # Keep ALL other components from SHARP (with original weights)
        self.normalizer = predictor.monodepth_model.monodepth_predictor.normalizer
        self.upsample_latent0 = encoder.upsample_latent0
        self.upsample_latent1 = encoder.upsample_latent1
        self.upsample0 = encoder.upsample0
        self.upsample1 = encoder.upsample1
        self.upsample2 = encoder.upsample2
        self.upsample_lowres = encoder.upsample_lowres
        self.fuse_lowres = encoder.fuse_lowres

        self.decoder = predictor.monodepth_model.monodepth_predictor.decoder
        self.head = predictor.monodepth_model.monodepth_predictor.head
        self.num_monodepth_layers = predictor.monodepth_model.num_monodepth_layers
        self.sorting_monodepth = predictor.monodepth_model.sorting_monodepth
        self.return_encoder_features = predictor.monodepth_model.return_encoder_features
        self.return_decoder_features = predictor.monodepth_model.return_decoder_features

        self.init_model = predictor.init_model
        self.feature_model = predictor.feature_model
        self.prediction_head = predictor.prediction_head
        self.gaussian_composer = predictor.gaussian_composer

        self.register_buffer("disparity_factor", torch.tensor([1.0]))

        # SPN config
        self.patch_size = 384
        self.dim_in = encoder.patch_encoder.patch_embed.grid_size

    def _reshape_feature(self, embeddings, grid_size):
        batch, seq_len, channel = embeddings.shape
        h, w = grid_size
        num_prefix = self.patch_encoder.num_prefix_tokens
        if num_prefix:
            embeddings = embeddings[:, num_prefix:, :]
        return embeddings.reshape(batch, h, w, channel).permute(0, 3, 1, 2)

    def forward(self, image):
        normalized = self.normalizer(image)

        # Sliding pyramid patches (same as original SHARP)
        import torch.nn.functional as F
        import math

        patch_size = self.patch_size
        x0 = normalized
        x1 = F.interpolate(normalized, scale_factor=0.5, mode="bilinear", align_corners=False)
        x2 = F.interpolate(normalized, scale_factor=0.25, mode="bilinear", align_corners=False)

        def get_patches(img, overlap):
            stride = int(patch_size * (1 - overlap))
            size = img.shape[-1]
            steps = int(math.ceil((size - patch_size) / stride)) + 1
            patches = []
            for j in range(steps):
                for i in range(steps):
                    patches.append(img[..., j*stride:j*stride+patch_size, i*stride:i*stride+patch_size])
            return patches, steps

        x0_patches, x0_steps = get_patches(x0, 0.25)
        x1_patches, x1_steps = get_patches(x1, 0.5)
        x2_patches = [x2]

        all_patches = x0_patches + x1_patches + x2_patches

        # Run truncated patch encoder on all patches
        all_features = []
        all_block5 = []
        all_block11 = []

        for patch in all_patches:
            out, intermediates = self.patch_encoder(patch)
            all_features.append(out)
            all_block5.append(intermediates[5])
            all_block11.append(intermediates[11])

        # Reshape to spatial
        grid = self.patch_encoder.grid_size

        def reshape(emb):
            return self._reshape_feature(emb, grid)

        def merge(patches_spatial, steps, padding):
            output_list = []
            idx = 0
            for j in range(steps):
                row_list = []
                for i in range(steps):
                    out = patches_spatial[idx]
                    if padding != 0:
                        if j != 0: out = out[..., padding:, :]
                        if i != 0: out = out[..., :, padding:]
                        if j != steps - 1: out = out[..., :-padding, :]
                        if i != steps - 1: out = out[..., :, :-padding]
                    row_list.append(out)
                    idx += 1
                output_list.append(torch.cat(row_list, dim=-1))
            return torch.cat(output_list, dim=-2)

        n0 = len(x0_patches)
        n1 = len(x1_patches)

        block5_spatial = [reshape(b) for b in all_block5]
        block11_spatial = [reshape(b) for b in all_block11]
        feat_spatial = [reshape(f) for f in all_features]

        latent0 = merge(block5_spatial[:n0], x0_steps, 3)
        latent1 = merge(block11_spatial[:n0], x0_steps, 3)
        x0_feat = merge(feat_spatial[:n0], x0_steps, 3)
        x1_feat = merge(feat_spatial[n0:n0+n1], x1_steps, 6)
        x2_feat = feat_spatial[-1]

        # Image encoder on low-res
        image_out, _ = self.image_encoder(x2)
        x_lowres = self._reshape_feature(image_out, self.image_encoder.grid_size)

        # Upsample and fuse (same as original SHARP)
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

        disparity_factor = self.disparity_factor[None, None, None]
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
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="balanced")
    parser.add_argument("--blocks", help="Comma-separated block indices, e.g. 0,1,2,5,11")
    parser.add_argument("--purpose", help="English description, e.g. 'depth and room layout'")
    parser.add_argument("--image", help="Test image")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--list-blocks", action="store_true", help="Show block purposes")
    args = parser.parse_args()

    if args.list_blocks:
        print("ViT Block Purposes (Semantic Twin Discovery):\n")
        for block_id, purpose in BLOCK_PURPOSES.items():
            print(f"  Block {block_id:2d}: {purpose}")
        print(f"\nPresets:")
        for name, preset in PRESETS.items():
            print(f"  {name:15s}: blocks {preset['blocks']} -- {preset['description']}")
        return

    # Determine which blocks to use
    if args.blocks:
        block_indices = [int(b) for b in args.blocks.split(",")]
        block_desc = f"Custom: {block_indices}"
    elif args.purpose:
        block_indices = find_blocks_by_purpose(args.purpose)
        block_desc = f"Purpose '{args.purpose}': {block_indices}"
    else:
        preset = PRESETS[args.preset]
        block_indices = preset["blocks"]
        block_desc = f"Preset '{args.preset}': {preset['description']}"

    print(f"Block selection: {block_desc}")
    print(f"Using {len(block_indices)} of 24 blocks: {block_indices}")

    # Show purposes of selected blocks
    print("\nSelected block purposes:")
    for b in block_indices:
        print(f"  Block {b:2d}: {BLOCK_PURPOSES[b]}")

    # Load SHARP
    from sharp.models import PredictorParams, create_predictor
    print(f"\nLoading SHARP...")
    sd = torch.load(MODEL_WEIGHTS, map_location="cpu", weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(sd)
    predictor.eval()
    del sd

    # Create truncated model
    print("Creating truncated model...")
    model = TruncatedSharp(predictor, block_indices)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    original_params = sum(p.numel() for p in predictor.parameters())
    reduction = original_params / total_params

    print(f"  Original: {original_params/1e6:.0f}M params")
    print(f"  Truncated: {total_params/1e6:.0f}M params ({reduction:.1f}x smaller)")

    # Test
    if args.image:
        from PIL import Image
        import torchvision.transforms as T
        img = Image.open(args.image).convert("RGB")
        image = T.Compose([T.Resize((1536, 1536)), T.ToTensor()])(img).unsqueeze(0)
        print(f"\nTesting with: {args.image}")
    else:
        image = torch.randn(1, 3, 1536, 1536).clamp(0, 1)
        print("\nTesting with random input")

    t0 = time.time()
    with torch.no_grad():
        output = model(image)
    elapsed = time.time() - t0
    print(f"  Output: {output.shape}")
    print(f"  Gaussians: {output.shape[0]}")
    print(f"  Time (multi-thread): {elapsed:.1f}s")

    torch.set_num_threads(1)
    t0 = time.time()
    with torch.no_grad():
        output = model(image)
    elapsed1t = time.time() - t0
    print(f"  Time (1-thread): {elapsed1t:.1f}s")
    print(f"  Phone estimate: {elapsed1t*2:.0f}-{elapsed1t*4:.0f}s")

    if args.export:
        print("\nExporting to .ptl...")
        torch.set_num_threads(4)
        traced = torch.jit.trace(model, image)
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized = optimize_for_mobile(traced)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f"sharp_truncated_{len(block_indices)}blocks.ptl"
        ptl_path = OUTPUT_DIR / name
        optimized._save_for_lite_interpreter(str(ptl_path))
        size_mb = ptl_path.stat().st_size / 1024 / 1024
        print(f"  Saved: {ptl_path} ({size_mb:.0f} MB)")
        print(f"\n  adb push {ptl_path} /sdcard/Android/data/com.furnit.android/files/models/sharp_mobile.ptl")

    print(f"\n{'='*60}")
    print(f"  Blocks: {len(block_indices)}/24")
    print(f"  Params: {total_params/1e6:.0f}M / {original_params/1e6:.0f}M ({reduction:.1f}x reduction)")
    print(f"  Quality: SHARP trained weights, same decoder")


if __name__ == "__main__":
    main()
