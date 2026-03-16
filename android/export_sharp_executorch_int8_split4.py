#!/usr/bin/env python3
"""
Export SHARP as 4 split INT8 ExecuTorch .pte parts with XNNPACK backend.

Each part is individually quantized via PT2E + XNNPACKQuantizer, then exported
with greedy AOT memory planning. INT8 reduces weight bandwidth by ~4x, directly
targeting the 96.7% Conv bottleneck identified by ORT profiling.

Output files:
  sharp_split_part1_int8.pte  (~145MB)  -- run 35x per patch
  sharp_split_part2_int8.pte  (~145MB)  -- run 35x per patch
  sharp_split_part3_int8.pte  (~145MB)  -- run 1x on full image
  sharp_split_part4b_int8.pte (optional) -- Part4b decoder INT8; C++ uses if present
  sharp_split_part4b_tile_00..15.pte -- Part4b INT8 tiles (C++ tiled path)
  sharp_split_part4b_tile_b4.pte -- Batched tile (batch=4, 4 forward calls instead of 16)
  sharp_split_part4_int8.pte  (~190MB)  -- full Part4 FP32 (app uses chunked Part4a+Part4b)

Same 4-part pipeline as FP32/FP16 split:
  1. Part 1 on 35 patches -> tokens, block5
  2. Part 2 on 35 token sets -> features
  3. Merge patches -> latent0, latent1, x0_feat, x1_feat, x2_feat
  4. Part 3 on full image -> image_tokens
  5. Part 4 -> packed [1, N, 14] Gaussians

Usage:
  cd android
  python export_sharp_executorch_int8_split4.py
  # Then push all 4 .pte files to device
"""

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 1536
PATCH_SIZE = 384
VIT_SPLIT_BLOCK = 12
FEATURE_DIM = 1024
SPATIAL_SIZE = 24
GRID_1X = 5
GRID_05X = 3
PADDING_1X = 3
PADDING_05X = 6


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


def quantize_and_export_pte(name, wrapper, sample_inputs, output_path, backend="xnnpack"):
    """Quantize a part with PT2E INT8 and export to .pte.
    backend: 'xnnpack' (INT8 CPU) or 'vulkan' (INT8 quantized graph lowered to Vulkan GPU when supported)."""
    from executorch.exir import EdgeCompileConfig

    backend_label = "Vulkan GPU" if backend == "vulkan" else "XNNPACK"
    print(f"\n{'='*60}")
    print(f"Exporting {name} (INT8 + {backend_label})")
    print(f"{'='*60}")

    start = time.time()

    # Step 1: Export to ATen IR
    print("  Exporting to ATen IR...")
    exported = torch.export.export(wrapper, sample_inputs, strict=False)

    # Step 2: PT2E quantization — use matched pair from torch.ao (quantizer + prepare/convert
    # must come from the same package; executorch's XNNPACKQuantizer produces QuantizationSpec
    # objects incompatible with torch.ao.quantization.quantize_pt2e.prepare_pt2e).
    print("  Quantizing (PT2E INT8 symmetric per-channel dynamic)...")
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    )

    graph_module = exported.module()
    prepared = prepare_pt2e(graph_module, quantizer)

    # Calibrate with sample input
    with torch.no_grad():
        prepared(*sample_inputs)
    print("  Calibration done")

    quantized = convert_pt2e(prepared)
    print("  INT8 conversion done")

    # Step 3: Re-export quantized model
    print("  Re-exporting quantized model...")
    quantized_exported = torch.export.export(quantized, sample_inputs, strict=False)

    # Step 4: Backend partitioning (XNNPACK or Vulkan)
    if backend == "vulkan":
        try:
            from executorch.exir import to_edge_transform_and_lower
            from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
            edge = to_edge_transform_and_lower(
                quantized_exported,
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
                partitioner=[VulkanPartitioner()],
            )
            print("  Vulkan delegate applied (INT8 quantized graph)")
        except Exception as e:
            print(f"  Vulkan partition failed: {e}")
            raise
        # Vulkan does its own AOT memory planning; skip greedy to avoid "TensorSpec memory offset" errors
        et_program = edge.to_executorch()
    else:
        try:
            from executorch.exir import to_edge_transform_and_lower
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            edge = to_edge_transform_and_lower(
                quantized_exported,
                partitioner=[XnnpackPartitioner()],
            )
            print("  XNNPACK delegate applied")
        except Exception as e:
            print(f"  to_edge_transform_and_lower failed ({e}), using legacy API")
            from executorch.exir import to_edge
            edge = to_edge(quantized_exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))
            try:
                from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
                edge = edge.to_backend(XnnpackPartitioner())
            except Exception:
                pass

        # Step 5: Greedy memory planning (reuse shared helper from FP32 export)
        from export_sharp_executorch_split4 import _apply_greedy_memory_planning
        et_program = _apply_greedy_memory_planning(edge)

    export_time = time.time() - start

    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  INT8 export: {export_time:.0f}s")
    print(f"  Saved: {output_path.name} ({size_mb:.0f} MB)")
    return size_mb


# ---- Part wrappers (same as export_sharp_executorch_split4.py) ----

class SinglePatchEncoderA(nn.Module):
    """Part 1: Normalizer + patch_embed + blocks[0:11]"""
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

    def forward(self, patch):
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


class SinglePatchEncoderB(nn.Module):
    """Part 2: blocks[12:23] + norm + reshape"""
    def __init__(self, predictor):
        super().__init__()
        spn = predictor.monodepth_model.monodepth_predictor.encoder
        pe = spn.patch_encoder
        self.blocks = nn.ModuleList(list(pe.blocks[VIT_SPLIT_BLOCK:]))
        self.norm = pe.norm
        self.num_prefix_tokens = pe.num_prefix_tokens
        self.grid_size = pe.patch_embed.grid_size

    def forward(self, tokens):
        x = tokens
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if self.num_prefix_tokens:
            x = x[:, self.num_prefix_tokens:, :]
        B, N, C = x.shape
        h, w = self.grid_size
        return x.reshape(B, h, w, C).permute(0, 3, 1, 2)


class ImageEncoderPartA(nn.Module):
    """Part 3: Image encoder blocks 0-11"""
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

    def forward(self, image):
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


class ImageEncoderPartBFull(nn.Module):
    """Part 4: Image encoder B + decoder + Gaussian output"""
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
        self.register_buffer("disparity_factor", torch.ones(1, 1, 1, 1))

    def _reshape_feature(self, embeddings):
        batch, seq_len, channel = embeddings.shape
        h, w = self.grid_size
        if self.num_prefix_tokens:
            embeddings = embeddings[:, self.num_prefix_tokens:, :]
        return embeddings.reshape(batch, h, w, channel).permute(0, 3, 1, 2)

    def forward(self, image_tokens, image, latent0, latent1, x0_feat, x1_feat, x2_feat):
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
        monodepth = self.disparity_factor / disparity.clamp(min=1e-4, max=1e4)
        init_output = self.init_model(image, monodepth)
        image_features = self.feature_model(init_output.feature_input, encodings=output_features)
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
        return torch.cat([positions, opacities, scales, quaternions, colors], dim=-1)


# ---- Helpers for validation ----

def split_patches_list(image, overlap_ratio, patch_size):
    patch_stride = int(patch_size * (1 - overlap_ratio))
    image_size = image.shape[-1]
    steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1
    patches = []
    for j in range(steps):
        for i in range(steps):
            j0 = j * patch_stride
            i0 = i * patch_stride
            patches.append(image[..., j0:j0+patch_size, i0:i0+patch_size])
    return patches


def reshape_feature(tokens, grid_h=24, grid_w=24, num_prefix=1):
    if num_prefix:
        tokens = tokens[:, num_prefix:, :]
    B, N, C = tokens.shape
    return tokens.reshape(B, grid_h, grid_w, C).permute(0, 3, 1, 2)


def merge_patches_from_list(features_list, padding=3, grid_h=24, grid_w=24):
    n = len(features_list)
    grid = int(math.sqrt(n))
    C = features_list[0].shape[1]
    inner_h = grid_h - 2 * padding if padding else grid_h
    inner_w = grid_w - 2 * padding if padding else grid_w
    out_h = grid * inner_h
    out_w = grid * inner_w
    merged = torch.zeros(1, C, out_h, out_w, device=features_list[0].device)
    for idx, feat in enumerate(features_list):
        row = idx // grid
        col = idx % grid
        src = feat[:, :, padding:padding+inner_h, padding:padding+inner_w] if padding else feat
        merged[:, :, row*inner_h:(row+1)*inner_h, col*inner_w:(col+1)*inner_w] = src
    return merged


def parse_args():
    pa = argparse.ArgumentParser(description="Export SHARP 4-part INT8 split to ExecuTorch .pte")
    pa.add_argument("--sharp-src",
        default=str(Path(__file__).resolve().parent / "third_party/ml-sharp/src"))
    pa.add_argument("--weights",
        default=str(Path(__file__).resolve().parent / "sharp_litert_models/sharp_2572gikvuh.pt"))
    pa.add_argument("--output-dir",
        default=str(Path(__file__).resolve().parent / "executorch_int8_models"),
        help="Output directory")
    pa.add_argument("--backend", choices=("xnnpack", "vulkan"), default="xnnpack",
        help="Backend: xnnpack (CPU INT8), vulkan (GPU; app must have executorch-android-vulkan)")
    return pa.parse_args()


def main():
    args = parse_args()
    sharp_src = Path(args.sharp_src)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)

    backend = args.backend
    print("=" * 60)
    print("ExecuTorch INT8 Split 4-Part Export")
    print("=" * 60)
    print(f"  Backend: {backend.upper()} ({'Vulkan GPU' if backend == 'vulkan' else 'XNNPACK CPU'})")
    print(f"  Memory planning: greedy AOT (XNNPACK); Vulkan default (vulkan)")
    print(f"  Quantization: PT2E symmetric per-channel dynamic INT8")

    if not sharp_src.exists():
        print(f"ERROR: SHARP source not found at {sharp_src}")
        return 1
    if not weights_path.exists():
        print(f"ERROR: Weights not found at {weights_path}")
        return 1

    sys.path.insert(0, str(sharp_src))
    from sharp.models import PredictorParams, create_predictor

    print("\nLoading SHARP...")
    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    del state_dict

    print("  Fusing Conv+BN layers...")
    fuse_conv_bn(predictor)

    part1 = SinglePatchEncoderA(predictor).eval()
    part2 = SinglePatchEncoderB(predictor).eval()
    part3 = ImageEncoderPartA(predictor).eval()
    part4 = ImageEncoderPartBFull(predictor).eval()

    print(f"  Part 1: {sum(p.numel() for p in part1.parameters())/1e6:.0f}M params")
    print(f"  Part 2: {sum(p.numel() for p in part2.parameters())/1e6:.0f}M params")
    print(f"  Part 3: {sum(p.numel() for p in part3.parameters())/1e6:.0f}M params")
    print(f"  Part 4: {sum(p.numel() for p in part4.parameters())/1e6:.0f}M params")

    # Generate sample tensors at the shapes the decoder expects (from ONNX model profiling).
    # The decoder's skip connections have hardcoded spatial dependencies:
    #   1x merge: 96x96 (not 90 from simplified merge), 0.5x merge: 48x48 (not 36)
    sample_image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    print("\nGenerating sample intermediate tensors (ONNX-validated shapes)...")
    with torch.no_grad():
        image_tokens = part3(sample_image)
    latent0 = torch.rand(1, FEATURE_DIM, 96, 96)
    latent1 = torch.rand(1, FEATURE_DIM, 96, 96)
    x0_feat = torch.rand(1, FEATURE_DIM, 96, 96)
    x1_feat = torch.rand(1, FEATURE_DIM, 48, 48)
    x2_feat = torch.rand(1, FEATURE_DIM, SPATIAL_SIZE, SPATIAL_SIZE)
    print(f"  image_tokens: {image_tokens.shape}")
    print(f"  latent0: {latent0.shape}, x1_feat: {x1_feat.shape}, x2_feat: {x2_feat.shape}")

    # Export all 4 parts as INT8
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    total_start = time.time()

    sample_patch = torch.rand(1, 3, PATCH_SIZE, PATCH_SIZE)
    sizes["part1"] = quantize_and_export_pte(
        "Part 1: Patch Encoder A (blocks 0-11)",
        part1, (sample_patch,),
        output_dir / "sharp_split_part1_int8.pte",
        backend=backend,
    )

    sample_tokens = torch.rand(1, 577, 1024)
    sizes["part2"] = quantize_and_export_pte(
        "Part 2: Patch Encoder B (blocks 12-23)",
        part2, (sample_tokens,),
        output_dir / "sharp_split_part2_int8.pte",
        backend=backend,
    )

    # Part1/Part2 batch=4 for native patch batching (35 → ~11 launches)
    PATCH_BATCH = 4
    sample_patch_b4 = torch.rand(PATCH_BATCH, 3, PATCH_SIZE, PATCH_SIZE)
    sample_tokens_b4 = torch.rand(PATCH_BATCH, 577, FEATURE_DIM)
    try:
        sizes["part1_b4"] = quantize_and_export_pte(
            f"Part 1 batch={PATCH_BATCH} (patch encoder)",
            part1, (sample_patch_b4,),
            output_dir / "sharp_split_part1_b4_int8.pte",
            backend=backend,
        )
        sizes["part2_b4"] = quantize_and_export_pte(
            f"Part 2 batch={PATCH_BATCH} (patch encoder B)",
            part2, (sample_tokens_b4,),
            output_dir / "sharp_split_part2_b4_int8.pte",
            backend=backend,
        )
        print(f"  Part1/Part2 batch={PATCH_BATCH} export OK (for native patch batching)")
    except Exception as e:
        print(f"  Part1/Part2 b4 export failed: {e}")
        sizes["part1_b4"] = sizes.get("part1_b4", 0.0)
        sizes["part2_b4"] = sizes.get("part2_b4", 0.0)

    sizes["part3"] = quantize_and_export_pte(
        "Part 3: Image Encoder A (blocks 0-11)",
        part3, (sample_image,),
        output_dir / "sharp_split_part3_int8.pte",
        backend=backend,
    )

    # Part 4b (decoder + Gaussians): try INT8 first.
    from export_sharp_executorch_split4 import (
        export_pte,
        ImageEncoderPartBFromTokens,
        ImageEncoderPartBFromTileInputs,
        get_part4b_tile_sample_inputs,
    )
    part4b = ImageEncoderPartBFromTokens(predictor).eval()
    tokens_after_blocks = torch.rand(1, 577, FEATURE_DIM)
    part4b_inputs = (tokens_after_blocks, sample_image, latent0, latent1, x0_feat, x1_feat, x2_feat)
    try:
        sizes["part4b_int8"] = quantize_and_export_pte(
            "Part 4b: Decoder + Gaussians (INT8)",
            part4b, part4b_inputs,
            output_dir / "sharp_split_part4b_int8.pte",
            backend=backend,
        )
        print("  Part4b INT8 export OK")
    except Exception as e:
        print(f"  Part4b INT8 export failed: {e}")
        sizes["part4b_int8"] = 0.0

    # Part 4b INT8 tiles (batch=1): export one, copy to tile_00..tile_15.
    NUM_PART4B_TILES = 16
    part4b_tile = ImageEncoderPartBFromTileInputs(predictor).eval()
    part4b_tile_inputs = get_part4b_tile_sample_inputs(batch_size=1)
    tile_00_path = output_dir / "sharp_split_part4b_tile_00.pte"
    try:
        size_tile = quantize_and_export_pte(
            "Part 4b tile (INT8, batch=1)",
            part4b_tile, part4b_tile_inputs,
            tile_00_path,
            backend=backend,
        )
        sizes["part4b_tiles_int8"] = size_tile * NUM_PART4B_TILES
        for i in range(1, NUM_PART4B_TILES):
            dest = output_dir / f"sharp_split_part4b_tile_{i:02d}.pte"
            shutil.copy2(tile_00_path, dest)
        print(f"  Part4b INT8 tiles: 16 files, {size_tile:.0f} MB each")
    except Exception as e:
        print(f"  Part4b INT8 tile export failed: {e}")
        sizes["part4b_tiles_int8"] = 0.0

    # Part 4b INT8 batched tile (batch=4): 4 tiles per forward call = 4 calls for 16 tiles.
    BATCH_SIZE = 4
    part4b_tile_b4 = ImageEncoderPartBFromTileInputs(predictor).eval()
    part4b_tile_b4_inputs = get_part4b_tile_sample_inputs(batch_size=BATCH_SIZE)
    tile_b4_path = output_dir / "sharp_split_part4b_tile_b4.pte"
    try:
        sizes["part4b_tile_b4"] = quantize_and_export_pte(
            f"Part 4b tile (INT8, batch={BATCH_SIZE})",
            part4b_tile_b4, part4b_tile_b4_inputs,
            tile_b4_path,
            backend=backend,
        )
        print(f"  Part4b INT8 batched tile (batch={BATCH_SIZE}): {sizes['part4b_tile_b4']:.0f} MB")
    except Exception as e:
        print(f"  Part4b INT8 batched tile export failed: {e}")
        sizes["part4b_tile_b4"] = 0.0

    # Part 4 full (single .pte): export FP32 for compatibility.
    sizes["part4"] = export_pte(
        "Part 4: Decoder + Gaussians (FP32, greedy planning)",
        part4, (image_tokens, sample_image, latent0, latent1, x0_feat, x1_feat, x2_feat),
        output_dir / "sharp_split_part4_int8.pte",
        use_fp16=False,
        backend=backend,
        use_greedy_memory_planning=(backend != "vulkan"),
    )

    total_time = time.time() - total_start
    total_size = sum(sizes.values())

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, size in sizes.items():
        print(f"  {name}: {size:.0f} MB")
    print(f"  Total: {total_size:.0f} MB (vs ~2.4GB FP32, ~1.2GB FP16)")
    print(f"  Export time: {total_time:.0f}s")
    print(f"\nPush to device:")
    print(f"  ./push_sharp_executorch_int8_models.sh")
    if tile_b4_path.exists():
        print(f"  (sharp_split_part4b_tile_b4.pte present – C++ batched tile path: 4 forward calls)")
    if tile_00_path.exists():
        print(f"  (tile_00..15.pte present – C++ sequential tile path: 16 forward calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
