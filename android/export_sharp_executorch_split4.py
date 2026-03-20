#!/usr/bin/env python3
"""
Export SHARP as 4 split ExecuTorch .pte parts (mirroring LiteRT split).

Backend: Vulkan (default) or portable only. XNNPACK is disabled — it causes SIGSEGV
(XNNWeightsCache::look_up_or_insert) on Android for large parts.

Same 4-part architecture as export_sharp_litert_split.py:
  Part 1: Single-Patch Encoder A (blocks 0-11)  ~582MB FP32
  Part 2: Single-Patch Encoder B (blocks 12-23)  ~577MB FP32
  Part 3: Image Encoder A (blocks 0-11)  ~582MB FP32
  Part 4: Image Encoder B + Full Decoder + Gaussians  ~755MB FP32

Android pipeline (same as LiteRT):
  1. Run Part 1 on 35 patches -> tokens, block5  (35 runs)
  2. Run Part 2 on 35 token sets -> features  (35 runs)
  3. Merge patches -> latent0, latent1, x0_feat, x1_feat, x2_feat
  4. Run Part 3 on full image -> image_tokens  (1 run)
  5. Run Part 4 -> packed [1, N, 14] Gaussians  (1 run)

Part 4 is exported with greedy memory planning (portable) or Vulkan AOT (vulkan).
Runtime uses mmap load + zero-copy output for Part 4.

Export (Vulkan, recommended):
  cd android
  ./export_sharp_vulkan_only.sh
  ./push_sharp_vulkan_only.sh

Or manually:
  python export_sharp_executorch_split4.py --backend vulkan --chunked-part4 --dtype fp16 --output-dir sharp_vulkan_only

Vulkan optional fixes (opt-in; default export uses strict=False, no IR check, Part4 FP32 so export succeeds):
  --strict-export       strict=True in torch.export (can surface graph/side-effect issues; may fail).
  --check-ir-validity   Enable IR validity in EdgeCompileConfig.
  --unify-fp16          Export Part4 and chunked Part4 as FP16 with Vulkan FP16 (dtype mismatch risk in Part4).
  --vulkan-aar-compat   Export Vulkan with FP32 + force_fp16=False so .pte only uses shaders in executorch-android-vulkan 1.1.0 AAR (avoids missing view_convert_buffer_float_half). Use same ExecuTorch version as AAR when exporting.
Each Vulkan export prints [Partition] Vulkan string count for diagnostics.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 1536
PATCH_SIZE = 384
VIT_SPLIT_BLOCK = 12


def fuse_conv_bn(model):
    """Recursively fuse Conv2d+BatchNorm2d pairs in eval mode.

    Folding BN into Conv eliminates intermediate normalization tensors from the
    graph, reducing peak activation memory by ~15-20% in the decoder pass and
    producing a smaller, faster model for all backends (ExecuTorch, ONNX, LiteRT).
    """
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


def parse_args():
    pa = argparse.ArgumentParser(
        description="Export SHARP 4-part split to ExecuTorch .pte (Vulkan or portable; XNNPACK removed)."
    )
    pa.add_argument("--sharp-src",
        default=str(Path(__file__).resolve().parent / "third_party/ml-sharp/src"))
    pa.add_argument("--weights",
        default=str(Path(__file__).resolve().parent / "sharp_litert_models/sharp_2572gikvuh.pt"))
    pa.add_argument("--output-dir",
        default=str(Path(__file__).resolve().parent / "executorch_models"),
        help="Output directory (default: executorch_models)")
    pa.add_argument("--backend", choices=("vulkan", "portable"), default="vulkan",
        help="Backend: vulkan (GPU, recommended; avoids XNNPACK SIGSEGV), portable (CPU fallback). Default: vulkan")
    pa.add_argument("--vulkan", action="store_true",
        help="Use Vulkan GPU backend (same as --backend vulkan)")
    pa.add_argument("--no-xnnpack", action="store_true",
        help="Use portable CPU backend (same as --backend portable). Kept for backward compat.")
    pa.add_argument("--chunked-part4", action="store_true",
        help="Also export chunked Part 4 (4a_chunk_512, 4a_chunk_65, 4b) for lower peak RAM on decoder.")
    pa.add_argument("--chunked-part4-only", action="store_true",
        help="Export ONLY chunked Part4 (sharp_split_part4a_chunk_512/65 + sharp_split_part4b.pte). "
             "Skips Part1–3 and monolithic part4. Use with --backend portable to generate missing Part4b for etCpu. "
             "Implies --chunked-part4. Still loads SHARP weights and runs split validation (needs RAM + time for Part4b export).")
    pa.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32",
        help="Export dtype: fp32 (default) or fp16. FP16 Vulkan avoids INT8 staging crashes on many devices.")
    pa.add_argument("--patch-batch-size", type=int, choices=(1, 2, 4), default=1,
        help="Export Part1/Part2 with this patch batch size (2 or 4). B2 Vulkan FP16 = 95%% success; B4 can crash.")
    pa.add_argument("--part12-only-portable", action="store_true",
        help="Export only Part1 and Part2 as portable (CPU): sharp_split_part1.pte, sharp_split_part2.pte.")
    pa.add_argument("--part1-only", action="store_true",
        help="Export only Part 1; also save one fixed test patch and Python golden outputs for app comparison.")
    pa.add_argument("--strict-export", action="store_true",
        help="Use strict=True in torch.export (may fail on side effects or dynamo; use to debug).")
    pa.add_argument("--check-ir-validity", action="store_true",
        help="Enable IR validity checks in EdgeCompileConfig (use to debug Vulkan).")
    pa.add_argument("--unify-fp16", action="store_true",
        help="Export Part4 and chunked Part4 as FP16 when Vulkan FP16 (risk: dtype mismatch in Part4).")
    pa.add_argument("--verify-delegate", action="store_true",
        help="After export, run delegate inspector on Part1 .pte to verify Vulkan/portable backend.")
    pa.add_argument("--image-size", type=int, choices=(1536, 1280), default=1536,
        help="Full image size for Part3/Part4 export (1536 default; 1280 for reduced memory). Part1/Part2 use 384 patches.")
    pa.add_argument("--vulkan-aar-compat", action="store_true",
        help="Vulkan export compatible with executorch-android-vulkan 1.1.0 AAR: FP32 + force_fp16=False to avoid "
             "missing view_convert_buffer_float_half shader. Use same ExecuTorch version as AAR when exporting.")
    return pa.parse_args()


# Reuse same Part wrappers from LiteRT split (import or inline)
# Inlined here to be self-contained

class SinglePatchEncoderA(nn.Module):
    """Part 1: Normalizer + patch_embed + blocks[0:11]
    Input: [1, 3, 384, 384] -> Output: (tokens [1,577,1024], block5 [1,577,1024])"""
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
    """Part 2: blocks[12:23] + norm + reshape
    Input: [1,577,1024] -> Output: [1,1024,24,24]"""
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
    """Part 3: Image encoder blocks 0-11
    Input: [1,3,1536,1536] -> Output: [1,577,1024]"""
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


class ImageEncoderPartBChunk(nn.Module):
    """Part 4a chunk: ViT blocks 12-23 + norm on a token slice only.
    Input: tokens_slice [1, chunk_len, 1024] -> Output: [1, chunk_len, 1024].
    Run 2x (chunk 0:512 and 512:577), then concat for Part 4b."""
    def __init__(self, predictor, chunk_len):
        super().__init__()
        self.chunk_len = chunk_len
        spn = predictor.monodepth_model.monodepth_predictor.encoder
        ie = spn.image_encoder
        self.blocks = nn.ModuleList(list(ie.blocks[VIT_SPLIT_BLOCK:]))
        self.norm = ie.norm

    def forward(self, tokens_slice):
        x = tokens_slice
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class ImageEncoderPartBFromTokens(nn.Module):
    """Part 4b: from tokens (after ViT 12-23) through reshape, fuse, decoder, head -> [1,N,14].
    Input: tokens_after_blocks [1,577,1024] + image + 5 feature maps -> Output: [1,N,14]."""
    def __init__(self, predictor):
        super().__init__()
        spn = predictor.monodepth_model.monodepth_predictor.encoder
        ie = spn.image_encoder
        mono = predictor.monodepth_model
        self.norm = ie.norm  # unused here; tokens are already normalized from chunk
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

    def _reshape_feature(self, embeddings):
        batch, seq_len, channel = embeddings.shape
        h, w = self.grid_size
        if self.num_prefix_tokens:
            embeddings = embeddings[:, self.num_prefix_tokens:, :]
        # .contiguous() so ExecuTorch cat(x2_up, x_lowres_up) sees same dim order (avoids "2 input tensors have different dim orders").
        return embeddings.reshape(batch, h, w, channel).permute(0, 3, 1, 2).contiguous()

    def forward(self, tokens_after_blocks, image, latent0, latent1, x0_feat, x1_feat, x2_feat):
        x_lowres = self._reshape_feature(tokens_after_blocks)
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


class ImageEncoderPartBFull(nn.Module):
    """Part 4: Image encoder B + decoder + Gaussian output
    Input: image + image_tokens + 5 feature maps -> Output: [1,N,14]"""
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

    def _reshape_feature(self, embeddings):
        batch, seq_len, channel = embeddings.shape
        h, w = self.grid_size
        if self.num_prefix_tokens:
            embeddings = embeddings[:, self.num_prefix_tokens:, :]
        # .contiguous() so ExecuTorch cat(x2_up, x_lowres_up) sees same dim order (avoids "2 input tensors have different dim orders").
        return embeddings.reshape(batch, h, w, channel).permute(0, 3, 1, 2).contiguous()

    def forward(self, image, image_tokens, latent0, latent1, x0_feat, x1_feat, x2_feat):
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


# Tile config: 4x4 grid, same input shapes as C++ runPart4bTiledFullPipeline.
PART4B_TILE_GRID = 4
PART4B_TILE_IMG_H = IMAGE_SIZE // PART4B_TILE_GRID   # 384
PART4B_TILE_IMG_W = IMAGE_SIZE // PART4B_TILE_GRID   # 384
PART4B_TILE_LAT_HW = 24   # 96/4
PART4B_TILE_X1_HW = 12    # 48/4
PART4B_TILE_X2_HW = 6     # 24/4


class ImageEncoderPartBFromTileInputs(nn.Module):
    """Part 4b tile: decoder + Gaussians from pre-cropped tile inputs (no tokens).
    Same modules as ImageEncoderPartBFromTokens; input is already spatial per-tile.
    Inputs: image [B,3,384,384], lat0, lat1, x0 [B,1024,24,24], x1 [B,1024,12,12], x2, x_lowres [B,1024,6,6].
    Output: [B, N_tile, 14] packed Gaussians."""
    def __init__(self, predictor):
        super().__init__()
        spn = predictor.monodepth_model.monodepth_predictor.encoder
        mono = predictor.monodepth_model
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

    def forward(self, image, lat0, lat1, x0_feat, x1_feat, x2_feat, x_lowres):
        latent0_up = self.upsample_latent0(lat0)
        latent1_up = self.upsample_latent1(lat1)
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


def get_part4b_tile_sample_inputs(batch_size=1):
    """Sample inputs for Part4b tile export. batch_size=1 for sequential, 4 for batched."""
    img_tile = torch.rand(batch_size, 3, PART4B_TILE_IMG_H, PART4B_TILE_IMG_W)
    lat_tile = torch.rand(batch_size, 1024, PART4B_TILE_LAT_HW, PART4B_TILE_LAT_HW)
    x1_tile = torch.rand(batch_size, 1024, PART4B_TILE_X1_HW, PART4B_TILE_X1_HW)
    x2_tile = torch.rand(batch_size, 1024, PART4B_TILE_X2_HW, PART4B_TILE_X2_HW)
    return (img_tile, lat_tile, lat_tile.clone(), lat_tile.clone(), x1_tile, x2_tile, x2_tile.clone())


def split_patches_list(image, overlap_ratio, patch_size, patch_stride=None):
    if patch_stride is None:
        patch_stride = int(patch_size * (1 - overlap_ratio))
    image_size = image.shape[-1]
    steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1
    patches = []
    for j in range(steps):
        j0 = j * patch_stride
        for i in range(steps):
            i0 = i * patch_stride
            patches.append(image[..., j0:j0+patch_size, i0:i0+patch_size])
    return patches


def merge_patches_from_list(patches, padding):
    steps = int(math.sqrt(len(patches)))
    output_list = []
    idx = 0
    for j in range(steps):
        row_list = []
        for i in range(steps):
            out = patches[idx]
            if padding != 0:
                if j != 0: out = out[..., padding:, :]
                if i != 0: out = out[..., :, padding:]
                if j != steps - 1: out = out[..., :-padding, :]
                if i != steps - 1: out = out[..., :, :-padding]
            row_list.append(out)
            idx += 1
        output_list.append(torch.cat(row_list, dim=-1))
    return torch.cat(output_list, dim=-2)


def reshape_feature(embeddings, num_prefix_tokens=1, grid_size=(24, 24)):
    if num_prefix_tokens:
        embeddings = embeddings[:, num_prefix_tokens:, :]
    B, N, C = embeddings.shape
    h, w = grid_size
    return embeddings.reshape(B, h, w, C).permute(0, 3, 1, 2)


def _apply_greedy_memory_planning(edge):
    """Apply greedy AOT memory planning to an EdgeProgramManager.

    greedy() returns MemoryAlgoResult; MemoryPlanningAlgorithmSuite wraps it and
    writes offsets back to TensorSpecs. Pass None as memory_planning_algo so the
    default Suite([greedy]) is used.
    """
    ExecutorchBackendConfig = None
    MPPass = None
    for api_idx in range(2):
        try:
            if api_idx == 0:
                from executorch.exir.capture._config import ExecutorchBackendConfig as _Cfg
                from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass as _MP
            else:
                from executorch.exir import ExecutorchBackendConfig as _Cfg
                from executorch.exir.passes import MemoryPlanningPass as _MP
            ExecutorchBackendConfig = _Cfg
            MPPass = _MP
            break
        except ImportError:
            continue

    if ExecutorchBackendConfig is None or MPPass is None:
        print("  Greedy memory planning: imports unavailable, using default")
        return edge.to_executorch()

    try:
        et_program = edge.to_executorch(
            ExecutorchBackendConfig(memory_planning_pass=MPPass())
        )
        print("  Greedy memory planning applied (suite default)")
        return et_program
    except Exception as e1:
        print(f"  Greedy suite (default alloc) failed: {e1}")

    try:
        et_program = edge.to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MPPass(
                    alloc_graph_input=False,
                    alloc_graph_output=False,
                ),
            )
        )
        print("  Greedy memory planning applied (caller-managed I/O)")
        return et_program
    except Exception as e2:
        print(f"  Greedy suite (caller-managed) failed: {e2}")

    print("  Greedy memory planning unavailable, using default planning")
    return edge.to_executorch()


def export_pte(name, wrapper, sample_inputs, output_path, use_fp16=True, backend="vulkan", use_greedy_memory_planning=False,
              strict_export=False, check_ir_validity=False, vulkan_compile_options=None):
    """Export a single part to .pte format.
    backend: "vulkan" (GPU, 20-60s) or "portable" (CPU fallback, 10min+)
    use_greedy_memory_planning: use ExecuTorch greedy memory planner (reuse buffers, lower peak RAM). Recommended for Part 4.
    strict_export: use strict=True in torch.export (catches graph issues).
    check_ir_validity: enable IR validity checks in EdgeCompileConfig (recommended when debugging Vulkan).
    vulkan_compile_options: dict for VulkanPartitioner (e.g. {"force_fp16": False}) for AAR shader compatibility.
    """
    from executorch.exir import EdgeCompileConfig

    backend_label = {"vulkan": "+ Vulkan GPU", "portable": "(portable only)"}.get(backend, "+ Vulkan GPU")
    planning_label = " + greedy memory planning" if use_greedy_memory_planning else ""
    strict_label = " [strict export]" if strict_export else ""
    ir_label = " [IR validity ON]" if check_ir_validity else ""
    aar_label = " [Vulkan AAR compat]" if (backend == "vulkan" and vulkan_compile_options) else ""
    print(f"\n{'='*60}")
    print(f"Exporting {name} {backend_label}{planning_label}{strict_label}{ir_label}{aar_label}")
    print(f"{'='*60}")

    if use_fp16:
        wrapper = wrapper.half()
        sample_inputs = tuple(
            inp.half() if inp.is_floating_point() else inp for inp in sample_inputs
        )

    start = time.time()
    exported = torch.export.export(wrapper, sample_inputs, strict=strict_export)
    # XNNPACK removed: XNNWeightsCache::look_up_or_insert causes SIGSEGV on Android for large parts.
    compile_config = EdgeCompileConfig(_check_ir_validity=check_ir_validity)
    if backend == "vulkan":
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        from executorch.exir import to_edge_transform_and_lower
        opts = vulkan_compile_options if vulkan_compile_options else {}
        edge = to_edge_transform_and_lower(
            exported,
            compile_config=compile_config,
            partitioner=[VulkanPartitioner(opts)],
        )
    else:
        from executorch.exir import to_edge
        edge = to_edge(exported, compile_config=compile_config)

    # Greedy memory planning: use alloc_graph_input=False, alloc_graph_output=False so I/O
    # buffers are caller-managed and don't inflate the plan (fixes "Misallocate graph input: False v.s. True").
    # Export must use static shapes (no torch.export.Dim).
    # Skip greedy for Vulkan: VulkanPartitioner does its own AOT memory planning; our greedy pass
    # can fail with "TensorSpec(...) should have specified memory offset" on Vulkan-partitioned graphs.
    if use_greedy_memory_planning and backend != "vulkan":
        try:
            from executorch.exir.capture._config import ExecutorchBackendConfig
            from executorch.exir.memory_planning import greedy
            from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
            et_program = edge.to_executorch(
                ExecutorchBackendConfig(
                    memory_planning_pass=MemoryPlanningPass(
                        memory_planning_algo=greedy,
                        alloc_graph_input=False,
                        alloc_graph_output=False,
                    ),
                )
            )
            print("  Greedy memory planning applied (caller-managed I/O)")
        except Exception as e:
            try:
                from executorch.exir import ExecutorchBackendConfig
                from executorch.exir.memory_planning import greedy
                from executorch.exir.passes import MemoryPlanningPass
                et_program = edge.to_executorch(
                    ExecutorchBackendConfig(
                        memory_planning_pass=MemoryPlanningPass(
                            memory_planning_algo=greedy,
                            alloc_graph_input=False,
                            alloc_graph_output=False,
                        ),
                    )
                )
                print("  Greedy memory planning applied (alt API, caller-managed I/O)")
            except Exception as e2:
                print(f"  Greedy memory planning not available: {e2}, using default planning")
                et_program = edge.to_executorch()
    else:
        if backend == "vulkan":
            print("  Vulkan: using default planning (VulkanPartitioner AOT handles memory)")
        et_program = edge.to_executorch()
    export_time = time.time() - start

    # Partition diagnostics (Vulkan): help spot fragmented/many subgraphs
    if backend == "vulkan" and hasattr(et_program, "buffer"):
        buf = et_program.buffer
        n_vulkan = buf.count(b"vulkan") + buf.count(b"Vulkan")
        n_backend = buf.count(b"VulkanBackend")
        print(f"  [Partition] Vulkan strings in .pte: {n_vulkan}, VulkanBackend id: {n_backend}")

    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    precision = "FP16" if use_fp16 else "FP32"
    print(f"  {precision} export: {export_time:.0f}s")
    print(f"  Saved: {output_path.name} ({size_mb:.0f} MB)")
    return size_mb


def main():
    overall_start = time.time()
    args = parse_args()

    # Backend: vulkan (default) or portable. XNNPACK removed — causes SIGSEGV (XNNWeightsCache) on Android.
    if args.vulkan:
        backend = "vulkan"
    elif args.no_xnnpack:
        backend = "portable"
    else:
        backend = args.backend
    if backend not in ("vulkan", "portable"):
        backend = "vulkan"
        print("WARNING: XNNPACK disabled (SIGSEGV on Android). Using Vulkan.")

    # --vulkan-aar-compat: FP32 + force_fp16=False so .pte only uses shaders in executorch-android-vulkan 1.1.0 AAR (avoids view_convert_buffer_float_half).
    vulkan_aar_compat = getattr(args, "vulkan_aar_compat", False)
    use_fp16_export = (args.dtype == "fp16") and not vulkan_aar_compat
    patch_batch = getattr(args, "patch_batch_size", 1)
    vulkan_fp16 = (backend == "vulkan" and use_fp16_export)
    # Keep _vulkan_fp16 suffix in filenames so the app finds them (Part1/2/3/4 vulkan); content may be FP32 when aar-compat.
    suffix = "_vulkan_fp16" if (backend == "vulkan") else ""
    vulkan_opts = {"force_fp16": False, "skip_memory_planning": False} if (backend == "vulkan" and vulkan_aar_compat) else None
    # Vulkan optional fixes: opt-in (defaults keep export working: strict=False, no IR check, Part4 FP32)
    strict_export = getattr(args, "strict_export", False)
    check_ir = getattr(args, "check_ir_validity", False)
    unify_fp16 = getattr(args, "unify_fp16", False)
    part4_use_fp16 = use_fp16_export if (vulkan_fp16 and unify_fp16) else False

    backend_labels = {"vulkan": "Vulkan GPU (20-60s)", "portable": "Portable (CPU fallback, 10min+)"}
    print("=" * 60)
    print("Export 4-Part Split SHARP to ExecuTorch .pte (" + (args.dtype or "fp32").upper() + ")")
    print("Same architecture as LiteRT split - Android runs same pipeline")
    print("Backend: " + backend_labels.get(backend, backend))
    if vulkan_fp16:
        print("Vulkan FP16: avoids INT8 staging crashes; patch_batch=%d" % patch_batch)
    if vulkan_aar_compat:
        print("Vulkan AAR compat: FP32 + force_fp16=False (shaders in executorch-android-vulkan 1.1.0 AAR only)")
    if strict_export or check_ir or unify_fp16:
        print("Options: strict_export=%s, check_ir_validity=%s, unify_fp16=%s" % (strict_export, check_ir, unify_fp16))
    image_size = getattr(args, "image_size", 1536)
    if image_size != IMAGE_SIZE:
        print("WARNING: --image-size 1280 requested but Part3/Part4 decoder is fixed to 1536; exporting at 1536.")
        image_size = IMAGE_SIZE
    print("=" * 60)

    sharp_src = Path(args.sharp_src)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)

    if not sharp_src.exists():
        print(f"ERROR: SHARP source not found at {sharp_src}")
        return 1
    if not weights_path.exists():
        print(f"ERROR: Weights not found at {weights_path}")
        return 1

    sys.path.insert(0, str(sharp_src))

    from sharp.models import PredictorParams, create_predictor

    print("\nLoading SHARP...")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
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

    # Validate split pipeline matches full model
    print("\nValidating split pipeline...")
    sample_image = torch.rand(1, 3, image_size, image_size)

    with torch.no_grad():
        x0_raw = sample_image
        x1_raw = F.interpolate(sample_image, scale_factor=0.5, mode="bilinear", align_corners=False)
        x2_raw = F.interpolate(sample_image, scale_factor=0.25, mode="bilinear", align_corners=False)

        x0_patches = split_patches_list(x0_raw, 0.25, PATCH_SIZE)
        x1_patches = split_patches_list(x1_raw, 0.5, PATCH_SIZE)
        all_patches = x0_patches + x1_patches + [x2_raw]
        print(f"  {len(all_patches)} patches")

        all_tokens = []
        all_block5 = []
        for patch in all_patches:
            tokens_i, block5_i = part1(patch)
            all_tokens.append(tokens_i)
            all_block5.append(block5_i)

        all_features = []
        for tokens_i in all_tokens:
            all_features.append(part2(tokens_i))

        all_block5_spatial = [reshape_feature(b) for b in all_block5]
        all_block11_spatial = [reshape_feature(t) for t in all_tokens]

        latent0 = merge_patches_from_list(all_block5_spatial[:25], padding=3)
        latent1 = merge_patches_from_list(all_block11_spatial[:25], padding=3)
        x0_feat = merge_patches_from_list(all_features[:25], padding=3)
        x1_feat = merge_patches_from_list(all_features[25:34], padding=6)
        x2_feat = all_features[34]

        image_tokens = part3(sample_image)
        packed = part4(sample_image, image_tokens, latent0, latent1, x0_feat, x1_feat, x2_feat)
        gaussianCount = packed.shape[1]
        print(f"  Split pipeline: {gaussianCount:,} Gaussians")

    # Export parts. Vulkan FP16: use_fp16 for Part1/2/3 to avoid INT8 staging crashes.
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}

    chunked_part4_only = getattr(args, "chunked_part4_only", False)
    if chunked_part4_only:
        args.chunked_part4 = True
        print("\n--chunked-part4-only: will export only Part4a (512/65) + Part4b single; skipping Part1–3 monolithic exports.\n")

    part12_only_portable = getattr(args, "part12_only_portable", False)
    if part12_only_portable:
        strict_export = False
        check_ir = False
        print("Exporting Part1+Part2 only as portable (CPU) FP32: sharp_split_part1.pte, sharp_split_part2.pte (app feeds Float)")
        sample_patch = torch.rand(1, 3, PATCH_SIZE, PATCH_SIZE)
        sample_tokens = torch.rand(1, 577, 1024)
        sizes["part1"] = export_pte(
            "Part 1 (portable CPU)",
            part1, (sample_patch,),
            output_dir / "sharp_split_part1.pte",
            use_fp16=False,
            backend="portable",
            use_greedy_memory_planning=True,
            strict_export=strict_export,
            check_ir_validity=check_ir,
        )
        sizes["part2"] = export_pte(
            "Part 2 (portable CPU)",
            part2, (sample_tokens,),
            output_dir / "sharp_split_part2.pte",
            use_fp16=False,
            backend="portable",
            use_greedy_memory_planning=True,
            strict_export=strict_export,
            check_ir_validity=check_ir,
        )
        print("Done. Push sharp_split_part1.pte and sharp_split_part2.pte to device; app uses them for Part1/Part2.")
        return 0

    # Part-1-only export: one .pte + fixed test patch + golden outputs (for app-side compare).
    if getattr(args, "part1_only", False):
        torch.manual_seed(42)
        sample_patch = torch.rand(1, 3, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save test input for app
        torch.save(sample_patch, output_dir / "part1_test_patch.pt")
        sample_patch.numpy().tofile(output_dir / "part1_test_patch_f32.bin")
        if use_fp16_export:
            sample_patch.half().numpy().tofile(output_dir / "part1_test_patch_f16.bin")
        # Output filename
        if backend == "vulkan":
            out_name = "sharp_split_part1_vulkan_fp16.pte" if use_fp16_export else "sharp_split_part1_vulkan_fp32.pte"
        else:
            out_name = "sharp_split_part1.pte"
        print("Exporting Part 1 only")
        export_pte(
            "Part 1 only",
            part1, (sample_patch,),
            output_dir / out_name,
            use_fp16=use_fp16_export,
            backend=backend,
            use_greedy_memory_planning=(backend != "vulkan"),
            strict_export=strict_export,
            check_ir_validity=check_ir,
            vulkan_compile_options=vulkan_opts,
        )
        # Golden outputs: run eager with same dtype as export, save as f32 for app compare
        with torch.no_grad():
            patch_eval = sample_patch.half() if use_fp16_export else sample_patch
            model_eval = part1.half() if use_fp16_export else part1
            tokens, block5 = model_eval(patch_eval)
        tokens_f32 = tokens.cpu().float()
        block5_f32 = block5.cpu().float()
        tokens_f32.numpy().tofile(output_dir / "part1_tokens_golden_f32.bin")
        block5_f32.numpy().tofile(output_dir / "part1_block5_golden_f32.bin")
        print("Part 1 golden outputs (eager, same input as export):")
        print("  tokens ", tokens_f32.shape, tokens_f32.dtype, "min={:.6f} max={:.6f} mean={:.6f}".format(
            tokens_f32.min().item(), tokens_f32.max().item(), tokens_f32.mean().item()))
        print("  tokens first 8:", tokens_f32.flatten()[:8].tolist())
        print("  block5 ", block5_f32.shape, block5_f32.dtype, "min={:.6f} max={:.6f} mean={:.6f}".format(
            block5_f32.min().item(), block5_f32.max().item(), block5_f32.mean().item()))
        print("  block5 first 8:", block5_f32.flatten()[:8].tolist())
        print(f"Done. Exported {out_name}; saved part1_test_patch*.pt/bin, part1_*_golden_f32.bin")
        return 0

    if not chunked_part4_only:
        sample_patch = torch.rand(1, 3, PATCH_SIZE, PATCH_SIZE)
        part1_name = "sharp_split_part1" + suffix + ".pte"
        sizes["part1"] = export_pte(
            "Part 1: Single-Patch Encoder A (blocks 0-11)",
            part1, (sample_patch,),
            output_dir / part1_name,
            use_fp16=use_fp16_export,
            backend=backend,
            use_greedy_memory_planning=(backend != "vulkan"),
            strict_export=strict_export,
            check_ir_validity=check_ir,
            vulkan_compile_options=vulkan_opts,
        )

        sample_tokens = torch.rand(1, 577, 1024)
        part2_name = "sharp_split_part2" + suffix + ".pte"
        sizes["part2"] = export_pte(
            "Part 2: Single-Patch Encoder B (blocks 12-23)",
            part2, (sample_tokens,),
            output_dir / part2_name,
            use_fp16=use_fp16_export,
            backend=backend,
            use_greedy_memory_planning=(backend != "vulkan"),
            strict_export=strict_export,
            check_ir_validity=check_ir,
            vulkan_compile_options=vulkan_opts,
        )

        part3_name = "sharp_split_part3" + suffix + ".pte"
        sizes["part3"] = export_pte(
            "Part 3: Image Encoder A (blocks 0-11)",
            part3, (sample_image,),
            output_dir / part3_name,
            use_fp16=use_fp16_export,
            backend=backend,
            use_greedy_memory_planning=(backend != "vulkan"),
            strict_export=strict_export,
            check_ir_validity=check_ir,
            vulkan_compile_options=vulkan_opts,
        )

        sizes["part4"] = export_pte(
            "Part 4: Image Encoder B + Full Decoder + Gaussians",
            part4, (sample_image, image_tokens, latent0, latent1, x0_feat, x1_feat, x2_feat),
            output_dir / "sharp_split_part4.pte",
            use_fp16=part4_use_fp16,
            backend=backend,
            use_greedy_memory_planning=(backend != "vulkan"),
            strict_export=strict_export,
            check_ir_validity=check_ir,
            vulkan_compile_options=vulkan_opts,
        )

        # Vulkan FP16 batch-2 Part1/Part2: B2 avoids INT8 staging crash, 95% success rate.
        if vulkan_fp16 and patch_batch >= 2:
            batch_sz = min(patch_batch, 4)
            sample_patch_b = torch.rand(batch_sz, 3, PATCH_SIZE, PATCH_SIZE)
            sample_tokens_b = torch.rand(batch_sz, 577, 1024)
            sizes["part1_b%d" % batch_sz] = export_pte(
                "Part 1 batch=%d (patch encoder A, Vulkan FP16)" % batch_sz,
                part1, (sample_patch_b,),
                output_dir / ("sharp_split_part1_b%d_vulkan_fp16.pte" % batch_sz),
                use_fp16=True,
                backend=backend,
                strict_export=strict_export,
                check_ir_validity=check_ir,
                vulkan_compile_options=vulkan_opts,
            )
            sizes["part2_b%d" % batch_sz] = export_pte(
                "Part 2 batch=%d (patch encoder B, Vulkan FP16)" % batch_sz,
                part2, (sample_tokens_b,),
                output_dir / ("sharp_split_part2_b%d_vulkan_fp16.pte" % batch_sz),
                use_fp16=True,
                backend=backend,
                strict_export=strict_export,
                check_ir_validity=check_ir,
                vulkan_compile_options=vulkan_opts,
            )
            print("  Part1/Part2 batch=%d Vulkan FP16 exported (use in C++ when useVulkan)" % batch_sz)

    # Chunked Part 4: run ViT 12-23 on token slices (512 + 65), then single decoder pass. Reduces peak RAM.
    if getattr(args, "chunked_part4", False):
        CHUNK_LEN_FIRST = 512
        CHUNK_LEN_LAST = 577 - CHUNK_LEN_FIRST  # 65
        part4a_512 = ImageEncoderPartBChunk(predictor, CHUNK_LEN_FIRST).eval()
        part4a_65 = ImageEncoderPartBChunk(predictor, CHUNK_LEN_LAST).eval()
        part4b = ImageEncoderPartBFromTokens(predictor).eval()
        sample_tokens_512 = torch.rand(1, CHUNK_LEN_FIRST, 1024)
        sample_tokens_65 = torch.rand(1, CHUNK_LEN_LAST, 1024)
        with torch.no_grad():
            # tokens_after_blocks from chunked run: concat(part4a_512(tokens[0:512]), part4a_65(tokens[512:577]))
            tokens_after_blocks = torch.cat([
                part4a_512(image_tokens[:, :CHUNK_LEN_FIRST]),
                part4a_65(image_tokens[:, CHUNK_LEN_FIRST:]),
            ], dim=1)
        sizes["part4a_chunk_512"] = export_pte(
            "Part 4a chunk (512 tokens): ViT blocks 12-23",
            part4a_512, (sample_tokens_512,),
            output_dir / ("sharp_split_part4a_chunk_512%s.pte" % ("_vulkan" if backend == "vulkan" else "")),
            use_fp16=part4_use_fp16,
            backend=backend,
            strict_export=strict_export,
            check_ir_validity=check_ir,
            vulkan_compile_options=vulkan_opts,
        )
        sizes["part4a_chunk_65"] = export_pte(
            "Part 4a chunk (65 tokens): ViT blocks 12-23",
            part4a_65, (sample_tokens_65,),
            output_dir / ("sharp_split_part4a_chunk_65%s.pte" % ("_vulkan" if backend == "vulkan" else "")),
            use_fp16=part4_use_fp16,
            backend=backend,
            strict_export=strict_export,
            check_ir_validity=check_ir,
            vulkan_compile_options=vulkan_opts,
        )
        sizes["part4b"] = export_pte(
            "Part 4b: From tokens (577) + decoder + Gaussians",
            part4b, (tokens_after_blocks, sample_image, latent0, latent1, x0_feat, x1_feat, x2_feat),
            output_dir / ("sharp_split_part4b%s.pte" % ("_vulkan" if backend == "vulkan" else "")),
            use_fp16=part4_use_fp16,
            backend=backend,
            use_greedy_memory_planning=(backend != "vulkan"),
            strict_export=strict_export,
            check_ir_validity=check_ir,
            vulkan_compile_options=vulkan_opts,
        )
        # Validate chunked pipeline runs and shape matches (numerical diff expected: chunked attention is per-slice)
        with torch.no_grad():
            packed_chunked = part4b(tokens_after_blocks, sample_image, latent0, latent1, x0_feat, x1_feat, x2_feat)
        assert packed_chunked.shape == packed.shape, f"Chunked {packed_chunked.shape} vs full {packed.shape}"
        print(f"  Chunked Part 4 output shape OK (Gaussians: {packed_chunked.shape[1]:,})")

    total_mb = sum(sizes.values())
    elapsed = time.time() - overall_start

    print(f"\n{'='*60}")
    print(f"Export complete in {elapsed:.0f}s")
    print(f"{'='*60}")
    for name, size in sizes.items():
        print(f"  {name}: {size:.0f} MB")
    print(f"  Total: {total_mb:.0f} MB")
    print(f"  Gaussians: {gaussianCount:,}")
    print(f"\nPush to device (match APK flavor):")
    print("  etCpu:    adb shell mkdir -p /sdcard/Android/data/com.furnit.android/files/models_cpu")
    print("  etVulkan: adb shell mkdir -p /sdcard/Android/data/com.furnit.android/files/models_vulkan")
    sub = "models_vulkan" if backend == "vulkan" else "models_cpu"
    for pte in sorted(output_dir.glob("sharp_split_part*.pte")):
        print(f"  adb push {pte} /sdcard/Android/data/com.furnit.android/files/{sub}/")

    if getattr(args, "verify_delegate", False):
        part1_candidates = sorted(output_dir.glob("sharp_split_part1*.pte"))
        part1_pte = part1_candidates[0] if part1_candidates else None
        if part1_pte and part1_pte.exists():
            print(f"\nVerify delegate: {part1_pte.name}")
            script_dir = Path(__file__).resolve().parent
            inspect_script = script_dir / "inspect_pte_delegates.py"
            if inspect_script.exists():
                import subprocess
                subprocess.run([sys.executable, str(inspect_script), str(part1_pte)], cwd=script_dir)
            else:
                print(f"  Run: python inspect_pte_delegates.py {part1_pte}")
        else:
            print("\nVerify delegate: no Part1 .pte found to inspect.")


if __name__ == "__main__":
    sys.exit(main() or 0)
