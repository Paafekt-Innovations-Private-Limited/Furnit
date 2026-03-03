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
  sharp_split_part4_int8.pte  (~190MB)  -- run 1x (decoder + Gaussians)

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

IMAGE_SIZE = 1536  # overridden by --image-size
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


def quantize_and_export_pte(name, wrapper, sample_inputs, output_path):
    """Quantize a part with PT2E INT8 and export to .pte with XNNPACK + greedy memory planning."""
    from executorch.exir import EdgeCompileConfig

    print(f"\n{'='*60}")
    print(f"Exporting {name} (INT8 + XNNPACK + greedy)")
    print(f"{'='*60}")

    start = time.time()

    # Step 1: Export to ATen IR
    print("  Exporting to ATen IR...")
    exported = torch.export.export(wrapper, sample_inputs, strict=False)

    # Step 2: PT2E quantization
    print("  Quantizing (PT2E INT8 symmetric per-channel dynamic)...")
    try:
        from torchao.quantization.pt2e import prepare_pt2e, convert_pt2e
        from torchao.quantization.pt2e.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
    except ImportError:
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

    # Step 4: XNNPACK partitioning
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

    # Step 5: Greedy memory planning
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
        print("  Greedy memory planning applied")
    except Exception:
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
            print("  Greedy memory planning applied (alt API)")
        except Exception as e2:
            print(f"  Greedy planning unavailable ({e2}), using default")
            et_program = edge.to_executorch()

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
    pa.add_argument("--chunked-part4", action="store_true",
        help="Export chunked Part 4 (4a_chunk_512, 4a_chunk_65, 4b) for lower peak RAM. Part4a INT8, Part4b FP16 or FP32.")
    pa.add_argument("--chunked-part4b", action="store_true",
        help="Further split Part4b into 2 stages: depth decoder + Gaussian generator (better progress UX, lower peak memory).")
    pa.add_argument("--part4b-fp16", action="store_true",
        help="Export Part4b as FP16 (recommended for quality; avoids INT8 artifacts on positions/scales/rotations). Implies --chunked-part4.")
    pa.add_argument("--part4b-backend", choices=("xnnpack", "vulkan"), default="xnnpack",
        help="Part4b partitioner: xnnpack (recommended, Ultralytics) or vulkan. Vulkan often OOM on large decoders; XNNPACK avoids UBO/memory issues.")
    pa.add_argument("--part4b-tiled", action="store_true",
        help="Export tiled Part4b (depth decoder in 2x2 windows) as _depth_tiled.pte + _gauss_tiled.pte for lower memory.")
    pa.add_argument("--image-size", type=int, default=1536,
        help="Input image size (default 1536). Lower values (e.g. 768) produce fewer Gaussians but run ~4x faster. Requires matching Android runtime IMAGE_SIZE.")
    return pa.parse_args()


def main():
    global IMAGE_SIZE
    args = parse_args()
    sharp_src = Path(args.sharp_src)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    IMAGE_SIZE = args.image_size

    print("=" * 60)
    print("ExecuTorch INT8 Split 4-Part Export")
    print("=" * 60)
    print(f"  Backend: XNNPACK (INT8 NEON kernels)")
    print(f"  Memory planning: greedy AOT (all parts)")
    print(f"  Quantization: PT2E symmetric per-channel dynamic INT8")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")

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
    merge_1x = IMAGE_SIZE // 16   # 96 at 1536, 48 at 768
    merge_05x = IMAGE_SIZE // 32  # 48 at 1536, 24 at 768
    spatial = IMAGE_SIZE // 64    # 24 at 1536, 12 at 768
    latent0 = torch.rand(1, FEATURE_DIM, merge_1x, merge_1x)
    latent1 = torch.rand(1, FEATURE_DIM, merge_1x, merge_1x)
    x0_feat = torch.rand(1, FEATURE_DIM, merge_1x, merge_1x)
    x1_feat = torch.rand(1, FEATURE_DIM, merge_05x, merge_05x)
    x2_feat = torch.rand(1, FEATURE_DIM, spatial, spatial)
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
    )

    sample_tokens = torch.rand(1, 577, 1024)
    sizes["part2"] = quantize_and_export_pte(
        "Part 2: Patch Encoder B (blocks 12-23)",
        part2, (sample_tokens,),
        output_dir / "sharp_split_part2_int8.pte",
    )

    sizes["part3"] = quantize_and_export_pte(
        "Part 3: Image Encoder A (blocks 0-11)",
        part3, (sample_image,),
        output_dir / "sharp_split_part3_int8.pte",
    )

    from export_sharp_executorch_split4 import export_pte, ImageEncoderPartBChunk, ImageEncoderPartBFromTokens

    chunked_part4 = getattr(args, "chunked_part4", False) or getattr(args, "part4b_fp16", False)
    part4b_fp16 = getattr(args, "part4b_fp16", False)
    part4b_backend = getattr(args, "part4b_backend", "xnnpack")

    if chunked_part4:
        # Chunked Part 4: Part4a INT8 (512 + 65 tokens), Part4b FP16 or FP32 (recommended FP16 for quality).
        # See android/docs/PART4B_DEPLOYMENT.md (Ultralytics: keep decoder heads FP16).
        CHUNK_LEN_FIRST = 512
        CHUNK_LEN_LAST = 577 - CHUNK_LEN_FIRST  # 65
        part4a_512 = ImageEncoderPartBChunk(predictor, CHUNK_LEN_FIRST).eval()
        part4a_65 = ImageEncoderPartBChunk(predictor, CHUNK_LEN_LAST).eval()
        part4b = ImageEncoderPartBFromTokens(predictor).eval()
        sample_tokens_512 = torch.rand(1, CHUNK_LEN_FIRST, 1024)
        sample_tokens_65 = torch.rand(1, CHUNK_LEN_LAST, 1024)
        with torch.no_grad():
            tokens_after_blocks = torch.cat([
                part4a_512(image_tokens[:, :CHUNK_LEN_FIRST]),
                part4a_65(image_tokens[:, CHUNK_LEN_FIRST:]),
            ], dim=1)
        sizes["part4a_chunk_512"] = quantize_and_export_pte(
            "Part 4a chunk (512 tokens) INT8",
            part4a_512, (sample_tokens_512,),
            output_dir / "sharp_split_part4a_chunk_512.pte",
        )
        sizes["part4a_chunk_65"] = quantize_and_export_pte(
            "Part 4a chunk (65 tokens) INT8",
            part4a_65, (sample_tokens_65,),
            output_dir / "sharp_split_part4a_chunk_65.pte",
        )
        part4b_name = "Part4b (FP16)" if part4b_fp16 else "Part4b (FP32)"
        part4b_path = output_dir / "sharp_split_part4b_fp16.pte" if part4b_fp16 else output_dir / "sharp_split_part4b.pte"
        sizes["part4b"] = export_pte(
            f"{part4b_name} + {part4b_backend}",
            part4b, (tokens_after_blocks, sample_image, latent0, latent1, x0_feat, x1_feat, x2_feat),
            part4b_path,
            use_fp16=part4b_fp16,
            backend=part4b_backend,
            use_greedy_memory_planning=True,
        )
        print(f"  Chunked Part 4: Part4a INT8, Part4b {'FP16' if part4b_fp16 else 'FP32'} ({part4b_backend})")

        # Further split Part4b into depth decoder (stage 1) + Gaussian generator (stage 2)
        if getattr(args, "chunked_part4b", False):
            from export_sharp_executorch_split4 import Part4bDepthDecoder, Part4bGaussGenerator
            depth_decoder = Part4bDepthDecoder(predictor).eval()
            gauss_generator = Part4bGaussGenerator(predictor).eval()
            # Validation forward in float32 so dtypes match (part4b_fp16 export uses half in .pte only).
            with torch.no_grad():
                d_f32 = depth_decoder.float()
                t_f32 = tokens_after_blocks.float() if tokens_after_blocks.is_floating_point() else tokens_after_blocks
                inter_inputs = (latent0.float(), latent1.float(), x0_feat.float(), x1_feat.float(), x2_feat.float())
                intermediates = d_f32(t_f32, *inter_inputs)
            monodepth_sample, f0, f1, f2, f3, f4, df = intermediates
            print(f"\n  Part4b depth decoder outputs: monodepth={monodepth_sample.shape}")
            print(f"    feat0={f0.shape} feat1={f1.shape} feat2={f2.shape} feat3={f3.shape} feat4={f4.shape} decoder={df.shape}")

            sizes["part4b_depth"] = export_pte(
                f"Part4b stage 1 (depth decoder) + {part4b_backend}",
                depth_decoder,
                (tokens_after_blocks, latent0, latent1, x0_feat, x1_feat, x2_feat),
                output_dir / "sharp_split_part4b_depth.pte",
                use_fp16=part4b_fp16,
                backend=part4b_backend,
            )
            sizes["part4b_gauss"] = export_pte(
                f"Part4b stage 2 (Gaussian generator) + {part4b_backend}",
                gauss_generator,
                (sample_image, monodepth_sample, f0, f1, f2, f3, f4, df),
                output_dir / "sharp_split_part4b_gauss.pte",
                use_fp16=part4b_fp16,
                backend=part4b_backend,
                use_greedy_memory_planning=True,
            )
            with torch.no_grad():
                # Validation in float32 (exported .pte may be FP16)
                g_f32 = gauss_generator.float()
                packed_staged = g_f32(
                    sample_image.float(), monodepth_sample.float(),
                    f0.float(), f1.float(), f2.float(), f3.float(), f4.float(), df.float(),
                )
                packed_single = part4b.float()(
                    t_f32, sample_image.float(), *[x.float() for x in (latent0, latent1, x0_feat, x1_feat, x2_feat)]
                )
            assert packed_staged.shape == packed_single.shape, f"Staged {packed_staged.shape} vs single {packed_single.shape}"
            print(f"  Chunked Part4b validated: {packed_staged.shape[1]:,} Gaussians")

            # Tiled Part4b: fused upsample+decode+gauss per tile (no full-res tensors in Java)
            if getattr(args, "part4b_tiled", False):
                from export_sharp_executorch_split4 import (
                    Part4bTileFull,
                    Part4bUpsample,
                    _window_partition_nxn,
                    PART4B_TILED_GRID,
                )
                n_tiles = PART4B_TILED_GRID
                num_tiles = n_tiles * n_tiles  # 16

                tile_full_model = Part4bTileFull(predictor).eval()
                print(f"  Part4b tile full (upsample+decode+gauss fused): {sum(p.numel() for p in tile_full_model.parameters())/1e6:.1f}M params")

                # Compute x_lowres from tokens via upsample model's reshape
                upsample_model = Part4bUpsample(predictor).eval()
                with torch.no_grad():
                    x_lowres = upsample_model._reshape_feature(t_f32)
                print(f"  x_lowres shape: {x_lowres.shape}")

                # Tile the LOW-RES inputs (tiny: latent0/1/x0 [1,1024,24,24], x1 [1,1024,12,12], x2/x_lowres [1,1024,6,6])
                latent0_tiles = _window_partition_nxn(latent0.float(), n_tiles)
                latent1_tiles = _window_partition_nxn(latent1.float(), n_tiles)
                x0_tiles = _window_partition_nxn(x0_feat.float(), n_tiles)
                x1_tiles = _window_partition_nxn(x1_feat.float(), n_tiles)
                x2_tiles = _window_partition_nxn(x2_feat.float(), n_tiles)
                x_lowres_tiles = _window_partition_nxn(x_lowres.float(), n_tiles)
                image_tiles = _window_partition_nxn(sample_image.float(), n_tiles)

                tile_sample_inputs = (
                    image_tiles[0], latent0_tiles[0], latent1_tiles[0],
                    x0_tiles[0], x1_tiles[0], x2_tiles[0], x_lowres_tiles[0],
                )
                print(f"  Tile full sample input shapes: image={image_tiles[0].shape} latent0={latent0_tiles[0].shape} "
                      f"x1={x1_tiles[0].shape} x2={x2_tiles[0].shape} x_lowres={x_lowres_tiles[0].shape}")

                # Export tile full model
                tile_full_path = output_dir / "sharp_split_part4b_tile_full.pte"
                sizes["part4b_tile_full"] = export_pte(
                    f"Part4b tile full (upsample+decode+gauss, run 16x) + {part4b_backend}",
                    tile_full_model,
                    tile_sample_inputs,
                    tile_full_path,
                    use_fp16=False,
                    backend=part4b_backend,
                    use_greedy_memory_planning=True,
                )
                # Copy to 16 tile files for load/unload per tile
                tile_full_mb = sizes["part4b_tile_full"]
                for i in range(num_tiles):
                    dest = output_dir / f"sharp_split_part4b_tile_{i:02d}.pte"
                    shutil.copy2(tile_full_path, dest)
                    sizes[f"part4b_tile_{i:02d}"] = tile_full_mb
                print(f"  Part4b: 16 tile files (fused, XNNPACK) written: sharp_split_part4b_tile_00.pte .. tile_15.pte")

                # Validate: run tile full on all 16 tiles
                with torch.no_grad():
                    tf_f32 = tile_full_model.float()
                    tile_gaussians = []
                    for i in range(num_tiles):
                        packed_tile = tf_f32(
                            image_tiles[i].float(),
                            latent0_tiles[i].float(), latent1_tiles[i].float(),
                            x0_tiles[i].float(), x1_tiles[i].float(),
                            x2_tiles[i].float(), x_lowres_tiles[i].float(),
                        )
                        tile_gaussians.append(packed_tile.float())
                        print(f"    Tile {i}: {packed_tile.shape[1]:,} Gaussians")
                total_tiled = sum(t.shape[1] for t in tile_gaussians)
                print(f"  Part4b tiled validated: {total_tiled:,} Gaussians across 16 tiles")
    else:
        # Part 4 decoder (full): export FP32. PT2E quantization breaks the decoder's
        # skip connections. Parts 1-3 get the INT8 win.
        sizes["part4"] = export_pte(
            "Part 4: Decoder + Gaussians (FP32, greedy planning)",
            part4, (image_tokens, sample_image, latent0, latent1, x0_feat, x1_feat, x2_feat),
            output_dir / "sharp_split_part4_int8.pte",
            use_fp16=False,
            backend="xnnpack",
            use_greedy_memory_planning=True,
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
    print(f"\nPush to device (16-piece: upsample + 16 tile files + gauss_tiled):")
    print(f"  for f in {output_dir}/sharp_split_part*_int8.pte {output_dir}/sharp_split_part4b_upsample.pte {output_dir}/sharp_split_part4b_tile_*.pte {output_dir}/sharp_split_part4b_gauss_tiled.pte; do")
    print(f"    [ -f \"$f\" ] && adb push \"$f\" /storage/emulated/0/Android/data/com.furnit.android/files/models/")
    print(f"  done")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
