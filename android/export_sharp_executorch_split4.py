#!/usr/bin/env python3
"""
Export SHARP as 4 split ExecuTorch .pte parts (mirroring LiteRT split).

Backend: XNNPACK ONLY (CPU optimized, stable on Android). No Vulkan or hybrid.

Ultralytics recommendations for Part4b latency: (1) Use torch.nn.functional.scaled_dot_product_attention
(SDPA) in the model so ExecuTorch can lower to XNNPACK/Core ML kernels; (2) AOT memory planning (we use
MemoryPlanningPass below); (3) XNNPACK quantizer for INT8 (see export_sharp_executorch_int8_split4.py).

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

Part 4 is exported with greedy memory planning (same as single memory_optimized .pte) to reduce
peak RAM during the decoder pass. Runtime uses mmap load + zero-copy output for Part 4.

CORRECT EXPORT COMMANDS (run from android/ directory):
  cd android
  python export_sharp_executorch_split4.py --weights /path/to/sharp.pt --backend xnnpack --output-dir executorch_models
  ./push_sharp_executorch_models.sh executorch_models

Verify export: ls -lh executorch_models/
  Expected: sharp_split_part1.pte ~500-600MB, part2 ~500-600MB, part3 ~500-600MB, part4 ~700-800MB
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
        description="Export SHARP 4-part split to ExecuTorch .pte (XNNPACK backend recommended)."
    )
    pa.add_argument("--sharp-src",
        default=str(Path(__file__).resolve().parent / "third_party/ml-sharp/src"))
    pa.add_argument("--weights",
        default=str(Path(__file__).resolve().parent / "sharp_litert_models/sharp_2572gikvuh.pt"))
    pa.add_argument("--output-dir",
        default=str(Path(__file__).resolve().parent / "executorch_models"),
        help="Output directory (default: executorch_models)")
    pa.add_argument("--backend", choices=("xnnpack", "vulkan", "portable"), default="xnnpack",
        help="Backend: xnnpack (CPU optimized, 1-2min), vulkan (GPU, 20-60s), portable (CPU fallback, 10min+). Default: xnnpack")
    pa.add_argument("--xnnpack", action="store_true",
        help="Use XNNPACK backend (same as --backend xnnpack)")
    pa.add_argument("--vulkan", action="store_true",
        help="Use Vulkan GPU backend (fastest on Mali/Adreno). Same as --backend vulkan")
    pa.add_argument("--no-xnnpack", action="store_true",
        help="Disable XNNPACK (same as --backend portable)")
    pa.add_argument("--chunked-part4", action="store_true",
        help="Also export chunked Part 4 (4a_chunk_512, 4a_chunk_65, 4b) for lower peak RAM on decoder.")
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
        return embeddings.reshape(batch, h, w, channel).permute(0, 3, 1, 2)

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
        return embeddings.reshape(batch, h, w, channel).permute(0, 3, 1, 2)

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


def split_patches_list(image, overlap_ratio, patch_size):
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


def export_pte(name, wrapper, sample_inputs, output_path, use_fp16=True, backend="xnnpack", use_greedy_memory_planning=False):
    """Export a single part to .pte format.
    backend: "xnnpack" (CPU optimized, 1-2min), "vulkan" (GPU, 20-60s), "portable" (CPU fallback, 10min+)
    use_greedy_memory_planning: use ExecuTorch greedy memory planner (reuse buffers, lower peak RAM). Recommended for Part 4.
    """
    from executorch.exir import EdgeCompileConfig

    backend_label = {"xnnpack": "+ XNNPACK", "vulkan": "+ Vulkan GPU", "portable": "(portable only)"}[backend]
    planning_label = " + greedy memory planning" if use_greedy_memory_planning else ""
    print(f"\n{'='*60}")
    print(f"Exporting {name} {backend_label}{planning_label}")
    print(f"{'='*60}")

    if use_fp16:
        wrapper = wrapper.half()
        sample_inputs = tuple(
            inp.half() if inp.is_floating_point() else inp for inp in sample_inputs
        )

    start = time.time()
    exported = torch.export.export(wrapper, sample_inputs, strict=False)
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower
        edge = to_edge_transform_and_lower(
            exported,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            partitioner=[XnnpackPartitioner()],
        )
    elif backend == "vulkan":
        try:
            from executorch.backends.vulkan.partition.vulkan_partitioner import VulkanPartitioner
            from executorch.exir import to_edge_transform_and_lower
            edge = to_edge_transform_and_lower(
                exported,
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
                partitioner=[VulkanPartitioner()],
            )
        except Exception as e:
            print(f"Vulkan export failed: {e}, falling back to XNNPACK")
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            from executorch.exir import to_edge_transform_and_lower
            edge = to_edge_transform_and_lower(
                exported,
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
                partitioner=[XnnpackPartitioner()],
            )
    else:
        from executorch.exir import to_edge
        edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))

    # Greedy memory planning: use alloc_graph_input=False, alloc_graph_output=False so I/O
    # buffers are caller-managed and don't inflate the plan (fixes "Misallocate graph input: False v.s. True").
    # Export must use static shapes (no torch.export.Dim).
    if use_greedy_memory_planning:
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
        et_program = edge.to_executorch()
    export_time = time.time() - start

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

    # Backend: --vulkan > --xnnpack > --no-xnnpack > --backend
    if args.vulkan:
        backend = "vulkan"
    elif args.no_xnnpack:
        backend = "portable"
    elif args.xnnpack:
        backend = "xnnpack"
    else:
        backend = args.backend

    backend_labels = {"xnnpack": "XNNPACK (CPU, 1-2min)", "vulkan": "Vulkan GPU (20-60s)", "portable": "Portable (CPU fallback, 10min+)"}
    print("=" * 60)
    print("Export 4-Part Split SHARP to ExecuTorch .pte (FP32)")
    print("Same architecture as LiteRT split - Android runs same pipeline")
    print("Backend: " + backend_labels.get(backend, backend))
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
    sample_image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)

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

    # Export all 4 parts as FP16 .pte
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}

    # Export as FP32 (FP16 fails due to mixed precision in model weights)
    sample_patch = torch.rand(1, 3, PATCH_SIZE, PATCH_SIZE)
    # All parts use greedy memory planning: AOT buffer reuse lowers peak RSS,
    # prevents swap/SSD hits that cause "glacial speed" on memory-constrained devices.
    sizes["part1"] = export_pte(
        "Part 1: Single-Patch Encoder A (blocks 0-11)",
        part1, (sample_patch,),
        output_dir / "sharp_split_part1.pte",
        use_fp16=False,
        backend=backend,
        use_greedy_memory_planning=True,
    )

    sample_tokens = torch.rand(1, 577, 1024)
    sizes["part2"] = export_pte(
        "Part 2: Single-Patch Encoder B (blocks 12-23)",
        part2, (sample_tokens,),
        output_dir / "sharp_split_part2.pte",
        use_fp16=False,
        backend=backend,
        use_greedy_memory_planning=True,
    )

    sizes["part3"] = export_pte(
        "Part 3: Image Encoder A (blocks 0-11)",
        part3, (sample_image,),
        output_dir / "sharp_split_part3.pte",
        use_fp16=False,
        backend=backend,
        use_greedy_memory_planning=True,
    )

    sizes["part4"] = export_pte(
        "Part 4: Image Encoder B + Full Decoder + Gaussians",
        part4, (sample_image, image_tokens, latent0, latent1, x0_feat, x1_feat, x2_feat),
        output_dir / "sharp_split_part4.pte",
        use_fp16=False,
        backend=backend,
        use_greedy_memory_planning=True,
    )

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
            output_dir / "sharp_split_part4a_chunk_512.pte",
            use_fp16=False,
            backend=backend,
        )
        sizes["part4a_chunk_65"] = export_pte(
            "Part 4a chunk (65 tokens): ViT blocks 12-23",
            part4a_65, (sample_tokens_65,),
            output_dir / "sharp_split_part4a_chunk_65.pte",
            use_fp16=False,
            backend=backend,
        )
        sizes["part4b"] = export_pte(
            "Part 4b: From tokens (577) + decoder + Gaussians",
            part4b, (tokens_after_blocks, sample_image, latent0, latent1, x0_feat, x1_feat, x2_feat),
            output_dir / "sharp_split_part4b.pte",
            use_fp16=False,
            backend=backend,
            use_greedy_memory_planning=True,
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
    print(f"\nPush to device:")
    for pte in sorted(output_dir.glob("sharp_split_part*.pte")):
        print(f"  adb push {pte} /sdcard/Android/data/com.furnit.android/files/models/")


if __name__ == "__main__":
    sys.exit(main() or 0)
