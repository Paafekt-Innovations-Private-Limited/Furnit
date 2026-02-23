#!/usr/bin/env python3
"""
Export SHARP as 4 split FP16 ExecuTorch .pte parts with XNNPACK backend.

Same 4-part split architecture as export_sharp_executorch_split4.py but with
all parts exported as FP16 (model.half() + half inputs). Produces ~50% smaller
.pte files and may run faster on ARM with native FP16 SIMD.

Output files:
  sharp_split_part1_fp16.pte  (~290MB)
  sharp_split_part2_fp16.pte  (~290MB)
  sharp_split_part3_fp16.pte  (~290MB)
  sharp_split_part4_fp16.pte  (~380MB)

Usage:
  cd android
  python export_sharp_executorch_fp16.py --weights /path/to/sharp.pt
  # then: adb push executorch_fp16_models/*.pte /sdcard/Android/data/com.furnit.android/files/models/
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

# Monkey-patch F.interpolate to preserve FP16 dtype on CPU.
# Bilinear interpolation on CPU upcasts FP16→FP32; this wraps it
# to cast back, preventing mixed-precision errors in conv layers.
_orig_interpolate = F.interpolate
def _fp16_safe_interpolate(*args, **kwargs):
    input_tensor = args[0] if args else kwargs.get("input")
    input_dtype = input_tensor.dtype
    out = _orig_interpolate(*args, **kwargs)
    if out.dtype != input_dtype:
        out = out.to(input_dtype)
    return out
F.interpolate = _fp16_safe_interpolate


def parse_args():
    pa = argparse.ArgumentParser(
        description="Export SHARP 4-part split to FP16 ExecuTorch .pte (XNNPACK)."
    )
    pa.add_argument("--sharp-src",
        default=str(Path(__file__).resolve().parent / "third_party/ml-sharp/src"))
    pa.add_argument("--weights",
        default=str(Path(__file__).resolve().parent / "sharp_litert_models/sharp_2572gikvuh.pt"))
    pa.add_argument("--output-dir",
        default=str(Path(__file__).resolve().parent / "executorch_fp16_models"),
        help="Output directory (default: executorch_fp16_models)")
    return pa.parse_args()


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


# ---- Utility functions ----

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


def export_pte_fp16(name, wrapper, sample_inputs, output_path, use_greedy_memory_planning=False, use_fp16=True):
    """Export a single part to .pte with XNNPACK. FP16 by default, FP32 fallback for parts with interpolate."""
    from executorch.exir import EdgeCompileConfig

    precision = "FP16" if use_fp16 else "FP32"
    planning_label = " + greedy memory planning" if use_greedy_memory_planning else ""
    print(f"\n{'='*60}")
    print(f"Exporting {name} + XNNPACK {precision}{planning_label}")
    print(f"{'='*60}")

    if use_fp16:
        wrapper = wrapper.half()
        sample_inputs = tuple(
            inp.half() if inp.is_floating_point() else inp for inp in sample_inputs
        )

    start = time.time()
    exported = torch.export.export(wrapper, sample_inputs, strict=False)

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    edge = to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[XnnpackPartitioner()],
    )

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
                print(f"  Greedy memory planning not available: {e2}, using default")
                et_program = edge.to_executorch()
    else:
        et_program = edge.to_executorch()

    export_time = time.time() - start
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  {precision} export: {export_time:.0f}s")
    print(f"  Saved: {output_path.name} ({size_mb:.0f} MB)")
    return size_mb


def main():
    overall_start = time.time()
    args = parse_args()

    print("=" * 60)
    print("Export 4-Part Split SHARP to ExecuTorch .pte (FP16 + XNNPACK)")
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

    part1 = SinglePatchEncoderA(predictor).eval()
    part2 = SinglePatchEncoderB(predictor).eval()
    part3 = ImageEncoderPartA(predictor).eval()
    part4 = ImageEncoderPartBFull(predictor).eval()

    print(f"  Part 1: {sum(p.numel() for p in part1.parameters())/1e6:.0f}M params")
    print(f"  Part 2: {sum(p.numel() for p in part2.parameters())/1e6:.0f}M params")
    print(f"  Part 3: {sum(p.numel() for p in part3.parameters())/1e6:.0f}M params")
    print(f"  Part 4: {sum(p.numel() for p in part4.parameters())/1e6:.0f}M params")

    # Validate split pipeline in FP32 first
    print("\nValidating split pipeline (FP32)...")
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

    sample_patch = torch.rand(1, 3, PATCH_SIZE, PATCH_SIZE)
    sizes["part1_fp16"] = export_pte_fp16(
        "Part 1: Single-Patch Encoder A (blocks 0-11)",
        part1, (sample_patch,),
        output_dir / "sharp_split_part1_fp16.pte",
    )

    sample_tokens = torch.rand(1, 577, 1024)
    sizes["part2_fp16"] = export_pte_fp16(
        "Part 2: Single-Patch Encoder B (blocks 12-23)",
        part2, (sample_tokens,),
        output_dir / "sharp_split_part2_fp16.pte",
    )

    sizes["part3_fp16"] = export_pte_fp16(
        "Part 3: Image Encoder A (blocks 0-11)",
        part3, (sample_image,),
        output_dir / "sharp_split_part3_fp16.pte",
    )

    # Chunked Part 4: split into ViT chunks (FP16) + decoder (FP32).
    # Part 4a chunks run ViT blocks 12-23 on token slices -- no interpolate, FP16 works.
    # Part 4b is the decoder with interpolate/upsample -- must be FP32.
    # This lowers peak RAM vs single Part 4 by loading/unloading each chunk.
    CHUNK_LEN_FIRST = 512
    CHUNK_LEN_LAST = 577 - CHUNK_LEN_FIRST  # 65

    from export_sharp_executorch_split4 import ImageEncoderPartBChunk, ImageEncoderPartBFromTokens

    part4a_512 = ImageEncoderPartBChunk(predictor, CHUNK_LEN_FIRST).eval()
    part4a_65 = ImageEncoderPartBChunk(predictor, CHUNK_LEN_LAST).eval()
    part4b = ImageEncoderPartBFromTokens(predictor).eval()

    # Run validation to get sample tensors for Part 4b export
    sample_tokens_512 = torch.rand(1, CHUNK_LEN_FIRST, 1024)
    sample_tokens_65 = torch.rand(1, CHUNK_LEN_LAST, 1024)
    with torch.no_grad():
        tokens_after_blocks = torch.cat([
            part4a_512(image_tokens[:, :CHUNK_LEN_FIRST]),
            part4a_65(image_tokens[:, CHUNK_LEN_FIRST:]),
        ], dim=1)

    # Part 4a chunks: ViT only, no interpolate → FP16 works
    sizes["part4a_chunk_512_fp16"] = export_pte_fp16(
        "Part 4a chunk (512 tokens): ViT blocks 12-23",
        part4a_512, (sample_tokens_512,),
        output_dir / "sharp_split_part4a_chunk_512_fp16.pte",
        use_fp16=True,
    )
    sizes["part4a_chunk_65_fp16"] = export_pte_fp16(
        "Part 4a chunk (65 tokens): ViT blocks 12-23",
        part4a_65, (sample_tokens_65,),
        output_dir / "sharp_split_part4a_chunk_65_fp16.pte",
        use_fp16=True,
    )

    # Part 4b: decoder with interpolate → FP32, greedy memory planning
    sizes["part4b_fp16"] = export_pte_fp16(
        "Part 4b: From tokens + decoder + Gaussians",
        part4b, (tokens_after_blocks, sample_image, latent0, latent1, x0_feat, x1_feat, x2_feat),
        output_dir / "sharp_split_part4b_fp16.pte",
        use_greedy_memory_planning=True,
        use_fp16=False,
    )

    # Validate chunked output shape
    with torch.no_grad():
        packed_chunked = part4b(tokens_after_blocks, sample_image, latent0, latent1, x0_feat, x1_feat, x2_feat)
    assert packed_chunked.shape == packed.shape, f"Chunked {packed_chunked.shape} vs full {packed.shape}"
    print(f"  Chunked Part 4 output shape OK (Gaussians: {packed_chunked.shape[1]:,})")

    total_mb = sum(sizes.values())
    elapsed = time.time() - overall_start

    print(f"\n{'='*60}")
    print(f"FP16 Export complete in {elapsed:.0f}s")
    print(f"{'='*60}")
    for name, size in sizes.items():
        print(f"  {name}: {size:.0f} MB")
    print(f"  Total: {total_mb:.0f} MB")
    print(f"  Gaussians: {gaussianCount:,}")
    print(f"\nPush to device:")
    for pte in sorted(output_dir.glob("sharp_split_part*_fp16.pte")):
        print(f"  adb push {pte} /sdcard/Android/data/com.furnit.android/files/models/")


if __name__ == "__main__":
    sys.exit(main() or 0)
