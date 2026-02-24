#!/usr/bin/env python3
"""
Hybrid ONNX+NativePt feasibility test.

Validates whether ONNX Parts 1-3 intermediate tensors can feed NativePt Part 4
by comparing the 7 semantic tensors produced by both pipelines.

Usage:
  # Step 1: Inspect ONNX Part 4 inputs (needs ONNX files - adb pull first)
  python tests/test_hybrid_onnx_nativept.py --inspect-onnx --onnx-dir /path/to/onnx/parts

  # Step 2: Full validation with SHARP weights (no ONNX files needed)
  python tests/test_hybrid_onnx_nativept.py

  # Step 3: Timing comparison
  python tests/test_hybrid_onnx_nativept.py --timing
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ANDROID_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ANDROID_DIR / "sharp_litert_models" / "sharp_2572gikvuh.pt"
DEFAULT_PTL_DIR = ANDROID_DIR / "sharp_litert_models"
DEFAULT_SHARP_SRC = ANDROID_DIR / "third_party" / "ml-sharp" / "src"

IMAGE_SIZE = 1536
PATCH_SIZE = 384
VIT_SPLIT_BLOCK = 12


# ---------- Patch helpers (from export_sharp_torchscript_split.py) ----------

def split_patches_list(image, overlap_ratio, patch_size):
    patch_stride = int(patch_size * (1 - overlap_ratio))
    image_size = image.shape[-1]
    steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1
    patches = []
    for j in range(steps):
        j0 = j * patch_stride
        for i in range(steps):
            i0 = i * patch_stride
            patches.append(image[..., j0:j0 + patch_size, i0:i0 + patch_size])
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


# ---------- Part wrappers (same as export script) ----------

class SinglePatchEncoderA(nn.Module):
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


# ---------- Step 1: Inspect ONNX Part 4 inputs ----------

def inspect_onnx_part4(onnx_dir: Path):
    """Print all ONNX Part 4 input names and shapes, classify into weight vs semantic."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: pip install onnxruntime")
        return

    part4_path = onnx_dir / "sharp_part4.onnx"
    if not part4_path.exists():
        print(f"ERROR: {part4_path} not found. Pull from device:")
        print(f"  adb pull /storage/emulated/0/Android/data/com.furnit.android/files/models/sharp_part4.onnx {onnx_dir}/")
        return

    session = ort.InferenceSession(str(part4_path))
    inputs = session.get_inputs()

    print(f"\n{'='*80}")
    print(f"ONNX Part 4: {len(inputs)} inputs")
    print(f"{'='*80}")

    semantic_inputs = []
    weight_inputs = []

    for inp in inputs:
        shape = [d if isinstance(d, int) else -1 for d in inp.shape]
        numel = 1
        for d in shape:
            numel *= abs(d)
        if numel > 2048:
            semantic_inputs.append((inp.name, shape))
        else:
            weight_inputs.append((inp.name, shape))

    print(f"\nSemantic inputs ({len(semantic_inputs)}):")
    print(f"  {'Name':<70} Shape")
    print(f"  {'-'*70} {'-'*20}")
    for name, shape in semantic_inputs:
        print(f"  {name:<70} {shape}")

    print(f"\nWeight inputs ({len(weight_inputs)}):")
    for name, shape in weight_inputs[:5]:
        print(f"  {name:<70} {shape}")
    if len(weight_inputs) > 5:
        print(f"  ... and {len(weight_inputs) - 5} more weight tensors")

    print(f"\n{'='*80}")
    print("MAPPING (by shape):")
    print(f"{'='*80}")
    shape_to_nativept = {
        (1, 3, 1536, 1536): "image",
        (1, 577, 1024): "image_tokens",
        (1, 1024, 96, 96): "latent0 / latent1 / x0_feat (AMBIGUOUS - 3 candidates)",
        (1, 1024, 48, 48): "x1_feat",
        (1, 1024, 24, 24): "x2_feat",
    }
    for name, shape in semantic_inputs:
        tshape = tuple(shape)
        mapping = shape_to_nativept.get(tshape, f"UNKNOWN shape {tshape}")
        print(f"  {name:<70} -> {mapping}")

    return semantic_inputs, weight_inputs


# ---------- Step 2-4: Run NativePt pipeline and validate ----------

def run_nativept_pipeline(weights_path: Path, sharp_src: Path):
    """Run the full NativePt 4-part pipeline on a random image, return intermediate tensors and final output."""
    sys.path.insert(0, str(sharp_src))
    from sharp.models import PredictorParams, create_predictor

    print(f"\nLoading SHARP from {weights_path.name}...")
    t0 = time.time()
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    del state_dict
    print(f"  Loaded in {time.time() - t0:.1f}s")

    part1 = SinglePatchEncoderA(predictor).eval()
    part2 = SinglePatchEncoderB(predictor).eval()
    part3 = ImageEncoderPartA(predictor).eval()
    part4 = ImageEncoderPartBFull(predictor).eval()

    torch.manual_seed(42)
    sample_image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    x0_raw = sample_image
    x1_raw = F.interpolate(sample_image, scale_factor=0.5, mode="bilinear", align_corners=False)
    x2_raw = F.interpolate(sample_image, scale_factor=0.25, mode="bilinear", align_corners=False)
    x0_patches = split_patches_list(x0_raw, 0.25, PATCH_SIZE)
    x1_patches = split_patches_list(x1_raw, 0.5, PATCH_SIZE)
    all_patches = x0_patches + x1_patches + [x2_raw]
    print(f"  Patches: {len(x0_patches)} (1x) + {len(x1_patches)} (0.5x) + 1 (0.25x) = {len(all_patches)}")

    with torch.no_grad():
        # Parts 1+2: patch encoding
        print("\n--- Parts 1+2: Patch Encoding ---")
        t1 = time.time()
        all_tokens = []
        all_block5 = []
        for i, patch in enumerate(all_patches):
            tokens_i, block5_i = part1(patch)
            all_tokens.append(tokens_i)
            all_block5.append(block5_i)
        part1_time = time.time() - t1
        print(f"  Part 1: {len(all_patches)} patches in {part1_time:.1f}s ({part1_time/len(all_patches)*1000:.0f}ms/patch)")

        t2 = time.time()
        all_features = [part2(t) for t in all_tokens]
        part2_time = time.time() - t2
        print(f"  Part 2: {len(all_tokens)} tokens in {part2_time:.1f}s ({part2_time/len(all_tokens)*1000:.0f}ms/token)")

        # Merge patches
        all_block5_spatial = [reshape_feature(b) for b in all_block5]
        all_block11_spatial = [reshape_feature(t) for t in all_tokens]
        latent0 = merge_patches_from_list(all_block5_spatial[:25], padding=3)
        latent1 = merge_patches_from_list(all_block11_spatial[:25], padding=3)
        x0_feat = merge_patches_from_list([all_features[i] for i in range(25)], padding=3)
        x1_feat = merge_patches_from_list([all_features[i] for i in range(25, 34)], padding=6)
        x2_feat = all_features[34]

        print(f"\n  Intermediate tensor shapes:")
        print(f"    latent0:      {list(latent0.shape)}")
        print(f"    latent1:      {list(latent1.shape)}")
        print(f"    x0_feat:      {list(x0_feat.shape)}")
        print(f"    x1_feat:      {list(x1_feat.shape)}")
        print(f"    x2_feat:      {list(x2_feat.shape)}")

        # Part 3: Image encoder
        print("\n--- Part 3: Image Encoder ---")
        t3 = time.time()
        image_tokens = part3(sample_image)
        part3_time = time.time() - t3
        print(f"  Part 3: {part3_time:.1f}s")
        print(f"    image_tokens: {list(image_tokens.shape)}")

        # Part 4: Decoder
        print("\n--- Part 4: Decoder + Gaussians ---")
        t4 = time.time()
        output = part4(sample_image, image_tokens, latent0, latent1, x0_feat, x1_feat, x2_feat)
        part4_time = time.time() - t4
        print(f"  Part 4: {part4_time:.1f}s")
        print(f"    output: {list(output.shape)} ({output.shape[1]} Gaussians x {output.shape[2]} params)")

        total_time = part1_time + part2_time + part3_time + part4_time
        print(f"\n{'='*60}")
        print(f"TIMING SUMMARY (Python, CPU, FP32)")
        print(f"{'='*60}")
        print(f"  Part 1 (35 patches):  {part1_time:6.1f}s  ({part1_time/total_time*100:4.1f}%)")
        print(f"  Part 2 (35 tokens):   {part2_time:6.1f}s  ({part2_time/total_time*100:4.1f}%)")
        print(f"  Part 3 (image enc):   {part3_time:6.1f}s  ({part3_time/total_time*100:4.1f}%)")
        print(f"  Part 4 (decoder):     {part4_time:6.1f}s  ({part4_time/total_time*100:4.1f}%)")
        print(f"  TOTAL:                {total_time:6.1f}s")

    intermediates = {
        "image": sample_image,
        "image_tokens": image_tokens,
        "latent0": latent0,
        "latent1": latent1,
        "x0_feat": x0_feat,
        "x1_feat": x1_feat,
        "x2_feat": x2_feat,
    }
    return intermediates, output, part4


def validate_ptl_part4(ptl_dir: Path, intermediates: dict, reference_output: torch.Tensor):
    """Load the TorchScript .ptl Part 4 and run it with the intermediate tensors. Compare to reference."""
    ptl_path = ptl_dir / "sharp_scripted_part4.ptl"
    if not ptl_path.exists():
        print(f"\nWARNING: {ptl_path} not found, skipping .ptl validation")
        return

    print(f"\n{'='*60}")
    print(f"VALIDATING TorchScript Part 4 (.ptl)")
    print(f"{'='*60}")

    t0 = time.time()
    part4_ptl = torch.jit.load(str(ptl_path), map_location="cpu")
    print(f"  Loaded {ptl_path.name} in {time.time() - t0:.1f}s")

    with torch.no_grad():
        t1 = time.time()
        ptl_output = part4_ptl(
            intermediates["image"],
            intermediates["image_tokens"],
            intermediates["latent0"],
            intermediates["latent1"],
            intermediates["x0_feat"],
            intermediates["x1_feat"],
            intermediates["x2_feat"],
        )
        ptl_time = time.time() - t1
        print(f"  Forward: {ptl_time:.1f}s")

    diff = (ptl_output - reference_output).abs()
    print(f"  Output shape: {list(ptl_output.shape)}")
    print(f"  Max diff vs Python Part 4:  {diff.max().item():.6e}")
    print(f"  Mean diff vs Python Part 4: {diff.mean().item():.6e}")

    if diff.max().item() < 1e-3:
        print(f"  RESULT: MATCH - .ptl Part 4 produces identical output")
    elif diff.max().item() < 1.0:
        print(f"  RESULT: CLOSE - small numerical differences (FP32 tracing artifacts)")
    else:
        print(f"  RESULT: MISMATCH - outputs differ significantly")


def validate_onnx_intermediates(onnx_dir: Path, intermediates: dict):
    """Run ONNX Parts 1-3 on the same random input. Compare intermediate tensors to NativePt."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nWARNING: onnxruntime not installed, skipping ONNX comparison")
        return

    required = [onnx_dir / f"sharp_part{i}.onnx" for i in range(1, 4)]
    missing = [f for f in required if not f.exists()]
    if missing:
        print(f"\nWARNING: Missing ONNX files: {[f.name for f in missing]}")
        print("  Pull from device: adb pull /storage/emulated/0/Android/data/com.furnit.android/files/models/ .")
        return

    print(f"\n{'='*60}")
    print(f"ONNX Parts 1-3 vs NativePt: Intermediate Tensor Comparison")
    print(f"{'='*60}")

    image_np = intermediates["image"].numpy()
    all_outputs = {"image": image_np}

    for part_idx in range(3):
        part_path = onnx_dir / f"sharp_part{part_idx + 1}.onnx"
        print(f"\n  Loading ONNX Part {part_idx + 1}...")
        session = ort.InferenceSession(str(part_path))
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]

        feed = {}
        for name in input_names:
            if name in all_outputs:
                feed[name] = all_outputs[name]
            else:
                print(f"    WARNING: input '{name}' not found in accumulated outputs")

        if len(feed) != len(input_names):
            print(f"    SKIP: only {len(feed)}/{len(input_names)} inputs available")
            continue

        t0 = time.time()
        results = session.run(None, feed)
        print(f"    Part {part_idx + 1}: {time.time() - t0:.1f}s, {len(output_names)} outputs")

        for name, arr in zip(output_names, results):
            all_outputs[name] = arr
            if arr.size > 2048:
                print(f"    -> {name}: shape={list(arr.shape)}")

    # Now try to match ONNX outputs to NativePt intermediates
    print(f"\n  Matching ONNX outputs to NativePt intermediates by shape...")
    shape_buckets = {}
    for name, arr in all_outputs.items():
        if isinstance(arr, np.ndarray) and arr.size > 2048:
            key = tuple(arr.shape)
            shape_buckets.setdefault(key, []).append((name, arr))

    nativept_tensors = {
        "image_tokens": ((1, 577, 1024), intermediates["image_tokens"].numpy()),
        "latent0": ((1, 1024, 96, 96), intermediates["latent0"].numpy()),
        "latent1": ((1, 1024, 96, 96), intermediates["latent1"].numpy()),
        "x0_feat": ((1, 1024, 96, 96), intermediates["x0_feat"].numpy()),
        "x1_feat": ((1, 1024, 48, 48), intermediates["x1_feat"].numpy()),
        "x2_feat": ((1, 1024, 24, 24), intermediates["x2_feat"].numpy()),
    }

    for npt_name, (expected_shape, npt_arr) in nativept_tensors.items():
        candidates = shape_buckets.get(expected_shape, [])
        if not candidates:
            print(f"\n  {npt_name} {expected_shape}: NO ONNX candidates found")
            continue

        print(f"\n  {npt_name} {expected_shape}: {len(candidates)} ONNX candidates")
        for onnx_name, onnx_arr in candidates:
            diff = np.abs(onnx_arr - npt_arr)
            corr = np.corrcoef(onnx_arr.flatten()[:10000], npt_arr.flatten()[:10000])[0, 1]
            print(f"    vs {onnx_name:<60} max_diff={diff.max():.4f} mean_diff={diff.mean():.4f} corr={corr:.4f}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Hybrid ONNX+NativePt feasibility test")
    parser.add_argument("--inspect-onnx", action="store_true",
                        help="Only inspect ONNX Part 4 inputs (Step 1)")
    parser.add_argument("--onnx-dir", type=Path, default=None,
                        help="Directory containing sharp_part*.onnx files")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help=f"SHARP weights file (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--ptl-dir", type=Path, default=DEFAULT_PTL_DIR,
                        help=f"Directory containing .ptl files (default: {DEFAULT_PTL_DIR})")
    parser.add_argument("--sharp-src", type=Path, default=DEFAULT_SHARP_SRC,
                        help=f"SHARP source directory (default: {DEFAULT_SHARP_SRC})")
    parser.add_argument("--timing", action="store_true",
                        help="Include detailed timing breakdown")
    args = parser.parse_args()

    if args.inspect_onnx:
        if args.onnx_dir is None:
            print("ERROR: --onnx-dir required with --inspect-onnx")
            return 1
        inspect_onnx_part4(args.onnx_dir)
        return 0

    if not args.weights.exists():
        print(f"ERROR: Weights not found at {args.weights}")
        print("  Expected: android/sharp_litert_models/sharp_2572gikvuh.pt")
        return 1

    if not args.sharp_src.exists():
        print(f"ERROR: SHARP source not found at {args.sharp_src}")
        return 1

    print("="*60)
    print("HYBRID ONNX+NativePt FEASIBILITY TEST")
    print("="*60)

    # Run NativePt pipeline (Python reference)
    intermediates, reference_output, python_part4 = run_nativept_pipeline(args.weights, args.sharp_src)

    # Validate .ptl Part 4 matches Python Part 4
    validate_ptl_part4(args.ptl_dir, intermediates, reference_output)

    # If ONNX files available, compare intermediates
    if args.onnx_dir and args.onnx_dir.exists():
        validate_onnx_intermediates(args.onnx_dir, intermediates)

    # Summary
    print(f"\n{'='*60}")
    print("HYBRID FEASIBILITY CONCLUSION")
    print(f"{'='*60}")
    print("""
The hybrid approach works IF:
  1. .ptl Part 4 produces same output as Python Part 4 (validated above)
  2. ONNX Parts 1-3 intermediate tensors match NativePt intermediates
     (validate with --onnx-dir after adb pull)

Implementation:
  SplitOnnxSharp.kt runs ONNX Parts 1-3 (fast, single-pass encoder)
  Then loads sharp_scripted_part4.ptl via LiteModuleLoader
  Maps the 7 ONNX output tensors to Part 4 positional inputs
  Runs NativePt Part 4 (weights baked in, 7 inputs instead of 60)

Expected benefit:
  - No 53 weight tensors as Part 4 inputs (baked into .ptl)
  - PyTorch Mobile native allocator (not ORT arena)
  - Potentially lower peak memory for Part 4 decoder activations
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
