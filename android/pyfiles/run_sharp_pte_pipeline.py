#!/usr/bin/env python3
"""
Run the full SHARP 4-part pipeline using ExecuTorch .pte models (Vulkan or portable).
Run with PYTHONUNBUFFERED=1 to see progress (e.g. Part1 35 patches can take minutes on CPU).

Loads one image, builds 35 patches, runs Part1 -> Part2 -> merge -> Part3 -> Part4
(or Part4a chunked + Part4b if those .pte exist), and prints/saves the Gaussian output.

Usage:
  python run_sharp_pte_pipeline.py --image /path/to/room.jpeg --models-dir executorch_models
  python run_sharp_pte_pipeline.py --image app/build/intermediates/assets/debug/mergeDebugAssets/room.jpeg

Note: Vulkan .pte will use the Vulkan delegate when executed; on macOS Vulkan may be
unavailable and execution can fail. Use portable .pte for CPU-only validation, or run
on a Vulkan-capable system (e.g. Linux) to test Vulkan models.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Default image path (mergeDebugAssets from Android debug build); override with --image
_ANDROID = Path(__file__).resolve().parent
DEFAULT_IMAGE = _ANDROID / "app/build/intermediates/assets/debug/mergeDebugAssets/room.jpeg"
IMAGE_SIZE = 1536
PATCH_SIZE = 384
VIT_SPLIT_BLOCK = 12
CHUNK_LEN_FIRST = 512
CHUNK_LEN_LAST = 577 - CHUNK_LEN_FIRST  # 65


def load_image_as_tensor(image_path: Path, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Load image, resize to size x size, return NCHW float [0, 1]."""
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("pip install Pillow")
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    import numpy as np
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    # HWC -> NCHW
    arr = arr.permute(2, 0, 1).unsqueeze(0)
    return arr


def split_patches_list(image: torch.Tensor, overlap_ratio: float, patch_size: int, patch_stride: int | None = None) -> list:
    if patch_stride is None:
        patch_stride = int(patch_size * (1 - overlap_ratio))
    image_size = image.shape[-1]
    steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1
    patches = []
    for j in range(steps):
        for i in range(steps):
            j0, i0 = j * patch_stride, i * patch_stride
            patches.append(image[..., j0 : j0 + patch_size, i0 : i0 + patch_size])
    return patches


def merge_patches_from_list(patches: list, padding: int) -> torch.Tensor:
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


def reshape_feature(embeddings: torch.Tensor, num_prefix_tokens: int = 1, grid_size: tuple = (24, 24)) -> torch.Tensor:
    if num_prefix_tokens:
        embeddings = embeddings[:, num_prefix_tokens:, :]
    B, N, C = embeddings.shape
    h, w = grid_size
    return embeddings.reshape(B, h, w, C).permute(0, 3, 1, 2)


def find_pte(models_dir: Path, base: str, prefer_vulkan: bool = True) -> Path | None:
    """Return path to .pte; prefer Vulkan or portable (CPU) variants."""
    if prefer_vulkan:
        candidates = [
            models_dir / f"{base}_vulkan_fp16.pte",
            models_dir / f"{base}_vulkan.pte",
            models_dir / f"{base}.pte",
        ]
    else:
        candidates = [
            models_dir / f"{base}_fp16.pte",
            models_dir / f"{base}.pte",
        ]
    for p in candidates:
        if p.exists() and (not p.is_symlink() or p.resolve().exists()):
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Run SHARP 4-part pipeline from .pte models and one image.")
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input image path")
    ap.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parent / "executorch_models", help="Directory containing Part1–4 .pte files")
    ap.add_argument("--output", type=Path, default=None, help="Optional: save packed Gaussians as .pt")
    ap.add_argument("--no-chunked", action="store_true", help="Use full Part4 only (ignore Part4a/4b if present)")
    ap.add_argument("--portable", action="store_true", help="Use portable (CPU) .pte instead of Vulkan (for macOS / no-Vulkan hosts)")
    ap.add_argument("--one-patch", action="store_true", help="Run only one Part1 patch then exit (fast isolation test)")
    args = ap.parse_args()
    prefer_vulkan = not args.portable

    if not args.image.exists():
        print(f"Error: image not found: {args.image}")
        return 1
    if not args.models_dir.exists():
        print(f"Error: models dir not found: {args.models_dir}")
        return 1

    try:
        from executorch.runtime import Runtime, Program, Method
    except ImportError:
        print("Error: executorch not installed. pip install executorch")
        return 1

    runtime = Runtime.get()
    print("Loading image...", flush=True)
    image = load_image_as_tensor(args.image)
    if prefer_vulkan:
        pass  # Vulkan FP16 .pte often accept float; runtime may convert
    else:
        # Portable _fp16.pte expect Half input
        image = image.to(torch.float16)
    print(f"  Image shape: {image.shape} (dtype={image.dtype})", flush=True)

    # Resolutions for patches
    x0_raw = image
    x1_raw = F.interpolate(image, scale_factor=0.5, mode="bilinear", align_corners=False)
    x2_raw = F.interpolate(image, scale_factor=0.25, mode="bilinear", align_corners=False)
    x0_patches = split_patches_list(x0_raw, 0.25, PATCH_SIZE)
    x1_patches = split_patches_list(x1_raw, 0.5, PATCH_SIZE)
    all_patches = x0_patches + x1_patches + [x2_raw]
    print(f"  Patches: {len(all_patches)}", flush=True)

    # Load Part1
    p1_path = find_pte(args.models_dir, "sharp_split_part1", prefer_vulkan)
    if not p1_path:
        print("Error: sharp_split_part1*.pte not found", flush=True)
        return 1
    print(f"  Part1: {p1_path.name}", flush=True)
    t0 = time.time()
    prog1 = runtime.load_program(p1_path)
    print(f"  Part1 program loaded in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    forward1: Method = prog1.load_method("forward")
    print(f"  Part1 method loaded in {time.time() - t0:.1f}s", flush=True)

    all_tokens = []
    all_block5 = []
    n_patches = 1 if args.one_patch else len(all_patches)
    if args.one_patch:
        print("  Running one Part1 test patch (--one-patch)...", flush=True)
    for i in range(n_patches):
        patch = all_patches[i]
        t0 = time.time()
        out = forward1.execute((patch.contiguous(),))
        print(f"  Part1 patch {i + 1}/{n_patches} done in {time.time() - t0:.1f}s", flush=True)
        if not out or len(out) < 2:
            print(f"  Part1 patch {i}: expected 2 outputs, got {len(out) if out else 0}", flush=True)
            return 1
        all_tokens.append(out[0])
        all_block5.append(out[1])
    if args.one_patch:
        print("  One Part1 test patch finished. Exiting (--one-patch).", flush=True)
        return 0
    print("  Part1: 35 patches OK", flush=True)

    # Load Part2
    p2_path = find_pte(args.models_dir, "sharp_split_part2", prefer_vulkan)
    if not p2_path:
        print("Error: sharp_split_part2*.pte not found", flush=True)
        return 1
    print(f"  Part2: {p2_path.name}", flush=True)
    t0 = time.time()
    prog2 = runtime.load_program(p2_path)
    print(f"  Part2 program loaded in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    forward2: Method = prog2.load_method("forward")
    print(f"  Part2 method loaded in {time.time() - t0:.1f}s", flush=True)
    all_features = []
    for i, t in enumerate(all_tokens):
        t0 = time.time()
        out = forward2.execute((t.contiguous(),))
        if i == 0 or (i + 1) % 10 == 0 or i == len(all_tokens) - 1:
            print(f"  Part2 patch {i + 1}/{len(all_tokens)} done in {time.time() - t0:.1f}s", flush=True)
        all_features.append(out[0])
    all_block5_spatial = [reshape_feature(b) for b in all_block5]
    all_block11_spatial = [reshape_feature(t) for t in all_tokens]
    latent0 = merge_patches_from_list(all_block5_spatial[:25], 3)
    latent1 = merge_patches_from_list(all_block11_spatial[:25], 3)
    x0_feat = merge_patches_from_list(all_features[:25], 3)
    x1_feat = merge_patches_from_list(all_features[25:34], 6)
    x2_feat = all_features[34]
    print("  Part2 + merge OK", flush=True)

    # Part3
    p3_path = find_pte(args.models_dir, "sharp_split_part3", prefer_vulkan)
    if not p3_path:
        print("Error: sharp_split_part3*.pte not found")
        return 1
    print(f"  Part3: {p3_path.name}", flush=True)
    t0 = time.time()
    prog3 = runtime.load_program(p3_path)
    print(f"  Part3 program loaded in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    forward3: Method = prog3.load_method("forward")
    print(f"  Part3 method loaded in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    out3 = forward3.execute((image.contiguous(),))
    print(f"  Part3 execute in {time.time() - t0:.1f}s", flush=True)
    image_tokens = out3[0]
    print(f"  Part3 output: {image_tokens.shape}", flush=True)

    # Part4: full or chunked
    use_chunked = not args.no_chunked
    p4a_512 = find_pte(args.models_dir, "sharp_split_part4a_chunk_512", prefer_vulkan)
    p4a_65 = find_pte(args.models_dir, "sharp_split_part4a_chunk_65", prefer_vulkan)
    p4b = find_pte(args.models_dir, "sharp_split_part4b", prefer_vulkan)
    if use_chunked and p4a_512 and p4a_65 and p4b:
        print("  Part4: chunked (4a_512 + 4a_65 + 4b)", flush=True)
        prog4a_512 = runtime.load_program(p4a_512)
        prog4a_65 = runtime.load_program(p4a_65)
        prog4b = runtime.load_program(p4b)
        f4a_512 = prog4a_512.load_method("forward")
        f4a_65 = prog4a_65.load_method("forward")
        f4b = prog4b.load_method("forward")
        tok_512 = image_tokens[:, :CHUNK_LEN_FIRST]
        tok_65 = image_tokens[:, CHUNK_LEN_FIRST:]
        out_512 = f4a_512.execute((tok_512.contiguous(),))[0]
        out_65 = f4a_65.execute((tok_65.contiguous(),))[0]
        tokens_after_blocks = torch.cat([out_512, out_65], dim=1)
        out4 = f4b.execute((tokens_after_blocks.contiguous(), image.contiguous(), latent0.contiguous(), latent1.contiguous(), x0_feat.contiguous(), x1_feat.contiguous(), x2_feat.contiguous()))
    else:
        p4_path = find_pte(args.models_dir, "sharp_split_part4", prefer_vulkan)
        if not p4_path:
            print("Error: sharp_split_part4*.pte (or Part4a/4b) not found")
            return 1
        print(f"  Part4: full {p4_path.name}", flush=True)
        prog4 = runtime.load_program(p4_path)
        forward4: Method = prog4.load_method("forward")
        out4 = forward4.execute((image.contiguous(), image_tokens.contiguous(), latent0.contiguous(), latent1.contiguous(), x0_feat.contiguous(), x1_feat.contiguous(), x2_feat.contiguous()))
    packed = out4[0]
    n_gauss = packed.shape[1]
    print(f"  Packed Gaussians: {packed.shape} ({n_gauss:,} Gaussians)", flush=True)

    if args.output:
        torch.save({"packed": packed}, args.output)
        print(f"  Saved to {args.output}", flush=True)

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
