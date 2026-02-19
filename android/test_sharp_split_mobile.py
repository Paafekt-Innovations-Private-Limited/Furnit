#!/usr/bin/env python3
"""
Test ExecuTorch SHARP Part1 on a room image with mobile-like settings.

Simulates mobile hardware by limiting threads. Outputs JSON with timing and
output checksums for parity comparison with Kotlin/Android.

Usage:
  python test_sharp_split_mobile.py --image /Users/al/Downloads/PXL_20260209_032207120.jpg
  python test_sharp_split_mobile.py --image room.jpg --output results.json

Output: JSON with load_ms, forward_ms, tokens_shape, tokens_checksum, block5_checksum.
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

# Simulate mobile: limit threads before importing torch/executorch
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
import torch

PATCH_SIZE = 384
IMAGE_SIZE = 1536
PART1_PTE = Path(__file__).resolve().parent / "executorch_models" / "sharp_split_part1.pte"


def load_and_preprocess_patch(image_path: str, patch_idx: int = 0) -> np.ndarray:
    """Load image, resize to 1536, extract patch 0 (top-left), return CHW float [0,1]."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC [0,1]

    # Extract patch 0: top-left 384x384 (same as Android stride for grid 0,0)
    stride = (IMAGE_SIZE - PATCH_SIZE) // 4  # 288 for 5x5 grid
    y, x = 0, 0  # patch 0
    patch = arr[y : y + PATCH_SIZE, x : x + PATCH_SIZE, :]  # 384,384,3

    # HWC -> CHW
    patch_chw = np.transpose(patch, (2, 0, 1))  # 3, 384, 384
    return patch_chw.astype(np.float32)


def checksum_floats(arr: np.ndarray, n: int = 32) -> list:
    """First n values for parity check."""
    flat = arr.flatten()
    return [float(flat[i]) for i in range(min(n, len(flat)))]


def run_part1_mobile(image_path: str, output_json: str | None) -> dict:
    pte_path = PART1_PTE
    if not pte_path.exists():
        raise FileNotFoundError(f"Part1 .pte not found: {pte_path}")

    # Limit PyTorch threads (mobile-like)
    torch.set_num_threads(4)

    # Load patch
    patch_chw = load_and_preprocess_patch(image_path)
    patch_batch = np.expand_dims(patch_chw, axis=0)  # 1, 3, 384, 384
    tensor = torch.from_numpy(patch_batch)

    # Load ExecuTorch Part1
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    t0 = time.perf_counter()
    program = runtime.load_program(pte_path)
    forward_method = program.load_method("forward")
    load_ms = (time.perf_counter() - t0) * 1000

    # Forward
    t1 = time.perf_counter()
    outputs = forward_method.execute([tensor])
    forward_ms = (time.perf_counter() - t1) * 1000

    out0 = outputs[0]
    out1 = outputs[1]
    tokens = out0.numpy() if hasattr(out0, "numpy") else np.array(out0)
    block5 = out1.numpy() if hasattr(out1, "numpy") else np.array(out1)

    result = {
        "image": image_path,
        "load_ms": round(load_ms, 1),
        "forward_ms": round(forward_ms, 1),
        "tokens_shape": list(tokens.shape),
        "block5_shape": list(block5.shape),
        "tokens_checksum": checksum_floats(tokens),
        "block5_checksum": checksum_floats(block5),
    }

    print("=" * 60)
    print("Python ExecuTorch Part1 (mobile-like: 4 threads)")
    print("=" * 60)
    print(f"Image:       {image_path}")
    print(f"Load:        {result['load_ms']} ms")
    print(f"Forward:     {result['forward_ms']} ms")
    print(f"Tokens:      {result['tokens_shape']}")
    print(f"Block5:      {result['block5_shape']}")
    print(f"tokens_checksum[:8] = {result['tokens_checksum'][:8]}")

    if output_json:
        Path(output_json).write_text(json.dumps(result, indent=2))
        print(f"\nWrote {output_json}")

    return result


def main():
    ap = argparse.ArgumentParser(description="Test SHARP Part1 with mobile-like settings")
    ap.add_argument("--image", "-i", default="/Users/al/Downloads/PXL_20260209_032207120.jpg", help="Room image path")
    ap.add_argument("--output", "-o", help="Write results JSON for Kotlin parity comparison")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    run_part1_mobile(args.image, args.output)


if __name__ == "__main__":
    main()
