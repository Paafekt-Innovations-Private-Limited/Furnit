#!/usr/bin/env python3
"""
Generate Part 1 test fixtures only: one fixed input patch + golden outputs (no export).

Use when you already have a Part 1 .pte and only need the test patch and Python
golden outputs to compare in the app. Same fixed seed and shapes as --part1-only export.

Output in --output-dir:
  part1_test_patch.pt
  part1_test_patch_f32.bin
  part1_test_patch_f16.bin (if --fp16)
  part1_tokens_golden_f32.bin
  part1_block5_golden_f32.bin

Usage:
  cd android
  python generate_part1_test_fixtures.py --output-dir executorch_models
  python generate_part1_test_fixtures.py --output-dir executorch_models --fp16
"""

import argparse
import sys
from pathlib import Path

import torch

PATCH_SIZE = 384
SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Generate Part 1 test patch + golden outputs (no .pte export)")
    ap.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "executorch_models")
    ap.add_argument("--sharp-src", type=Path, default=SCRIPT_DIR / "third_party/ml-sharp/src")
    ap.add_argument("--weights", type=Path, default=SCRIPT_DIR / "sharp_litert_models/sharp_2572gikvuh.pt")
    ap.add_argument("--fp16", action="store_true", help="Also save f16 patch and run Part1 in half() for goldens")
    args = ap.parse_args()

    if not args.sharp_src.exists():
        print("ERROR: sharp_src not found:", args.sharp_src)
        return 1
    if not args.weights.exists():
        print("ERROR: weights not found:", args.weights)
        return 1

    sys.path.insert(0, str(args.sharp_src))
    from export_sharp_executorch_split4 import SinglePatchEncoderA, fuse_conv_bn
    from sharp.models import PredictorParams, create_predictor

    print("Loading SHARP...")
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    fuse_conv_bn(predictor)
    part1 = SinglePatchEncoderA(predictor).eval()
    del state_dict

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    sample_patch = torch.rand(1, 3, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32)

    # Save test input
    torch.save(sample_patch, args.output_dir / "part1_test_patch.pt")
    sample_patch.numpy().tofile(args.output_dir / "part1_test_patch_f32.bin")
    if args.fp16:
        sample_patch.half().numpy().tofile(args.output_dir / "part1_test_patch_f16.bin")

    # Golden outputs (eager Part1); match dtype to what you export
    with torch.no_grad():
        patch_eval = sample_patch.half() if args.fp16 else sample_patch
        model_eval = part1.half() if args.fp16 else part1
        tokens, block5 = model_eval(patch_eval)
    tokens_f32 = tokens.cpu().float()
    block5_f32 = block5.cpu().float()
    tokens_f32.numpy().tofile(args.output_dir / "part1_tokens_golden_f32.bin")
    block5_f32.numpy().tofile(args.output_dir / "part1_block5_golden_f32.bin")

    print("Part 1 golden outputs (eager, fixed seed 42):")
    print("  tokens ", tokens_f32.shape, tokens_f32.dtype,
          "min={:.6f} max={:.6f} mean={:.6f}".format(
              tokens_f32.min().item(), tokens_f32.max().item(), tokens_f32.mean().item()))
    print("  tokens first 8:", tokens_f32.flatten()[:8].tolist())
    print("  block5 ", block5_f32.shape, block5_f32.dtype,
          "min={:.6f} max={:.6f} mean={:.6f}".format(
              block5_f32.min().item(), block5_f32.max().item(), block5_f32.mean().item()))
    print("  block5 first 8:", block5_f32.flatten()[:8].tolist())
    print("Saved to", args.output_dir, ": part1_test_patch*.pt/bin, part1_*_golden_f32.bin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
