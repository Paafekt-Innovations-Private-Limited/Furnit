#!/usr/bin/env python3
"""
Brutal narrow test: Part1/Part2 eager vs exported .pte on the SAME input.

Proves whether export is broken before touching the app:
  - Test A: Part1 eager vs Part1 .pte (same patch input)
  - Test B: Part2 eager vs Part2 .pte (same tokens input = Part1 eager output)

Logs only: input shape/dtype, output shape/dtype, first 8 values, min/max, error code.
Saves fixtures to verify_export_fixtures/ so runs are reproducible.

Usage:
  cd android
  python verify_export_part1_part2.py [--output-dir executorch_models] [--sharp-src third_party/ml-sharp/src] [--weights sharp_litert_models/sharp_2572gikvuh.pt]
  python verify_export_part1_part2.py --part1-only   # only Part1
  python verify_export_part1_part2.py --part2-only   # only Part2 (requires fixtures from a prior full run)
"""

import argparse
import sys
from pathlib import Path

import torch

# Same as export
PATCH_SIZE = 384

SCRIPT_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = SCRIPT_DIR / "verify_export_fixtures"


def log_line(label: str, shape, dtype, first8=None, min_max=None, error=None):
    parts = [f"  {label}: shape={tuple(shape)} dtype={dtype}"]
    if first8 is not None:
        flat = first8.flatten()
        n = min(8, flat.numel())
        parts.append(f" first8={flat[:n].tolist()}")
    if min_max is not None:
        parts.append(f" min={min_max[0]:.6f} max={min_max[1]:.6f}")
    if error is not None:
        parts.append(f" error={error}")
    print("".join(parts))


def summarize_tensor(t, name: str):
    s = t.shape
    d = str(t.dtype)
    first8 = t.flatten()[:8] if t.numel() >= 8 else t.flatten()
    mn, mx = t.min().item(), t.max().item()
    log_line(name, s, d, first8=first8, min_max=(mn, mx))


def compare_outputs(eager_out, pte_out, name: str) -> bool:
    """Compare eager vs .pte output; log and return True if match (shape/dtype/values)."""
    if isinstance(eager_out, (list, tuple)):
        if not isinstance(pte_out, (list, tuple)) or len(pte_out) != len(eager_out):
            print(f"  {name}: output structure mismatch: eager={len(eager_out)} vs pte={type(pte_out)} len={len(pte_out) if hasattr(pte_out, '__len__') else 'n/a'}")
            return False
        ok = True
        for i, (e, p) in enumerate(zip(eager_out, pte_out)):
            if not compare_outputs(e, p, f"{name}[{i}]"):
                ok = False
        return ok
    # single tensor
    if not isinstance(pte_out, torch.Tensor):
        pte_out = torch.tensor(pte_out) if pte_out is not None else None
    if pte_out is None:
        print(f"  {name}: pte output is None")
        return False
    if tuple(eager_out.shape) != tuple(pte_out.shape):
        print(f"  {name}: shape mismatch eager={eager_out.shape} pte={pte_out.shape}")
        return False
    if eager_out.dtype != pte_out.dtype:
        print(f"  {name}: dtype mismatch eager={eager_out.dtype} pte={pte_out.dtype}")
    summarize_tensor(eager_out, f"{name}_eager")
    summarize_tensor(pte_out, f"{name}_pte")
    diff = (eager_out.float() - pte_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  {name}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    return max_diff < 0.01  # allow small numerical difference


def run_part1_eager(part1, patch_input: torch.Tensor, fixtures_dir: Path):
    """Run Part1 eager, save input and outputs, return (tokens, block5)."""
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        tokens, block5 = part1(patch_input)
    torch.save(patch_input.cpu(), fixtures_dir / "part1_input.pt")
    torch.save(tokens.cpu(), fixtures_dir / "part1_eager_tokens.pt")
    torch.save(block5.cpu(), fixtures_dir / "part1_eager_block5.pt")
    return tokens, block5


def run_part2_eager(part2, tokens_input: torch.Tensor, fixtures_dir: Path):
    """Run Part2 eager, save output, return feature tensor."""
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        out = part2(tokens_input)
    torch.save(tokens_input.cpu(), fixtures_dir / "part2_input.pt")
    torch.save(out.cpu(), fixtures_dir / "part2_eager_output.pt")
    return out


def _vulkan_backend_hint(pte_path: Path, err: str) -> str:
    """If this looks like a Vulkan backend not registered / BackendFailed, append hint."""
    if "vulkan" not in pte_path.name.lower():
        return err
    if any(x in err.lower() for x in ("not registered", "backendfailed", "backend failed", "vulkanbackend")):
        return err + " [Vulkan .pte requires executorch built/linked with VulkanBackend; see android/docs/EXECUTORCH_EXPORT_VERIFY.md]"
    return err


def run_pte_forward(pte_path: Path, *inputs) -> tuple:
    """Load .pte and run forward; return tuple of outputs. On error return (None, error_str)."""
    try:
        from executorch.runtime import Runtime
        runtime = Runtime.get()
        program = runtime.load_program(pte_path)
        if "forward" not in program.method_names:
            return (None, f"no 'forward' in method_names: {program.method_names}")
        forward = program.load_method("forward")
        # inputs: tuple of tensors
        result = forward.execute(inputs)
        if result is None:
            return (None, "execute returned None")
        # result may be tuple or single tensor
        if isinstance(result, (list, tuple)):
            return tuple(result)
        return (result,)
    except Exception as e:
        err = str(e)
        return (None, _vulkan_backend_hint(pte_path, err))


def main():
    ap = argparse.ArgumentParser(description="Verify Part1/Part2 export: eager vs .pte on same input")
    ap.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "executorch_models", help="Directory with .pte files")
    ap.add_argument("--sharp-src", type=Path, default=SCRIPT_DIR / "third_party/ml-sharp/src")
    ap.add_argument("--weights", type=Path, default=SCRIPT_DIR / "sharp_litert_models/sharp_2572gikvuh.pt")
    ap.add_argument("--fixtures-dir", type=Path, default=FIXTURES_DIR)
    ap.add_argument("--part1-only", action="store_true")
    ap.add_argument("--part2-only", action="store_true")
    ap.add_argument("--batch", type=int, default=1, choices=(1, 2), help="Batch size (1 = single patch; 2 = Vulkan app path)")
    ap.add_argument("--portable-only", action="store_true", help="Only use portable .pte (sharp_split_part1.pte, part2.pte); no Vulkan. Use when executorch has no Vulkan support.")
    args = ap.parse_args()

    args.fixtures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Verify export: Part1 & Part2 eager vs .pte (same input)")
    print("  output_dir (pte):", args.output_dir)
    print("  fixtures_dir:    ", args.fixtures_dir)
    print("  batch:           ", args.batch)
    print("  portable_only:    ", args.portable_only)
    print("=" * 60)

    # Resolve .pte names (Vulkan FP16 vs portable)
    if args.portable_only:
        part1_pte = args.output_dir / "sharp_split_part1.pte"
        part2_pte = args.output_dir / "sharp_split_part2.pte"
    elif args.batch == 1:
        part1_pte = args.output_dir / "sharp_split_part1_vulkan_fp16.pte"
        part2_pte = args.output_dir / "sharp_split_part2_vulkan_fp16.pte"
        if not part1_pte.exists():
            part1_pte = args.output_dir / "sharp_split_part1.pte"
        if not part2_pte.exists():
            part2_pte = args.output_dir / "sharp_split_part2.pte"
    else:
        part1_pte = args.output_dir / "sharp_split_part1_b2_vulkan_fp16.pte"
        part2_pte = args.output_dir / "sharp_split_part2_b2_vulkan_fp16.pte"

    if not args.portable_only and ("vulkan" in part1_pte.name or "vulkan" in part2_pte.name):
        print("  Note: Vulkan .pte requires executorch installed/built with Vulkan support (VulkanBackend). Use --portable-only to verify with portable .pte only.")
    print("  Part1 .pte:", part1_pte.name if part1_pte.exists() else part1_pte, "(exists:", part1_pte.exists(), ")")
    print("  Part2 .pte:", part2_pte.name if part2_pte.exists() else part2_pte, "(exists:", part2_pte.exists(), ")")

    part1 = part2 = None
    if not args.part2_only:
        if not args.sharp_src.exists():
            print("ERROR: sharp_src not found:", args.sharp_src)
            return 1
        if not args.weights.exists():
            print("ERROR: weights not found:", args.weights)
            return 1
        sys.path.insert(0, str(args.sharp_src))
        from export_sharp_executorch_split4 import (
            SinglePatchEncoderA,
            SinglePatchEncoderB,
            fuse_conv_bn,
        )
        from sharp.models import PredictorParams, create_predictor

        print("\nLoading SHARP...")
        state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()
        fuse_conv_bn(predictor)
        part1 = SinglePatchEncoderA(predictor).eval()
        part2 = SinglePatchEncoderB(predictor).eval()
        del state_dict

    # Deterministic input
    torch.manual_seed(42)
    patch_input = torch.rand(args.batch, 3, PATCH_SIZE, PATCH_SIZE)

    # --- Test A: Part1 ---
    if not args.part2_only:
        print("\n--- Test A: Part1 eager vs Part1 .pte ---")
        log_line("part1_input", patch_input.shape, str(patch_input.dtype))

        tokens_eager, block5_eager = run_part1_eager(part1, patch_input, args.fixtures_dir)
        print("  Part1 eager:")
        summarize_tensor(tokens_eager, "  tokens")
        summarize_tensor(block5_eager, "  block5")

        if not part1_pte.exists():
            print("  Part1 .pte not found; skip .pte run. Export with: --backend vulkan --chunked-part4 --dtype fp16")
        else:
            pte_in = patch_input  # same dtype as export (fp32 or fp16); runtime usually accepts both
            out = run_pte_forward(part1_pte, pte_in)
            if out[0] is None and len(out) > 1:
                print("  Part1 .pte error:", out[1])
            else:
                print("  Part1 .pte:")
                pte_out = (out[0], out[1]) if len(out) >= 2 else (out[0],)
                compare_outputs((tokens_eager, block5_eager), pte_out, "Part1")

    # --- Test B: Part2 ---
    if not args.part1_only:
        print("\n--- Test B: Part2 eager vs Part2 .pte ---")
        tokens_path = args.fixtures_dir / "part1_eager_tokens.pt"
        part2_eager_path = args.fixtures_dir / "part2_eager_output.pt"
        if args.part2_only and (not tokens_path.exists() or not part2_eager_path.exists()):
            print("  For --part2-only need part1_eager_tokens.pt and part2_eager_output.pt. Run full verify once first.")
        elif not tokens_path.exists() and part2 is None:
            print("  part1_eager_tokens.pt not found. Run without --part2-only first.")
        else:
            if tokens_path.exists():
                tokens_input = torch.load(tokens_path)
            else:
                run_part1_eager(part1, patch_input, args.fixtures_dir)
                tokens_input = torch.load(args.fixtures_dir / "part1_eager_tokens.pt")
            if args.batch != tokens_input.shape[0]:
                tokens_input = tokens_input[: args.batch]
            log_line("part2_input (tokens)", tokens_input.shape, str(tokens_input.dtype))

            if part2 is not None:
                feat_eager = run_part2_eager(part2, tokens_input, args.fixtures_dir)
                print("  Part2 eager:")
                summarize_tensor(feat_eager, "  output")
            else:
                feat_eager = torch.load(part2_eager_path)

            if not part2_pte.exists():
                print("  Part2 .pte not found; skip .pte run.")
            else:
                out = run_pte_forward(part2_pte, tokens_input)
                if out[0] is None and len(out) > 1:
                    print("  Part2 .pte error:", out[1])
                else:
                    print("  Part2 .pte:")
                    compare_outputs(feat_eager, out[0] if isinstance(out, (list, tuple)) else out, "Part2")

    print("\nDone. Fixtures in", args.fixtures_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
