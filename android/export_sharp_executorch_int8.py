#!/usr/bin/env python3
"""
Export SHARP as a single INT8 ExecuTorch .pte model with XNNPACK backend.

Uses PT2E quantization flow: export -> XNNPACKQuantizer -> calibrate -> convert -> XNNPACK delegate.
The INT8 model targets ARM NEON INT8 kernels for bandwidth reduction.

Supports variable input resolution via --imgsz:
  1536 (default): Full quality, 35 patches, ~4GB activations (may OOM on mobile)
  768:  Reduced quality, 11 patches, ~1GB activations (fits most devices)
  640:  Lower quality, 6 patches, ~600MB activations (safe for 4GB RAM devices)

The sliding pyramid is kept INSIDE the model for export simplicity. Scales that
produce sub-384 images are zero-padded to 384 so the ViT patch_embed still works.

Output: sharp_int8.pte + metadata.yaml

Usage:
  cd android
  python export_sharp_executorch_int8.py --imgsz 768
  adb push sharp_int8.pte /storage/emulated/0/Android/data/com.furnit.android/files/models/
  adb push sharp_int8_metadata.yaml /storage/emulated/0/Android/data/com.furnit.android/files/models/
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PATCH_SIZE = 384


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


def valid_grid_size(image_size, patch_size=PATCH_SIZE, overlap_ratio=0.25):
    """Compute the smallest padded size that produces uniform patch_size patches."""
    if image_size < patch_size:
        return patch_size
    stride = int(patch_size * (1 - overlap_ratio))
    steps = int(math.ceil((image_size - patch_size) / stride)) + 1
    return (steps - 1) * stride + patch_size


def pad_to_grid(tensor, overlap_ratio=0.25):
    """Pad spatial dims so split() produces only full-size patches (no edge clipping)."""
    _, _, h, w = tensor.shape
    target_h = valid_grid_size(h, PATCH_SIZE, overlap_ratio)
    target_w = valid_grid_size(w, PATCH_SIZE, overlap_ratio)
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return tensor


class SharpFullPipeline(nn.Module):
    """Complete SHARP model: image in -> [N, 14] Gaussian params out.

    Wraps the predictor with optional padding for sub-1536 input sizes.
    The predictor internally handles the sliding pyramid, patch encoding,
    image encoding, decoding, and Gaussian composition.
    """

    def __init__(self, predictor, default_disparity_factor: float = 1.0):
        super().__init__()
        self.predictor = predictor
        self.register_buffer("disparity_factor", torch.tensor([default_disparity_factor]))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        result = self.predictor(image, self.disparity_factor)
        means = result.mean_vectors
        scales = result.singular_values
        rotations = result.quaternions
        colors = result.colors
        opacities = result.opacities
        if opacities.dim() == 2:
            opacities = opacities.unsqueeze(-1)
        params = torch.cat([means, scales, rotations, opacities, colors], dim=-1)
        return params.squeeze(0)


def compute_patch_info(imgsz):
    """Compute expected patch counts and memory for a given input size (after grid padding)."""
    def steps_for(size, overlap):
        padded = valid_grid_size(size, PATCH_SIZE, overlap)
        stride = int(PATCH_SIZE * (1 - overlap))
        if padded < PATCH_SIZE:
            return 1, padded
        steps = int(math.ceil((padded - PATCH_SIZE) / stride)) + 1
        return steps, padded

    x0_size = imgsz
    x1_size = imgsz // 2
    x2_size = imgsz // 4

    x0_steps, x0_padded = steps_for(x0_size, 0.25)
    x1_steps, x1_padded = steps_for(x1_size, 0.5)

    x0_patches = x0_steps * x0_steps
    x1_patches = x1_steps * x1_steps
    x2_patches = 1
    total = x0_patches + x1_patches + x2_patches

    print(f"  Input: {imgsz}x{imgsz}")
    print(f"  1x scale: {x0_size}->{x0_padded}px, {x0_steps}x{x0_steps} = {x0_patches} patches")
    print(f"  0.5x scale: {x1_size}->{x1_padded}px, {x1_steps}x{x1_steps} = {x1_patches} patches")
    print(f"  0.25x scale: {x2_size}->{max(x2_size, PATCH_SIZE)}px, 1x1 = {x2_patches} patch")
    print(f"  Total patches: {total} (vs 35 at 1536)")
    return total


def parse_args():
    pa = argparse.ArgumentParser(description="Export SHARP as single INT8 ExecuTorch .pte")
    pa.add_argument(
        "--sharp-src",
        default=str(Path(__file__).resolve().parent / "third_party/ml-sharp/src"),
    )
    pa.add_argument(
        "--weights",
        default=str(Path(__file__).resolve().parent / "sharp_litert_models/sharp_2572gikvuh.pt"),
    )
    pa.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "sharp_int8.pte"),
        help="Output .pte path (default: android/sharp_int8.pte)",
    )
    pa.add_argument(
        "--imgsz",
        type=int,
        default=768,
        help="Input image size (default: 768). Lower = less memory, fewer patches, faster but lower quality. Native: 1536.",
    )
    pa.add_argument(
        "--calibration-steps",
        type=int,
        default=1,
        help="Number of calibration forward passes",
    )
    return pa.parse_args()


def main():
    args = parse_args()
    sharp_src = Path(args.sharp_src)
    weights_path = Path(args.weights)
    output_path = Path(args.output)
    imgsz = args.imgsz

    print("=" * 60)
    print(f"ExecuTorch INT8 Single Model Export (imgsz={imgsz})")
    print("=" * 60)

    if imgsz < 384:
        print(f"ERROR: imgsz must be >= 384 (patch_size). Got {imgsz}")
        return 1

    if imgsz < 1536:
        print(f"\n  NOTE: imgsz={imgsz} < 1536 (native). Quality will be reduced.")
        print(f"  Scales smaller than 384 will be padded to 384 for ViT compatibility.")

    print("\nPatch analysis:")
    total_patches = compute_patch_info(imgsz)
    est_encoder_mb = total_patches * 577 * 1024 * 4 / 1024 / 1024
    print(f"  Estimated encoder activation: ~{est_encoder_mb:.0f}MB (vs ~{35 * 577 * 1024 * 4 / 1024 / 1024:.0f}MB at 1536)")

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

    # Monkey-patch the SPN encoder's _create_pyramid to pad all scales to valid
    # grid-aligned sizes so split() never produces clipped patches.
    spn = predictor.monodepth_model.monodepth_predictor.encoder
    original_create_pyramid = spn._create_pyramid

    def _padded_create_pyramid(self_spn, x):
        x0, x1, x2 = original_create_pyramid(x)
        x0 = pad_to_grid(x0, overlap_ratio=0.25)
        x1 = pad_to_grid(x1, overlap_ratio=0.5)
        x2 = pad_to_grid(x2, overlap_ratio=0.25)
        return x0, x1, x2

    import types
    spn._create_pyramid = types.MethodType(_padded_create_pyramid, spn)
    print(f"  Patched _create_pyramid to pad scales to valid grid sizes")

    model = SharpFullPipeline(predictor)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: {param_count / 1e6:.0f}M parameters")

    example_input = torch.randn(1, 3, imgsz, imgsz)

    # Validate FP32 output
    print(f"\nValidating FP32 forward pass at {imgsz}x{imgsz}...")
    t0 = time.time()
    with torch.no_grad():
        fp32_output = model(example_input)
    fp32_time = time.time() - t0
    print(f"  FP32 output: {fp32_output.shape} ({fp32_output.shape[0]} Gaussians) in {fp32_time:.1f}s")
    print(f"  Range: [{fp32_output.min():.4f}, {fp32_output.max():.4f}]")

    # PT2E quantization
    print("\n" + "=" * 60)
    print("Step 1: Export to ATen IR")
    print("=" * 60)
    t0 = time.time()
    exported = torch.export.export(model, (example_input,), strict=False)
    print(f"  Exported in {time.time() - t0:.1f}s")

    print("\n" + "=" * 60)
    print("Step 2: PT2E INT8 Quantization (XNNPACKQuantizer)")
    print("=" * 60)
    try:
        from torchao.quantization.pt2e import prepare_pt2e, convert_pt2e
        from torchao.quantization.pt2e.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        print("  Using torchao PT2E quantization API")
    except ImportError:
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        print("  Using torch.ao PT2E quantization API (legacy)")

    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    )
    print("  Quantizer: symmetric per-channel dynamic INT8")

    t0 = time.time()
    graph_module = exported.module()
    prepared = prepare_pt2e(graph_module, quantizer)
    print(f"  Prepared for quantization in {time.time() - t0:.1f}s")

    print(f"  Calibrating ({args.calibration_steps} step(s))...")
    t0 = time.time()
    with torch.no_grad():
        for step in range(args.calibration_steps):
            calibration_input = torch.randn(1, 3, imgsz, imgsz)
            prepared(calibration_input)
            print(f"    Step {step + 1}/{args.calibration_steps} done")
    print(f"  Calibration done in {time.time() - t0:.1f}s")

    t0 = time.time()
    quantized = convert_pt2e(prepared)
    print(f"  Converted to INT8 in {time.time() - t0:.1f}s")

    # Re-export and convert to ExecuTorch
    print("\n" + "=" * 60)
    print("Step 3: ExecuTorch export (XNNPACK + greedy memory planning)")
    print("=" * 60)

    t0 = time.time()
    print("  Re-exporting quantized model to ATen IR...")
    quantized_exported = torch.export.export(quantized, (example_input,), strict=False)

    try:
        from executorch.exir import to_edge_transform_and_lower
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        edge = to_edge_transform_and_lower(
            quantized_exported,
            partitioner=[XnnpackPartitioner()],
        )
        print("  to_edge_transform_and_lower + XNNPACK applied")
    except Exception as e:
        print(f"  to_edge_transform_and_lower failed ({e}), trying legacy API...")
        from executorch.exir import EdgeCompileConfig, to_edge
        edge = to_edge(quantized_exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            edge = edge.to_backend(XnnpackPartitioner())
            print("  XNNPACK delegate applied (legacy API)")
        except Exception as e2:
            print(f"  XNNPACK delegate failed: {e2}")

    # Greedy memory planning
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
            print(f"  Greedy memory planning not available: {e2}, using default")
            et_program = edge.to_executorch()

    export_time = time.time() - t0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  Saved: {output_path} ({size_mb:.1f} MB)")
    print(f"  Export time: {export_time:.1f}s")

    # Write metadata YAML for the runtime to read
    metadata_path = output_path.with_name(output_path.stem + "_metadata.yaml")
    metadata = {
        "imgsz": imgsz,
        "quantization": "int8_symmetric_per_channel_dynamic",
        "backend": "xnnpack",
        "gaussians": int(fp32_output.shape[0]),
        "patches": total_patches,
        "model_size_mb": round(size_mb, 1),
    }
    with open(metadata_path, "w") as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")
    print(f"  Metadata: {metadata_path}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Model: {output_path.name} ({size_mb:.1f} MB)")
    print(f"  Input: {imgsz}x{imgsz} ({total_patches} patches)")
    print(f"  Quantization: INT8 symmetric per-channel dynamic (XNNPACK)")
    print(f"  Memory planning: greedy AOT")
    print(f"  Gaussians: {fp32_output.shape[0]}")
    print(f"\nPush to device:")
    print(f"  adb push {output_path} /storage/emulated/0/Android/data/com.furnit.android/files/models/")
    print(f"  adb push {metadata_path} /storage/emulated/0/Android/data/com.furnit.android/files/models/")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
