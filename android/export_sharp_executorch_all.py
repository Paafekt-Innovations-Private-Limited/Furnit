#!/usr/bin/env python3
"""
Export FULL SHARP model to ExecuTorch .pte format in 4 main variants:
  1. FP32 (baseline, largest, most accurate)
  2. FP16 (half size, slightly lower precision)
  3. memory_optimized (single .pte + greedy memory planning + FP16 + XNNPACK; recommended for low RAM)
  4. INT8 (quarter size, fastest on mobile GPU)

Each variant exports the COMPLETE pipeline:
  Input: [1, 3, 1536, 1536] RGB image
  Output: [N, 14] Gaussian parameters (pos3 + scale3 + rot4 + opacity1 + color3)

Usage:
  python export_sharp_executorch_all.py                           # Export all
  python export_sharp_executorch_all.py --variant memory_optimized  # Single .pte, greedy planning
  python export_sharp_executorch_all.py --variant fp32
  python export_sharp_executorch_all.py --variant fp16
  python export_sharp_executorch_all.py --variant int8
  python export_sharp_executorch_all.py --test                    # Test inference
"""

import sys
import argparse
from pathlib import Path

SHARP_SRC = Path("/Users/al/Documents/tries01/Furnit/android/third_party/ml-sharp/src")
sys.path.insert(0, str(SHARP_SRC))

import torch
import torch.nn as nn

MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/android/sharp_litert_models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/executorch_models")


class SharpFullPipeline(nn.Module):
    """
    Complete SHARP model for ExecuTorch export.

    Wraps the full predictor pipeline:
      image -> encoder (patch pyramid + ViT) -> decoder -> Gaussian params

    Output format: [N, 14] where each row is:
      [x, y, z, sx, sy, sz, qw, qx, qy, qz, opacity, r, g, b]
    """

    def __init__(self, predictor, default_disparity_factor: float = 1.0):
        super().__init__()
        self.predictor = predictor
        self.register_buffer('disparity_factor', torch.tensor([default_disparity_factor]))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [1, 3, 1536, 1536] normalized RGB image in [0, 1]
        Returns:
            params: [N, 14] Gaussian parameters
        """
        result = self.predictor(image, self.disparity_factor)

        means = result.mean_vectors        # [B, N, 3]
        scales = result.singular_values    # [B, N, 3]
        rotations = result.quaternions     # [B, N, 4]
        colors = result.colors             # [B, N, 3]
        opacities = result.opacities       # [B, N] or [B, N, 1]

        if opacities.dim() == 2:
            opacities = opacities.unsqueeze(-1)

        # [B, N, 14]: pos(3) + scale(3) + rot(4) + opacity(1) + color(3)
        params = torch.cat([means, scales, rotations, opacities, colors], dim=-1)
        return params.squeeze(0)  # [N, 14]


def load_sharp_model():
    """Load SHARP predictor from checkpoint."""
    from sharp.models import PredictorParams, create_predictor

    print(f"Loading SHARP model from {MODEL_WEIGHTS}...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    model = SharpFullPipeline(predictor)
    model.eval()
    return model


def test_pytorch_inference(model):
    """Run a test forward pass to verify model works."""
    print("\nRunning PyTorch test inference...")
    example_input = torch.randn(1, 3, 1536, 1536)
    with torch.no_grad():
        output = model(example_input)
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Gaussians: {output.shape[0]}")
    return output


def export_fp32(model, example_input):
    """Export FP32 (baseline) .pte model."""
    from executorch.exir import EdgeCompileConfig, to_edge

    print("\n" + "=" * 60)
    print("Exporting FP32 variant")
    print("=" * 60)

    exported = torch.export.export(model, (example_input,), strict=False)
    edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    et_program = edge.to_executorch()

    output_path = OUTPUT_DIR / "sharp_full_fp32.pte"
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def export_fp16(model, example_input):
    """Export FP16 (half precision) .pte model."""
    from executorch.exir import EdgeCompileConfig, to_edge

    print("\n" + "=" * 60)
    print("Exporting FP16 variant")
    print("=" * 60)

    # Convert model to FP16
    model_fp16 = model.half()
    example_fp16 = example_input.half()

    exported = torch.export.export(model_fp16, (example_fp16,), strict=False)
    edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    et_program = edge.to_executorch()

    output_path = OUTPUT_DIR / "sharp_full_fp16.pte"
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def export_memory_optimized(model, example_input):
    """
    Export a single .pte with greedy memory planning + FP16 + XNNPACK (Priority 1+2+4).
    Keeps peak activation memory ~one layer instead of sum of all layers (no split).
    """
    from executorch.exir import EdgeCompileConfig, to_edge

    print("\n" + "=" * 60)
    print("Exporting memory-optimized variant (single .pte, greedy planning, FP16, XNNPACK)")
    print("=" * 60)

    model_fp16 = model.half()
    example_fp16 = example_input.half()

    exported = torch.export.export(model_fp16, (example_fp16,), strict=False)
    edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))

    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        edge = edge.to_backend(XnnpackPartitioner())
        print("  XNNPACK partitioner applied (operator fusion)")
    except Exception as e:
        print(f"  XNNPACK partitioner skipped: {e}")

    # Greedy memory planning: reuse buffers across ops. Use alloc_graph_input=False,
    # alloc_graph_output=False so I/O buffers are caller-managed and don't inflate the plan.
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
            print(f"  Greedy memory planning not available: {e2}")
            et_program = edge.to_executorch()

    output_path = OUTPUT_DIR / "sharp_full_memory_optimized.pte"
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def export_int8(model, example_input):
    """Export INT8 (quantized) .pte model using PT2E quantization flow."""
    from executorch.exir import EdgeCompileConfig, to_edge

    print("\n" + "=" * 60)
    print("Exporting INT8 variant (PT2E quantization)")
    print("=" * 60)

    try:
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
        from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

        # Step 1: Export to ATen IR first
        print("  Exporting to ATen IR...")
        exported = torch.export.export(model, (example_input,), strict=False)

        # Step 2: Prepare quantization with XNNPACK quantizer
        print("  Preparing INT8 quantization (XNNPACK symmetric)...")
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        prepared = prepare_pt2e(exported, quantizer)

        # Step 3: Calibrate with example input
        print("  Calibrating with example input...")
        with torch.no_grad():
            prepared.module()(example_input)

        # Step 4: Convert to quantized model
        print("  Converting to quantized model...")
        quantized = convert_pt2e(prepared)

        # Step 5: Export to edge
        print("  Converting to edge format...")
        edge = to_edge(quantized, compile_config=EdgeCompileConfig(_check_ir_validity=False))

        # Step 6: Try XNNPACK delegate
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            edge = edge.to_backend(XnnpackPartitioner())
            print("  XNNPACK delegate applied for INT8 ops")
        except Exception as e:
            print(f"  XNNPACK delegate failed: {e}")

        et_program = edge.to_executorch()

        output_path = OUTPUT_DIR / "sharp_full_int8.pte"
        with open(output_path, "wb") as f:
            f.write(et_program.buffer)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
        return output_path

    except Exception as e:
        print(f"  INT8 PT2E quantization failed: {e}")
        print("  This may require specific ExecuTorch version or calibration data.")
        print("  Skipping INT8 variant. Use FP16 instead (good balance of speed + quality).")
        return None


def export_int8_xnnpack(model, example_input):
    """Export INT8 with XNNPACK delegate (CPU SIMD, most compatible)."""
    from executorch.exir import EdgeCompileConfig, to_edge

    print("\n" + "=" * 60)
    print("Exporting INT8 + XNNPACK variant")
    print("=" * 60)

    from torch.ao.quantization import quantize_dynamic
    model_int8 = quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    exported = torch.export.export(model_int8, (example_input,), strict=False)
    edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))

    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        edge = edge.to_backend(XnnpackPartitioner())
        print("  XNNPACK delegate applied")
    except Exception as e:
        print(f"  XNNPACK delegate not available: {e}")

    et_program = edge.to_executorch()

    output_path = OUTPUT_DIR / "sharp_full_int8_xnnpack.pte"
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export SHARP to ExecuTorch")
    parser.add_argument("--variant", choices=["fp32", "fp16", "memory_optimized", "int8", "int8_xnnpack", "all"], default="all")
    parser.add_argument("--test", action="store_true", help="Only run PyTorch test inference")
    args = parser.parse_args()

    model = load_sharp_model()

    if args.test:
        test_pytorch_inference(model)
        return

    example_input = torch.randn(1, 3, 1536, 1536)

    # Verify model works before exporting
    test_pytorch_inference(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.variant in ("fp32", "all"):
        export_fp32(model, example_input)

    if args.variant in ("fp16", "all"):
        export_fp16(model, example_input)

    if args.variant in ("memory_optimized", "all"):
        export_memory_optimized(model, example_input)

    if args.variant in ("int8", "all"):
        export_int8(model, example_input)

    if args.variant in ("int8_xnnpack", "all"):
        export_int8_xnnpack(model, example_input)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nModels saved to: {OUTPUT_DIR}")
    print("\nPush to device:")
    for pte in OUTPUT_DIR.glob("*.pte"):
        print(f"  adb push {pte} /sdcard/Android/data/com.furnit.android/files/models/")


if __name__ == "__main__":
    main()
