#!/usr/bin/env python3
"""
Export SHARP model components for ExecuTorch with Vulkan GPU acceleration.

This version uses the VulkanPartitioner to enable GPU execution on Android.
The Vulkan backend provides significant speedup over CPU execution.

Components:
1. sharp_single_patch_vulkan.pte - Process single 384x384 patch through ViT (Vulkan GPU)
"""

import sys
from pathlib import Path

SHARP_SRC = Path("/tmp/ml-sharp/src")
sys.path.insert(0, str(SHARP_SRC))

import torch
import torch.nn as nn

MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/ml_experiments/models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")


class SinglePatchEncoder(nn.Module):
    """
    Process a single 384x384 patch through the ViT encoder.

    Input: [1, 3, 384, 384] - Single RGB patch
    Output: [1, 1024, 24, 24] - Encoded features

    This is ~1.2GB but processes one patch at a time on device.
    With Vulkan GPU, inference should be much faster than CPU.
    """
    def __init__(self, vit):
        super().__init__()
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.blocks = vit.blocks
        self.norm = vit.norm

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed.proj(patch)  # [1, 1024, 24, 24]
        x = x.flatten(2).transpose(1, 2)  # [1, 576, 1024]

        # Add CLS token
        cls = self.cls_token.expand(1, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [1, 577, 1024]

        # Add position embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Remove CLS, reshape back to spatial
        x = x[:, 1:, :]  # [1, 576, 1024]
        x = x.transpose(1, 2).reshape(1, 1024, 24, 24)  # [1, 1024, 24, 24]

        return x


def load_sharp_model():
    """Load the full SHARP model."""
    from sharp.models import PredictorParams, create_predictor

    print(f"Loading SHARP model from {MODEL_WEIGHTS}...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    return predictor


def export_to_pte_vulkan(model, example_input, output_path: Path, name: str, use_fp16: bool = False):
    """Export a model to ExecuTorch .pte format with Vulkan GPU backend.

    Args:
        use_fp16: If True, use half precision (FP16) to reduce memory by ~2x.
                  NOTE: FP16 requires shader support (view_convert_buffer_half_float).
                  Default is False (FP32) for better compatibility.
    """
    try:
        from executorch.exir import to_edge_transform_and_lower
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    except ImportError as e:
        print(f"ERROR: Vulkan partitioner not available: {e}")
        print("Install executorch with Vulkan support or use basic export.")
        return None

    print(f"\n{'='*50}")
    print(f"Exporting {name} with Vulkan GPU backend (FP16={use_fp16})")
    print(f"{'='*50}")
    print(f"Input: {example_input.shape}, dtype: {example_input.dtype}")

    # Convert model to FP16 if requested (reduces memory by ~2x)
    # WARNING: FP16 may fail at runtime if Vulkan shaders are missing
    if use_fp16:
        print("Converting model to FP16 (half precision)...")
        print("WARNING: FP16 requires view_convert_buffer_half_float shader support")
        model = model.half()
        example_input = example_input.half()

    # Test inference
    with torch.no_grad():
        output = model(example_input)
        print(f"Output: {output.shape}, dtype: {output.dtype}")

    # Export with Vulkan partitioner
    print("torch.export...")
    exported = torch.export.export(model, (example_input,), strict=False)

    print("to_edge_transform_and_lower with VulkanPartitioner...")
    try:
        # Use Vulkan partitioner for GPU acceleration
        edge_program = to_edge_transform_and_lower(
            exported,
            partitioner=[VulkanPartitioner()],
        )

        # Log partition info if available
        try:
            print("\n--- Partition Report ---")
            if hasattr(edge_program, 'exported_program'):
                prog = edge_program.exported_program()
                if hasattr(prog, 'graph_module'):
                    gm = prog.graph_module
                    vulkan_nodes = sum(1 for n in gm.graph.nodes if 'vulkan' in str(n.target).lower())
                    total_nodes = sum(1 for n in gm.graph.nodes if n.op == 'call_function')
                    print(f"  Vulkan-partitioned nodes: {vulkan_nodes}")
                    print(f"  Total call_function nodes: {total_nodes}")
        except Exception as pe:
            print(f"  (Could not get partition details: {pe})")
        print("------------------------\n")

        print("to_executorch...")
        et = edge_program.to_executorch()

        print(f"Saving {output_path.name}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(et.buffer)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[Vulkan FP{'16' if use_fp16 else '32'}] {name}: {size_mb:.1f} MB")

        return output_path

    except Exception as e:
        print(f"Vulkan export failed: {e}")
        print("Some operators may not be supported by Vulkan backend.")
        print("Falling back to CPU export...")
        return export_to_pte_cpu(model, example_input, output_path, name)


def export_to_pte_cpu(model, example_input, output_path: Path, name: str):
    """Fallback: Export to ExecuTorch without Vulkan (CPU only)."""
    from executorch.exir import EdgeCompileConfig, to_edge

    print(f"Exporting {name} (CPU fallback)...")

    exported = torch.export.export(model, (example_input,), strict=False)
    edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    et = edge.to_executorch()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[CPU] {name}: {size_mb:.1f} MB")

    return output_path


def export_to_pte_xnnpack(model, example_input, output_path: Path, name: str):
    """Export to ExecuTorch with XNNPACK backend (optimized CPU with better memory management).

    XNNPACK provides:
    - Optimized memory management (streams data instead of loading all at once)
    - SIMD acceleration on ARM CPUs
    - Better compatibility than Vulkan for large models
    """
    try:
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    except ImportError as e:
        print(f"ERROR: XNNPACK partitioner not available: {e}")
        print("Falling back to basic CPU export...")
        return export_to_pte_cpu(model, example_input, output_path, name)

    print(f"\n{'='*50}")
    print(f"Exporting {name} with XNNPACK backend (optimized CPU)")
    print(f"{'='*50}")
    print(f"Input: {example_input.shape}, dtype: {example_input.dtype}")

    # Test inference
    with torch.no_grad():
        output = model(example_input)
        print(f"Output: {output.shape}, dtype: {output.dtype}")

    # Export with XNNPACK partitioner
    print("torch.export...")
    exported = torch.export.export(model, (example_input,), strict=False)

    print("to_edge_transform_and_lower with XnnpackPartitioner...")
    try:
        # Use XNNPACK partitioner for optimized CPU execution
        edge_program = to_edge_transform_and_lower(
            exported,
            partitioner=[XnnpackPartitioner()],
        )

        # Log partition info if available
        try:
            print("\n--- Partition Report ---")
            if hasattr(edge_program, 'exported_program'):
                prog = edge_program.exported_program()
                if hasattr(prog, 'graph_module'):
                    gm = prog.graph_module
                    xnnpack_nodes = sum(1 for n in gm.graph.nodes if 'xnnpack' in str(n.target).lower())
                    total_nodes = sum(1 for n in gm.graph.nodes if n.op == 'call_function')
                    print(f"  XNNPACK-partitioned nodes: {xnnpack_nodes}")
                    print(f"  Total call_function nodes: {total_nodes}")
        except Exception as pe:
            print(f"  (Could not get partition details: {pe})")
        print("------------------------\n")

        print("to_executorch...")
        et = edge_program.to_executorch()

        print(f"Saving {output_path.name}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(et.buffer)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[XNNPACK] {name}: {size_mb:.1f} MB")

        return output_path

    except Exception as e:
        print(f"XNNPACK export failed: {e}")
        print("Some operators may not be supported by XNNPACK backend.")
        print("Falling back to basic CPU export...")
        return export_to_pte_cpu(model, example_input, output_path, name)


def main():
    print("=" * 60)
    print("SHARP Export for ExecuTorch with Vulkan GPU")
    print("=" * 60)

    if not SHARP_SRC.exists():
        print(f"ERROR: ml-sharp not found at {SHARP_SRC}")
        print("Clone ml-sharp repository to /tmp/ml-sharp first.")
        return 1

    if not MODEL_WEIGHTS.exists():
        print(f"ERROR: Model weights not found at {MODEL_WEIGHTS}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    predictor = load_sharp_model()

    # Get components
    encoder = predictor.monodepth_model.monodepth_predictor.encoder
    patch_vit = encoder.patch_encoder

    print(f"\nComponent sizes:")
    print(f"  Patch encoder: {sum(p.numel() for p in patch_vit.parameters()) / 1e6:.1f}M params")

    # === Part 1: Single Patch Encoder ===
    single_patch = SinglePatchEncoder(patch_vit)
    single_patch.eval()
    dummy_patch = torch.randn(1, 3, 384, 384)

    # === Export XNNPACK version (optimized CPU with better memory management) ===
    xnnpack_path = OUTPUT_DIR / "sharp_single_patch_xnnpack.pte"
    if xnnpack_path.exists():
        print(f"\nRemoving old {xnnpack_path.name}...")
        xnnpack_path.unlink()

    try:
        export_to_pte_xnnpack(single_patch, dummy_patch, xnnpack_path, "Single Patch Encoder")
    except Exception as e:
        print(f"XNNPACK export failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Find all exported models
    exported = list(OUTPUT_DIR.glob("sharp_single_patch*.pte"))
    if exported:
        print("\nExported ExecuTorch files:")
        for f in sorted(exported):
            size = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size:.1f} MB")

        print("""
Push XNNPACK model to device (recommended - better memory management):
  adb push sharp_ncnn_models/sharp_single_patch_xnnpack.pte \\
    /sdcard/Android/data/com.furnit.android/files/models/

XNNPACK uses optimized CPU execution with better memory streaming.
Slower than Vulkan GPU but won't crash on memory-constrained devices.
""")
    else:
        print("\nNo models exported. Check errors above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
