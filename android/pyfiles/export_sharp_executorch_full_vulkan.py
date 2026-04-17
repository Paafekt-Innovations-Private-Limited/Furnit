#!/usr/bin/env python3
"""
Export SHARP as a single full ExecuTorch .pte with Vulkan backend (sharp_full_vulkan.pte).

The Android app uses this when Settings backend is "ExecuTorch Vulkan (single model)".
One forward: input [1, 3, 1536, 1536] -> output [N, 14] Gaussians.

Memory management (ExecuTorch + Vulkan):
- Vulkan AOT: VulkanPartitioner(skip_memory_planning=False) so Vulkan preprocess plans
  tensor locations and reuses buffers (works with Vulkan's explicit memory API).
- Greedy planning: to_executorch(ExecutorchBackendConfig(memory_planning_pass=
  MemoryPlanningPass(memory_planning_algo=greedy, alloc_graph_input=False,
  alloc_graph_output=False))) so intermediate tensors share memory (best-fit reuse),
  reducing peak RAM vs naive (linear concatenation). I/O buffers are caller-managed.
- force_fp16: True for lower bandwidth and GPU memory.

Usage:
  cd android
  python export_sharp_executorch_full_vulkan.py --weights /path/to/sharp.pt --output-dir executorch_models
  ./push_sharp_executorch_models.sh executorch_models
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(description="Export single full SHARP ExecuTorch .pte with Vulkan")
    p.add_argument("--sharp-src", default=str(Path(__file__).resolve().parent / "third_party/ml-sharp/src"))
    p.add_argument("--weights", default=str(Path(__file__).resolve().parent / "sharp_litert_models/sharp_2572gikvuh.pt"))
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "executorch_models"))
    return p.parse_args()


class SharpForExport(nn.Module):
    """Full SHARP: image -> Gaussians [N, 14]. Same as export_sharp_executorch.py."""

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


def main():
    args = parse_args()
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
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from sharp.models import PredictorParams, create_predictor

    IMAGE_SIZE = 1536
    OUTPUT_FILENAME = "sharp_full_vulkan.pte"

    print("=" * 60)
    print("Export single full SHARP ExecuTorch + Vulkan")
    print("  Output: " + OUTPUT_FILENAME)
    print("=" * 60)

    print("\nLoading SHARP...")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    del state_dict

    model = SharpForExport(predictor).eval()
    example_input = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print("Test forward...")
    with torch.no_grad():
        out = model(example_input)
    print(f"  Output shape: {out.shape}")

    print("\nExporting (torch.export)...")
    t0 = time.time()
    exported = torch.export.export(model, (example_input,), strict=False)

    # Vulkan AOT memory planning: skip_memory_planning=False so Vulkan preprocess runs
    # greedy-style planning (buffer reuse, non-overlapping tensors share memory).
    vulkan_options = {"skip_memory_planning": False, "force_fp16": True}
    edge = to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[VulkanPartitioner(vulkan_options)],
    )
    # ExecuTorch memory planning pass: use greedy algorithm (best-fit reuse) to reduce
    # fragmentation and peak usage. I/O buffers caller-managed (alloc_graph_input/output=False).
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
        print("  Greedy memory planning applied (AOT tensor reuse)")
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
            print("  Greedy memory planning applied (alt API)")
        except Exception as e2:
            print(f"  Greedy memory planning not available: {e2}, using default")
            et_program = edge.to_executorch()
    export_time = time.time() - t0

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Export: {export_time:.0f}s")
    print(f"  Saved: {output_path} ({size_mb:.0f} MB)")
    print("\nPush to device:")
    print(f"  ./push_sharp_executorch_models.sh {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
