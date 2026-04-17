#!/usr/bin/env python3
"""
Re-export sharp_single_patch.pte with XNNPACK backend for fast mobile inference.

XNNPACK is ExecuTorch's optimized CPU backend that uses NEON/SSE intrinsics,
making it 5-20x faster than the default portable backend.
"""

import sys
from pathlib import Path

SHARP_SRC = Path("/tmp/ml-sharp/src")
sys.path.insert(0, str(SHARP_SRC))

import torch
import torch.nn as nn

MODEL_WEIGHTS = Path("/Users/al/Library/Mobile Documents/com~apple~CloudDocs/ml_experiments/models/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")


class SinglePatchEncoder(nn.Module):
    """
    Process a single 384x384 patch through the ViT encoder.
    Input: [1, 3, 384, 384] → Output: [1, 1024, 24, 24]
    """
    def __init__(self, vit):
        super().__init__()
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.blocks = vit.blocks
        self.norm = vit.norm

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed.proj(patch)  # [1, 1024, 24, 24]
        x = x.flatten(2).transpose(1, 2)  # [1, 576, 1024]
        cls = self.cls_token.expand(1, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [1, 577, 1024]
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 1:, :]  # [1, 576, 1024]
        x = x.transpose(1, 2).reshape(1, 1024, 24, 24)
        return x


def main():
    print("=" * 60)
    print("SHARP Patch Encoder Export with XNNPACK")
    print("=" * 60)

    if not SHARP_SRC.exists():
        print(f"ERROR: ml-sharp not found at {SHARP_SRC}")
        return 1

    if not MODEL_WEIGHTS.exists():
        print(f"ERROR: Model weights not found at {MODEL_WEIGHTS}")
        return 1

    # Load SHARP model
    from sharp.models import PredictorParams, create_predictor

    print(f"Loading SHARP model from {MODEL_WEIGHTS}...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    patch_vit = predictor.monodepth_model.monodepth_predictor.encoder.patch_encoder
    model = SinglePatchEncoder(patch_vit)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Test inference
    dummy = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        out = model(dummy)
        print(f"Test: {dummy.shape} → {out.shape}")

    # Export with XNNPACK
    print("\nExporting with XNNPACK backend...")

    from executorch.exir import EdgeCompileConfig, to_edge
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    print("  torch.export...")
    exported = torch.export.export(model, (dummy,), strict=False)

    print("  to_edge...")
    edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))

    print("  Lowering to XNNPACK...")
    edge = edge.to_backend(XnnpackPartitioner())

    print("  to_executorch...")
    et = edge.to_executorch()

    output_path = OUTPUT_DIR / "sharp_single_patch_xnnpack.pte"
    print(f"  Saving {output_path.name}...")
    with open(output_path, "wb") as f:
        f.write(et.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nDone: {output_path.name} = {size_mb:.1f} MB")

    # Also export without XNNPACK as baseline (in case XNNPACK has issues)
    print("\nAlso exporting portable backend (fallback)...")
    exported2 = torch.export.export(model, (dummy,), strict=False)
    edge2 = to_edge(exported2, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    et2 = edge2.to_executorch()

    output_path2 = OUTPUT_DIR / "sharp_single_patch.pte"
    with open(output_path2, "wb") as f:
        f.write(et2.buffer)

    size_mb2 = output_path2.stat().st_size / (1024 * 1024)
    print(f"Done: {output_path2.name} = {size_mb2:.1f} MB")

    print(f"\nPush to device:")
    print(f"  adb push {output_path} /sdcard/Android/data/com.furnit.android/files/models/")
    print(f"  adb push {output_path} /data/local/tmp/furnit/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
