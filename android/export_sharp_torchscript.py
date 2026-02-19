#!/usr/bin/env python3
"""
Export SHARP to TorchScript for Native .pt path on Android.

Pipeline: sharp.pt -> torch.jit.trace -> sharp_scripted.pt -> sharp_scripted.ptl
Native path uses PyTorch Mobile (LibTorch) runtime - no ONNX/ExecuTorch/TFLite.

Usage:
    python export_sharp_torchscript.py
"""
import sys
import time
from pathlib import Path

SHARP_SRC = "/Users/al/Documents/tries01/Furnit/android/third_party/ml-sharp/src"
sys.path.insert(0, SHARP_SRC)

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

MW = Path("/Users/al/Documents/tries01/Furnit/android/sharp_litert_models/sharp_2572gikvuh.pt")
OUT = Path("/Users/al/Documents/tries01/Furnit/android/executorch_models")


class SharpForMobile(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.register_buffer("disparity_factor", torch.tensor([1.0]))

    def forward(self, image):
        result = self.predictor(image, self.disparity_factor)
        means = result.mean_vectors
        scales = result.singular_values
        rotations = result.quaternions
        colors = result.colors
        opacities = result.opacities
        if opacities.dim() == 2:
            opacities = opacities.unsqueeze(-1)
        return torch.cat([means, scales, rotations, opacities, colors], dim=-1).squeeze(0)


def main():
    from sharp.models import PredictorParams, create_predictor

    print("Loading SHARP...")
    t0 = time.time()
    sd = torch.load(MW, map_location="cpu", weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(sd)
    predictor.eval()
    del sd
    model = SharpForMobile(predictor)
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Test forward pass
    print("\nTesting forward pass...")
    example = torch.randn(1, 3, 1536, 1536)
    t0 = time.time()
    with torch.no_grad():
        output = model(example)
    print(f"  {output.shape[0]} Gaussians in {time.time()-t0:.1f}s")

    # TorchScript trace
    print("\nTracing to TorchScript...")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    print(f"  Traced in {time.time()-t0:.1f}s")

    # Validate traced model
    print("\nValidating traced model...")
    with torch.no_grad():
        traced_output = traced(example)
    diff = (output - traced_output).abs().max().item()
    print(f"  Max diff: {diff}")
    if diff < 1e-3:
        print("  MATCH")
    else:
        print("  WARNING: mismatch!")

    # Optimize for mobile
    print("\nOptimizing for mobile...")
    t0 = time.time()
    optimized = optimize_for_mobile(traced)
    print(f"  Optimized in {time.time()-t0:.1f}s")

    # Save models for both PyTorch Mobile and Native .pt paths
    OUT.mkdir(parents=True, exist_ok=True)

    # sharp_scripted.pt - TorchScript static graph (for Native .pt path)
    scripted_pt = OUT / "sharp_scripted.pt"
    print(f"\nSaving TorchScript: {scripted_pt}")
    traced.save(str(scripted_pt))
    print(f"  Size: {scripted_pt.stat().st_size / 1024 / 1024:.0f} MB")

    # sharp_scripted.ptl - mobile-optimized for LiteModuleLoader (Native .pt + PyTorch Mobile)
    scripted_ptl = OUT / "sharp_scripted.ptl"
    print(f"Saving mobile-optimized: {scripted_ptl}")
    optimized._save_for_lite_interpreter(str(scripted_ptl))
    print(f"  Size: {scripted_ptl.stat().st_size / 1024 / 1024:.0f} MB")

    # Legacy naming for PyTorch Mobile backend
    mobile_ptl = OUT / "sharp_mobile.ptl"
    if scripted_ptl != mobile_ptl:
        import shutil
        shutil.copy(scripted_ptl, mobile_ptl)
        print(f"  Copy: {mobile_ptl}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print("\nPush for Native .pt path:")
    print(f"  adb push {scripted_ptl} /sdcard/Android/data/com.furnit.android/files/models/")


if __name__ == "__main__":
    main()
