#!/usr/bin/env python3
"""
Split the 4585-channel class conv in YOLO-E into two halves (each <4096)
so ANE can execute them. Then re-export to CoreML with half=True.

Path 3: bypasses all coremltools spec-patching — the fix is in the model itself.

Usage (from /tmp with clean PYTHONPATH):
  cd /tmp && env PYTHONPATH=/opt/miniconda3/envs/coreml-py311/lib/python3.11/site-packages \
    /opt/miniconda3/envs/coreml-py311/bin/python \
    /Users/al/Documents/tries01/Furnit/scripts/split_classhead_for_ane.py \
    --pt /Users/al/Documents/tries01/Furnit/android/yoloe-11l-seg-pf.pt --export
"""

import torch
import torch.nn as nn
import copy
import numpy as np
from pathlib import Path

NUM_CLASSES = 4585
ANE_CHANNEL_LIMIT = 4096


def find_class_convs(model):
    """Find Conv2d/Linear layers with out_channels/out_features == NUM_CLASSES."""
    targets = []

    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels == NUM_CLASSES:
            targets.append((name, module, "conv"))
            print(f"  Found: {name} — Conv2d({module.in_channels}, {module.out_channels}, k={module.kernel_size})")
        elif isinstance(module, nn.Linear) and module.out_features == NUM_CLASSES:
            targets.append((name, module, "linear"))
            print(f"  Found: {name} — Linear({module.in_features}, {module.out_features})")

    if not targets:
        print("\n  No exact 4585-channel layers found. Listing all large layers:")
        for name, module in model.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 1000:
                print(f"    {name}: Conv2d(in={module.in_channels}, out={module.out_channels})")
            if isinstance(module, nn.Linear) and module.out_features > 1000:
                print(f"    {name}: Linear(in={module.in_features}, out={module.out_features})")

    return targets


class SplitConv2d(nn.Module):
    """Drop-in Conv2d replacement split into two halves, each under ANE_CHANNEL_LIMIT."""

    def __init__(self, original: nn.Conv2d):
        super().__init__()
        out_ch = original.out_channels
        in_ch = original.in_channels
        self.split_at = out_ch // 2

        self.conv_a = nn.Conv2d(
            in_ch, self.split_at,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=original.bias is not None,
        )
        self.conv_b = nn.Conv2d(
            in_ch, out_ch - self.split_at,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=original.bias is not None,
        )

        with torch.no_grad():
            self.conv_a.weight.copy_(original.weight[:self.split_at])
            self.conv_b.weight.copy_(original.weight[self.split_at:])
            if original.bias is not None:
                self.conv_a.bias.copy_(original.bias[:self.split_at])
                self.conv_b.bias.copy_(original.bias[self.split_at:])

        print(f"    Split Conv2d({in_ch}, {out_ch}) → ({in_ch}, {self.split_at}) + ({in_ch}, {out_ch - self.split_at})")

    def forward(self, x):
        return torch.cat([self.conv_a(x), self.conv_b(x)], dim=1)


class SplitLinear(nn.Module):
    """Drop-in Linear replacement split into two halves."""

    def __init__(self, original: nn.Linear):
        super().__init__()
        out_f = original.out_features
        in_f = original.in_features
        self.split_at = out_f // 2

        self.linear_a = nn.Linear(in_f, self.split_at, bias=original.bias is not None)
        self.linear_b = nn.Linear(in_f, out_f - self.split_at, bias=original.bias is not None)

        with torch.no_grad():
            self.linear_a.weight.copy_(original.weight[:self.split_at])
            self.linear_b.weight.copy_(original.weight[self.split_at:])
            if original.bias is not None:
                self.linear_a.bias.copy_(original.bias[:self.split_at])
                self.linear_b.bias.copy_(original.bias[self.split_at:])

        print(f"    Split Linear({in_f}, {out_f}) → ({in_f}, {self.split_at}) + ({in_f}, {out_f - self.split_at})")

    def forward(self, x):
        return torch.cat([self.linear_a(x), self.linear_b(x)], dim=-1)


def replace_module_by_name(model, target_name: str, new_module: nn.Module):
    """Replace a nested module by its dot-separated name."""
    parts = target_name.split(".")
    parent = model.model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def split_and_verify(pt_path: str):
    """Split class convs and verify outputs match original exactly."""
    from ultralytics import YOLOE

    print(f"Loading {pt_path}...")
    model = YOLOE(str(pt_path))

    print("\nSearching for 4585-channel layers...")
    targets = find_class_convs(model)

    if not targets:
        print("\n❌ No 4585-channel layers found")
        return None

    print(f"\n✅ Found {len(targets)} layer(s) to split")

    print("\nRunning inference BEFORE split...")
    test_input = torch.randn(1, 3, 1280, 1280)
    model.model.eval()
    with torch.no_grad():
        out_before = model.model(test_input)

    print("\nApplying splits...")
    for name, module, kind in targets:
        if kind == "conv":
            new_module = SplitConv2d(module)
        else:
            new_module = SplitLinear(module)
        replace_module_by_name(model, name, new_module)

    print("\nRunning inference AFTER split...")
    model.model.eval()
    with torch.no_grad():
        out_after = model.model(test_input)

    def compare_tensors(label, a, b):
        if isinstance(a, (tuple, list)):
            for i, (ai, bi) in enumerate(zip(a, b)):
                compare_tensors(f"{label}[{i}]", ai, bi)
        elif isinstance(a, torch.Tensor):
            diff = torch.max(torch.abs(a - b)).item()
            match = "✅" if diff < 1e-4 else "❌"
            print(f"  {match} {label}: shape={a.shape} max_diff={diff:.8f}")
            if diff >= 1e-4:
                raise AssertionError(f"Output mismatch on {label}: max_diff={diff}")

    compare_tensors("output", out_before, out_after)
    print("\n✅ Split verification PASSED — outputs match")

    return model


def export_to_coreml(model, pt_path: str, output_dir: str):
    """Export split model to CoreML via Ultralytics."""
    print("\nFusing model...")
    model.fuse()

    print("Exporting to CoreML (half=True, nms=False)...")
    exported = model.export(
        format="coreml",
        imgsz=1280,
        batch=1,
        nms=False,
        half=True,
        simplify=True,
    )
    if isinstance(exported, (list, tuple)):
        exported = exported[0] if exported else None
    print(f"✅ Exported: {exported}")
    return exported


def validate_coreml(model_path: str, imgsz: int = 1280):
    """Run a test prediction on the exported CoreML model."""
    import coremltools as ct
    from PIL import Image

    print(f"\nValidating CoreML model: {model_path}")
    model = ct.models.MLModel(model_path)

    test_image = Image.fromarray(
        np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    )
    output = model.predict({"image": test_image})

    print("Outputs:")
    for name, arr in output.items():
        shape = arr.shape if hasattr(arr, "shape") else type(arr)
        dtype = arr.dtype if hasattr(arr, "dtype") else type(arr)
        print(f"  {name}: shape={shape} dtype={dtype}")

    det_found = False
    proto_found = False
    for name, arr in output.items():
        if not hasattr(arr, "shape"):
            continue
        if len(arr.shape) == 3 and arr.shape[1] == 4 + NUM_CLASSES + 32:
            print(f"  ✅ Detection tensor '{name}': {arr.shape}")
            det_found = True
        elif len(arr.shape) == 4 and arr.shape[1] == 32:
            print(f"  ✅ Proto tensor '{name}': {arr.shape}")
            proto_found = True

    if det_found and proto_found:
        print("\n✅ CoreML validation PASSED")
        return True
    else:
        print(f"\n⚠️  det={det_found} proto={proto_found}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Split YOLO-E class head for ANE (<4096 ch per conv), then export CoreML"
    )
    parser.add_argument("--pt", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--export", action="store_true", help="Export to CoreML after split")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for mlpackage")
    args = parser.parse_args()

    model = split_and_verify(args.pt)
    if model is None:
        return 1

    if args.export:
        exported = export_to_coreml(model, args.pt, args.output_dir)
        if exported:
            validate_coreml(exported)

    return 0


if __name__ == "__main__":
    exit(main())
