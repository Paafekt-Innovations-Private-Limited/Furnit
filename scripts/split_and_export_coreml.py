#!/usr/bin/env python3
"""
Split the 4585-channel conv in PyTorch, then export via Ultralytics to CoreML.
Replaces the ONNX→CoreML step that ct 9.0 doesn't support.
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
import sys

def find_and_split(model):
    """Find the 4585-channel conv and replace with split version."""
    found = False
    
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels == 4585:
            print(f"Found: {name} → Conv2d(in={module.in_channels}, out={module.out_channels})")
            
            # Create split replacement
            split_a = nn.Conv2d(
                module.in_channels, 2292,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None
            )
            split_b = nn.Conv2d(
                module.in_channels, 2293,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None
            )
            
            # Transfer weights
            with torch.no_grad():
                split_a.weight.copy_(module.weight[:2292])
                split_b.weight.copy_(module.weight[2292:])
                if module.bias is not None:
                    split_a.bias.copy_(module.bias[:2292])
                    split_b.bias.copy_(module.bias[2292:])
            
            # Replace with wrapper that splits and concats
            class SplitConcat(nn.Module):
                def __init__(self, a, b):
                    super().__init__()
                    self.a = a
                    self.b = b
                def forward(self, x):
                    return torch.cat([self.a(x), self.b(x)], dim=1)
            
            replacement = SplitConcat(split_a, split_b)
            
            # Navigate to parent and replace
            parts = name.split('.')
            parent = model.model
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            
            last = parts[-1]
            if last.isdigit():
                parent[int(last)] = replacement
            else:
                setattr(parent, last, replacement)
            
            print(f"  ✅ Replaced with SplitConcat(2292 + 2293)")
            found = True
    
    return found


def verify(model, imgsz=1280):
    """Quick forward pass to make sure model still works."""
    model.model.eval()
    test_input = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        try:
            output = model.model(test_input)
            print(f"  ✅ Forward pass OK")
            return True
        except Exception as e:
            print(f"  ❌ Forward pass failed: {e}")
            return False


if __name__ == "__main__":
    pt_path = "/Users/al/Documents/tries01/Furnit/android/yoloe-11l-seg-pf.pt"
    
    print("Loading model...")
    model = YOLO(pt_path)
    
    print("\nSplitting 4585-channel conv...")
    found = find_and_split(model)
    
    if not found:
        print("❌ No 4585-channel conv found!")
        # Diagnostic: show all large convs
        for name, module in model.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 1000:
                print(f"  {name}: Conv2d(out={module.out_channels})")
            if isinstance(module, nn.Linear) and module.out_features > 1000:
                print(f"  {name}: Linear(out={module.out_features})")
        sys.exit(1)
    
    print("\nVerifying forward pass...")
    if not verify(model):
        sys.exit(1)
    
    print("\nExporting to CoreML...")
    model.export(
        format="coreml",
        imgsz=1280,
        half=False,
        nms=False,
        simplify=True,
    )
    
    print("\nExporting to CoreML completed.")