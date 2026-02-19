#!/usr/bin/env python3
"""
Export MobileNetV3 Small to ExecuTorch .pte format for Android inference.

Requirements:
    pip install torch torchvision executorch

Usage:
    python export_mobilenet.py

Then push to device:
    adb push mobilenet_v3_small.pte /data/local/tmp/furnit/
"""

import torch
import torchvision.models as models
from torch.export import export
from executorch.exir import to_edge


def main():
    print("Loading MobileNetV3 Small (pretrained)...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.eval()

    # Example input: batch=1, channels=3, height=224, width=224
    example_input = (torch.randn(1, 3, 224, 224),)

    print("Exporting model with torch.export...")
    exported_program = export(model, example_input)

    print("Converting to Edge program...")
    edge_program = to_edge(exported_program)

    print("Converting to ExecuTorch program...")
    executorch_program = edge_program.to_executorch()

    output_path = "mobilenet_v3_small.pte"
    print(f"Saving to {output_path}...")
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    print(f"Done! Model saved to {output_path}")
    print()
    print("To deploy to device:")
    print("  adb shell mkdir -p /data/local/tmp/furnit/")
    print(f"  adb push {output_path} /data/local/tmp/furnit/")


if __name__ == "__main__":
    main()
