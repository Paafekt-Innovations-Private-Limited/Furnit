#!/usr/bin/env python3
"""Validate .pte export: compare PyTorch direct vs torch.export graph."""
import sys
import argparse
import time
from pathlib import Path

SHARP_SRC = "/Users/al/Documents/tries01/Furnit/android/third_party/ml-sharp/src"
sys.path.insert(0, SHARP_SRC)

import torch
import torch.nn as nn
import numpy as np

MW = "/Users/al/Documents/tries01/Furnit/android/sharp_litert_models/sharp_2572gikvuh.pt"


class SharpFull(nn.Module):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    args = parser.parse_args()

    from sharp.models import PredictorParams, create_predictor

    print("Loading SHARP...")
    stateDict = torch.load(MW, map_location="cpu", weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(stateDict)
    predictor.eval()
    model = SharpFull(predictor)
    model.eval()

    if args.image:
        from PIL import Image
        import torchvision.transforms as T

        rawImage = Image.open(args.image).convert("RGB")
        transformPipeline = T.Compose([T.Resize((1536, 1536)), T.ToTensor()])
        image = transformPipeline(rawImage).unsqueeze(0)
        print("Image: " + args.image)
    else:
        image = torch.randn(1, 3, 1536, 1536).clamp(0, 1)
        print("Random input")

    print("\n=== PyTorch Direct ===")
    startTime = time.time()
    with torch.no_grad():
        pytorchOutput = model(image)
    elapsedPytorch = time.time() - startTime
    gaussianCount = pytorchOutput.shape[0]
    minVal = pytorchOutput.min().item()
    maxVal = pytorchOutput.max().item()
    print("  " + str(gaussianCount) + " Gaussians in " + str(round(elapsedPytorch, 1)) + "s")
    print("  Range: [" + str(round(minVal, 4)) + ", " + str(round(maxVal, 4)) + "]")

    print("\n=== torch.export (same graph as .pte) ===")
    startTime = time.time()
    exportedProgram = torch.export.export(model, (image,), strict=False)
    exportTime = time.time() - startTime
    print("  Export: " + str(round(exportTime, 1)) + "s")

    startTime = time.time()
    exportOutput = exportedProgram.module()(image)
    inferTime = time.time() - startTime
    exportCount = exportOutput.shape[0]
    exportMin = exportOutput.min().item()
    exportMax = exportOutput.max().item()
    print("  " + str(exportCount) + " Gaussians in " + str(round(inferTime, 1)) + "s")
    print("  Range: [" + str(round(exportMin, 4)) + ", " + str(round(exportMax, 4)) + "]")

    print("\n=== Comparison ===")
    pytorchArray = pytorchOutput.detach().numpy()
    exportArray = exportOutput.detach().numpy()
    print("  Count: PT=" + str(pytorchArray.shape[0]) + " EX=" + str(exportArray.shape[0]))

    if pytorchArray.shape[0] != exportArray.shape[0]:
        print("  ERROR: count mismatch!")
        return

    fieldNames = [("Pos", 0, 3), ("Scale", 3, 6), ("Rot", 6, 10), ("Opac", 10, 11), ("Color", 11, 14)]
    for fieldName, startIdx, endIdx in fieldNames:
        fieldDiff = np.abs(pytorchArray[:, startIdx:endIdx] - exportArray[:, startIdx:endIdx])
        maxDiff = fieldDiff.max()
        meanDiff = fieldDiff.mean()
        print("  " + fieldName.ljust(6) + " max=" + str(round(maxDiff, 8)) + " mean=" + str(round(meanDiff, 8)))

    totalMaxDiff = np.abs(pytorchArray - exportArray).max()
    print("\n  Max diff: " + str(round(totalMaxDiff, 8)))
    if totalMaxDiff < 1e-3:
        print("  EXACT MATCH. Safe to push .pte to device.")
    elif totalMaxDiff < 1e-1:
        print("  CLOSE MATCH. Safe to push .pte to device.")
    else:
        print("  MISMATCH! Do NOT push to device.")


if __name__ == "__main__":
    main()
