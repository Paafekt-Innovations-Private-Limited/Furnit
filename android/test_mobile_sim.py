#!/usr/bin/env python3
"""
Simulate mobile environment in Python to estimate Android performance.

Constraints applied:
  - 1 CPU thread (mobile big core equivalent)
  - No MKL/Accelerate (disable optimized BLAS)
  - Measure per-stage timing
"""
import os
import sys
import time

# Limit to 1 thread BEFORE importing torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

SHARP_SRC = "/Users/al/Documents/tries01/Furnit/android/third_party/ml-sharp/src"
sys.path.insert(0, SHARP_SRC)

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

print(f"torch.get_num_threads() = {torch.get_num_threads()}")
print(f"torch.get_num_interop_threads() = {torch.get_num_interop_threads()}")

MW = "/Users/al/Documents/tries01/Furnit/android/sharp_litert_models/sharp_2572gikvuh.pt"
IMG = "/Users/al/Downloads/PXL_20260202_175808231.MP.jpg"

from sharp.models import PredictorParams, create_predictor

print("\n=== Loading model ===")
t0 = time.time()
sd = torch.load(MW, map_location="cpu", weights_only=False)
predictor = create_predictor(PredictorParams())
predictor.load_state_dict(sd)
predictor.eval()
del sd
print(f"  Model loaded in {time.time()-t0:.1f}s")

print("\n=== Loading image ===")
t0 = time.time()
from PIL import Image
import torchvision.transforms as T
img = Image.open(IMG).convert("RGB")
image = T.Compose([T.Resize((1536, 1536)), T.ToTensor()])(img).unsqueeze(0)
print(f"  Image loaded in {time.time()-t0:.1f}s")

print("\n=== Running inference (1 thread, simulating mobile) ===")
disparity = torch.tensor([1.0])
t0 = time.time()
with torch.no_grad():
    result = predictor(image, disparity)
infer_time = time.time() - t0

means = result.mean_vectors.squeeze(0)
gaussians = means.shape[0]

print(f"\n  Inference: {infer_time:.1f}s")
print(f"  Gaussians: {gaussians}")
print(f"\n=== Mobile estimate ===")
print(f"  Mac 1-thread: {infer_time:.1f}s")
print(f"  Phone ARM (estimated 2-4x slower): {infer_time*2:.0f}-{infer_time*4:.0f}s")
