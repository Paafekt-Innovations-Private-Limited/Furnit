#!/usr/bin/env python3
"""
Check how the CoreML model was exported and if bbox decoding is included.
"""

import coremltools as ct
import numpy as np
from PIL import Image

def analyze_output_ranges(model_path, image_path):
    """Analyze raw output value ranges to understand the format."""

    print("="*60)
    print("COREML OUTPUT VALUE ANALYSIS")
    print("="*60)

    model = ct.models.MLModel(model_path)

    # Load and letterbox image
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size

    target_size = 1280
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    resized = img.resize((new_w, new_h), Image.BILINEAR)
    letterboxed = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    letterboxed.paste(resized, (pad_x, pad_y))

    output = model.predict({'image': letterboxed})

    det = output['var_2346'][0]
    if det.shape[0] < det.shape[1]:
        det = det.T

    print(f"\nDetections shape: {det.shape}")

    # Check value ranges to understand format
    print("\n--- RAW VALUE RANGES ---")
    for i in range(min(8, det.shape[1])):
        col = det[:, i]
        print(f"Col {i}: min={col.min():12.4f}, max={col.max():12.4f}, mean={col.mean():12.4f}")

    # Check if coords are normalized (0-1) or absolute
    print("\n--- FORMAT DETECTION ---")

    col0_max = det[:, 0].max()
    col1_max = det[:, 1].max()
    col2_max = det[:, 2].max()
    col3_max = det[:, 3].max()

    print(f"\nColumn max values: {col0_max:.1f}, {col1_max:.1f}, {col2_max:.1f}, {col3_max:.1f}")

    if col0_max <= 1.0 and col1_max <= 1.0:
        print("  -> Columns 0-1 appear NORMALIZED (0-1)")
    elif col0_max <= 1280 and col1_max <= 1280:
        print("  -> Columns 0-1 appear to be in MODEL SPACE (0-1280)")

    if col2_max <= 1.0 and col3_max <= 1.0:
        print("  -> Columns 2-3 appear NORMALIZED (0-1)")
    elif col2_max <= 1280 and col3_max <= 1280:
        print("  -> Columns 2-3 appear to be in MODEL SPACE (0-1280)")
    else:
        print("  -> Columns 2-3 values seem UNUSUAL")

    # Check if maybe cols 2-3 need different interpretation
    print("\n--- TRYING DIFFERENT WIDTH/HEIGHT INTERPRETATIONS ---")

    # Get top confident detection
    conf = det[:, 4]
    best_idx = np.argmax(conf)
    x, y, w, h = det[best_idx, :4]

    print(f"\nBest detection raw: x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f}")

    # Maybe w,h are deltas or log-scaled?
    print("\n1. Direct (current interpretation):")
    print(f"   Width={w:.1f}, Height={h:.1f}")

    print("\n2. If w,h are exponential (exp(w), exp(h)):")
    print(f"   Width={np.exp(w/100):.1f}, Height={np.exp(h/100):.1f}")

    print("\n3. If values need to be divided by stride (e.g., 8):")
    print(f"   Width={w/8:.1f}, Height={h/8:.1f}")

    print("\n4. If values are in different scale (divide by 10):")
    print(f"   Width={w/10:.1f}, Height={h/10:.1f}")

    # Check the relationship between x,y and w,h
    print("\n--- CHECKING IF x,y,w,h MIGHT BE x1,y1,x2,y2 ---")
    print(f"If xyxy: box would be ({x:.0f},{y:.0f}) to ({w:.0f},{h:.0f})")
    print(f"  Width would be: {w-x:.1f}")
    print(f"  Height would be: {h-y:.1f}")

    # This might actually be the case!
    if 0 < (w-x) < 500 and 0 < (h-y) < 500:
        print("  -> This interpretation gives REASONABLE box sizes!")

    # Let's verify by checking multiple detections
    print("\n--- VERIFYING XYXY INTERPRETATION ---")
    print("Top 5 detections if interpreted as x1,y1,x2,y2:")

    sorted_idx = np.argsort(conf)[::-1]
    for i in range(5):
        idx = sorted_idx[i]
        x1, y1, x2, y2 = det[idx, :4]
        c = conf[idx]

        if x2 > x1 and y2 > y1:  # Valid box
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Un-letterbox
            cx_orig = (cx - pad_x) / scale
            cy_orig = (cy - pad_y) / scale
            w_orig = w / scale
            h_orig = h / scale

            print(f"  {i+1}. conf={c:.3f}, center=({cx_orig:.0f},{cy_orig:.0f}), size=({w_orig:.0f}x{h_orig:.0f})")
        else:
            print(f"  {i+1}. conf={c:.3f}, INVALID box (x2<=x1 or y2<=y1)")

if __name__ == "__main__":
    model_path = "/Users/al/Documents/tries01/Furnit/yoloe-26l-seg-pf.mlpackage"
    image_path = "/Users/al/Documents/tries01/Furnit/android/TestRoom.jpg"

    analyze_output_ranges(model_path, image_path)
