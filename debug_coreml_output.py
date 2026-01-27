#!/usr/bin/env python3
"""
Debug CoreML output format vs PyTorch.
"""

import torch
import coremltools as ct
import numpy as np
from PIL import Image
from ultralytics import YOLO

def main():
    pt_path = "/Users/al/Documents/tries01/Furnit/yoloe-26l-seg-pf.pt"
    coreml_path = "/Users/al/Documents/tries01/Furnit/yoloe-26l-seg-pf.mlpackage"
    image_path = "/Users/al/Documents/tries01/Furnit/android/TestRoom.jpg"

    # Load image
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    print(f"Original image: {orig_w}x{orig_h}")

    # =========== PyTorch ===========
    print("\n" + "="*60)
    print("PYTORCH OUTPUT FORMAT")
    print("="*60)

    model = YOLO(pt_path)
    results = model(image_path, imgsz=1280, conf=0.25, verbose=False)
    result = results[0]

    print(f"\nBoxes shape: {result.boxes.xyxy.shape}")
    print(f"Confidence shape: {result.boxes.conf.shape}")

    if result.masks is not None:
        print(f"Masks data shape: {result.masks.data.shape}")

        # Get best detection
        best_idx = result.boxes.conf.argmax().item()
        best_box = result.boxes.xyxy[best_idx].cpu().numpy()
        best_conf = result.boxes.conf[best_idx].item()
        best_mask = result.masks.data[best_idx].cpu().numpy()

        print(f"\nBest detection (idx={best_idx}):")
        print(f"  Box (xyxy, original coords): {best_box}")
        print(f"  Confidence: {best_conf:.4f}")
        print(f"  Mask shape: {best_mask.shape}")
        print(f"  Mask coverage: {best_mask.sum() / best_mask.size * 100:.2f}%")

    # =========== CoreML ===========
    print("\n" + "="*60)
    print("COREML OUTPUT FORMAT")
    print("="*60)

    coreml_model = ct.models.MLModel(coreml_path)

    # Letterbox
    target_size = 1280
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    resized = img.resize((new_w, new_h), Image.BILINEAR)
    letterboxed = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    letterboxed.paste(resized, (pad_x, pad_y))

    print(f"\nLetterbox: scale={scale:.4f}, pad=({pad_x},{pad_y}), content={new_w}x{new_h}")

    output = coreml_model.predict({'image': letterboxed})

    det = output['var_2346'][0]  # [300, 38]
    proto = output['var_2429'][0]  # [32, 320, 320]

    if det.shape[0] < det.shape[1]:
        det = det.T

    print(f"\nDetections shape: {det.shape}")
    print(f"Prototypes shape: {proto.shape}")

    # Analyze each column
    print("\n--- COLUMN ANALYSIS ---")
    for i in range(min(10, det.shape[1])):
        col = det[:, i]
        print(f"  Col {i}: min={col.min():10.4f}, max={col.max():10.4f}, mean={col.mean():10.4f}")

    print("  ...")
    for i in range(det.shape[1]-3, det.shape[1]):
        col = det[:, i]
        print(f"  Col {i}: min={col.min():10.4f}, max={col.max():10.4f}, mean={col.mean():10.4f}")

    # Get best detection by confidence (col 4)
    conf = det[:, 4]
    best_idx = np.argmax(conf)
    best_det = det[best_idx]

    print(f"\n--- BEST COREML DETECTION (idx={best_idx}) ---")
    print(f"  x (center): {best_det[0]:.1f}")
    print(f"  y (center): {best_det[1]:.1f}")
    print(f"  w: {best_det[2]:.1f}")
    print(f"  h: {best_det[3]:.1f}")
    print(f"  confidence: {best_det[4]:.4f}")
    print(f"  col5 (class?): {best_det[5]:.4f}")
    print(f"  mask coeffs [6:10]: {best_det[6:10]}")

    # Convert to original image coordinates
    cx, cy, w, h = best_det[0], best_det[1], best_det[2], best_det[3]

    # Remove padding and scale back
    cx_unpad = (cx - pad_x) / scale
    cy_unpad = (cy - pad_y) / scale
    w_orig = w / scale
    h_orig = h / scale

    x1 = cx_unpad - w_orig/2
    y1 = cy_unpad - h_orig/2
    x2 = cx_unpad + w_orig/2
    y2 = cy_unpad + h_orig/2

    print(f"\n  Converted to original coords:")
    print(f"  Box (xyxy): ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    # =========== COMPARE ===========
    print("\n" + "="*60)
    print("COORDINATE COMPARISON")
    print("="*60)

    if result.masks is not None:
        pt_best_box = result.boxes.xyxy[result.boxes.conf.argmax()].cpu().numpy()
        print(f"\nPyTorch best box:  ({pt_best_box[0]:.1f}, {pt_best_box[1]:.1f}, {pt_best_box[2]:.1f}, {pt_best_box[3]:.1f})")
        print(f"CoreML best box:   ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

        # Check if they're close
        diff = np.abs(pt_best_box - np.array([x1, y1, x2, y2]))
        print(f"Difference: {diff}")

        if diff.max() < 50:
            print("\n✓ Boxes match! HEAD is working correctly.")
            print("  The issue is likely in MASK GENERATION, not detection.")
        else:
            print("\n✗ Boxes don't match!")
            print("  -> Check coordinate transformation")

    # =========== TEST MASK GENERATION ===========
    print("\n" + "="*60)
    print("MASK GENERATION TEST")
    print("="*60)

    # Get mask coefficients (columns 6-37)
    coeffs = best_det[6:38]
    print(f"\nMask coefficients shape: {coeffs.shape}")
    print(f"Coefficients range: {coeffs.min():.4f} to {coeffs.max():.4f}")

    # Generate mask from prototypes
    # mask = sigmoid(sum(proto[i] * coeff[i]))
    mask_logits = np.zeros((320, 320), dtype=np.float32)
    for i in range(32):
        mask_logits += proto[i] * coeffs[i]

    mask = 1 / (1 + np.exp(-np.clip(mask_logits, -500, 500)))

    print(f"\nGenerated mask:")
    print(f"  Logits range: {mask_logits.min():.4f} to {mask_logits.max():.4f}")
    print(f"  Sigmoid range: {mask.min():.4f} to {mask.max():.4f}")
    print(f"  Coverage (>0.5): {(mask > 0.5).sum() / mask.size * 100:.2f}%")

    # Save the mask for inspection
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save("/Users/al/Documents/tries01/Furnit/debug_coreml_mask.png")
    print(f"\n  Saved mask to: debug_coreml_mask.png")

    # Compare with PyTorch mask if available
    if result.masks is not None:
        pt_mask = result.masks.data[result.boxes.conf.argmax()].cpu().numpy()
        pt_coverage = (pt_mask > 0.5).sum() / pt_mask.size * 100

        print(f"\nPyTorch mask coverage: {pt_coverage:.2f}%")
        print(f"CoreML mask coverage: {(mask > 0.5).sum() / mask.size * 100:.2f}%")

if __name__ == "__main__":
    main()
