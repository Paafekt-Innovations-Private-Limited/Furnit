#!/usr/bin/env python3
"""
Compare ALL detections between PyTorch and CoreML to find matching objects.
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

    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    print(f"Original image: {orig_w}x{orig_h}")

    # =========== PyTorch ===========
    model = YOLO(pt_path)
    results = model(image_path, imgsz=1280, conf=0.25, verbose=False)
    result = results[0]

    pt_boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4] in original coords
    pt_confs = result.boxes.conf.cpu().numpy()

    # Sort by confidence
    pt_order = np.argsort(pt_confs)[::-1]
    pt_boxes = pt_boxes[pt_order]
    pt_confs = pt_confs[pt_order]

    print(f"\n=== PYTORCH TOP 10 DETECTIONS ===")
    print(f"Total detections: {len(pt_boxes)}")
    for i in range(min(10, len(pt_boxes))):
        x1, y1, x2, y2 = pt_boxes[i]
        cx, cy = (x1+x2)/2, (y1+y2)/2
        w, h = x2-x1, y2-y1
        print(f"  {i+1}. conf={pt_confs[i]:.3f}, center=({cx:.0f},{cy:.0f}), size=({w:.0f}x{h:.0f})")

    # =========== CoreML ===========
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

    output = coreml_model.predict({'image': letterboxed})
    det = output['var_2346'][0]
    if det.shape[0] < det.shape[1]:
        det = det.T

    # Extract coords and confidence
    cm_x = det[:, 0]  # center x in model space
    cm_y = det[:, 1]  # center y in model space
    cm_w = det[:, 2]  # width
    cm_h = det[:, 3]  # height
    cm_conf = det[:, 4]

    # Convert to original image coordinates
    # Un-pad and un-scale
    cm_x_orig = (cm_x - pad_x) / scale
    cm_y_orig = (cm_y - pad_y) / scale
    cm_w_orig = cm_w / scale
    cm_h_orig = cm_h / scale

    # Sort by confidence
    cm_order = np.argsort(cm_conf)[::-1]

    print(f"\n=== COREML TOP 10 DETECTIONS ===")
    print(f"Total with conf > 0.25: {np.sum(cm_conf > 0.25)}")
    for i in range(10):
        idx = cm_order[i]
        cx, cy = cm_x_orig[idx], cm_y_orig[idx]
        w, h = cm_w_orig[idx], cm_h_orig[idx]
        print(f"  {i+1}. conf={cm_conf[idx]:.3f}, center=({cx:.0f},{cy:.0f}), size=({w:.0f}x{h:.0f})")

    # =========== FIND MATCHING DETECTIONS ===========
    print(f"\n=== MATCHING ANALYSIS ===")
    print("Looking for CoreML detections that match PyTorch boxes...")

    def iou(box1_xyxy, cx, cy, w, h):
        """Calculate IoU between xyxy box and xywh box."""
        x1_a, y1_a, x2_a, y2_a = box1_xyxy
        x1_b = cx - w/2
        y1_b = cy - h/2
        x2_b = cx + w/2
        y2_b = cy + h/2

        xi1 = max(x1_a, x1_b)
        yi1 = max(y1_a, y1_b)
        xi2 = min(x2_a, x2_b)
        yi2 = min(y2_a, y2_b)

        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = w * h
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0

    # For each PyTorch detection, find best matching CoreML detection
    for i in range(min(5, len(pt_boxes))):
        pt_box = pt_boxes[i]

        best_iou = 0
        best_idx = -1

        for j in range(len(cm_conf)):
            if cm_conf[j] < 0.1:
                continue
            iou_val = iou(pt_box, cm_x_orig[j], cm_y_orig[j], cm_w_orig[j], cm_h_orig[j])
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = j

        if best_idx >= 0:
            print(f"\nPT#{i+1} (conf={pt_confs[i]:.3f}) -> Best CM match: IoU={best_iou:.3f}")
            print(f"  PT box: {pt_box}")
            print(f"  CM conf: {cm_conf[best_idx]:.3f}, center=({cm_x_orig[best_idx]:.0f},{cm_y_orig[best_idx]:.0f}), size=({cm_w_orig[best_idx]:.0f}x{cm_h_orig[best_idx]:.0f})")

            # Check if this is a reasonable match
            if best_iou > 0.5:
                print(f"  ✓ GOOD MATCH")
            elif best_iou > 0.3:
                print(f"  ~ PARTIAL MATCH")
            else:
                print(f"  ✗ POOR MATCH")
        else:
            print(f"\nPT#{i+1} (conf={pt_confs[i]:.3f}) -> NO MATCH FOUND")

    # =========== CHECK BBOX FORMAT ===========
    print(f"\n=== BBOX FORMAT VERIFICATION ===")

    # Check if CoreML might be using xyxy instead of xywh
    print("\nTrying different bbox interpretations for CoreML top detection:")
    idx = cm_order[0]
    c0, c1, c2, c3 = det[idx, 0], det[idx, 1], det[idx, 2], det[idx, 3]

    print(f"\nRaw values: {c0:.1f}, {c1:.1f}, {c2:.1f}, {c3:.1f}")

    print("\n1. As XYWH (center x, center y, width, height):")
    x1 = (c0 - c2/2 - pad_x) / scale
    y1 = (c1 - c3/2 - pad_y) / scale
    x2 = (c0 + c2/2 - pad_x) / scale
    y2 = (c1 + c3/2 - pad_y) / scale
    print(f"   xyxy: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    print("\n2. As XYXY (x1, y1, x2, y2):")
    x1 = (c0 - pad_x) / scale
    y1 = (c1 - pad_y) / scale
    x2 = (c2 - pad_x) / scale
    y2 = (c3 - pad_y) / scale
    print(f"   xyxy: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    print("\n3. As XYWH but model output coords (no letterbox):")
    # Maybe the model already outputs in a different coordinate system
    x1 = c0 - c2/2
    y1 = c1 - c3/2
    x2 = c0 + c2/2
    y2 = c1 + c3/2
    print(f"   Model space xyxy: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

if __name__ == "__main__":
    main()
