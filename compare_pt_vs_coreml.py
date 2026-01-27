#!/usr/bin/env python3
"""
Compare PyTorch vs CoreML model outputs to find where they differ.
"""

import torch
import coremltools as ct
import numpy as np
from PIL import Image
from ultralytics import YOLO

def letterbox_image(img, target_size=1280):
    """Resize image with letterboxing to square."""
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = img.resize((new_w, new_h), Image.BILINEAR)
    letterboxed = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    letterboxed.paste(resized, (pad_x, pad_y))

    return letterboxed, scale, pad_x, pad_y

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def test_pytorch(model_path, image_path):
    """Run PyTorch model and get outputs."""
    print("\n" + "="*60)
    print("PYTORCH MODEL TEST")
    print("="*60)

    model = YOLO(model_path)
    results = model(image_path, imgsz=1280, conf=0.25, verbose=False)

    result = results[0]

    print(f"\nDetections: {len(result.boxes)}")

    if len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        print("\nTop 5 detections:")
        for i in range(min(5, len(boxes))):
            x1, y1, x2, y2 = boxes[i]
            print(f"  {i+1}. conf={confs[i]:.3f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    # Check if masks are present
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        print(f"\nMasks shape: {masks.shape}")
        print(f"Mask coverage: {masks.sum() / masks.size * 100:.1f}%")
    else:
        print("\nNo masks in output!")

    return result

def test_coreml(model_path, image_path):
    """Run CoreML model and get raw outputs."""
    print("\n" + "="*60)
    print("COREML MODEL TEST")
    print("="*60)

    model = ct.models.MLModel(model_path)

    img = Image.open(image_path).convert('RGB')
    letterboxed, scale, pad_x, pad_y = letterbox_image(img, 1280)

    print(f"\nInput: {img.size} -> letterboxed 1280x1280 (scale={scale:.4f}, pad=({pad_x},{pad_y}))")

    output = model.predict({'image': letterboxed})

    # Get raw outputs
    detections = output.get('var_2346')
    prototypes = output.get('var_2429')

    print(f"\nRaw detections shape: {detections.shape}")
    print(f"Raw prototypes shape: {prototypes.shape}")

    # Parse detections
    det = detections[0]  # Remove batch dim
    if det.shape[0] < det.shape[1]:
        det = det.T  # Transpose to [N, features]

    num_dets, num_features = det.shape
    print(f"Parsed: {num_dets} detections x {num_features} features")

    # Extract components
    # Format: x, y, w, h, class_scores..., mask_coeffs (32)
    x = det[:, 0]
    y = det[:, 1]
    w = det[:, 2]
    h = det[:, 3]

    # Check what's in columns 4 onwards
    print(f"\nColumn 4 stats: min={det[:,4].min():.4f}, max={det[:,4].max():.4f}")
    print(f"Column 5 stats: min={det[:,5].min():.4f}, max={det[:,5].max():.4f}")
    print(f"Column 6 stats: min={det[:,6].min():.4f}, max={det[:,6].max():.4f}")
    print(f"Last column (37) stats: min={det[:,37].min():.4f}, max={det[:,37].max():.4f}")

    # Try different interpretations
    print("\n--- Interpretation 1: col 4 = confidence ---")
    conf1 = det[:, 4]
    above_thresh1 = np.sum(conf1 > 0.25)
    print(f"  Detections > 0.25: {above_thresh1}")
    print(f"  Top 5 conf: {sorted(conf1, reverse=True)[:5]}")

    print("\n--- Interpretation 2: col 4 = class score (needs sigmoid) ---")
    conf2 = sigmoid(det[:, 4])
    above_thresh2 = np.sum(conf2 > 0.25)
    print(f"  Detections > 0.25: {above_thresh2}")
    print(f"  Top 5 conf: {sorted(conf2, reverse=True)[:5]}")

    print("\n--- Interpretation 3: cols 4-5 = obj_conf * class_score ---")
    if num_features > 5:
        conf3 = det[:, 4] * det[:, 5]
        above_thresh3 = np.sum(conf3 > 0.25)
        print(f"  Detections > 0.25: {above_thresh3}")
        print(f"  Top 5 conf: {sorted(conf3, reverse=True)[:5]}")

    # Check bbox values
    print("\n--- Bbox analysis ---")
    print(f"  X range: {x.min():.1f} - {x.max():.1f}")
    print(f"  Y range: {y.min():.1f} - {y.max():.1f}")
    print(f"  W range: {w.min():.1f} - {w.max():.1f}")
    print(f"  H range: {h.min():.1f} - {h.max():.1f}")

    # Check if values are normalized (0-1) or pixel coords
    if x.max() <= 1.0 and y.max() <= 1.0:
        print("  -> Coords appear NORMALIZED (0-1)")
    elif x.max() <= 1280 and y.max() <= 1280:
        print("  -> Coords appear to be PIXEL values")
    else:
        print("  -> Coords appear to be UNUSUAL - check conversion!")

    # Prototype analysis
    print("\n--- Prototype analysis ---")
    proto = prototypes[0]  # Remove batch
    print(f"  Shape: {proto.shape}")
    print(f"  Value range: {proto.min():.4f} to {proto.max():.4f}")

    return output, det, proto

def compare_outputs(pt_result, coreml_det, coreml_proto):
    """Compare PyTorch and CoreML outputs."""
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    # PyTorch boxes
    if len(pt_result.boxes) > 0:
        pt_boxes = pt_result.boxes.xyxy.cpu().numpy()
        pt_confs = pt_result.boxes.conf.cpu().numpy()

        print(f"\nPyTorch: {len(pt_boxes)} detections")
        print(f"  Conf range: {pt_confs.min():.3f} - {pt_confs.max():.3f}")

        # Best detection
        best_idx = np.argmax(pt_confs)
        print(f"  Best: conf={pt_confs[best_idx]:.3f}, box={pt_boxes[best_idx]}")

    # CoreML - find best detection
    conf = coreml_det[:, 4]  # Assuming col 4 is confidence
    best_idx = np.argmax(conf)
    x, y, w, h = coreml_det[best_idx, :4]

    print(f"\nCoreML: analyzing top detection")
    print(f"  Raw values: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}, conf={conf[best_idx]:.3f}")

    # Convert xywh to xyxy
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    print(f"  As xyxy: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    # Check if there's a systematic difference
    if len(pt_result.boxes) > 0:
        print("\n--- DIAGNOSIS ---")

        # Compare confidence distributions
        pt_conf_mean = pt_confs.mean()
        coreml_conf_mean = conf.mean()

        print(f"  PyTorch mean conf: {pt_conf_mean:.4f}")
        print(f"  CoreML mean conf: {coreml_conf_mean:.4f}")

        if coreml_conf_mean < 0.01 and pt_conf_mean > 0.1:
            print("\n  ISSUE: CoreML confidence values are very low!")
            print("  -> Possible cause: Missing sigmoid activation on confidence output")
            print("  -> Or: Confidence is in different column")

        if conf.max() > 100:
            print("\n  ISSUE: CoreML confidence values are > 1!")
            print("  -> These are likely raw logits, need sigmoid")

def main():
    pt_path = "/Users/al/Documents/tries01/Furnit/yoloe-26l-seg-pf.pt"
    coreml_path = "/Users/al/Documents/tries01/Furnit/yoloe-26l-seg-pf.mlpackage"
    image_path = "/Users/al/Documents/tries01/Furnit/android/TestRoom.jpg"

    print("="*60)
    print("PyTorch vs CoreML COMPARISON")
    print("="*60)
    print(f"\nImage: {image_path}")

    # Test PyTorch
    pt_result = test_pytorch(pt_path, image_path)

    # Test CoreML
    coreml_output, coreml_det, coreml_proto = test_coreml(coreml_path, image_path)

    # Compare
    compare_outputs(pt_result, coreml_det, coreml_proto)

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

if __name__ == "__main__":
    main()
