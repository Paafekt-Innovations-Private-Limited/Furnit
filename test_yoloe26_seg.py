#!/usr/bin/env python3
"""
Test YOLOE-26l-seg-pf CoreML model with a test image and save output.
"""

import coremltools as ct
import numpy as np
from PIL import Image
import sys

def letterbox_image(img, target_size=1280):
    """Resize image with letterboxing to square."""
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize
    resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Create letterboxed image (gray background = 128)
    letterboxed = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    letterboxed.paste(resized, (pad_x, pad_y))

    return letterboxed, scale, pad_x, pad_y, new_w, new_h

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def main():
    model_path = "/Users/al/Documents/tries01/Furnit/yoloe-26l-seg-pf.mlpackage"
    image_path = "/Users/al/Documents/tries01/Furnit/android/TestRoom.jpg"
    output_path = "/Users/al/Documents/tries01/Furnit/yoloe26_room_output.png"

    print(f"Loading model: {model_path}")
    model = ct.models.MLModel(model_path)

    # Print model info
    spec = model.get_spec()
    print("\nModel inputs:")
    for inp in spec.description.input:
        print(f"  {inp.name}: {inp.type}")
    print("\nModel outputs:")
    for out in spec.description.output:
        print(f"  {out.name}: {out.type}")

    print(f"\nLoading image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    print(f"Original size: {orig_w}x{orig_h}")

    # Letterbox to 1280x1280
    letterboxed, scale, pad_x, pad_y, content_w, content_h = letterbox_image(img, 1280)
    print(f"Letterboxed: scale={scale:.4f}, pad=({pad_x},{pad_y}), content={content_w}x{content_h}")

    # Run inference
    print("\nRunning inference...")
    output = model.predict({'image': letterboxed})

    print("\nOutput keys:", list(output.keys()))
    for k, v in output.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v)}")

    # Get detections and prototypes (handle different output names)
    detections = output.get('detections') or output.get('var_2346')
    prototypes = output.get('prototypes') or output.get('var_2429')

    if detections is None or prototypes is None:
        print("ERROR: Missing detections or prototypes output")
        print("Available outputs:", list(output.keys()))
        return

    # Parse detections - shape is typically [1, num_features, num_detections]
    det_shape = detections.shape
    print(f"\nDetections shape: {det_shape}")

    if len(det_shape) == 3:
        if det_shape[0] == 1:
            detections = detections[0]  # Remove batch dim

    # Determine format: [features, detections] or [detections, features]
    if detections.shape[0] < detections.shape[1]:
        # [features, detections] -> transpose to [detections, features]
        detections = detections.T

    num_dets, num_features = detections.shape
    print(f"Detections: {num_dets} candidates x {num_features} features")

    # Features: x, y, w, h, conf, cls*num_classes (optional), 32 mask coefficients
    # For YOLOE with prompt-free: typically x,y,w,h + conf + 32 coeffs = 37 features
    # Or x,y,w,h + conf + class_scores + 32 coeffs

    # Parse prototypes
    proto_shape = prototypes.shape
    print(f"Prototypes shape: {proto_shape}")

    # Remove batch dim if present
    if len(proto_shape) == 4 and proto_shape[0] == 1:
        prototypes = prototypes[0]

    # Determine layout: [32, H, W] or [H, W, 32]
    if prototypes.shape[0] == 32:
        # [32, H, W] -> keep as is
        num_proto, proto_h, proto_w = prototypes.shape
    else:
        # [H, W, 32] -> transpose
        proto_h, proto_w, num_proto = prototypes.shape
        prototypes = prototypes.transpose(2, 0, 1)

    print(f"Prototypes: {num_proto} channels x {proto_h}x{proto_w}")

    # Filter by confidence
    conf_thresh = 0.25

    # CoreML YOLOE output format (38 features):
    # [0-3]: x1, y1, x2, y2 (XYXY format in model space 1280x1280)
    # [4]: confidence
    # [5]: class info (ignored for prompt-free)
    # [6-37]: 32 mask coefficients

    if num_features == 38:
        # XYXY format + conf + class + 32 coeffs
        x1_raw = detections[:, 0]
        y1_raw = detections[:, 1]
        x2_raw = detections[:, 2]
        y2_raw = detections[:, 3]
        # Convert to center + size format for bbox drawing
        x = (x1_raw + x2_raw) / 2  # center x
        y = (y1_raw + y2_raw) / 2  # center y
        w = x2_raw - x1_raw  # width
        h = y2_raw - y1_raw  # height
        conf = detections[:, 4]
        coeffs = detections[:, 6:38]  # Skip col 5 (class info), take 32 coeffs
        print(f"Format: XYXY (x1,y1,x2,y2) + conf + class + 32 mask coefficients")
    elif num_features == 37:
        # x,y,w,h,conf + 32 coeffs (older format)
        x = detections[:, 0]
        y = detections[:, 1]
        w = detections[:, 2]
        h = detections[:, 3]
        conf = detections[:, 4]
        coeffs = detections[:, 5:37]
        print(f"Format: XYWH (cx,cy,w,h) + conf + 32 mask coefficients")
    else:
        print(f"Unknown detection format with {num_features} features")
        return

    # Filter detections
    mask = conf > conf_thresh
    valid_indices = np.where(mask)[0]
    print(f"\nFound {len(valid_indices)} detections above {conf_thresh} confidence")

    if len(valid_indices) == 0:
        print("No detections found!")
        # Save original image
        img.save(output_path)
        print(f"Saved original image to: {output_path}")
        return

    # Sort by confidence and take top detections
    sorted_indices = valid_indices[np.argsort(conf[valid_indices])[::-1]]
    top_k = min(10, len(sorted_indices))
    selected = sorted_indices[:top_k]

    print(f"\nTop {top_k} detections:")
    for i, idx in enumerate(selected):
        print(f"  {i+1}. conf={conf[idx]:.3f}, bbox=({x[idx]:.0f},{y[idx]:.0f},{w[idx]:.0f},{h[idx]:.0f})")

    # Generate combined mask from all detections
    # mask = sigmoid(sum of (prototype @ coeffs) for each detection)
    combined_mask = np.zeros((proto_h, proto_w), dtype=np.float32)

    for idx in selected:
        c = coeffs[idx]  # 32 coefficients
        # Compute mask: sum over channels of (prototype[c] * coeff[c])
        det_mask = np.zeros((proto_h, proto_w), dtype=np.float32)
        for ch in range(32):
            det_mask += prototypes[ch] * c[ch]
        # Take max (union of masks)
        combined_mask = np.maximum(combined_mask, det_mask)

    # Apply sigmoid and threshold
    combined_mask = sigmoid(combined_mask)
    binary_mask = (combined_mask > 0.5).astype(np.uint8)

    print(f"\nMask stats: min={combined_mask.min():.3f}, max={combined_mask.max():.3f}")
    print(f"Binary mask: {binary_mask.sum()} positive pixels out of {binary_mask.size}")

    # Upscale mask to model input size (1280x1280)
    mask_pil = Image.fromarray((combined_mask * 255).astype(np.uint8))
    mask_1280 = mask_pil.resize((1280, 1280), Image.BILINEAR)
    mask_1280 = np.array(mask_1280) / 255.0

    # Crop to content region
    mask_content = mask_1280[pad_y:pad_y+content_h, pad_x:pad_x+content_w]

    # Resize mask to original image size
    mask_content_pil = Image.fromarray((mask_content * 255).astype(np.uint8))
    mask_orig = mask_content_pil.resize((orig_w, orig_h), Image.BILINEAR)
    mask_orig = np.array(mask_orig) / 255.0

    # Create output image with mask overlay
    img_array = np.array(img)

    # Create RGBA output with transparency
    output_rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    output_rgba[:, :, :3] = img_array
    output_rgba[:, :, 3] = (mask_orig * 255).astype(np.uint8)

    # Save as PNG with transparency
    output_img = Image.fromarray(output_rgba, 'RGBA')
    output_img.save(output_path)
    print(f"\nSaved output to: {output_path}")

    # Also save a visualization with bboxes
    viz_path = output_path.replace('.png', '_viz.png')
    viz_img = img.copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(viz_img)

    for idx in selected[:5]:  # Draw top 5 bboxes
        # Convert from model coords to image coords
        cx, cy, bw, bh = x[idx], y[idx], w[idx], h[idx]
        # Un-letterbox
        img_cx = (cx - pad_x) / scale
        img_cy = (cy - pad_y) / scale
        img_bw = bw / scale
        img_bh = bh / scale

        x1 = int(img_cx - img_bw/2)
        y1 = int(img_cy - img_bh/2)
        x2 = int(img_cx + img_bw/2)
        y2 = int(img_cy + img_bh/2)

        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, y1-15), f"{conf[idx]:.2f}", fill='red')

    viz_img.save(viz_path)
    print(f"Saved visualization to: {viz_path}")

if __name__ == "__main__":
    main()
