#!/usr/bin/env python3
"""
Fixed YOLOE-26L export to CoreML.
Key fix: Set export=True on head before tracing.
"""

import os
import torch
import numpy as np
from ultralytics import YOLOE
import coremltools as ct

CKPT_PATH = "/Users/al/Downloads/yoloe-26l-seg-pf.pt"
IMG_SIZE = 1280
OUT_PATH = "/Users/al/Documents/tries01/Furnit/yoloe_26l_seg_pf_1280_fixed.mlpackage"

def main():
    print("Loading YOLOE-26L model...")
    model = YOLOE(CKPT_PATH)
    net = model.model
    net.eval()
    net.to("cpu")

    # CRITICAL: Set export mode on the head
    head = net.model[-1]
    print(f"Head type: {type(head).__name__}")
    print(f"Head export BEFORE: {head.export}")
    head.export = True
    print(f"Head export AFTER: {head.export}")

    # Create wrapper that extracts clean outputs
    class YoloESegExportWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor):
            # With export=True, output is (det_tensor, proto_tensor)
            out = self.inner(x)

            # out should be tuple of (detections, prototypes)
            if isinstance(out, tuple) and len(out) == 2:
                det, proto = out
                if isinstance(det, torch.Tensor) and isinstance(proto, torch.Tensor):
                    return det, proto

            # Fallback: try to extract from nested structure
            if isinstance(out, tuple):
                if isinstance(out[0], tuple):
                    det = out[0][0]
                    proto = out[0][1]
                else:
                    det = out[0]
                    proto = out[1]
                return det, proto

            raise RuntimeError(f"Unexpected output structure: {type(out)}")

    wrapped = YoloESegExportWrapper(net)
    wrapped.eval()

    # Test forward pass
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
    print("\nTesting forward pass...")
    with torch.no_grad():
        det, proto = wrapped(dummy)
    print(f"Detection shape: {det.shape}")
    print(f"Proto shape: {proto.shape}")

    # Trace
    print("\nTracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, dummy, strict=False)
    traced.eval()

    # Verify traced outputs
    with torch.no_grad():
        det_t, proto_t = traced(dummy)
    print(f"Traced detection shape: {det_t.shape}")
    print(f"Traced proto shape: {proto_t.shape}")

    # Convert to CoreML
    print("\nConverting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=dummy.shape, dtype=np.float32)],
        outputs=[
            ct.TensorType(name="detections"),
            ct.TensorType(name="protos")
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL
    )

    mlmodel.save(OUT_PATH)
    print(f"\nSaved: {OUT_PATH}")

    # Verify CoreML model
    print("\nVerifying CoreML model...")
    mlmodel_loaded = ct.models.MLModel(OUT_PATH)

    # Test with actual image
    import cv2
    test_img = cv2.imread("/Users/al/Documents/tries01/Furnit/test_chair.jpg")
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    # Letterbox
    h, w = test_img_rgb.shape[:2]
    r = min(IMG_SIZE / h, IMG_SIZE / w)
    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(test_img_rgb, (new_w, new_h))
    padded = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    dw, dh = (IMG_SIZE - new_w) // 2, (IMG_SIZE - new_h) // 2
    padded[dh:dh+new_h, dw:dw+new_w] = resized

    # To tensor
    x = padded.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, 0)

    # PyTorch inference
    x_torch = torch.from_numpy(x)
    with torch.no_grad():
        pt_det, pt_proto = wrapped(x_torch)
    pt_det = pt_det.numpy()
    pt_proto = pt_proto.numpy()

    # CoreML inference
    coreml_out = mlmodel_loaded.predict({"image": x})
    coreml_det = coreml_out["detections"]
    coreml_proto = coreml_out["protos"]

    print(f"\nPyTorch top detection:")
    confs = pt_det[0, :, 4]
    top_idx = np.argmax(confs)
    d = pt_det[0, top_idx]
    print(f"  conf={d[4]:.3f} cls={int(d[5])} bbox=[{d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f}]")

    print(f"\nCoreML top detection:")
    confs = coreml_det[0, :, 4]
    top_idx = np.argmax(confs)
    d = coreml_det[0, top_idx]
    print(f"  conf={d[4]:.3f} cls={int(d[5])} bbox=[{d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f}]")

    # Compare
    det_diff = np.abs(pt_det - coreml_det).max()
    proto_diff = np.abs(pt_proto - coreml_proto).max()
    print(f"\nMax detection diff: {det_diff:.4f}")
    print(f"Max proto diff: {proto_diff:.4f}")

if __name__ == "__main__":
    main()
