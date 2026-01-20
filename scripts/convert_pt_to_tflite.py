#!/usr/bin/env python3
"""
Convert a PyTorch .pt (TorchScript or traced model) to TFLite via ONNX -> SavedModel -> TFLite.

Usage:
  python3 scripts/convert_pt_to_tflite.py --input android/yoloe-11l-seg-pf.pt --output android/app/src/main/assets/yoloe_11l_from_pt.tflite --dummy-shape 1,3,1536,1536

Notes:
- This script assumes the .pt is a TorchScript/traced model that can be loaded with `torch.jit.load`.
- If the .pt is only a state_dict, you must re-create the model class in Python and load the state dict.
- Install required packages: torch, onnx, onnx-tf, tensorflow. See `scripts/requirements.txt` for the coreml path; you may need additional torch/torchvision versions compatible with your CUDA or CPU.
"""

import argparse
import os
import sys
from pathlib import Path


def log(msg):
    print(msg)


def convert(pt_path: Path, out_tflite: Path, dummy_shape, tmpdir: Path):
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is not available in this environment: " + str(e))

    try:
        import onnx
    except Exception as e:
        raise RuntimeError("onnx is not available: " + str(e))

    try:
        from onnx_tf.backend import prepare as onnx_prepare
    except Exception:
        onnx_prepare = None

    try:
        import tensorflow as tf
    except Exception as e:
        tf = None

    log(f"Loading TorchScript model from {pt_path}")
    # Try to load as TorchScript
    try:
        model = torch.jit.load(str(pt_path), map_location='cpu')
        model.eval()
    except Exception as e:
        # Could be a state dict
        raise RuntimeError("Failed to load as TorchScript. If this is a state_dict, you must recreate the model class in Python and load it. Error: " + str(e))

    # Dummy input
    import numpy as np
    shape = tuple(int(x) for x in dummy_shape.split(","))
    dummy = torch.zeros(shape)

    onnx_path = tmpdir / (pt_path.stem + ".onnx")
    log(f"Exporting to ONNX: {onnx_path}")
    try:
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=13, do_constant_folding=True, input_names=['input'], output_names=['output'])
    except Exception as e:
        raise RuntimeError("torch.onnx.export failed: " + str(e))

    log("ONNX export complete. Converting ONNX -> SavedModel -> TFLite")
    if onnx_prepare is None or tf is None:
        raise RuntimeError("onnx-tf or tensorflow not available for ONNX->TFLite conversion")

    # Convert ONNX -> SavedModel
    model_onnx = onnx.load(str(onnx_path))
    tf_rep = onnx_prepare(model_onnx)
    saved_model_dir = tmpdir / "saved_model"
    if saved_model_dir.exists():
        import shutil
        shutil.rmtree(saved_model_dir)
    tf_rep.export_graph(str(saved_model_dir))

    # Convert SavedModel -> TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_model = converter.convert()
    out_tflite.parent.mkdir(parents=True, exist_ok=True)
    out_tflite.write_bytes(tflite_model)
    log(f"Wrote TFLite to {out_tflite}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dummy-shape", default="1,3,1536,1536", help="comma-separated tensor shape for dummy input")
    p.add_argument("--tmpdir", default="/tmp/pt_conv", help="temporary dir")
    args = p.parse_args()

    pt = Path(args.input)
    out = Path(args.output)
    tmp = Path(args.tmpdir)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        convert(pt, out, args.dummy_shape, tmp)
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()
