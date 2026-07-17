#!/usr/bin/env python3
"""
Attempt reproducible conversions: CoreML (.mlmodel / .mlpackage) or ONNX -> TensorFlow SavedModel -> TFLite.

Usage:
  python3 convert_coreml_to_tflite.py --input PATH_TO_MODEL --output out.tflite --tmpdir /tmp/conv

Notes:
- Converting CoreML -> ONNX is not always supported by existing tools. This script tries reasonable
  fallbacks and prints clear guidance if an automated path fails.
- Recommended environment (pip install -r requirements.txt in this repo's `scripts/` folder).

Supported flows implemented:
- .onnx -> TFLite using onnx + onnx-tf -> SavedModel -> TFLite
- Attempt CoreML (.mlmodel) -> ONNX using coremltools if available (best-effort), otherwise
  suggests manual steps and extracts the mlmodel for inspection.

This is a best-effort helper to make conversions reproducible; manual intervention may be required
for model-specific post-processing (e.g., custom ops, multiple outputs, prototype masks, etc.).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import coremltools as ct
except Exception:
    ct = None

try:
    import onnx
except Exception:
    onnx = None

try:
    from onnx_tf.backend import prepare as onnx_prepare
except Exception:
    onnx_prepare = None

try:
    import tensorflow as tf
except Exception:
    tf = None

import numpy as np


def log(msg: str):
    print(msg)


def convert_onnx_to_tflite(onnx_path: Path, out_tflite: Path, tmpdir: Path):
    log(f"Converting ONNX -> SavedModel (tmp: {tmpdir})")
    if onnx is None or onnx_prepare is None or tf is None:
        raise RuntimeError("Required libraries for ONNX->TFLite (onnx, onnx-tf, tensorflow) are not available. See scripts/requirements.txt")

    model = onnx.load(str(onnx_path))
    tf_rep = onnx_prepare(model)
    saved_model_dir = tmpdir / "saved_model"
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    tf_rep.export_graph(str(saved_model_dir))
    log(f"SavedModel exported to {saved_model_dir}")

    # Convert SavedModel -> TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    # Use float16 quantization if desired (optional)
    try:
        tflite_model = converter.convert()
    except Exception as e:
        raise RuntimeError(f"TFLite conversion failed: {e}")

    out_tflite.write_bytes(tflite_model)
    log(f"Wrote TFLite to {out_tflite}")


def attempt_coreml_to_onnx(coreml_path: Path, out_onnx: Path):
    log("Attempting CoreML -> ONNX conversion (best-effort)")
    if ct is None:
        raise RuntimeError("coremltools is not available in the environment")

    # Best-effort strategies (wrapped in try/except). coremltools does not officially provide
    # a general CoreML->ONNX converter in many versions; this step may fail and will provide
    # guidance if so.
    try:
        # Try to load MLModel and inspect spec
        mlmodel = ct.models.MLModel(str(coreml_path))
        spec = mlmodel.get_spec()
        log(f"Loaded CoreML model: spec type {type(spec)}")

        # Some versions of coremltools have experimental converters via MIL -> ONNX
        # Try the converter if available
        convert_fn = None
        # Typical place: ct.converters.mil.frontend_to_onnx (not guaranteed)
        if hasattr(ct, "converters"):
            conv_mod = getattr(ct, "converters")
            if hasattr(conv_mod, "mil") and hasattr(conv_mod.mil, "convert"):
                convert_fn = conv_mod.mil.convert

        if convert_fn is not None:
            log("Found coremltools MIL convert function; attempting convert to ONNX (experimental)")
            try:
                onnx_model = convert_fn(str(coreml_path), target='onnx')
                # If convert_fn returns an onnx model object, write it
                if hasattr(onnx_model, 'SerializeToString'):
                    out_onnx.write_bytes(onnx_model.SerializeToString())
                    log(f"Wrote ONNX to {out_onnx}")
                    return out_onnx
            except Exception as e:
                log(f"Experimental coremltools->ONNX conversion failed: {e}")

        # If no converter available or failed, write guidance
        raise RuntimeError("Automated CoreML->ONNX conversion not available in coremltools. Try alternative paths described below.")

    except Exception as e:
        # Reraise with guidance
        raise RuntimeError(str(e))


def main():
    p = argparse.ArgumentParser(description="Convert CoreML/ONNX to TFLite (best-effort)")
    p.add_argument("--input", required=True, help="Path to input model (.mlmodel, .mlpackage, .onnx)")
    p.add_argument("--output", required=True, help="Path to output .tflite file")
    p.add_argument("--tmpdir", default=None, help="Temporary working directory")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    tmpdir = Path(args.tmpdir) if args.tmpdir else Path(tempfile.mkdtemp(prefix="conv_"))
    tmpdir.mkdir(parents=True, exist_ok=True)

    try:
        if inp.suffix == ".onnx":
            onnx_path = inp
            convert_onnx_to_tflite(onnx_path, out, tmpdir)
            return

        # If it's a .mlpackage, attempt to find inner mlmodel file
        if inp.suffix == ".mlpackage":
            # Try to find 'best.mlmodel' or '*.mlmodel' inside
            candidates = list(inp.rglob("*.mlmodel"))
            if not candidates:
                log(f"No .mlmodel found inside {inp}. You may need to extract it from the package.")
                return
            coreml_path = candidates[0]
            log(f"Found internal .mlmodel: {coreml_path}")
        else:
            coreml_path = inp

        # Try automated CoreML -> ONNX
        onnx_out = tmpdir / (coreml_path.stem + ".onnx")
        try:
            attempt_coreml_to_onnx(coreml_path, onnx_out)
            # If ONNX created, continue to ONNX->TFLite
            convert_onnx_to_tflite(onnx_out, out, tmpdir)
            return
        except Exception as e:
            log(f"CoreML->ONNX automated step failed: {e}")
            log("\nManual fallback instructions:")
            log(" - If you have the original training model (PyTorch/TF/ONNX), convert that directly to TFLite or ONNX.")
            log(" - To convert an ONNX model to TFLite, install onnx + onnx-tf and run this script with --input model.onnx")
            log(" - coremltools currently lacks a robust general CoreML->ONNX converter; consider these options:")
            log("     * Re-export the model from your original training code to ONNX (preferred)")
            log("     * If your CoreML model was compiled from a TFLite/ONNX artifact, try to locate that original file in the repo or build artifacts")
            log("     * As a last resort, inspect the CoreML spec with coremltools and manually recreate the model in TF/PyTorch")
            return

    finally:
        log(f"Temporary workdir: {tmpdir}")


if __name__ == '__main__':
    main()
