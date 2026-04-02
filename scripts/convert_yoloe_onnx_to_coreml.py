#!/usr/bin/env python3
"""
Convert YOLOE ONNX (e.g. android/yoloe-11l-seg-pf.onnx) to Core ML .mlpackage for iOS.

Dependencies (use a clean venv; avoid repo ``executorch/`` on PYTHONPATH):
  pip install coremltools onnx

**Preferred path for iOS:** export Core ML directly from PyTorch with Ultralytics — tensors and
names match what ``FurnitureFitView`` expects (``var_2374`` / ``var_2412`` on typical 11l exports):

  python scripts/yoloe_export_and_backup.py --format coreml

See docs/YOLOE_COREML_FIX.md.

ONNX → Core ML can fail on dynamic shapes or unsupported ops; this script is a best-effort helper.

**coremltools 6+** removed ``ct.converters.onnx``. For YOLOE from the Android ONNX file, use the
legacy stack instead (see ``scripts/yoloe_onnx_to_coreml_ct5.py`` and its docstring).

Example (when your ``coremltools`` still exposes ``converters.onnx``):

  python scripts/convert_yoloe_onnx_to_coreml.py \\
    --onnx android/yoloe-11l-seg-pf.onnx \\
    --out ./yoloe-11l-seg-pf.mlpackage

Example (recommended for current YOLOE ONNX in this repo):

  # See scripts/requirements-yoloe-onnx-coreml.txt for a venv pin set.
  env -u PYTHONPATH .build/conda_ct_onnx/bin/python scripts/yoloe_onnx_to_coreml_ct5.py \\
    --onnx android/yoloe-11l-seg-pf.onnx \\
    --out .build/yoloe-from-onnx.mlmodel --verify-ort
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX → Core ML for YOLOE (best-effort).")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "android" / "yoloe-11l-seg-pf.onnx",
        help="Input ONNX file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path (.mlpackage directory)",
    )
    args = parser.parse_args()

    if not args.onnx.is_file():
        print(f"Missing ONNX: {args.onnx}", file=sys.stderr)
        return 1

    try:
        import coremltools as ct
    except ImportError as e:
        print("Install coremltools: pip install coremltools onnx", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    out = args.out
    if out.suffix == ".mlmodel" or (out.exists() and out.is_file()):
        print("Use --out path ending in .mlpackage (directory bundle).", file=sys.stderr)
        return 1

    if not hasattr(ct.converters, "onnx"):
        print(
            "This coremltools build has no ONNX converter. For YOLOE ONNX use:\n"
            "  scripts/yoloe_onnx_to_coreml_ct5.py\n"
            "(coremltools 5.2 + NumPy 1.23.x; see that file's docstring).\n",
            file=sys.stderr,
        )
        return 1

    print(f"Converting {args.onnx} → {out} …")
    try:
        ml = ct.converters.onnx.convert(
            model=str(args.onnx),
            minimum_deployment_target=ct.target.iOS15,
        )
    except Exception as e:
        print(
            "Conversion failed (common for seg heads / dynamic shapes). "
            "Try scripts/yoloe_onnx_to_coreml_ct5.py or "
            "scripts/yoloe_export_and_backup.py --format coreml from the .pt weights.\n",
            file=sys.stderr,
        )
        print(repr(e), file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(out))
    print(f"Saved {out}")
    print("Add the .mlpackage to the Xcode target and keep the default name yoloe-11l-seg-pf.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
