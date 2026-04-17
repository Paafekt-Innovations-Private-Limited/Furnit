#!/usr/bin/env python3
"""
Re-export YOLO-E 11L seg-pf through the delayed-concat ANE pipeline.

Workflow:
1. Export PT -> ONNX with static shapes and `nms=False`
2. Split large class-head ops with delayed concat
3. Verify original vs split ONNX outputs match
4. Convert the split ONNX to Core ML Float16
5. Print output names for `YoloEDetectionParser.swift`

Usage:
  python3 scripts/reexport_yoloe_ane.py \
      --pt android/yoloe-11l-seg-pf.pt \
      --output-onnx android/yoloe-11l-seg-pf_split.onnx \
      --output-mlpackage yoloe-11l-seg-pf-ane.mlpackage

Environment (avoid local executorch/ shadowing coremltools):
  cd /tmp && env PYTHONPATH=/opt/miniconda3/envs/coreml-py311/lib/python3.11/site-packages \
    /opt/miniconda3/envs/coreml-py311/bin/python /path/to/repo/scripts/reexport_yoloe_ane.py ...
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from split_onnx_delayed_concat import (
    convert_split_onnx_to_coreml,
    patch_onnx_delayed_concat,
    print_coreml_output_names,
    verify_onnx_outputs_match,
)


def export_to_onnx(pt_path: str, onnx_path: str, imgsz: int = 1280) -> str:
    """Export .pt to ONNX with static shapes and no NMS."""
    from ultralytics import YOLO

    model = YOLO(pt_path)
    model.fuse()
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=False,
        nms=False,
        simplify=True,
        opset=17,
        dynamic=False,
    )

    exported_onnx_path = Path(exported_path)
    requested_onnx_path = Path(onnx_path)
    if exported_onnx_path.resolve() != requested_onnx_path.resolve():
        shutil.copy2(exported_onnx_path, requested_onnx_path)
    print(f"✅ ONNX exported with nms=False: {requested_onnx_path}")
    return str(requested_onnx_path)


def validate_model(model_path: str, imgsz: int = 1280, num_classes: int = 4585) -> bool:
    """Validate output shapes and run a test prediction."""
    import coremltools as ct
    from PIL import Image

    print(f"\nValidating {model_path}...")
    model = ct.models.MLModel(model_path)
    spec = model.get_spec()

    print("\nOutputs:")
    for output in spec.description.output:
        array_type = output.type.multiArrayType
        shape = list(array_type.shape)
        data_type = array_type.dataType
        data_type_name = {65552: "Float16", 65568: "Float32"}.get(data_type, str(data_type))
        print(f"  {output.name}: {shape} ({data_type_name})")

    try:
        input_description = spec.description.input[0]
        input_name = input_description.name
        if input_description.type.WhichOneof("Type") == "imageType":
            test_value = Image.fromarray(
                np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
            )
        else:
            test_value = np.ascontiguousarray(
                np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
            )

        outputs = model.predict({input_name: test_value})
        detection_output_name = None
        proto_output_name = None

        for name, array in outputs.items():
            shape = array.shape if hasattr(array, "shape") else type(array)
            print(f"  {name}: {shape}")
            if not hasattr(array, "shape"):
                continue
            if len(array.shape) == 3 and (4 + num_classes + 32) in array.shape:
                detection_output_name = name
            elif len(array.shape) == 4 and 32 in array.shape:
                proto_output_name = name

        if detection_output_name:
            print(f"  ✅ Detection tensor candidate: {detection_output_name}")
        else:
            print("  ⚠️  Could not identify detection tensor by shape")

        if proto_output_name:
            print(f"  ✅ Proto tensor candidate: {proto_output_name}")
        else:
            print("  ⚠️  Could not identify proto tensor by shape")

        print("\n✅ Validation PASSED")
        return True
    except Exception as error:
        print(f"\n❌ Validation FAILED: {error}")
        import traceback
        traceback.print_exc()
        return False


def convert_with_legacy_onnx_frontend(
    split_onnx_path: str,
    output_path: str,
    legacy_python_path: str,
) -> str:
    output_model_path = Path(output_path)
    if output_model_path.suffix.lower() != ".mlmodel":
        output_model_path = output_model_path.with_suffix(".mlmodel")

    legacy_converter_script = Path(__file__).with_name("yoloe_onnx_to_coreml_ct5.py")
    command = [
        legacy_python_path,
        str(legacy_converter_script),
        "--onnx",
        split_onnx_path,
        "--out",
        str(output_model_path),
    ]
    environment = dict(**__import__("os").environ)
    environment.pop("PYTHONPATH", None)

    print("Falling back to legacy coremltools 5.2 ONNX converter...")
    completed_process = subprocess.run(command, check=False, env=environment)
    if completed_process.returncode != 0:
        raise RuntimeError("Legacy ONNX->Core ML conversion failed")
    print(f"✅ Legacy converter saved: {output_model_path}")
    return str(output_model_path)


def resolve_onnx_path(args) -> str:
    if args.onnx and Path(args.onnx).exists():
        print(f"Using existing ONNX: {args.onnx}")
        return args.onnx

    default_onnx_path = str(Path(args.pt).with_suffix(".onnx"))
    if Path(default_onnx_path).exists():
        print(f"ONNX already exists: {default_onnx_path}")
        return default_onnx_path

    return export_to_onnx(args.pt, default_onnx_path, args.imgsz)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run delayed-concat YOLO-E export for ANE/Core ML"
    )
    parser.add_argument(
        "--pt",
        type=str,
        default="android/yoloe-11l-seg-pf.pt",
        help="Path to Ultralytics .pt checkpoint",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Path to existing ONNX (skip pt->onnx export)",
    )
    parser.add_argument(
        "--output-onnx",
        type=str,
        default=None,
        help="Output split ONNX path (default: <input>_split.onnx)",
    )
    parser.add_argument(
        "--output-mlpackage",
        "--output",
        dest="output_mlpackage",
        type=str,
        default="yoloe-11l-seg-pf-ane.mlpackage",
        help="Output .mlpackage path",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--num-classes", type=int, default=4585)
    parser.add_argument(
        "--legacy-onnx-python",
        type=str,
        default=".build/conda_ct5_onnx/bin/python",
        help="Python executable for legacy coremltools 5.2 ONNX conversion fallback",
    )
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-coreml", action="store_true")
    parser.add_argument(
        "--validate-only",
        type=str,
        default=None,
        help="Just validate an existing .mlpackage",
    )
    args = parser.parse_args()

    if args.validate_only:
        if not validate_model(args.validate_only, args.imgsz, args.num_classes):
            sys.exit(1)
        return

    original_onnx_path = resolve_onnx_path(args)
    split_onnx_path = args.output_onnx or str(
        Path(original_onnx_path).with_name(Path(original_onnx_path).stem + "_split.onnx")
    )

    print("\n" + "=" * 60)
    print("Step 1: Delayed-concat ONNX split")
    print("=" * 60)
    if not patch_onnx_delayed_concat(original_onnx_path, split_onnx_path):
        sys.exit(1)

    if not args.skip_verify:
        print("\n" + "=" * 60)
        print("Step 2: Verify original vs split ONNX")
        print("=" * 60)
        try:
            if verify_onnx_outputs_match(original_onnx_path, split_onnx_path, args.imgsz):
                print("✅ Verification PASSED")
            else:
                print("❌ Verification FAILED — outputs do not match")
                sys.exit(1)
        except ImportError:
            print("⚠️  onnxruntime not installed — skipping verification")

    if not args.skip_coreml:
        print("\n" + "=" * 60)
        print("Step 3: Convert split ONNX to CoreML")
        print("=" * 60)
        coreml_output_path = args.output_mlpackage
        try:
            model = convert_split_onnx_to_coreml(
                split_onnx_path,
                args.output_mlpackage,
                args.imgsz,
            )
            print_coreml_output_names(model, args.imgsz)
        except Exception as conversion_error:
            legacy_python_path = Path(args.legacy_onnx_python)
            if not legacy_python_path.exists():
                raise conversion_error
            print(f"⚠️  Modern ONNX->Core ML conversion failed: {conversion_error}")
            coreml_output_path = convert_with_legacy_onnx_frontend(
                split_onnx_path,
                args.output_mlpackage,
                str(legacy_python_path),
            )

        if not validate_model(coreml_output_path, args.imgsz, args.num_classes):
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("✅ RE-EXPORT COMPLETE")
    print(f"{'=' * 60}")
    print(f"Original ONNX: {original_onnx_path}")
    print(f"Split ONNX:    {split_onnx_path}")
    if not args.skip_coreml:
        print(f"Core ML:       {coreml_output_path}")
    print("Export contract: nms=False (unchanged for parser compatibility)")


if __name__ == "__main__":
    main()
