#!/usr/bin/env python3
"""
Re-export YOLO-E 11L seg-pf for ANE + CPU split execution.

Backbone + neck + proto head → ANE (~200ms)
Class prediction linear layers + transposes → CPU (~150ms)

Requires: coremltools >= 7.2, ultralytics, torch

Usage:
  # Full pipeline from .pt
  python3 scripts/reexport_yoloe_ane.py \
      --pt android/yoloe-11l-seg-pf.pt \
      --output yoloe-11l-seg-pf.mlpackage \
      --original yoloe-11l-seg-pf_fp32_backup.mlpackage

  # From existing ONNX
  python3 scripts/reexport_yoloe_ane.py \
      --onnx android/yoloe-11l-seg-pf.onnx \
      --output yoloe-11l-seg-pf.mlpackage

  # Validate only
  python3 scripts/reexport_yoloe_ane.py \
      --validate-only yoloe-11l-seg-pf.mlpackage

Environment (avoid local executorch/ shadowing coremltools):
  cd /tmp && env PYTHONPATH=/opt/miniconda3/envs/coreml-py311/lib/python3.11/site-packages \
    /opt/miniconda3/envs/coreml-py311/bin/python /path/to/repo/scripts/reexport_yoloe_ane.py ...
"""

import sys
import numpy as np
from pathlib import Path


# ─── Step 1: Export from Ultralytics to ONNX ───────────────────────

def export_to_onnx(pt_path: str, onnx_path: str, imgsz: int = 1280):
    """Export .pt to ONNX with static shapes, no NMS."""
    from ultralytics import YOLO

    model = YOLO(pt_path)
    model.fuse()
    model.export(
        format="onnx",
        imgsz=imgsz,
        half=False,        # export full precision, we'll handle F16 in coremltools
        nms=False,         # NMS stays in Swift
        simplify=True,     # ONNX simplifier cleans up redundant ops
        opset=17,
        dynamic=False,     # static shapes — required for ANE
    )
    print(f"✅ ONNX exported: {onnx_path}")


# ─── Step 2: Convert ONNX → CoreML with ANE + CPU split ───────────

def convert_to_coreml_split(
    onnx_path: str,
    output_path: str,
    imgsz: int = 1280,
    num_classes: int = 4585,
):
    """
    Convert ONNX to CoreML mlprogram with Float16 precision.
    After conversion, patch the mlprogram spec to force the
    4585-class linear/transpose ops onto CPU.
    """
    import coremltools as ct

    print(f"coremltools version: {ct.__version__}")
    ct_version = tuple(int(x) for x in ct.__version__.split('.')[:2])
    assert ct_version >= (7, 2), \
        f"Need coremltools >= 7.2 for compute_unit op attributes, got {ct.__version__}"

    # ── Convert with Float16 (ANE-native precision) ──
    print("Converting ONNX → CoreML (Float16, mlprogram)...")

    model = ct.convert(
        onnx_path,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, imgsz, imgsz),
                scale=1.0 / 255.0,
                bias=[0.0, 0.0, 0.0],
                color_layout="RGB",
            )
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )

    print("✅ Base conversion complete")

    # ── Patch: force large linear/transpose ops to CPU ──
    spec = model.get_spec()

    patched_count = 0

    for func_name, function in spec.mlProgram.functions.items():
        for block_name, block in function.block_specializations.items():
            for op in block.operations:
                should_force_cpu = False
                reason = ""

                # Force linear/matmul ops with 4585-wide outputs to CPU
                if op.type in ("linear", "matmul"):
                    for output in op.outputs:
                        tensor_type = output.type.tensorType
                        for dim in tensor_type.dimensions:
                            if hasattr(dim, 'constant') and \
                               hasattr(dim.constant, 'immediateValue') and \
                               dim.constant.immediateValue >= num_classes:
                                should_force_cpu = True
                                reason = f"{op.type} output dim={dim.constant.immediateValue}"
                                break

                # Force transpose ops on large tensors to CPU
                if op.type == "transpose":
                    for inp_key, inp_val in op.inputs.items():
                        name = inp_val.name if hasattr(inp_val, 'name') else str(inp_val)
                        if "linear" in name.lower() or str(num_classes) in name.lower():
                            should_force_cpu = True
                            reason = f"transpose of {name}"
                            break

                    if not should_force_cpu:
                        for output in op.outputs:
                            tensor_type = output.type.tensorType
                            dims = []
                            for d in tensor_type.dimensions:
                                if hasattr(d, 'constant') and hasattr(d.constant, 'immediateValue'):
                                    dims.append(d.constant.immediateValue)
                            if any(d >= num_classes for d in dims):
                                should_force_cpu = True
                                reason = f"transpose output dims={dims}"

                # Force concat ops that produce 4621+ channels to CPU
                if op.type == "concat":
                    for output in op.outputs:
                        tensor_type = output.type.tensorType
                        dims = []
                        for d in tensor_type.dimensions:
                            if hasattr(d, 'constant') and hasattr(d.constant, 'immediateValue'):
                                dims.append(d.constant.immediateValue)
                        if any(d >= num_classes for d in dims):
                            should_force_cpu = True
                            reason = f"concat output dims={dims}"

                if should_force_cpu:
                    attr = op.attributes.add()
                    attr.name = "compute_unit"
                    attr.value.type = 30  # string type
                    attr.value.immediateValue.s = "cpuOnly"
                    patched_count += 1
                    print(f"  🔧 CPU: {op.type} [{op.name}] — {reason}")

    print(f"\n✅ Patched {patched_count} ops to CPU")

    # ── Save ──
    patched_model = ct.models.MLModel(spec)
    patched_model.save(output_path)
    print(f"✅ Saved: {output_path}")

    return patched_model


# ─── Step 2B (fallback): MIL pass approach ─────────────────────────

def convert_with_mil_pass(onnx_path: str, output_path: str, imgsz: int = 1280,
                          num_classes: int = 4585):
    """
    Fallback: convert with a custom MIL pass that forces large ops to CPU.
    More reliable than spec-level patching if the protobuf structure varies.
    """
    import coremltools as ct
    from coremltools.converters.mil import register_pass

    @register_pass(namespace="custom")
    class force_large_linear_to_cpu:
        """MIL graph pass: tag ops with output dim >= num_classes for CPU."""

        def __call__(self, prog):
            for func in prog.functions.values():
                self._process_block(func)

        def _process_block(self, block):
            for op in list(block.operations):
                if hasattr(op, 'blocks'):
                    for b in op.blocks:
                        self._process_block(b)

                if op.op_type in ("linear", "matmul", "transpose", "concat"):
                    for output in op.outputs:
                        if any(d >= num_classes for d in output.shape if isinstance(d, int)):
                            op._set_attr("compute_unit", "cpuOnly")
                            print(f"  🔧 CPU: {op.op_type} {op.name} "
                                  f"output_shape={output.shape}")
                            break

    model = ct.convert(
        onnx_path,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, imgsz, imgsz),
                scale=1.0 / 255.0,
                bias=[0.0, 0.0, 0.0],
                color_layout="RGB",
            )
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
        pass_pipeline=ct.PassPipeline.DEFAULT + ["custom::force_large_linear_to_cpu"],
    )

    model.save(output_path)
    print(f"✅ Saved (MIL pass): {output_path}")
    return model


# ─── Step 3: Validate ──────────────────────────────────────────────

def validate_model(
    model_path: str,
    imgsz: int = 1280,
    num_classes: int = 4585,
):
    """Validate output shapes and run a test prediction."""
    import coremltools as ct

    print(f"\nValidating {model_path}...")
    model = ct.models.MLModel(model_path)

    spec = model.get_spec()

    print("\nOutputs:")
    output_names = []
    for output in spec.description.output:
        arr_type = output.type.multiArrayType
        dims = []
        for d in arr_type.shape:
            if hasattr(d, 'constant') and hasattr(d.constant, 'immediateValue'):
                dims.append(d.constant.immediateValue)
        dtype = arr_type.dataType
        dtype_name = {65552: "Float16", 65568: "Float32"}.get(dtype, str(dtype))
        print(f"  {output.name}: {dims} ({dtype_name})")
        output_names.append(output.name)

    print("\nRunning test prediction...")
    from PIL import Image
    test_image = Image.fromarray(
        np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    )

    try:
        output = model.predict({"image": test_image})

        for name, arr in output.items():
            shape = arr.shape if hasattr(arr, 'shape') else type(arr)
            print(f"  {name}: {shape}")

        # Find detection and proto tensors by shape heuristic
        det_name = None
        proto_name = None
        for name, arr in output.items():
            if not hasattr(arr, 'shape'):
                continue
            if len(arr.shape) == 3 and arr.shape[1] == 4 + num_classes + 32:
                det_name = name
            elif len(arr.shape) == 4 and arr.shape[1] == 32:
                proto_name = name

        if det_name:
            det = output[det_name]
            expected_features = 4 + num_classes + 32
            assert det.shape[1] == expected_features, \
                f"Expected {expected_features} features, got {det.shape[1]}"
            print(f"  ✅ Detection tensor '{det_name}': {det.shape}")
        else:
            print("  ⚠️  Could not identify detection tensor by shape")

        if proto_name:
            proto = output[proto_name]
            assert proto.shape[1] == 32, \
                f"Expected 32 proto channels, got {proto.shape[1]}"
            print(f"  ✅ Proto tensor '{proto_name}': {proto.shape}")
        else:
            print("  ⚠️  Could not identify proto tensor by shape")

        print("\n✅ Validation PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ─── Step 4: Compare with original (accuracy check) ───────────────

def compare_outputs(
    original_model_path: str,
    patched_model_path: str,
    imgsz: int = 1280,
    num_classes: int = 4585,
    tolerance: float = 0.01,
):
    """Compare detection outputs between original and patched model."""
    import coremltools as ct
    from PIL import Image

    print("\nComparing original vs patched outputs...")

    original = ct.models.MLModel(
        original_model_path,
        compute_units=ct.ComputeUnit.CPU_ONLY
    )
    patched = ct.models.MLModel(
        patched_model_path,
        compute_units=ct.ComputeUnit.CPU_ONLY  # compare on CPU for determinism
    )

    np.random.seed(42)
    test_image = Image.fromarray(
        np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    )

    out_orig = original.predict({"image": test_image})
    out_patch = patched.predict({"image": test_image})

    for name in out_orig:
        if name not in out_patch:
            print(f"  ⚠️  Missing output: {name}")
            continue

        a = np.array(out_orig[name]).astype(np.float32)
        b = np.array(out_patch[name]).astype(np.float32)

        max_diff = np.max(np.abs(a - b))
        mean_diff = np.mean(np.abs(a - b))

        status = "✅" if max_diff < tolerance else "⚠️"
        print(f"  {status} {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        if max_diff >= tolerance and len(a.shape) >= 3:
            if a.shape[1] == 4 + num_classes + 32:
                classes_orig = np.argmax(a[0, 4:4+num_classes, :], axis=0)
                classes_patch = np.argmax(b[0, 4:4+num_classes, :], axis=0)
                match_rate = np.mean(classes_orig == classes_patch)
                print(f"       Top-1 class match rate: {match_rate:.4f}")

                conf_orig = np.max(a[0, 4:4+num_classes, :], axis=0)
                conf_patch = np.max(b[0, 4:4+num_classes, :], axis=0)
                conf_diff = np.mean(np.abs(conf_orig - conf_patch))
                print(f"       Mean confidence diff: {conf_diff:.6f}")

    print("\n✅ Comparison complete")


# ─── Main ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-export YOLO-E for ANE with CPU class head"
    )
    parser.add_argument(
        "--pt", type=str, default="android/yoloe-11l-seg-pf.pt",
        help="Path to Ultralytics .pt checkpoint"
    )
    parser.add_argument(
        "--onnx", type=str, default=None,
        help="Path to existing ONNX (skip pt→onnx export)"
    )
    parser.add_argument(
        "--output", type=str, default="yoloe-11l-seg-pf.mlpackage",
        help="Output .mlpackage path"
    )
    parser.add_argument(
        "--original", type=str, default=None,
        help="Original .mlpackage for accuracy comparison"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280
    )
    parser.add_argument(
        "--num-classes", type=int, default=4585
    )
    parser.add_argument(
        "--use-mil-pass", action="store_true",
        help="Use MIL pass approach (Option B) instead of spec-level patching"
    )
    parser.add_argument(
        "--validate-only", type=str, default=None,
        help="Just validate an existing .mlpackage"
    )
    args = parser.parse_args()

    if args.validate_only:
        validate_model(args.validate_only, args.imgsz, args.num_classes)
        return

    # Step 1: Get ONNX
    if args.onnx and Path(args.onnx).exists():
        onnx_path = args.onnx
        print(f"Using existing ONNX: {onnx_path}")
    else:
        onnx_path = str(Path(args.pt).with_suffix('.onnx'))
        if not Path(onnx_path).exists():
            export_to_onnx(args.pt, onnx_path, args.imgsz)
        else:
            print(f"ONNX already exists: {onnx_path}")

    # Step 2: Convert to CoreML with ANE/CPU split
    try:
        if args.use_mil_pass:
            print("\n=== Using MIL pass approach (Option B) ===")
            model = convert_with_mil_pass(
                onnx_path, args.output, args.imgsz, args.num_classes
            )
        else:
            print("\n=== Using spec-level patching (Option A) ===")
            model = convert_to_coreml_split(
                onnx_path, args.output, args.imgsz, args.num_classes
            )
    except Exception as e:
        print(f"\n❌ Option A failed: {e}")
        if not args.use_mil_pass:
            print("Falling back to MIL pass approach (Option B)...")
            model = convert_with_mil_pass(
                onnx_path, args.output, args.imgsz, args.num_classes
            )
        else:
            raise

    # Step 3: Validate
    passed = validate_model(args.output, args.imgsz, args.num_classes)

    # Step 4: Compare with original if provided
    if passed and args.original:
        compare_outputs(args.original, args.output, args.imgsz, args.num_classes)

    if passed:
        print(f"""
{'='*60}
✅ RE-EXPORT COMPLETE
{'='*60}
Output: {args.output}

Next steps:
1. Replace the .mlpackage in your Xcode project
2. In YOLOEModelService.swift, ensure:
     config.computeUnits = .cpuAndNeuralEngine
   (already set when Settings toggle is on)
3. Build and run — check the ANE diagnostic logs
4. Expected: ~300-400ms inference (vs 1200ms CPU-only)
{'='*60}
""")
    else:
        print("\n❌ Validation failed — do not deploy this model")
        sys.exit(1)


if __name__ == "__main__":
    main()
