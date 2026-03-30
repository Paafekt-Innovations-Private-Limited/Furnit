#!/usr/bin/env python3
"""
Convert YOLOE ONNX → Core ML neuralNetwork .mlmodel using coremltools 5.2 ONNX frontend.

Why this exists: coremltools 6+ removed ``ct.converters.onnx``. Newer stacks need a dedicated
environment (and NumPy 1.23.x) for the legacy converter. This model also needs small fixes:

  * Strip ``dilations`` from MaxPool nodes when they are ``[1,1]`` (legacy converter bug).
  * Patch ONNX op handlers: Resize scales on input slot 2 (empty ROI slot 1); Unsqueeze axes
    from opset-13+ second input; constant-shape Reshape uses ``reshape_static`` (avoids
    ``rank_preserving_reshape`` quirks for 4D→4D).

Run with **PYTHONPATH unset** so a repo ``executorch/`` tree does not shadow site-packages::

  env -u PYTHONPATH \\
    /path/to/venv/bin/python scripts/yoloe_onnx_to_coreml_ct5.py \\
    --onnx android/yoloe-11l-seg-pf.onnx \\
    --out .build/yoloe-from-onnx.mlmodel

Suggested venv pins::

  pip install 'coremltools==5.2.0' 'numpy>=1.21,<1.24' onnx onnxruntime

**Core ML NeuralNetwork runtime:** On current macOS + coremltools 5.2, ``MLModel.predict`` may
still fail with error **-1** for this architecture: the legacy converter emits
``batchedMatMul`` outputs that do not combine cleanly with following ``reshape_static`` /
``flatten`` in the attention blocks (verified by slicing the graph: MatMul-only outputs predict
OK; MatMul→Reshape subgraphs fail). The exported ``.mlmodel`` is valid protobuf and
``coremlcompiler compile`` succeeds; fixing execution requires a different graph (e.g. re-export
with an op mix the ML program converter accepts) or running ONNX on-device via **ONNX Runtime**
instead of Core ML for this checkpoint.

Use ``--verify-coreml`` to attempt a local ``predict`` (non-zero exit if it fails).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _strip_maxpool_dilations(onnx_in: Path, onnx_out: Path) -> int:
    """Remove MaxPool ``dilations`` attributes when present and equal to [1, 1]. Returns edit count."""
    import onnx
    from onnx import helper

    model = onnx.load(str(onnx_in))
    edited = 0
    for node in model.graph.node:
        if node.op_type != "MaxPool":
            continue
        by_name = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if "dilations" not in by_name:
            continue
        dil = list(by_name["dilations"])
        if dil not in ([1, 1], [1, 1, 1, 1]):
            continue
        new_attrs = [a for a in node.attribute if a.name != "dilations"]
        del node.attribute[:]
        node.attribute.extend(new_attrs)
        edited += 1
    onnx.save(model, str(onnx_out))
    return edited


def _apply_ct52_onnx_patches() -> None:
    import numpy as np
    import coremltools.converters.onnx._operators_nd as ops_nd

    _original_convert_reshape = ops_nd._ONNX_NODE_REGISTRY_ND["Reshape"]

    def _convert_reshape_static_for_constant_shape(builder, node, graph, err):
        shape_node = node.inputs[1]
        if shape_node in node.input_tensors:
            output_shape = np.asarray(node.input_tensors[shape_node]).astype(np.int64).flatten()
            builder.add_reshape_static(
                name=node.name,
                input_name=node.inputs[0],
                output_name=node.outputs[0],
                output_shape=[int(x) for x in output_shape],
            )
            return
        return _original_convert_reshape(builder, node, graph, err)

    def _convert_resize_scales_any_slot(builder, node, graph, err):
        mode = node.attrs.get("mode", "nearest")
        scale_tensor_name = None
        for candidate in node.inputs[1:]:
            if not candidate:
                continue
            if candidate in node.input_tensors:
                scale_tensor_name = candidate
                break
        if scale_tensor_name is None:
            return err.unsupported_op_configuration(
                builder,
                node,
                graph,
                "Scaling factor unknown!! CoreML does not support dynamic scaling for Resize",
            )
        mode = "NN" if mode == "nearest" else "BILINEAR"
        scale = node.input_tensors[scale_tensor_name]
        if scale.size == 0:
            input_shape = graph.shape_dict[node.inputs[0]]
            output_shape = graph.shape_dict[node.outputs[0]]
            scale = (output_shape[2] // input_shape[2], output_shape[3] // input_shape[3])
        builder.add_upsample(
            name=node.name,
            scaling_factor_h=float(scale[-2]),
            scaling_factor_w=float(scale[-1]),
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            mode=mode,
        )

    def _convert_unsqueeze_axes_input(builder, node, graph, err):
        axes = node.attrs.get("axes")
        if axes is None and len(node.inputs) > 1:
            n2 = node.inputs[1]
            if n2 in node.input_tensors:
                t = np.asarray(node.input_tensors[n2]).flatten()
                axes = [int(x) for x in t]
        if axes is None:
            return err.unsupported_op_configuration(
                builder,
                node,
                graph,
                "Unsqueeze axes missing (opset 13+ needs input axes)",
            )
        builder.add_expand_dims(
            name=node.name,
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            axes=axes,
        )

    ops_nd._ONNX_NODE_REGISTRY_ND["Reshape"] = _convert_reshape_static_for_constant_shape
    ops_nd._ONNX_NODE_REGISTRY_ND["Resize"] = _convert_resize_scales_any_slot
    ops_nd._ONNX_NODE_REGISTRY_ND["Unsqueeze"] = _convert_unsqueeze_axes_input


def _spec_only_mlmodel_class():
    class SpecOnlyMLModel:
        def __init__(self, spec):
            self._spec = spec

        @property
        def user_defined_metadata(self):
            return self._spec.description.metadata.userDefined

        def save(self, path: str) -> None:
            from coremltools.models.utils import save_spec

            save_spec(self._spec, str(path))

    return SpecOnlyMLModel


def _verify_coreml_predict(mlmodel_path: Path, python_exe: Path | None) -> None:
    """Run ``MLModel.predict`` in another interpreter (coremltools 5.2 conda env often lacks libcoremlpython)."""
    exe = python_exe
    if exe is None:
        env_p = os.environ.get("FURNIT_COREML_PREDICT_PYTHON")
        if env_p:
            exe = Path(env_p)
        else:
            which = shutil.which("python3")
            exe = Path(which) if which else None
    if exe is None or not exe.is_file():
        print(
            "Core ML predict smoke test: set --coreml-predict-python or FURNIT_COREML_PREDICT_PYTHON",
            file=sys.stderr,
        )
        raise SystemExit(1)

    snippet = (
        "import os, numpy as np, coremltools as ct\n"
        "p = os.environ['FURNIT_MLMODEL_PATH']\n"
        "m = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_ONLY)\n"
        "x = np.ascontiguousarray(np.random.randn(1, 3, 1280, 1280).astype(np.float32))\n"
        "out = m.predict({'images': x})\n"
        "for k, v in out.items():\n"
        "    a = np.asarray(v)\n"
        "    print(k, tuple(a.shape), float(a.mean()))\n"
        "print('COREML_PREDICT_OK')\n"
    )
    env = os.environ.copy()
    env["FURNIT_MLMODEL_PATH"] = str(mlmodel_path.resolve())
    print(f"Core ML predict smoke test via {exe} …")
    proc = subprocess.run(
        [str(exe), "-c", snippet],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0 or "COREML_PREDICT_OK" not in proc.stdout:
        print(
            "Core ML predict smoke test failed (known issue for batchedMatMul→reshape in this model).",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _verify_onnxruntime(onnx_path: Path) -> None:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    x = np.random.randn(1, 3, 1280, 1280).astype(np.float32)
    outputs = session.run(None, {input_meta.name: x})
    print(
        "ORT OK:",
        input_meta.name,
        x.shape,
        "→",
        [(i, o.shape, float(o.mean()), float(o.min()), float(o.max())) for i, o in enumerate(outputs)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOE ONNX → Core ML via coremltools 5.2")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "android" / "yoloe-11l-seg-pf.onnx",
        help="Source ONNX model",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path (must end with .mlmodel); .mlpackage needs libmodelpackage in this env",
    )
    parser.add_argument(
        "--prepared-onnx",
        type=Path,
        default=None,
        help="Write MaxPool-fixed ONNX here (default: alongside --out with -nodil.onnx suffix)",
    )
    parser.add_argument(
        "--skip-maxpool-fix",
        action="store_true",
        help="Do not strip MaxPool dilations (use if you already have a prepared ONNX)",
    )
    parser.add_argument(
        "--verify-ort",
        action="store_true",
        help="After conversion, run a CPU ONNX Runtime inference smoke test on prepared ONNX",
    )
    parser.add_argument(
        "--verify-coreml",
        action="store_true",
        help="After conversion, run MLModel.predict (needs --coreml-predict-python or env / PATH)",
    )
    parser.add_argument(
        "--coreml-predict-python",
        type=Path,
        default=None,
        help="Python executable with working Core ML bindings (e.g. Miniconda python with coremltools 7+)",
    )
    parser.add_argument(
        "--ios-target",
        default="13",
        help='minimum_ios_deployment_target string for ct.converters.onnx.convert (default: "13")',
    )
    args = parser.parse_args()

    if not args.onnx.is_file():
        print(f"Missing ONNX: {args.onnx}", file=sys.stderr)
        return 1

    out = args.out
    if out.suffix.lower() != ".mlmodel":
        print("Use --out ending in .mlmodel (conda coremltools often cannot save .mlpackage).", file=sys.stderr)
        return 1

    try:
        import coremltools as ct
    except ImportError as e:
        print("Install coremltools 5.2 in a clean venv.", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    if not ct.__version__.startswith("5.2"):
        print(
            f"Expected coremltools 5.2.x, got {ct.__version__}. "
            "Create a venv: pip install 'coremltools==5.2.0' 'numpy>=1.21,<1.24' onnx",
            file=sys.stderr,
        )
        return 1

    import numpy as np

    if tuple(int(x) for x in np.__version__.split(".")[:2]) >= (1, 24):
        print("Use NumPy 1.23.x with coremltools 5.2 (e.g. pip install 'numpy>=1.21,<1.24').", file=sys.stderr)
        return 1

    prepared = args.prepared_onnx
    if prepared is None:
        prepared = out.with_name(out.stem + "-nodil-source.onnx")

    onnx_to_convert = args.onnx
    if not args.skip_maxpool_fix:
        n = _strip_maxpool_dilations(args.onnx, prepared)
        print(f"MaxPool dilations stripped: {n} node(s); wrote {prepared}")
        onnx_to_convert = prepared
    else:
        print("Skipping MaxPool dilations strip; using --onnx as-is")

    _apply_ct52_onnx_patches()

    import coremltools.converters.onnx._converter as onnx_conv

    SpecOnlyMLModel = _spec_only_mlmodel_class()
    onnx_conv.MLModel = SpecOnlyMLModel

    print(f"Converting {onnx_to_convert} → {out} …")
    ml = ct.converters.onnx.convert(
        model=str(onnx_to_convert),
        minimum_ios_deployment_target=args.ios_target,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(out))
    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"Saved {out} ({size_mb:.1f} MiB)")

    if args.verify_ort:
        _verify_onnxruntime(onnx_to_convert)

    if args.verify_coreml:
        _verify_coreml_predict(out, args.coreml_predict_python)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
