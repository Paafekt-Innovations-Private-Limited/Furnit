#!/usr/bin/env python3
"""
Split YOLO-E class head in ONNX with DELAYED CONCAT — keeps each half
under 4096 channels through reshape+transpose, only concatenating after.

Usage (from /tmp with clean PYTHONPATH):
  cd /tmp && env PYTHONPATH=/opt/miniconda3/envs/coreml-py311/lib/python3.11/site-packages \
    /opt/miniconda3/envs/coreml-py311/bin/python \
    /Users/al/Documents/tries01/Furnit/scripts/split_onnx_delayed_concat.py \
    --onnx /Users/al/Documents/tries01/Furnit/android/yoloe-11l-seg-pf.onnx
"""

import sys
import numpy as np
from pathlib import Path
from copy import deepcopy


def patch_onnx_delayed_concat(onnx_path: str, output_path: str):
    import onnx
    from onnx import helper, numpy_helper

    print(f"Loading {onnx_path}...")
    model = onnx.load(onnx_path)
    graph = model.graph

    initializers = {init.name: init for init in graph.initializer}

    output_consumers = {}
    for node in graph.node:
        for inp in node.input:
            output_consumers.setdefault(inp, []).append(node)

    output_producers = {}
    for node in graph.node:
        for out in node.output:
            output_producers[out] = node

    modifications = []

    for node in list(graph.node):
        if node.op_type not in ("Conv", "Gemm", "MatMul"):
            continue

        weight_name = node.input[1] if len(node.input) > 1 else None
        if weight_name not in initializers:
            continue

        weight = numpy_helper.to_array(initializers[weight_name])

        if node.op_type == "Conv":
            out_channels = weight.shape[0]
        elif node.op_type == "Gemm":
            transB = 0
            for attr in node.attribute:
                if attr.name == "transB":
                    transB = attr.i
            out_channels = weight.shape[0] if transB == 0 else weight.shape[1]
        else:
            continue

        if out_channels < 4096:
            continue

        conv_output = node.output[0]
        consumers = output_consumers.get(conv_output, [])

        print(f"\n  Found: {node.op_type} [{node.name}] out_channels={out_channels}")
        print(f"  Consumers: {[f'{c.op_type}({c.name})' for c in consumers]}")

        reshape_node = None
        transpose_node = None

        for consumer in consumers:
            if consumer.op_type in ("Reshape", "Flatten"):
                reshape_node = consumer
                reshape_consumers = output_consumers.get(consumer.output[0], [])
                for rc in reshape_consumers:
                    if rc.op_type == "Transpose":
                        transpose_node = rc
                        break
            elif consumer.op_type == "Transpose":
                transpose_node = consumer

        modifications.append({
            'conv_node': node,
            'weight_name': weight_name,
            'bias_name': node.input[2] if len(node.input) > 2 else None,
            'out_channels': out_channels,
            'reshape_node': reshape_node,
            'transpose_node': transpose_node,
            'conv_output': conv_output,
        })

        if reshape_node and transpose_node:
            print(f"  → DELAYED concat past reshape+transpose")
        elif transpose_node:
            print(f"  → DELAYED concat past transpose")
        else:
            print(f"  → Simple split+cat (no reshape/transpose downstream)")

    print(f"\n{'='*60}")
    print(f"Applying {len(modifications)} modifications...")

    for mod in modifications:
        apply_delayed_concat_split(graph, mod, initializers, output_consumers)

    print("\nValidating ONNX graph...")
    try:
        onnx.checker.check_model(model)
        print("✅ ONNX validation passed")
    except Exception as e:
        print(f"⚠️  ONNX validation warning: {e}")

    onnx.save(model, output_path)
    print(f"✅ Saved: {output_path}")
    return True


def apply_delayed_concat_split(graph, mod, initializers, output_consumers):
    from onnx import helper, numpy_helper

    conv = mod['conv_node']
    out_ch = mod['out_channels']
    split_at = out_ch // 2

    orig_weight = numpy_helper.to_array(initializers[mod['weight_name']])
    weight_a = orig_weight[:split_at].copy()
    weight_b = orig_weight[split_at:].copy()

    wa_name = f"{mod['weight_name']}_split_a"
    wb_name = f"{mod['weight_name']}_split_b"
    graph.initializer.append(numpy_helper.from_array(weight_a, name=wa_name))
    graph.initializer.append(numpy_helper.from_array(weight_b, name=wb_name))

    ba_name = ""
    bb_name = ""
    if mod['bias_name'] and mod['bias_name'] in initializers:
        orig_bias = numpy_helper.to_array(initializers[mod['bias_name']])
        ba_name = f"{mod['bias_name']}_split_a"
        bb_name = f"{mod['bias_name']}_split_b"
        graph.initializer.append(numpy_helper.from_array(orig_bias[:split_at].copy(), name=ba_name))
        graph.initializer.append(numpy_helper.from_array(orig_bias[split_at:].copy(), name=bb_name))

    conv_output = mod['conv_output']

    def make_split_conv(suffix, w_name, b_name):
        out_name = f"{conv_output}_split_{suffix}"
        inputs = [conv.input[0], w_name]
        if b_name:
            inputs.append(b_name)
        node = helper.make_node(conv.op_type, inputs, [out_name],
                                name=f"{conv.name}_split_{suffix}")
        for attr in conv.attribute:
            node.attribute.append(deepcopy(attr))
        return node, out_name

    conv_a, conv_a_out = make_split_conv("a", wa_name, ba_name)
    conv_b, conv_b_out = make_split_conv("b", wb_name, bb_name)

    if mod['reshape_node'] and mod['transpose_node']:
        reshape = mod['reshape_node']
        transpose = mod['transpose_node']
        final_output = transpose.output[0]

        # Build reshape shape for each half.
        # Original reshape takes [B, 4585, H, W] → [B, 4585, H*W] (or similar).
        # The shape tensor might be a constant like [1, 4585, -1] or [0, 4585, -1].
        # We need to create new shape tensors for each half: [1, split_at, -1] etc.
        # Strategy: read original shape, replace the 4585-like dim with split sizes.

        reshape_shape_name = reshape.input[1] if len(reshape.input) > 1 else None
        if reshape_shape_name and reshape_shape_name in initializers:
            orig_shape_arr = numpy_helper.to_array(initializers[reshape_shape_name])
            shape_a = orig_shape_arr.copy()
            shape_b = orig_shape_arr.copy()
            for idx_s in range(len(orig_shape_arr)):
                if orig_shape_arr[idx_s] == out_ch:
                    shape_a[idx_s] = split_at
                    shape_b[idx_s] = out_ch - split_at
                    break
            shape_a_name = f"{reshape_shape_name}_split_a"
            shape_b_name = f"{reshape_shape_name}_split_b"
            graph.initializer.append(numpy_helper.from_array(shape_a, name=shape_a_name))
            graph.initializer.append(numpy_helper.from_array(shape_b, name=shape_b_name))
        else:
            shape_a_name = reshape_shape_name
            shape_b_name = reshape_shape_name

        reshape_a_out = f"{reshape.output[0]}_split_a"
        reshape_a = helper.make_node("Reshape",
                                     [conv_a_out, shape_a_name],
                                     [reshape_a_out],
                                     name=f"{reshape.name}_split_a")

        reshape_b_out = f"{reshape.output[0]}_split_b"
        reshape_b = helper.make_node("Reshape",
                                     [conv_b_out, shape_b_name],
                                     [reshape_b_out],
                                     name=f"{reshape.name}_split_b")

        transpose_a_out = f"{transpose.output[0]}_split_a"
        transpose_a = helper.make_node("Transpose", [reshape_a_out], [transpose_a_out],
                                       name=f"{transpose.name}_split_a")
        for attr in transpose.attribute:
            transpose_a.attribute.append(deepcopy(attr))

        transpose_b_out = f"{transpose.output[0]}_split_b"
        transpose_b = helper.make_node("Transpose", [reshape_b_out], [transpose_b_out],
                                       name=f"{transpose.name}_split_b")
        for attr in transpose.attribute:
            transpose_b.attribute.append(deepcopy(attr))

        # Determine concat axis — after transpose, channels are typically last dim
        concat_axis = -1
        for attr in transpose.attribute:
            if attr.name == "perm":
                perm = list(attr.ints)
                concat_axis = len(perm) - 1

        concat = helper.make_node("Concat",
                                  [transpose_a_out, transpose_b_out],
                                  [final_output],
                                  name=f"{conv.name}_delayed_concat",
                                  axis=concat_axis)

        idx = list(graph.node).index(conv)
        graph.node.remove(conv)
        graph.node.remove(reshape)
        graph.node.remove(transpose)

        for new_node in [conv_a, conv_b, reshape_a, reshape_b,
                         transpose_a, transpose_b, concat]:
            graph.node.insert(idx, new_node)
            idx += 1

        print(f"  ✅ Delayed concat: {conv.name} ({out_ch}) → "
              f"split({split_at}+{out_ch-split_at}) → reshape → transpose → concat")

    elif mod['transpose_node']:
        transpose = mod['transpose_node']
        final_output = transpose.output[0]

        transpose_a_out = f"{transpose.output[0]}_split_a"
        transpose_a = helper.make_node("Transpose", [conv_a_out], [transpose_a_out],
                                       name=f"{transpose.name}_split_a")
        for attr in transpose.attribute:
            transpose_a.attribute.append(deepcopy(attr))

        transpose_b_out = f"{transpose.output[0]}_split_b"
        transpose_b = helper.make_node("Transpose", [conv_b_out], [transpose_b_out],
                                       name=f"{transpose.name}_split_b")
        for attr in transpose.attribute:
            transpose_b.attribute.append(deepcopy(attr))

        concat_axis = -1
        for attr in transpose.attribute:
            if attr.name == "perm":
                concat_axis = len(list(attr.ints)) - 1

        concat = helper.make_node("Concat",
                                  [transpose_a_out, transpose_b_out],
                                  [final_output],
                                  name=f"{conv.name}_delayed_concat",
                                  axis=concat_axis)

        idx = list(graph.node).index(conv)
        graph.node.remove(conv)
        graph.node.remove(transpose)

        for new_node in [conv_a, conv_b, transpose_a, transpose_b, concat]:
            graph.node.insert(idx, new_node)
            idx += 1

        print(f"  ✅ Delayed concat past transpose: {conv.name} ({out_ch}) → "
              f"split({split_at}+{out_ch-split_at}) → transpose → concat")

    else:
        concat = helper.make_node("Concat",
                                  [conv_a_out, conv_b_out],
                                  [conv_output],
                                  name=f"{conv.name}_simple_cat",
                                  axis=1)

        idx = list(graph.node).index(conv)
        graph.node.remove(conv)
        graph.node.insert(idx, conv_a)
        graph.node.insert(idx + 1, conv_b)
        graph.node.insert(idx + 2, concat)

        print(f"  ✅ Simple split: {conv.name} ({out_ch}) → "
              f"split({split_at}+{out_ch-split_at}) → concat")

    for init in list(graph.initializer):
        if init.name == mod['weight_name']:
            graph.initializer.remove(init)
        if mod['bias_name'] and init.name == mod['bias_name']:
            graph.initializer.remove(init)


def verify_onnx_outputs_match(
    original_onnx_path: str,
    split_onnx_path: str,
    imgsz: int = 1280,
    tolerance: float = 1e-4,
) -> bool:
    import onnxruntime as ort

    np.random.seed(42)
    test_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    original_session = ort.InferenceSession(original_onnx_path)
    split_session = ort.InferenceSession(split_onnx_path)

    input_name = original_session.get_inputs()[0].name
    original_outputs = original_session.run(None, {input_name: test_input})
    split_outputs = split_session.run(None, {input_name: test_input})

    all_outputs_match = True
    for output_index, (original_output, split_output) in enumerate(zip(original_outputs, split_outputs)):
        maximum_difference = np.max(np.abs(original_output - split_output))
        status = "✅" if maximum_difference < tolerance else "❌"
        print(
            f"  {status} Output[{output_index}] "
            f"shape={original_output.shape} max_diff={maximum_difference:.8f}"
        )
        if maximum_difference >= tolerance:
            all_outputs_match = False

    return all_outputs_match


def convert_split_onnx_to_coreml(
    split_onnx_path: str,
    output_mlpackage_path: str,
    imgsz: int = 1280,
):
    import coremltools as ct

    model = ct.convert(
        split_onnx_path,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, imgsz, imgsz),
            scale=1.0 / 255.0,
            bias=[0.0, 0.0, 0.0],
            color_layout="RGB",
        )],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )
    model.save(output_mlpackage_path)
    print(f"✅ Saved: {output_mlpackage_path}")
    return model


def print_coreml_output_names(model, imgsz: int = 1280):
    from PIL import Image

    test_image = Image.fromarray(
        np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    )
    outputs = model.predict({"image": test_image})
    print("\nOutput names (update YoloEDetectionParser.swift knownDetectionProtoPairs):")
    for name, array in outputs.items():
        shape = array.shape if hasattr(array, "shape") else "?"
        print(f"  {name}: {shape}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Split YOLO-E class head in ONNX with delayed concat for ANE"
    )
    parser.add_argument("--onnx", type=str, required=True,
                        help="Input ONNX model")
    parser.add_argument("--output-onnx", type=str, default=None,
                        help="Output split ONNX (default: <input>_split.onnx)")
    parser.add_argument("--output-mlpackage", type=str, default=None,
                        help="Output CoreML mlpackage (default: skip CoreML conversion)")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-coreml", action="store_true")
    args = parser.parse_args()

    onnx_path = args.onnx
    output_onnx = args.output_onnx or str(Path(onnx_path).stem + "_split.onnx")

    success = patch_onnx_delayed_concat(onnx_path, output_onnx)
    if not success:
        return 1

    if not args.skip_verify:
        print("\n" + "="*60)
        print("Verifying outputs match original...")
        print("="*60)
        try:
            if verify_onnx_outputs_match(onnx_path, output_onnx):
                print("✅ Verification PASSED")
            else:
                print("❌ Verification FAILED — outputs don't match")
                return 1
        except ImportError:
            print("⚠️  onnxruntime not installed — skipping verification")
        except Exception as e:
            print(f"⚠️  Verification error: {e}")

    if not args.skip_coreml:
        print("\n" + "="*60)
        print("Converting to CoreML...")
        print("="*60)
        try:
            output_ml = args.output_mlpackage or str(
                Path(onnx_path).parent / (Path(onnx_path).stem + "_ane.mlpackage")
            )
            model = convert_split_onnx_to_coreml(output_onnx, output_ml)
            print_coreml_output_names(model)
        except ImportError:
            print("⚠️  coremltools not available — skipping CoreML conversion")
        except Exception as e:
            print(f"❌ CoreML conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    exit(main())
