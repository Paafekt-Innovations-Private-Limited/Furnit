#!/usr/bin/env python3
"""
Split SHARP Part 1 ONNX model into 2 sub-chunks using ONNX graph surgery.

Sub-chunks:
  Part 1a: Preprocessing + patch encoder blocks 0-11 (~974 nodes)
           Input: image [1,3,1536,1536]
           Outputs: Resize_1, blocks.5/Add_1, blocks.11/Add_1, + 98 weight pass-throughs

  Part 1b: Patch encoder blocks 12-18 + MLP (~365 nodes)
           Input: blocks.11/Add_1_output_0 [35,577,1024]
           Outputs: blocks.18/Add, blocks.18/mlp/act/Mul

This reduces peak memory: Part 1a session (~485 MB) is destroyed before Part 1b (~461 MB) loads,
vs loading all 946 MB at once.

Usage:
  cd android
  python export_onnx_part1_chunks.py [--input sharp_part1.onnx] [--output-dir .]

Then push to device:
  adb push sharp_part1a.onnx      /storage/emulated/0/Android/data/com.furnit.android/files/models/
  adb push sharp_part1a.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
  adb push sharp_part1b.onnx      /storage/emulated/0/Android/data/com.furnit.android/files/models/
  adb push sharp_part1b.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
"""

import argparse
import os
import sys
import time
from collections import OrderedDict

import onnx
from onnx import TensorProto, helper

BOUNDARY_TENSOR = "/predictor/monodepth_model/encoder/patch_encoder/blocks.11/Add_1_output_0"

PART1A_ACTIVATION_OUTPUTS = [
    "/predictor/monodepth_model/encoder/Resize_1_output_0",
    "/predictor/monodepth_model/encoder/patch_encoder/blocks.5/Add_1_output_0",
    "/predictor/monodepth_model/encoder/patch_encoder/blocks.11/Add_1_output_0",
]

PART1B_ACTIVATION_OUTPUTS = [
    "/predictor/monodepth_model/encoder/patch_encoder/blocks.18/Add_output_0",
    "/predictor/monodepth_model/encoder/patch_encoder/blocks.18/mlp/act/Mul_output_0",
]


def partition_nodes(graph, boundary_tensor):
    """Split graph nodes: part_a produces boundary_tensor and everything before; part_b is after."""
    part_a_names = set()
    part_b_names = set()
    found = False
    for node in graph.node:
        if boundary_tensor in node.output:
            part_a_names.add(node.name)
            found = True
            continue
        if not found:
            part_a_names.add(node.name)
        else:
            part_b_names.add(node.name)
    return part_a_names, part_b_names


def collect_needed_tensors(graph, node_names):
    """Tensor names consumed by node_names but not produced internally."""
    internal_outputs = set()
    for node in graph.node:
        if node.name in node_names:
            for o in node.output:
                internal_outputs.add(o)

    needed = OrderedDict()
    for node in graph.node:
        if node.name in node_names:
            for inp in node.input:
                if inp and inp not in internal_outputs and inp not in needed:
                    needed[inp] = True
    return list(needed.keys())


def find_initializer(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return init
    return None


def find_graph_input(graph, name):
    for inp in graph.input:
        if inp.name == name:
            return inp
    return None


def build_submodel(original_model, node_names, new_input_specs, new_output_specs,
                   model_doc_string=""):
    graph = original_model.graph
    sub_nodes = [n for n in graph.node if n.name in node_names]
    needed_tensor_names = collect_needed_tensors(graph, node_names)

    sub_initializers = []
    for name in needed_tensor_names:
        init = find_initializer(graph, name)
        if init is not None:
            sub_initializers.append(init)

    initializer_names = {init.name for init in sub_initializers}
    sub_inputs = []
    for name, elem_type, shape in new_input_specs:
        if shape is not None:
            sub_inputs.append(helper.make_tensor_value_info(name, elem_type, shape))
        else:
            sub_inputs.append(helper.make_tensor_value_info(name, elem_type, None))

    for name in needed_tensor_names:
        if name not in initializer_names and not any(i.name == name for i in sub_inputs):
            orig_inp = find_graph_input(graph, name)
            if orig_inp is not None:
                sub_inputs.append(orig_inp)

    sub_outputs = []
    for name, elem_type, shape in new_output_specs:
        if shape is not None:
            sub_outputs.append(helper.make_tensor_value_info(name, elem_type, shape))
        else:
            sub_outputs.append(helper.make_tensor_value_info(name, elem_type, None))

    sub_graph = helper.make_graph(
        sub_nodes, "sub_graph", sub_inputs, sub_outputs, initializer=sub_initializers,
    )
    sub_model = helper.make_model(sub_graph, opset_imports=original_model.opset_import)
    sub_model.ir_version = original_model.ir_version
    sub_model.doc_string = model_doc_string
    return sub_model


def split_part1(input_path, output_dir):
    print(f"Loading {input_path} (with external data)...")
    t0 = time.time()
    model = onnx.load(input_path)
    print(f"  Loaded in {time.time() - t0:.1f}s ({len(model.graph.node)} nodes)")

    graph = model.graph
    original_outputs = [out.name for out in graph.output]
    print(f"  Total outputs: {len(original_outputs)}")

    weight_outputs = [o for o in original_outputs if "/" not in o]
    activation_outputs = [o for o in original_outputs if "/" in o]
    print(f"  Weight pass-throughs: {len(weight_outputs)}")
    print(f"  Activation outputs: {len(activation_outputs)}")
    for name in activation_outputs:
        print(f"    {name}")

    part_a_names, part_b_names = partition_nodes(graph, BOUNDARY_TENSOR)
    print(f"\n  Part 1a nodes: {len(part_a_names)}, Part 1b nodes: {len(part_b_names)}")

    part_b_needed = collect_needed_tensors(graph, part_b_names)
    part_b_ext_inputs = []
    for name in part_b_needed:
        if name == BOUNDARY_TENSOR:
            continue
        if find_initializer(graph, name) is None:
            part_b_ext_inputs.append(name)
    print(f"  Part 1b external inputs (besides boundary): {part_b_ext_inputs}")

    os.makedirs(output_dir, exist_ok=True)

    # --- Part 1a: Preprocessing + blocks 0-11 ---
    print(f"\nBuilding Part 1a (preprocess + blocks 0-11)...")
    t0 = time.time()

    part1a_input_specs = []
    for inp in graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value > 0 else None)
        part1a_input_specs.append((inp.name, inp.type.tensor_type.elem_type, shape if shape else None))

    part1a_output_specs = []
    for out in graph.output:
        if out.name in weight_outputs or out.name in PART1A_ACTIVATION_OUTPUTS:
            shape = []
            for d in out.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.dim_value > 0 else None)
            part1a_output_specs.append((out.name, out.type.tensor_type.elem_type, shape if shape else None))

    print(f"  Part 1a outputs: {len(part1a_output_specs)} ({len(weight_outputs)} weights + {len(PART1A_ACTIVATION_OUTPUTS)} activations)")

    part1a_model = build_submodel(
        model, part_a_names,
        new_input_specs=part1a_input_specs,
        new_output_specs=part1a_output_specs,
        model_doc_string="SHARP Part 1a: Preprocessing + patch encoder blocks 0-11",
    )

    part1a_path = os.path.join(output_dir, "sharp_part1a.onnx")
    onnx.save(part1a_model, part1a_path,
              save_as_external_data=True, all_tensors_to_one_file=True,
              location="sharp_part1a.onnx.data")
    elapsed_a = time.time() - t0

    part1a_size = os.path.getsize(part1a_path) / 1e6
    part1a_data_path = part1a_path + ".data"
    part1a_data_size = os.path.getsize(part1a_data_path) / 1e6 if os.path.exists(part1a_data_path) else 0
    print(f"  Part 1a: {part1a_size:.1f} MB graph + {part1a_data_size:.1f} MB weights ({elapsed_a:.1f}s)")
    print(f"  Part 1a nodes: {len(part1a_model.graph.node)}, inputs: {len(part1a_model.graph.input)}, outputs: {len(part1a_model.graph.output)}")

    # --- Part 1b: Blocks 12-18 ---
    print(f"\nBuilding Part 1b (blocks 12-18)...")
    t0 = time.time()

    part1b_input_specs = [(BOUNDARY_TENSOR, TensorProto.FLOAT, [35, 577, 1024])]
    for name in part_b_ext_inputs:
        orig_inp = find_graph_input(graph, name)
        if orig_inp is not None:
            shape = []
            for d in orig_inp.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.dim_value > 0 else None)
            part1b_input_specs.append((name, orig_inp.type.tensor_type.elem_type, shape if shape else None))

    part1b_output_specs = []
    for name in PART1B_ACTIVATION_OUTPUTS:
        for out in graph.output:
            if out.name == name:
                shape = []
                for d in out.type.tensor_type.shape.dim:
                    shape.append(d.dim_value if d.dim_value > 0 else None)
                part1b_output_specs.append((name, out.type.tensor_type.elem_type, shape if shape else None))

    part1b_model = build_submodel(
        model, part_b_names,
        new_input_specs=part1b_input_specs,
        new_output_specs=part1b_output_specs,
        model_doc_string="SHARP Part 1b: Patch encoder blocks 12-18",
    )

    part1b_path = os.path.join(output_dir, "sharp_part1b.onnx")
    onnx.save(part1b_model, part1b_path,
              save_as_external_data=True, all_tensors_to_one_file=True,
              location="sharp_part1b.onnx.data")
    elapsed_b = time.time() - t0

    part1b_size = os.path.getsize(part1b_path) / 1e6
    part1b_data_path = part1b_path + ".data"
    part1b_data_size = os.path.getsize(part1b_data_path) / 1e6 if os.path.exists(part1b_data_path) else 0
    print(f"  Part 1b: {part1b_size:.1f} MB graph + {part1b_data_size:.1f} MB weights ({elapsed_b:.1f}s)")
    print(f"  Part 1b nodes: {len(part1b_model.graph.node)}, inputs: {len(part1b_model.graph.input)}, outputs: {len(part1b_model.graph.output)}")

    total_mb = part1a_size + part1a_data_size + part1b_size + part1b_data_size
    print(f"\n{'='*60}")
    print("Split complete")
    print(f"{'='*60}")
    print(f"  Part 1a (preprocess+blocks 0-11): {part1a_size + part1a_data_size:.0f} MB")
    print(f"  Part 1b (blocks 12-18):           {part1b_size + part1b_data_size:.0f} MB")
    print(f"  Total:                            {total_mb:.0f} MB")
    print(f"\nPush to device:")
    dest = "/storage/emulated/0/Android/data/com.furnit.android/files/models"
    for f in sorted(os.listdir(output_dir)):
        if (f.startswith("sharp_part1a") or f.startswith("sharp_part1b")) and \
           (f.endswith(".onnx") or f.endswith(".onnx.data")):
            print(f"  adb push {os.path.join(output_dir, f)} {dest}/")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Split SHARP Part 1 ONNX into blocks 0-11 + blocks 12-18")
    parser.add_argument("--input", default="sharp_part1.onnx",
        help="Input Part 1 ONNX model (default: sharp_part1.onnx)")
    parser.add_argument("--output-dir", default=".",
        help="Output directory (default: current directory)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        print("Pull from device first:")
        print("  adb pull /storage/emulated/0/Android/data/com.furnit.android/files/models/sharp_part1.onnx .")
        print("  adb pull .../sharp_part1.onnx.data .")
        return 1

    return split_part1(args.input, args.output_dir)


if __name__ == "__main__":
    sys.exit(main() or 0)
