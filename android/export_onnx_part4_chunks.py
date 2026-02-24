#!/usr/bin/env python3
"""
Split SHARP Part 4 ONNX model into 2 sub-chunks using manual ONNX graph surgery.

Sub-chunks:
  Part 4a: ViT blocks (second half of block 11 through block 23 + norm)
           Takes original 60 inputs -> produces norm output [1,577,1024].
  Part 4b: Decoder + Gaussian head.
           Takes norm output + 6 feature maps -> 5 Gaussian outputs.

This reduces peak memory: Part 4a session is destroyed before Part 4b loads.
Compatible with existing ONNX Parts 1-3 (same input tensor names).

Usage:
  cd android
  python export_onnx_part4_chunks.py [--input sharp_part4.onnx] [--output-dir .]

Then push to device:
  adb push sharp_part4a.onnx      /storage/emulated/0/Android/data/com.furnit.android/files/models/
  adb push sharp_part4a.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
  adb push sharp_part4b.onnx      /storage/emulated/0/Android/data/com.furnit.android/files/models/
  adb push sharp_part4b.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
"""

import argparse
import os
import sys
import time
from collections import OrderedDict

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np


NORM_OUTPUT = "/predictor/monodepth_model/encoder/image_encoder/norm/LayerNormalization_output_0"


def partition_nodes(graph, boundary_tensor):
    """Split graph nodes into vit_nodes (produce boundary_tensor) and decoder_nodes (consume it)."""
    produced_by = {}
    for node in graph.node:
        for o in node.output:
            produced_by[o] = node.name

    vit_names = set()
    decoder_names = set()
    found = False
    for node in graph.node:
        if boundary_tensor in node.output:
            vit_names.add(node.name)
            found = True
            continue
        if not found:
            vit_names.add(node.name)
        else:
            decoder_names.add(node.name)

    return vit_names, decoder_names


def collect_needed_tensors(graph, node_names):
    """Collect all tensor names that nodes in node_names consume but don't produce internally."""
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
    """Build a sub-model from selected nodes of the original graph.

    new_input_specs: list of (name, elem_type, shape_or_None)
    new_output_specs: list of (name, elem_type, shape_or_None)
    """
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
            inp = helper.make_tensor_value_info(name, elem_type, shape)
        else:
            inp = helper.make_tensor_value_info(name, elem_type, None)
        sub_inputs.append(inp)

    for name in needed_tensor_names:
        if name not in initializer_names and not any(i.name == name for i in sub_inputs):
            orig_inp = find_graph_input(graph, name)
            if orig_inp is not None:
                sub_inputs.append(orig_inp)

    sub_outputs = []
    for name, elem_type, shape in new_output_specs:
        if shape is not None:
            out = helper.make_tensor_value_info(name, elem_type, shape)
        else:
            out = helper.make_tensor_value_info(name, elem_type, None)
        sub_outputs.append(out)

    sub_graph = helper.make_graph(
        sub_nodes,
        "sub_graph",
        sub_inputs,
        sub_outputs,
        initializer=sub_initializers,
    )

    sub_model = helper.make_model(sub_graph, opset_imports=original_model.opset_import)
    sub_model.ir_version = original_model.ir_version
    sub_model.doc_string = model_doc_string

    return sub_model


def split_part4(input_path, output_dir):
    print(f"Loading {input_path} (with external data)...")
    t0 = time.time()
    model = onnx.load(input_path)
    print(f"  Loaded in {time.time() - t0:.1f}s ({len(model.graph.node)} nodes)")

    graph = model.graph
    original_output_names = [out.name for out in graph.output]
    print(f"  Outputs: {original_output_names}")

    vit_names, decoder_names = partition_nodes(graph, NORM_OUTPUT)
    print(f"  ViT nodes: {len(vit_names)}, Decoder nodes: {len(decoder_names)}")

    decoder_needed = collect_needed_tensors(graph, decoder_names)
    decoder_graph_inputs = []
    for name in decoder_needed:
        if name == NORM_OUTPUT:
            continue
        init = find_initializer(graph, name)
        if init is None:
            decoder_graph_inputs.append(name)

    print(f"\n  Part 4b (decoder) needs {len(decoder_graph_inputs)} original activation inputs + norm output:")
    for name in decoder_graph_inputs:
        print(f"    {name}")

    os.makedirs(output_dir, exist_ok=True)

    # --- Part 4a: ViT blocks ---
    print(f"\nBuilding Part 4a (ViT blocks)...")
    t0 = time.time()

    vit_input_specs = []
    for inp in graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(None)
        if not shape:
            shape = None
        vit_input_specs.append((inp.name, inp.type.tensor_type.elem_type, shape))

    part4a_model = build_submodel(
        model, vit_names,
        new_input_specs=vit_input_specs,
        new_output_specs=[(NORM_OUTPUT, TensorProto.FLOAT, [1, 577, 1024])],
        model_doc_string="SHARP Part 4a: ViT blocks 11.5-23 + norm",
    )

    part4a_path = os.path.join(output_dir, "sharp_part4a.onnx")
    onnx.save(part4a_model, part4a_path,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location="sharp_part4a.onnx.data")
    elapsed_a = time.time() - t0

    part4a_size = os.path.getsize(part4a_path) / 1e6
    part4a_data = part4a_path + ".data"
    part4a_data_size = os.path.getsize(part4a_data) / 1e6 if os.path.exists(part4a_data) else 0
    print(f"  Part 4a: {part4a_size:.1f} MB graph + {part4a_data_size:.1f} MB weights ({elapsed_a:.1f}s)")
    print(f"  Part 4a nodes: {len(part4a_model.graph.node)}, inputs: {len(part4a_model.graph.input)}")

    # --- Part 4b: Decoder ---
    print(f"\nBuilding Part 4b (decoder)...")
    t0 = time.time()

    part4b_input_specs = [(NORM_OUTPUT, TensorProto.FLOAT, [1, 577, 1024])]
    for name in decoder_graph_inputs:
        orig_inp = find_graph_input(graph, name)
        if orig_inp is not None:
            shape = []
            for d in orig_inp.type.tensor_type.shape.dim:
                if d.dim_value > 0:
                    shape.append(d.dim_value)
                else:
                    shape.append(None)
            if not shape:
                shape = None
            part4b_input_specs.append((name, orig_inp.type.tensor_type.elem_type, shape))

    part4b_output_specs = []
    for out in graph.output:
        shape = []
        for d in out.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(None)
        if not shape:
            shape = None
        part4b_output_specs.append((out.name, out.type.tensor_type.elem_type, shape))

    part4b_model = build_submodel(
        model, decoder_names,
        new_input_specs=part4b_input_specs,
        new_output_specs=part4b_output_specs,
        model_doc_string="SHARP Part 4b: Decoder + Gaussian head",
    )

    part4b_path = os.path.join(output_dir, "sharp_part4b.onnx")
    onnx.save(part4b_model, part4b_path,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location="sharp_part4b.onnx.data")
    elapsed_b = time.time() - t0

    part4b_size = os.path.getsize(part4b_path) / 1e6
    part4b_data = part4b_path + ".data"
    part4b_data_size = os.path.getsize(part4b_data) / 1e6 if os.path.exists(part4b_data) else 0
    print(f"  Part 4b: {part4b_size:.1f} MB graph + {part4b_data_size:.1f} MB weights ({elapsed_b:.1f}s)")
    print(f"  Part 4b nodes: {len(part4b_model.graph.node)}, inputs: {len(part4b_model.graph.input)}")

    total_mb = part4a_size + part4a_data_size + part4b_size + part4b_data_size
    print(f"\n{'='*60}")
    print("Split complete")
    print(f"{'='*60}")
    print(f"  Part 4a (ViT):     {part4a_size + part4a_data_size:.0f} MB")
    print(f"  Part 4b (decoder): {part4b_size + part4b_data_size:.0f} MB")
    print(f"  Total:             {total_mb:.0f} MB")
    print(f"\nPush to device:")
    dest = "/storage/emulated/0/Android/data/com.furnit.android/files/models"
    for f in sorted(os.listdir(output_dir)):
        if (f.startswith("sharp_part4a") or f.startswith("sharp_part4b")) and \
           (f.endswith(".onnx") or f.endswith(".onnx.data")):
            print(f"  adb push {os.path.join(output_dir, f)} {dest}/")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Split SHARP Part 4 ONNX into ViT + Decoder sub-chunks")
    parser.add_argument("--input", default="sharp_part4.onnx",
        help="Input Part 4 ONNX model (default: sharp_part4.onnx)")
    parser.add_argument("--output-dir", default=".",
        help="Output directory (default: current directory)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        print("Pull from device first:")
        print("  adb pull /storage/emulated/0/Android/data/com.furnit.android/files/models/sharp_part4.onnx .")
        print("  adb pull /storage/emulated/0/Android/data/com.furnit.android/files/models/sharp_part4.onnx.data .")
        return 1

    return split_part4(args.input, args.output_dir)


if __name__ == "__main__":
    sys.exit(main() or 0)
