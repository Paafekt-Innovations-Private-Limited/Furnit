#!/usr/bin/env python3
"""Finalize the Android RTMDet raw ONNX with the same graph contract as Swift.

The original Android export stops before two operations that are embedded in the
iOS Core ML graph:

* BGR mean/std normalization
* bilinear 80x80 -> 160x160 mask-feature upsampling (align_corners=False)

This deterministic graph rewrite adds those operations without changing the
checkpoint weights or the nine detection/kernel outputs. It also records model
metadata so Android knows to submit raw 0...255 BGR values.

PyTorch also exports shared RTMDet regression-head parameters through Identity
nodes. ONNX Runtime handles those aliases, but ONNX-to-LiteRT converters can
mistake the aliased NCHW convolution weights for ordinary tensors and skip the
OHWI weight transpose. Folding initializer-only aliases keeps ONNX numerics
identical and makes the graph safe to convert to LiteRT.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper


REPO_ANDROID = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ANDROID / "rtmdet_models/src/main/assets/rtmdet-ins-m-raw.onnx"
CONTRACT_KEY = "furnit.rtmdet.contract"
CONTRACT_VALUE = "swift_raw_v2"
PREPROCESS_KEY = "furnit.rtmdet.preprocess"
PREPROCESS_VALUE = "bgr_mean_std"
MASK_SIDE_KEY = "furnit.rtmdet.mask_feat_side"
CONVERTER_SAFE_KEY = "furnit.rtmdet.converter_safe_constants"
CONVERTER_SAFE_VALUE = "initializer_identity_folded_v1"


def _metadata(model: onnx.ModelProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in model.metadata_props}


def _shape(value: onnx.ValueInfoProto) -> list[int]:
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def _set_output_side(value: onnx.ValueInfoProto, side: int) -> None:
    dimensions = value.type.tensor_type.shape.dim
    if len(dimensions) != 4:
        raise ValueError(f"mask_feat must be rank 4, got {_shape(value)}")
    dimensions[2].dim_value = side
    dimensions[3].dim_value = side


def _fold_initializer_identity_aliases(model: onnx.ModelProto) -> int:
    """Remove Identity aliases whose source is a constant initializer.

    The RTMDet SepBN head intentionally shares convolution parameters. PyTorch
    represents the reused initializers with Identity nodes in ONNX. Replacing
    their consumers with the original initializer is an exact graph rewrite;
    it also lets TFLite conversion recognize and transpose every Conv weight.
    """
    graph = model.graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    aliases = {
        node.output[0]: node.input[0]
        for node in graph.node
        if node.op_type == "Identity"
        and len(node.input) == 1
        and len(node.output) == 1
        and node.input[0] in initializer_names
    }
    if not aliases:
        return 0

    graph_outputs = {output.name for output in graph.output}
    aliased_graph_outputs = graph_outputs & aliases.keys()
    if aliased_graph_outputs:
        raise ValueError(
            "Refusing to fold initializer aliases exposed as graph outputs: "
            f"{sorted(aliased_graph_outputs)}"
        )

    for node in graph.node:
        for index, name in enumerate(node.input):
            source = aliases.get(name)
            if source is not None:
                node.input[index] = source

    retained_nodes = [
        node
        for node in graph.node
        if not (node.op_type == "Identity" and node.output and node.output[0] in aliases)
    ]
    del graph.node[:]
    graph.node.extend(retained_nodes)
    return len(aliases)


def _rewrite(model: onnx.ModelProto) -> bool:
    metadata = _metadata(model)
    changed = False
    if metadata.get(CONTRACT_KEY) == CONTRACT_VALUE:
        mask_output = next((item for item in model.graph.output if item.name == "mask_feat"), None)
        if mask_output is None or _shape(mask_output) != [1, 8, 160, 160]:
            raise ValueError("Swift-parity metadata exists but mask_feat is not [1,8,160,160]")
    else:
        graph = model.graph
        if len(graph.input) != 1:
            raise ValueError(f"Expected one model input, found {len(graph.input)}")
        model_input = graph.input[0]
        if model_input.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise ValueError("Expected Float32 RTMDet input")
        if _shape(model_input) != [1, 3, 640, 640]:
            raise ValueError(f"Expected input [1,3,640,640], got {_shape(model_input)}")

        existing_names = {
            name
            for node in graph.node
            for name in (*node.input, *node.output)
            if name
        } | {initializer.name for initializer in graph.initializer}
        added_names = {
            "furnit_input_mean",
            "furnit_input_inv_std",
            "furnit_input_centered",
            "furnit_input_normalized",
            "furnit_mask_feat_80",
            "furnit_mask_feat_sizes",
        }
        collisions = existing_names & added_names
        if collisions:
            raise ValueError(f"Graph already contains reserved Furnit names: {sorted(collisions)}")

        original_input_name = model_input.name
        original_nodes = list(graph.node)
        for node in original_nodes:
            for index, name in enumerate(node.input):
                if name == original_input_name:
                    node.input[index] = "furnit_input_normalized"

        mean = np.asarray([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 3, 1, 1)
        inv_std = (1.0 / np.asarray([57.375, 57.12, 58.395], dtype=np.float32)).reshape(1, 3, 1, 1)
        graph.initializer.extend(
            [
                numpy_helper.from_array(mean, name="furnit_input_mean"),
                numpy_helper.from_array(inv_std, name="furnit_input_inv_std"),
            ]
        )
        preprocess_nodes = [
            helper.make_node(
                "Sub",
                [original_input_name, "furnit_input_mean"],
                ["furnit_input_centered"],
                name="FurnitBgrMean",
            ),
            helper.make_node(
                "Mul",
                ["furnit_input_centered", "furnit_input_inv_std"],
                ["furnit_input_normalized"],
                name="FurnitBgrInvStd",
            ),
        ]
        del graph.node[:]
        graph.node.extend(preprocess_nodes)
        graph.node.extend(original_nodes)

        mask_output = next((item for item in graph.output if item.name == "mask_feat"), None)
        if mask_output is None:
            raise ValueError("Missing mask_feat graph output")
        if _shape(mask_output) != [1, 8, 80, 80]:
            raise ValueError(f"Expected original mask_feat [1,8,80,80], got {_shape(mask_output)}")
        mask_producers = [node for node in graph.node if "mask_feat" in node.output]
        mask_consumers = [node for node in graph.node if "mask_feat" in node.input]
        if len(mask_producers) != 1 or mask_consumers:
            raise ValueError(
                f"mask_feat must be one terminal value; producers={len(mask_producers)} "
                f"consumers={len(mask_consumers)}"
            )
        producer = mask_producers[0]
        mask_output_index = next(index for index, name in enumerate(producer.output) if name == "mask_feat")
        producer.output[mask_output_index] = "furnit_mask_feat_80"

        sizes = np.asarray([1, 8, 160, 160], dtype=np.int64)
        graph.initializer.append(numpy_helper.from_array(sizes, name="furnit_mask_feat_sizes"))
        graph.node.append(
            helper.make_node(
                "Resize",
                ["furnit_mask_feat_80", "", "", "furnit_mask_feat_sizes"],
                ["mask_feat"],
                name="FurnitMaskFeatBilinear2x",
                mode="linear",
                coordinate_transformation_mode="half_pixel",
            )
        )
        _set_output_side(mask_output, 160)
        changed = True

    folded_alias_count = _fold_initializer_identity_aliases(model)
    if folded_alias_count:
        print(f"folded {folded_alias_count} initializer Identity aliases")
        changed = True

    metadata.update(
        {
            CONTRACT_KEY: CONTRACT_VALUE,
            PREPROCESS_KEY: PREPROCESS_VALUE,
            MASK_SIDE_KEY: "160",
            CONVERTER_SAFE_KEY: CONVERTER_SAFE_VALUE,
        }
    )
    helper.set_model_props(model, metadata)
    checker.check_model(model)
    return changed


def _bilinear_2x_align_corners_false(source: np.ndarray) -> np.ndarray:
    if source.shape != (1, 8, 80, 80):
        raise ValueError(f"Unexpected source mask shape: {source.shape}")
    coordinates = (np.arange(160, dtype=np.float32) + 0.5) * 0.5 - 0.5
    low = np.floor(coordinates).astype(np.int64)
    high = low + 1
    weight = coordinates - low
    low = np.clip(low, 0, 79)
    high = np.clip(high, 0, 79)

    top = source[:, :, low, :]
    bottom = source[:, :, high, :]
    vertical = top + (bottom - top) * weight.reshape(1, 1, 160, 1)
    left = vertical[:, :, :, low]
    right = vertical[:, :, :, high]
    return left + (right - left) * weight.reshape(1, 1, 1, 160)


def _verify(original_session, rewritten_path: Path) -> None:
    import onnxruntime as ort

    rng = np.random.default_rng(20260804)
    raw_bgr = rng.integers(0, 256, size=(1, 3, 640, 640), dtype=np.uint8).astype(np.float32)
    mean = np.asarray([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 3, 1, 1)
    normalized = (raw_bgr - mean) / std

    output_names = [output.name for output in original_session.get_outputs()]
    original_metadata = original_session.get_modelmeta().custom_metadata_map
    original_input = (
        raw_bgr
        if original_metadata.get(PREPROCESS_KEY) == PREPROCESS_VALUE
        else normalized
    )
    original = original_session.run(output_names, {original_session.get_inputs()[0].name: original_input})
    rewritten_session = ort.InferenceSession(
        str(rewritten_path),
        providers=["CPUExecutionProvider"],
    )
    rewritten = rewritten_session.run(output_names, {rewritten_session.get_inputs()[0].name: raw_bgr})

    for name, expected, actual in zip(output_names, original, rewritten):
        if name == "mask_feat" and expected.shape == (1, 8, 80, 80):
            expected = _bilinear_2x_align_corners_false(expected)
        if expected.shape != actual.shape:
            raise AssertionError(f"{name}: shape {expected.shape} != {actual.shape}")
        maximum_error = float(np.max(np.abs(expected - actual)))
        if not np.allclose(expected, actual, rtol=2e-4, atol=2e-4):
            raise AssertionError(f"{name}: max abs error {maximum_error:.6g}")
        print(f"verified {name}: shape={actual.shape} max_abs_error={maximum_error:.6g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    source = args.input.resolve()
    destination = (args.output or args.input).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    original_session = None
    if args.verify:
        import onnxruntime as ort

        original_session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])

    model = onnx.load(str(source))
    changed = _rewrite(model)
    if changed:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.swift-parity.tmp")
        onnx.save_model(model, str(temporary))
        os.replace(temporary, destination)
        print(f"wrote Swift-parity RTMDet ONNX: {destination}")
    else:
        print(f"already Swift-parity RTMDet ONNX: {source}")

    if args.verify:
        if not changed:
            print("model was already rewritten; structural validation passed")
        else:
            assert original_session is not None
            _verify(original_session, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
