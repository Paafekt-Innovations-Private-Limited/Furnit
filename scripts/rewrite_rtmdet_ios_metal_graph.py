#!/usr/bin/env python3
"""Build Paafekt's semantics-preserving iOS Metal RTMDet graph variant.

LiteRT 2.17's iOS Metal delegate does not accept RELU_0_TO_1 in this exported
graph. Each of the four occurrences is exactly ``clip(x, 0, 1)``. Replace it
with the equivalent supported pair ``maximum(x, 0)`` then ``minimum(x, 1)``.

The script is intentionally pinned to the reviewed Android source payload and
the reviewed deterministic output. It never changes Android's model.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
from collections import Counter
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema


SOURCE_SHA256 = "7edbd6692733d42a70344999aa5815762585c2a785b0e47cead4d786d4fb854d"
OUTPUT_SHA256 = "f13a4bf62e79284ae1b2f872c8ab7288767475fc2864af627c7dd79479bf1757"
EXPECTED_RELU_0_TO_1_COUNT = 4


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def operator_inventory(model: schema.ModelT) -> Counter[int]:
    inventory: Counter[int] = Counter()
    for subgraph in model.subgraphs:
        for operator in subgraph.operators:
            inventory[model.operatorCodes[operator.opcodeIndex].builtinCode] += 1
    return inventory


def opcode_index(model: schema.ModelT, builtin_code: int) -> int:
    for index, entry in enumerate(model.operatorCodes):
        if entry.builtinCode == builtin_code:
            return index

    entry = schema.OperatorCodeT()
    entry.builtinCode = builtin_code
    entry.deprecatedBuiltinCode = builtin_code if builtin_code < 127 else 127
    entry.version = 1
    model.operatorCodes.append(entry)
    return len(model.operatorCodes) - 1


def append_scalar(
    model: schema.ModelT,
    subgraph: schema.SubGraphT,
    value: float,
    name: bytes,
) -> int:
    buffer = schema.BufferT()
    buffer.data = np.frombuffer(struct.pack("<f", value), dtype=np.uint8).copy()
    model.buffers.append(buffer)

    # Rank-1 `[1]`, not rank-0 `[]`. Both are one element and are identical on CPU, but the
    # LiteRT Metal delegate derives a BHWC shape from the constant operand's dimension count and
    # has no rank-0 case, so a `[]` clamp bound silently reaches the GPU kernel as garbage. That
    # corrupts the four hard-sigmoid attention gates and, with them, the whole feature pyramid.
    tensor = schema.TensorT()
    tensor.shape = np.asarray([1], dtype=np.int32)
    tensor.shapeSignature = np.asarray([1], dtype=np.int32)
    tensor.type = schema.TensorType.FLOAT32
    tensor.buffer = len(model.buffers) - 1
    tensor.name = name
    tensor.hasRank = True
    subgraph.tensors.append(tensor)
    return len(subgraph.tensors) - 1


def binary_operator(
    opcode: int,
    left_tensor: int,
    right_tensor: int,
    output_tensor: int,
) -> schema.OperatorT:
    operator = schema.OperatorT()
    operator.opcodeIndex = opcode
    operator.inputs = np.asarray([left_tensor, right_tensor], dtype=np.int32)
    operator.outputs = np.asarray([output_tensor], dtype=np.int32)
    operator.builtinOptionsType = schema.BuiltinOptions.MaximumMinimumOptions
    operator.builtinOptions = schema.MaximumMinimumOptionsT()
    return operator


def rewrite(source_payload: bytes) -> bytes:
    source_hash = sha256(source_payload)
    if source_hash != SOURCE_SHA256:
        raise RuntimeError(
            f"Refusing unreviewed RTMDet source: sha256={source_hash}, "
            f"expected={SOURCE_SHA256}"
        )

    model = schema.ModelT.InitFromPackedBuf(bytearray(source_payload))
    before = operator_inventory(model)
    if before[schema.BuiltinOperator.RELU_0_TO_1] != EXPECTED_RELU_0_TO_1_COUNT:
        raise RuntimeError(
            "Expected exactly four RELU_0_TO_1 operators, found "
            f"{before[schema.BuiltinOperator.RELU_0_TO_1]}"
        )

    maximum_opcode = opcode_index(model, schema.BuiltinOperator.MAXIMUM)
    minimum_opcode = opcode_index(model, schema.BuiltinOperator.MINIMUM)
    replacement_index = 0

    for subgraph_index, subgraph in enumerate(model.subgraphs):
        matching_operators = [
            operator
            for operator in subgraph.operators
            if model.operatorCodes[operator.opcodeIndex].builtinCode
            == schema.BuiltinOperator.RELU_0_TO_1
        ]
        if not matching_operators:
            continue

        zero_tensor = append_scalar(
            model,
            subgraph,
            0.0,
            f"paafekt_metal_clip_zero_{subgraph_index}".encode(),
        )
        one_tensor = append_scalar(
            model,
            subgraph,
            1.0,
            f"paafekt_metal_clip_one_{subgraph_index}".encode(),
        )
        rewritten_operators: list[schema.OperatorT] = []

        for operator in subgraph.operators:
            builtin_code = model.operatorCodes[operator.opcodeIndex].builtinCode
            if builtin_code != schema.BuiltinOperator.RELU_0_TO_1:
                rewritten_operators.append(operator)
                continue
            if len(operator.inputs) != 1 or len(operator.outputs) != 1:
                raise RuntimeError("Unexpected RELU_0_TO_1 tensor contract")

            input_tensor = int(operator.inputs[0])
            output_tensor = int(operator.outputs[0])
            output_metadata = subgraph.tensors[output_tensor]

            intermediate = schema.TensorT()
            intermediate.shape = np.asarray(output_metadata.shape, dtype=np.int32)
            intermediate.shapeSignature = (
                None
                if output_metadata.shapeSignature is None
                else np.asarray(output_metadata.shapeSignature, dtype=np.int32)
            )
            intermediate.type = schema.TensorType.FLOAT32
            intermediate.buffer = 0
            intermediate.name = f"paafekt_metal_clip_max_{replacement_index}".encode()
            intermediate.hasRank = True
            subgraph.tensors.append(intermediate)
            intermediate_tensor = len(subgraph.tensors) - 1

            rewritten_operators.append(
                binary_operator(
                    maximum_opcode,
                    input_tensor,
                    zero_tensor,
                    intermediate_tensor,
                )
            )
            rewritten_operators.append(
                binary_operator(
                    minimum_opcode,
                    intermediate_tensor,
                    one_tensor,
                    output_tensor,
                )
            )
            replacement_index += 1

        subgraph.operators = rewritten_operators

    if replacement_index != EXPECTED_RELU_0_TO_1_COUNT:
        raise RuntimeError(f"Rewrote {replacement_index} clamp operators, expected four")

    after = operator_inventory(model)
    if after[schema.BuiltinOperator.RELU_0_TO_1] != 0:
        raise RuntimeError("RELU_0_TO_1 remains in rewritten graph")
    if after[schema.BuiltinOperator.MAXIMUM] - before[schema.BuiltinOperator.MAXIMUM] != 4:
        raise RuntimeError("Expected four new MAXIMUM operators")
    if after[schema.BuiltinOperator.MINIMUM] - before[schema.BuiltinOperator.MINIMUM] != 4:
        raise RuntimeError("Expected four new MINIMUM operators")

    builder = flatbuffers.Builder(len(source_payload) + 4096)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    output_payload = bytes(builder.Output())
    output_hash = sha256(output_payload)
    if output_hash != OUTPUT_SHA256:
        raise RuntimeError(
            f"Rewritten graph did not match reviewed payload: sha256={output_hash}, "
            f"expected={OUTPUT_SHA256}"
        )
    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Reviewed Android RTMDet .tflite")
    parser.add_argument("destination", type=Path, help="New iOS-specific .tflite output")
    args = parser.parse_args()

    if args.source.resolve() == args.destination.resolve():
        raise RuntimeError("Source and destination must be different paths")
    output_payload = rewrite(args.source.read_bytes())
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_bytes(output_payload)
    print(
        f"wrote={args.destination} bytes={len(output_payload)} "
        f"sha256={sha256(output_payload)}"
    )


if __name__ == "__main__":
    main()
