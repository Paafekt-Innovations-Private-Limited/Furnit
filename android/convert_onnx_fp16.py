#!/usr/bin/env python3
"""
Convert all 4 FP32 split ONNX models to FP16.

Input:  sharp_part{1,2,3,4}.onnx + .data (FP32, ~600MB each, ~2.6GB total)
Output: sharp_part{1,2,3,4}_fp16.onnx + .data (FP16, ~300MB each, ~1.3GB total)

Usage:
  cd android
  python convert_onnx_fp16.py

Then push to device:
  for i in 1 2 3 4; do
    adb push sharp_part${i}_fp16.onnx /storage/emulated/0/Android/data/com.furnit.android/files/models/
    adb push sharp_part${i}_fp16.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/
  done
"""

import os
import sys
import time

PARTS = [
    ("sharp_part1.onnx", "sharp_part1_fp16.onnx"),
    ("sharp_part2.onnx", "sharp_part2_fp16.onnx"),
    ("sharp_part3.onnx", "sharp_part3_fp16.onnx"),
    ("sharp_part4.onnx", "sharp_part4_fp16.onnx"),
]


def main():
    try:
        import onnx
        from onnxruntime.transformers import float16
    except ImportError:
        print("ERROR: Install dependencies: pip install onnx onnxruntime")
        return 1

    missing = [inp for inp, _ in PARTS if not os.path.exists(inp)]
    if missing:
        print(f"ERROR: Missing FP32 input files: {missing}")
        print("Pull from device first:")
        for f in missing:
            print(f"  adb pull /storage/emulated/0/Android/data/com.furnit.android/files/models/{f} .")
            data = f + ".data"
            print(f"  adb pull /storage/emulated/0/Android/data/com.furnit.android/files/models/{data} .")
        return 1

    total_fp32 = 0
    total_fp16 = 0

    for input_file, output_file in PARTS:
        fp32_size = os.path.getsize(input_file)
        fp32_data = input_file + ".data"
        fp32_data_size = os.path.getsize(fp32_data) if os.path.exists(fp32_data) else 0
        total_in = fp32_size + fp32_data_size

        print(f"\nConverting {input_file} ({total_in / 1e6:.0f} MB) -> FP16...")
        t0 = time.time()

        model = onnx.load(input_file)

        # Collect names of nodes with ops that lack FP16 CPU kernels in ORT Android.
        # op_block_list doesn't catch com.microsoft domain ops, so we use node_block_list.
        blocked_ops = {
            "Gelu", "FastGelu", "BiasGelu",
            "LayerNormalization", "SkipLayerNormalization",
            "EmbedLayerNormalization",
        }
        node_block_list = [
            n.name for n in model.graph.node
            if n.op_type in blocked_ops
        ]
        if node_block_list:
            print(f"  Keeping {len(node_block_list)} nodes in FP32 ({', '.join(blocked_ops & {n.op_type for n in model.graph.node})})")

        fp16_model = float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            node_block_list=node_block_list,
        )
        onnx.save(fp16_model, output_file)

        elapsed = time.time() - t0

        fp16_size = os.path.getsize(output_file)
        fp16_data = output_file + ".data"
        fp16_data_size = os.path.getsize(fp16_data) if os.path.exists(fp16_data) else 0
        total_out = fp16_size + fp16_data_size
        reduction = (1 - total_out / total_in) * 100 if total_in > 0 else 0

        print(f"  Output: {output_file} ({total_out / 1e6:.0f} MB, -{reduction:.0f}%) in {elapsed:.1f}s")

        total_fp32 += total_in
        total_fp16 += total_out

        del model, fp16_model

    print(f"\n{'='*60}")
    print(f"All parts converted to FP16")
    print(f"{'='*60}")
    print(f"  FP32 total: {total_fp32 / 1e6:.0f} MB")
    print(f"  FP16 total: {total_fp16 / 1e6:.0f} MB")
    print(f"  Reduction:  {(1 - total_fp16 / total_fp32) * 100:.0f}%")

    dest = "/storage/emulated/0/Android/data/com.furnit.android/files/models"
    print(f"\nPush to device:")
    for _, output_file in PARTS:
        print(f"  adb push {output_file} {dest}/")
        data_file = output_file + ".data"
        if os.path.exists(data_file):
            print(f"  adb push {data_file} {dest}/")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
