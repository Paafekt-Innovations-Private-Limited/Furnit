#!/usr/bin/env python3
"""
Quantize all 4 SHARP ONNX parts to INT8 (dynamic quantization).

Produces: sharp_part1_int8.onnx, sharp_part2_int8.onnx, sharp_part3_int8.onnx, sharp_part4_int8.onnx

Usage:
  cd android
  python quantize_all_parts_int8.py

Then push to device:
  for f in sharp_part{1,2,3,4}_int8.onnx; do
    adb push $f /storage/emulated/0/Android/data/com.furnit.android/files/models/
  done
"""

import os
import sys
import time
from onnxruntime.quantization import quantize_dynamic, QuantType

PARTS = [
    ("sharp_part1.onnx", "sharp_part1_int8.onnx"),
    ("sharp_part2.onnx", "sharp_part2_int8.onnx"),
    ("sharp_part3.onnx", "sharp_part3_int8.onnx"),
    ("sharp_part4.onnx", "sharp_part4_int8.onnx"),
]


def main():
    missing = [inp for inp, _ in PARTS if not os.path.exists(inp)]
    if missing:
        print(f"ERROR: Missing input files: {missing}")
        print("Pull from device first:")
        for f in missing:
            print(f"  adb pull /storage/emulated/0/Android/data/com.furnit.android/files/models/{f} .")
            print(f"  adb pull .../{f}.data .")
        return 1

    total_input = 0
    total_output = 0

    for input_file, output_file in PARTS:
        input_size = os.path.getsize(input_file)
        data_file = input_file + ".data"
        data_size = os.path.getsize(data_file) if os.path.exists(data_file) else 0
        total_in = input_size + data_size

        print(f"\nQuantizing {input_file} ({total_in / 1e6:.1f} MB)...")
        t0 = time.time()
        quantize_dynamic(
            model_input=input_file,
            model_output=output_file,
            weight_type=QuantType.QInt8,
        )
        elapsed = time.time() - t0

        output_size = os.path.getsize(output_file)
        out_data = output_file + ".data"
        out_data_size = os.path.getsize(out_data) if os.path.exists(out_data) else 0
        total_out = output_size + out_data_size
        reduction = (1 - total_out / total_in) * 100 if total_in > 0 else 0

        print(f"  Output: {output_file} ({total_out / 1e6:.1f} MB, -{reduction:.0f}%) in {elapsed:.1f}s")

        total_input += total_in
        total_output += total_out

    print(f"\n{'='*60}")
    print(f"All parts quantized")
    print(f"{'='*60}")
    print(f"  FP32 total: {total_input / 1e6:.0f} MB")
    print(f"  INT8 total: {total_output / 1e6:.0f} MB")
    print(f"  Reduction:  {(1 - total_output / total_input) * 100:.0f}%")

    print(f"\nPush to device:")
    dest = "/storage/emulated/0/Android/data/com.furnit.android/files/models"
    for _, output_file in PARTS:
        if os.path.exists(output_file):
            print(f"  adb push {output_file} {dest}/")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
