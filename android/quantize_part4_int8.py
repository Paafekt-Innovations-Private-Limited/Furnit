#!/usr/bin/env python3
"""
Quantize SHARP Part 4 ONNX model to INT8 using dynamic quantization.
No calibration data needed -- just rewrites MatMul/Linear weights to INT8.

Usage:
    cd android
    python quantize_part4_int8.py [--input sharp_part4.onnx] [--output sharp_part4_int8.onnx]

Then push to device:
    adb push sharp_part4_int8.onnx /storage/emulated/0/Android/data/com.furnit.android/files/models/
    adb push sharp_part4_int8.onnx.data /storage/emulated/0/Android/data/com.furnit.android/files/models/  # if generated
"""

import argparse
import os
from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    parser = argparse.ArgumentParser(description="Quantize SHARP Part 4 to INT8")
    parser.add_argument("--input", default="sharp_part4.onnx", help="Input FP32 ONNX model")
    parser.add_argument("--output", default="sharp_part4_int8.onnx", help="Output INT8 ONNX model")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Copy from device or models directory first.")
        return

    input_size = os.path.getsize(args.input)
    data_file = args.input + ".data"
    data_size = os.path.getsize(data_file) if os.path.exists(data_file) else 0
    print(f"Input: {args.input} ({input_size / 1e6:.1f} MB)")
    if data_size:
        print(f"External data: {data_file} ({data_size / 1e6:.1f} MB)")

    print("Quantizing to INT8 (dynamic, no calibration)...")
    quantize_dynamic(
        model_input=args.input,
        model_output=args.output,
        weight_type=QuantType.QInt8,
    )

    output_size = os.path.getsize(args.output)
    out_data = args.output + ".data"
    out_data_size = os.path.getsize(out_data) if os.path.exists(out_data) else 0
    print(f"Output: {args.output} ({output_size / 1e6:.1f} MB)")
    if out_data_size:
        print(f"External data: {out_data} ({out_data_size / 1e6:.1f} MB)")
    print(f"Size reduction: {(1 - (output_size + out_data_size) / (input_size + data_size)) * 100:.1f}%")
    print("Done. Push to device models directory.")


if __name__ == "__main__":
    main()
