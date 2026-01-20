#!/usr/bin/env python3
"""Convert YOLO ONNX model to TFLite using onnx2tf with shape overrides.

Usage:
  python docker_convert_onnx_to_tflite.py --input model.onnx --output model.tflite
"""
import argparse
import os
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Input model path (.pt or .onnx)')
    p.add_argument('--output', required=True, help='Output TFLite model path')
    p.add_argument('--tmpdir', default='/tmp/convert_output', help='Temp directory')
    args = p.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(args.tmpdir, exist_ok=True)

    print(f"Converting: {input_path} -> {output_path}")

    # If input is .pt, first export to ONNX using ultralytics
    if input_path.endswith('.pt'):
        from ultralytics import YOLO
        print("Loading YOLO model and exporting to ONNX...")
        model = YOLO(input_path)
        # Export to ONNX only (not TFLite, we'll do that manually)
        onnx_path = model.export(format='onnx', imgsz=768, simplify=True, opset=17)
        print(f"ONNX exported to: {onnx_path}")
        input_path = onnx_path

    # Convert ONNX to TFLite using onnx2tf CLI with special options
    print("Converting ONNX to TFLite using onnx2tf...")

    cmd = [
        'onnx2tf',
        '-i', input_path,
        '-o', args.tmpdir,
        '-oiqt',  # Output integer quantized tflite
        '-ioqd', 'uint8',  # Input/output quantization dtype
        '-cind', 'images', '1,3,768,768', '[[[[0.0,0.0,0.0]]]]', '[[[[1.0,1.0,1.0]]]]',  # Calibration
        '-coion',  # Copy ONNX input/output names
        '-b', '1',  # Batch size
        '-kat', 'images',  # Keep as-is for input tensor
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(result.stdout)
        if result.returncode != 0:
            print(f"onnx2tf failed: {result.stderr}")
            # Try simpler conversion without quantization
            print("Trying simpler conversion...")
            cmd_simple = [
                'onnx2tf',
                '-i', input_path,
                '-o', args.tmpdir,
                '-coion',
                '-b', '1',
            ]
            result = subprocess.run(cmd_simple, capture_output=True, text=True, timeout=600)
            print(result.stdout)
            if result.returncode != 0:
                print(f"Simple conversion also failed: {result.stderr}")
                raise RuntimeError("onnx2tf conversion failed")
    except subprocess.TimeoutExpired:
        raise RuntimeError("onnx2tf conversion timed out")

    # Find generated tflite file
    tflite_files = [f for f in os.listdir(args.tmpdir) if f.endswith('.tflite')]
    if tflite_files:
        # Prefer float32 version
        for f in tflite_files:
            if 'float32' in f:
                src = os.path.join(args.tmpdir, f)
                shutil.copy(src, output_path)
                print(f"TFLite model saved to {output_path}")
                break
        else:
            # Just use first one
            src = os.path.join(args.tmpdir, tflite_files[0])
            shutil.copy(src, output_path)
            print(f"TFLite model saved to {output_path}")
    else:
        # List directory to help debug
        print(f"Files in {args.tmpdir}:")
        for f in os.listdir(args.tmpdir):
            print(f"  {f}")
        raise FileNotFoundError(f"No TFLite file generated in {args.tmpdir}")

    print('Done')


if __name__ == '__main__':
    main()
