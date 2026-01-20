#!/usr/bin/env python3
"""Convert an ONNX model to TFLite using onnx-simplifier + onnx2tf.

Usage:
  python docker_convert_onnx_to_tflite.py --input model.onnx --output model.tflite --tmpdir /tmp/conv

This script:
1. Simplifies the ONNX model with static shapes using onnxsim
2. Converts to TFLite using onnx2tf
"""
import argparse
import os
import shutil
import subprocess

def simplify_onnx(input_path, output_path, input_shape):
    """Simplify ONNX model with static input shape."""
    print(f"Simplifying ONNX model with shape {input_shape}...")

    # Use onnxsim CLI to simplify with static shape
    cmd = [
        'onnxsim', input_path, output_path,
        '--overwrite-input-shape', input_shape,
        '--skip-shape-inference'  # Skip for complex models
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: onnxsim failed: {result.stderr}")
        # Try without skip-shape-inference
        cmd = ['onnxsim', input_path, output_path, '--overwrite-input-shape', input_shape]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: onnxsim also failed: {result.stderr}")
            # Fall back to original
            shutil.copy(input_path, output_path)
            return False

    print("ONNX simplification complete")
    return True

def main():
    import onnx2tf

    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Input ONNX model path')
    p.add_argument('--output', required=True, help='Output TFLite model path')
    p.add_argument('--tmpdir', default='/tmp/onnx2tf_output', help='Temp directory for intermediate files')
    args = p.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    os.makedirs(args.tmpdir, exist_ok=True)

    print(f"Converting ONNX: {input_path} -> TFLite: {output_path}")

    # Step 1: Simplify ONNX with static shape
    simplified_path = os.path.join(args.tmpdir, 'simplified.onnx')
    input_shape = 'images:1,3,768,768'  # Static shape for YOLO model
    simplify_onnx(input_path, simplified_path, input_shape)

    # Step 2: Convert simplified model to TFLite
    print("Converting to TFLite...")
    try:
        onnx2tf.convert(
            input_onnx_file_path=simplified_path,
            output_folder_path=args.tmpdir,
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=False,
        )
    except Exception as e:
        print(f"First attempt failed: {e}")
        print("Trying with original model and static shape override...")
        onnx2tf.convert(
            input_onnx_file_path=input_path,
            output_folder_path=args.tmpdir,
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=False,
            overwrite_input_shape=['images:1,3,768,768'],
        )

    # Find the generated tflite file
    tflite_files = [f for f in os.listdir(args.tmpdir) if f.endswith('.tflite')]
    if not tflite_files:
        # Check for saved_model and convert manually
        saved_model_path = os.path.join(args.tmpdir, 'saved_model')
        if os.path.exists(saved_model_path):
            import tensorflow as tf
            print(f"Converting SavedModel to TFLite...")
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            tflite_model = converter.convert()
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            print(f"TFLite model saved to {output_path}")
        else:
            raise FileNotFoundError(f"No TFLite file found in {args.tmpdir}")
    else:
        # Copy the first tflite file to output path
        src_tflite = os.path.join(args.tmpdir, tflite_files[0])
        shutil.copy(src_tflite, output_path)
        print(f"TFLite model saved to {output_path}")

    print('Done')

if __name__ == '__main__':
    main()
