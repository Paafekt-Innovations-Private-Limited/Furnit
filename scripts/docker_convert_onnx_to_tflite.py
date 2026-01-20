#!/usr/bin/env python3
"""Convert an ONNX model to TFLite using onnx2tf.

Usage:
  python docker_convert_onnx_to_tflite.py --input model.onnx --output model.tflite --tmpdir /tmp/conv

This script uses onnx2tf which handles version compatibility better than onnx-tf.
"""
import argparse
import os
import shutil
import onnx2tf

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Input ONNX model path')
    p.add_argument('--output', required=True, help='Output TFLite model path')
    p.add_argument('--tmpdir', default='/tmp/onnx2tf_output', help='Temp directory for intermediate files')
    args = p.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    print(f"Converting ONNX: {input_path} -> TFLite: {output_path}")

    # onnx2tf converts directly to TFLite
    # Use static batch size to avoid dynamic dimension issues
    onnx2tf.convert(
        input_onnx_file_path=input_path,
        output_folder_path=args.tmpdir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=False,
        overwrite_input_shape=['images:1,3,768,768'],  # Static input shape for YOLO model
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
