#!/usr/bin/env python3
"""Convert an ONNX model to TensorFlow SavedModel then to TFLite inside Docker.

Usage:
  python docker_convert_onnx_to_tflite.py --input model.onnx --output model.tflite --tmpdir /tmp/conv

This script assumes compatible versions of `onnx`, `onnx_tf`, and `tensorflow` are installed.
"""
import argparse
import os
import sys
import tempfile
import onnx
from onnx_tf.backend import prepare

def onnx_to_saved_model(onnx_path, saved_model_dir):
    print(f"Loading ONNX model {onnx_path}")
    model = onnx.load(onnx_path)
    print("Preparing TensorFlow representation via onnx-tf")
    tf_rep = prepare(model)
    print(f"Exporting SavedModel to {saved_model_dir}")
    tf_rep.export_graph(saved_model_dir)

def saved_model_to_tflite(saved_model_dir, tflite_path):
    import tensorflow as tf
    print(f"Converting SavedModel {saved_model_dir} -> TFLite {tflite_path}")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = []
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tmpdir', default=None)
    args = p.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix='onnx_conv_')
    os.makedirs(tmpdir, exist_ok=True)

    saved_model_dir = os.path.join(tmpdir, 'saved_model')
    os.makedirs(saved_model_dir, exist_ok=True)

    onnx_to_saved_model(input_path, saved_model_dir)
    saved_model_to_tflite(saved_model_dir, output_path)

    print('Done')

if __name__ == '__main__':
    main()
