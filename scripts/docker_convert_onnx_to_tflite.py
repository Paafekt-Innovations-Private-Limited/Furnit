#!/usr/bin/env python3
"""Convert a PyTorch YOLO model (.pt) to TFLite using ai-edge-torch.

Usage:
  python docker_convert_onnx_to_tflite.py --input model.pt --output model.tflite

This script uses Google's ai-edge-torch to convert directly from PyTorch to TFLite,
bypassing ONNX conversion issues.
"""
import argparse
import os
import torch

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

    # Check if input is .pt (PyTorch) or .onnx
    if input_path.endswith('.pt'):
        convert_pt_to_tflite(input_path, output_path, args.tmpdir)
    elif input_path.endswith('.onnx'):
        convert_onnx_to_tflite(input_path, output_path, args.tmpdir)
    else:
        raise ValueError(f"Unsupported input format: {input_path}")

    print('Done')


def convert_pt_to_tflite(input_path, output_path, tmpdir):
    """Convert PyTorch .pt model to TFLite using ai-edge-torch."""
    import ai_edge_torch
    from ultralytics import YOLO

    print("Loading YOLO model...")
    model = YOLO(input_path)

    # Get the PyTorch model
    pt_model = model.model
    pt_model.eval()

    # Create sample input (YOLO uses 768x768 for this model)
    sample_input = torch.randn(1, 3, 768, 768)

    print("Converting to TFLite using ai-edge-torch...")
    try:
        # Try direct conversion
        edge_model = ai_edge_torch.convert(pt_model, (sample_input,))
        edge_model.export(output_path)
        print(f"TFLite model saved to {output_path}")
    except Exception as e:
        print(f"ai-edge-torch conversion failed: {e}")
        print("Trying ultralytics built-in export...")

        # Fallback: Use ultralytics export
        export_path = model.export(format='tflite', imgsz=768)
        if export_path and os.path.exists(export_path):
            import shutil
            shutil.copy(export_path, output_path)
            print(f"TFLite model saved to {output_path}")
        else:
            raise RuntimeError("Both conversion methods failed")


def convert_onnx_to_tflite(input_path, output_path, tmpdir):
    """Fallback: Convert ONNX to TFLite using onnx2tf."""
    import onnx2tf
    import shutil

    print("Converting ONNX to TFLite using onnx2tf...")
    onnx2tf.convert(
        input_onnx_file_path=input_path,
        output_folder_path=tmpdir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=False,
    )

    # Find generated tflite
    tflite_files = [f for f in os.listdir(tmpdir) if f.endswith('.tflite')]
    if tflite_files:
        src = os.path.join(tmpdir, tflite_files[0])
        shutil.copy(src, output_path)
        print(f"TFLite model saved to {output_path}")
    else:
        raise FileNotFoundError(f"No TFLite file generated in {tmpdir}")


if __name__ == '__main__':
    main()
