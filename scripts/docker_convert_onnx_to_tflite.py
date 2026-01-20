#!/usr/bin/env python3
"""Convert a PyTorch YOLO model (.pt) to TFLite using ultralytics export.

Usage:
  python docker_convert_onnx_to_tflite.py --input model.pt --output model.tflite
"""
import argparse
import os
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Input model path (.pt)')
    p.add_argument('--output', required=True, help='Output TFLite model path')
    p.add_argument('--tmpdir', default='/tmp/convert_output', help='Temp directory')
    args = p.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(args.tmpdir, exist_ok=True)

    print(f"Converting: {input_path} -> {output_path}")

    from ultralytics import YOLO

    print("Loading YOLO model...")
    model = YOLO(input_path)

    print("Exporting to TFLite...")
    # ultralytics export creates the tflite file in the same directory as the .pt
    export_result = model.export(format='tflite', imgsz=768)

    print(f"Export result: {export_result}")

    # Find the generated tflite file
    if export_result and os.path.exists(export_result):
        shutil.copy(export_result, output_path)
        print(f"TFLite model saved to {output_path}")
    else:
        # Look for tflite in common locations
        base_dir = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        possible_paths = [
            os.path.join(base_dir, f"{base_name}_saved_model", f"{base_name}_float32.tflite"),
            os.path.join(base_dir, f"{base_name}_float32.tflite"),
            os.path.join(base_dir, f"{base_name}.tflite"),
        ]

        for p in possible_paths:
            if os.path.exists(p):
                shutil.copy(p, output_path)
                print(f"TFLite model saved to {output_path}")
                break
        else:
            # List directory to help debug
            print(f"Files in {base_dir}:")
            for f in os.listdir(base_dir):
                print(f"  {f}")
            raise FileNotFoundError(f"Could not find generated TFLite file")

    print('Done')


if __name__ == '__main__':
    main()
