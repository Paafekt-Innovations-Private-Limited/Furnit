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

    # Optionally simplify ONNX first (helps with dynamic shapes and weird Transpose)
    print("Running ONNX simplifier (onnxsim) to collapse dynamic shapes if available...")
    simplified_onnx = os.path.join(args.tmpdir, 'simplified.onnx')
    try:
        # Try using onnxsim python API if installed
        from onnxsim import simplify
        import onnx

        print(f"Loading ONNX model: {input_path}")
        model_onnx = onnx.load(input_path)
        print("Running simplify()...")
        model_simp, check = simplify(model_onnx)
        if check:
            onnx.save(model_simp, simplified_onnx)
            input_path = simplified_onnx
            print(f"Simplified ONNX saved to {simplified_onnx}")
        else:
            print("onnxsim simplify returned check=False; skipping simplified save")
    except Exception as e:
        print(f"onnxsim simplify step skipped or failed: {e}")

    # Convert ONNX to TFLite using onnx2tf CLI with special options
    print("Converting ONNX to TFLite using onnx2tf...")

    # allow overriding input size via env or detect common sizes (fallback 1536)
    default_size = 1536
    try:
        # quick heuristic: look for common size markers in filename
        if '768' in os.path.basename(input_path):
            default_size = 768
        elif '1536' in os.path.basename(input_path):
            default_size = 1536
    except Exception:
        pass

    cmd = [
        'onnx2tf',
        '-i', input_path,
        '-o', args.tmpdir,
        '-coion',  # Copy ONNX input/output names
        '-b', '1',  # Batch size
        '-ois',  # Override input shapes / rewrite dynamic dims
        '-kat', 'images',  # Keep-as-tensor for input
        '-kt', 'images',  # Also try keep-tensor variant for safety
        '-onwdt',  # Try rewriting NMS/post-processing where needed
    ]
    # Prepare an onnx2tf-style parameter-replacement JSON that targets the
    # problematic Transpose node. onnx2tf expects a top-level structure with
    # "format_version" and an "operations" list; each entry names the op and
    # specifies what attribute/input to replace.
    param_json_path = os.path.join(args.tmpdir, 'replace.json')
    param_replacement = {
        "format_version": 1,
        "operations": [
            {
                # Use the full node name observed in the CI logs so onnx2tf can
                # match it exactly.
                "op_name": "wa/model.23/lrpc.0/Transpose_1",
                "param_target": "attributes",
                "param_name": "perm",
                "values": [0, 1, 2, 3]
            }
        ]
    }
    try:
        import json
        with open(param_json_path, 'w') as f:
            json.dump(param_replacement, f, indent=2)
        print(f"Wrote onnx2tf replace.json to {param_json_path}")
    except Exception as e:
        print(f"Failed to write replace.json: {e}")

    def run_cmd(command):
        try:
            res = subprocess.run(command, capture_output=True, text=True, timeout=900)
            print(res.stdout)
            if res.returncode != 0:
                print(f"Command failed (rc={res.returncode}): {res.stderr}")
            return res
        except subprocess.TimeoutExpired:
            print("Command timed out")
            return None

    # First attempt: existing rich flags
    print("Running onnx2tf (primary flags)...")
    result = run_cmd(cmd)
    if not result or result.returncode != 0:
        print("Primary attempt failed; trying simpler conversion...")
        cmd_simple = [
            'onnx2tf',
            '-i', input_path,
            '-o', args.tmpdir,
            '-coion',
            '-b', '1',
        ]
        result = run_cmd(cmd_simple)

    # If still failed, try passing the parameter-replacement JSON using the
    # documented onnx2tf flag `-prf` (replace.json file). This is the flag used
    # in official examples and release assets.
    if not result or result.returncode != 0:
        print("Attempting onnx2tf with replace.json (-prf) ...")
        cmd_prf = [
            'onnx2tf',
            '-i', input_path,
            '-o', args.tmpdir,
            '-coion',
            '-b', '1',
            '-ois',
            '-kat', 'images',
            '-kt', 'images',
            '-prf', param_json_path,
        ]
        result = run_cmd(cmd_prf)
        if result and result.returncode == 0:
            print("onnx2tf succeeded with -prf")

    if not result or result.returncode != 0:
        raise RuntimeError("onnx2tf conversion failed")

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
