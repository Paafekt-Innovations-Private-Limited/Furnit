#!/usr/bin/env python3
"""
Test SHARP .pte model inference in Python (on Mac/Linux, NOT Android).

This validates the exported ExecuTorch model produces correct output
before pushing to Android device.

Usage:
  python test_sharp_pte.py sharp_full_fp32.pte
  python test_sharp_pte.py sharp_full_fp32.pte --image /path/to/room.jpg
  python test_sharp_pte.py sharp_full_fp32.pte --compare-pytorch  # Compare with PyTorch
"""

import sys
import argparse
import time
from pathlib import Path

import torch
import numpy as np

SHARP_SRC = Path("/Users/al/Documents/tries01/Furnit/android/third_party/ml-sharp/src")
sys.path.insert(0, str(SHARP_SRC))

MODEL_WEIGHTS = Path("/Users/al/Documents/tries01/Furnit/android/sharp_litert_models/sharp_2572gikvuh.pt")


def load_image(image_path: str, size: int = 1536) -> torch.Tensor:
    """Load and preprocess an image to [1, 3, size, size] in [0, 1]."""
    from PIL import Image
    import torchvision.transforms as T

    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),  # -> [3, H, W] in [0, 1]
    ])
    return transform(img).unsqueeze(0)  # [1, 3, H, W]


def run_pytorch_reference(image_tensor: torch.Tensor):
    """Run PyTorch reference inference for comparison."""
    from sharp.models import PredictorParams, create_predictor

    print("Loading PyTorch reference model...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    disparity_factor = torch.tensor([1.0])

    print("Running PyTorch inference...")
    start = time.time()
    with torch.no_grad():
        result = predictor(image_tensor, disparity_factor)
    pytorch_time = time.time() - start

    means = result.mean_vectors.squeeze(0)
    scales = result.singular_values.squeeze(0)
    rotations = result.quaternions.squeeze(0)
    colors = result.colors.squeeze(0)
    opacities = result.opacities.squeeze(0)
    if opacities.dim() == 1:
        opacities = opacities.unsqueeze(-1)

    params = torch.cat([means, scales, rotations, opacities, colors], dim=-1)
    print(f"PyTorch: {params.shape[0]} Gaussians in {pytorch_time:.1f}s")
    return params.numpy()


def run_executorch_inference(pte_path: str, image_tensor: torch.Tensor):
    """Run ExecuTorch inference using the Python runtime."""
    try:
        from executorch.runtime import Runtime
        runtime = Runtime.get()
        print(f"Loading ExecuTorch model: {pte_path}")
        loadStart = time.time()
        program = runtime.load_program(Path(pte_path))
        forwardMethod = program.load_method("forward")
        loadTime = time.time() - loadStart
        print(f"Model loaded in {loadTime:.1f}s")

        print("Running ExecuTorch inference...")
        inferStart = time.time()
        outputTensors = forwardMethod.execute([image_tensor])
        inferTime = time.time() - inferStart

        outputTensor = outputTensors[0]
        if isinstance(outputTensor, torch.Tensor):
            resultArray = outputTensor.numpy()
        else:
            resultArray = np.array(outputTensor)

        print(f"ExecuTorch: {resultArray.shape[0]} Gaussians in {inferTime:.1f}s")
        return resultArray
    except Exception as e:
        print(f"ExecuTorch runtime error: {e}")
        print(f"\nPush to device instead:")
        print(f"  adb push {pte_path} /sdcard/Android/data/com.furnit.android/files/models/")
        return None


def compare_outputs(pytorch_output, et_output):
    """Compare PyTorch and ExecuTorch outputs."""
    print("\n" + "=" * 60)
    print("Comparison: PyTorch vs ExecuTorch")
    print("=" * 60)

    if pytorch_output is None or et_output is None:
        print("Cannot compare - missing output")
        return

    n_pt = pytorch_output.shape[0]
    n_et = et_output.shape[0]
    print(f"Gaussian count: PyTorch={n_pt}, ExecuTorch={n_et}")

    if n_pt != n_et:
        print("WARNING: Different Gaussian counts!")
        n = min(n_pt, n_et)
    else:
        n = n_pt

    # Compare per-field
    fields = [
        ("Position X", 0), ("Position Y", 1), ("Position Z", 2),
        ("Scale X", 3), ("Scale Y", 4), ("Scale Z", 5),
        ("Rot W", 6), ("Rot X", 7), ("Rot Y", 8), ("Rot Z", 9),
        ("Opacity", 10),
        ("Color R", 11), ("Color G", 12), ("Color B", 13),
    ]

    print(f"\n{'Field':<15} {'Max Diff':>10} {'Mean Diff':>10} {'RMSE':>10}")
    print("-" * 50)

    for name, idx in fields:
        pt_vals = pytorch_output[:n, idx]
        et_vals = et_output[:n, idx]
        diff = np.abs(pt_vals - et_vals)
        max_diff = diff.max()
        mean_diff = diff.mean()
        rmse = np.sqrt((diff ** 2).mean())
        status = "OK" if max_diff < 0.01 else "WARN" if max_diff < 0.1 else "BAD"
        print(f"{name:<15} {max_diff:>10.6f} {mean_diff:>10.6f} {rmse:>10.6f}  [{status}]")

    total_diff = np.abs(pytorch_output[:n] - et_output[:n]).mean()
    print(f"\nOverall mean absolute difference: {total_diff:.6f}")
    if total_diff < 0.001:
        print("RESULT: Excellent match")
    elif total_diff < 0.01:
        print("RESULT: Good match (minor quantization differences)")
    elif total_diff < 0.1:
        print("RESULT: Acceptable (noticeable but usable)")
    else:
        print("RESULT: POOR - significant differences, investigate!")


def write_test_ply(params, output_path: str):
    """Write a simple PLY file from Gaussian params for visual validation."""
    import struct

    n = params.shape[0]
    print(f"\nWriting test PLY: {n} Gaussians to {output_path}")

    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n):
            x, y, z = params[i, 0], params[i, 1], params[i, 2]
            r = int(np.clip(params[i, 11] * 255, 0, 255))
            g = int(np.clip(params[i, 12] * 255, 0, 255))
            b = int(np.clip(params[i, 13] * 255, 0, 255))
            f.write(struct.pack('<fff', x, y, z))
            f.write(struct.pack('BBB', r, g, b))

    print(f"PLY written: {Path(output_path).stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Test SHARP .pte model")
    parser.add_argument("pte_path", nargs="?", help="Path to .pte model")
    parser.add_argument("--image", help="Input image (default: random noise)")
    parser.add_argument("--compare-pytorch", action="store_true", help="Compare with PyTorch reference")
    parser.add_argument("--write-ply", help="Write test PLY file")
    args = parser.parse_args()

    # Load or generate input
    if args.image:
        image_tensor = load_image(args.image)
        print(f"Loaded image: {args.image} -> {image_tensor.shape}")
    else:
        image_tensor = torch.randn(1, 3, 1536, 1536).clamp(0, 1)
        print("Using random noise input (no --image specified)")

    # PyTorch reference
    pytorch_output = None
    if args.compare_pytorch or args.pte_path is None:
        pytorch_output = run_pytorch_reference(image_tensor)
        if args.write_ply and pytorch_output is not None:
            write_test_ply(pytorch_output, args.write_ply.replace(".ply", "_pytorch.ply"))

    # ExecuTorch inference
    et_output = None
    if args.pte_path:
        et_output = run_executorch_inference(args.pte_path, image_tensor)
        if args.write_ply and et_output is not None:
            write_test_ply(et_output, args.write_ply.replace(".ply", "_executorch.ply"))

    # Compare
    if pytorch_output is not None and et_output is not None:
        compare_outputs(pytorch_output, et_output)


if __name__ == "__main__":
    main()
