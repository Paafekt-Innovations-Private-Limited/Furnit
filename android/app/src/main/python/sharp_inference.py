"""
SHARP inference via PyTorch on Android (Chaquopy).

Same code that runs in 19s on Mac. On Android ARM CPU, expect ~2-5 min.
No model conversion needed -- uses the original .pt checkpoint directly.

Called from Kotlin via:
  Python.getInstance().getModule("sharp_inference").callAttr("run_inference", imagePath, modelPath)
"""

import time
import os
import struct
import numpy as np

# Global model cache -- loaded once, reused across calls
_model = None
_model_path = None


def warmup(model_path):
    """Pre-load model during app startup (called from Application.onCreate)."""
    global _model, _model_path
    if _model is not None and _model_path == model_path:
        return "already_loaded"

    import torch
    start = time.time()
    _model_path = model_path

    # Load checkpoint
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    # Try loading as full predictor first
    try:
        import sys
        # SHARP source should be bundled or accessible
        sharp_src = os.path.join(os.path.dirname(model_path), "ml-sharp-src")
        if os.path.exists(sharp_src):
            sys.path.insert(0, sharp_src)

        from sharp.models import PredictorParams, create_predictor
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()
        _model = predictor
        elapsed = time.time() - start
        return f"loaded_predictor_in_{elapsed:.1f}s"
    except Exception as e:
        # If SHARP source not available, try as torchscript or state_dict
        elapsed = time.time() - start
        return f"load_failed:{str(e)[:100]}"


def run_inference(image_path, model_path, output_dir):
    """
    Run SHARP inference on an image.

    Args:
        image_path: path to input image (JPEG/PNG)
        model_path: path to sharp .pt checkpoint
        output_dir: directory to write PLY output

    Returns:
        JSON string with results: ply_path, gaussian_count, room_dims, timing
    """
    import torch
    import json

    global _model

    total_start = time.time()
    timings = {}

    # Step 1: Load model (or use cached)
    load_start = time.time()
    if _model is None:
        warmup(model_path)
    if _model is None:
        return json.dumps({"error": "Failed to load model"})
    timings["model_load"] = round(time.time() - load_start, 2)

    # Step 2: Load and preprocess image
    preprocess_start = time.time()
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((1536, 1536), Image.BILINEAR)

    # Convert to tensor [1, 3, 1536, 1536] normalized to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    # HWC -> CHW
    img_chw = np.transpose(img_array, (2, 0, 1))
    image_tensor = torch.from_numpy(img_chw).unsqueeze(0)
    timings["preprocess"] = round(time.time() - preprocess_start, 2)

    # Step 3: Run inference
    infer_start = time.time()
    disparity_factor = torch.tensor([1.0])
    with torch.no_grad():
        result = _model(image_tensor, disparity_factor)
    timings["inference"] = round(time.time() - infer_start, 2)

    # Step 4: Extract Gaussian parameters
    extract_start = time.time()
    means = result.mean_vectors.squeeze(0).numpy()       # [N, 3]
    scales = result.singular_values.squeeze(0).numpy()    # [N, 3]
    rotations = result.quaternions.squeeze(0).numpy()     # [N, 4]
    colors = result.colors.squeeze(0).numpy()             # [N, 3]
    opacities = result.opacities.squeeze(0).numpy()       # [N] or [N, 1]
    if opacities.ndim == 1:
        opacities = opacities[:, np.newaxis]

    gaussian_count = means.shape[0]
    timings["extract"] = round(time.time() - extract_start, 2)

    # Step 5: Write PLY
    ply_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, "room.ply")
    classic_ply_path = os.path.join(output_dir, "room_classic.ply")

    write_gaussian_ply(ply_path, means, scales, rotations, opacities, colors)

    # Copy for classic viewer
    import shutil
    shutil.copy2(ply_path, classic_ply_path)

    timings["ply_write"] = round(time.time() - ply_start, 2)

    # Room bounds
    min_bounds = means.min(axis=0)
    max_bounds = means.max(axis=0)
    room_width = float(max_bounds[0] - min_bounds[0])
    room_height = float(max_bounds[1] - min_bounds[1])
    room_depth = float(max_bounds[2] - min_bounds[2])

    timings["total"] = round(time.time() - total_start, 2)

    return json.dumps({
        "ply_path": ply_path,
        "classic_ply_path": classic_ply_path,
        "gaussian_count": gaussian_count,
        "room_width": round(room_width, 4),
        "room_height": round(room_height, 4),
        "room_depth": round(room_depth, 4),
        "timings": timings,
    })


def write_gaussian_ply(path, means, scales, rotations, opacities, colors):
    """Write Gaussian PLY in standard 3DGS format."""
    n = means.shape[0]
    sh_c0 = 0.28209479177387814

    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
"""
    for i in range(45):
        header += f"property float f_rest_{i}\n"
    header += """property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

    zero_sh = b'\x00' * (45 * 4)

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))

        for i in range(n):
            # Position (flip Y and Z for viewer convention)
            f.write(struct.pack('<fff', means[i, 0], -means[i, 1], -means[i, 2]))

            # Normals (zero)
            f.write(struct.pack('<fff', 0.0, 0.0, 0.0))

            # SH DC coefficients from color
            r, g, b = colors[i, 0], colors[i, 1], colors[i, 2]
            f.write(struct.pack('<fff',
                (float(r) - 0.5) / sh_c0,
                (float(g) - 0.5) / sh_c0,
                (float(b) - 0.5) / sh_c0))

            # Higher order SH (zeros)
            f.write(zero_sh)

            # Opacity (logit)
            op = float(np.clip(opacities[i, 0], 1e-4, 1.0 - 1e-4))
            f.write(struct.pack('<f', np.log(op / (1.0 - op))))

            # Scale (log)
            for j in range(3):
                sv = float(max(scales[i, j], 0.001))
                f.write(struct.pack('<f', np.log(sv)))

            # Rotation (normalize quaternion)
            qw, qx, qy, qz = rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]
            mag = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            if mag > 1e-8:
                qw, qx, qy, qz = qw/mag, qx/mag, qy/mag, qz/mag
            else:
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            f.write(struct.pack('<ffff', float(qw), float(qx), float(qy), float(qz)))
