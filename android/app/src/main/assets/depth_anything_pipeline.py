#!/usr/bin/env python3
"""
Depth Anything V2 Small (HuggingFace) — Full Room Mesh
=======================================================
Uses depth-anything/Depth-Anything-V2-Small-hf from HuggingFace transformers.
Produces one solid connected mesh of the entire room.
X/Y proportional to pixel position, Z = depth relief only.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import trimesh

IMAGE_PATH = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room.jpeg"
OUTPUT_GLB = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room_da2.glb"
OUTPUT_OBJ = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room_da2.obj"

# =====================================================================
# STEP 1: Load HuggingFace model
# =====================================================================
print("[1/4] Loading depth-anything/Depth-Anything-V2-Small-hf...")

model_name = "depth-anything/Depth-Anything-V2-Small-hf"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForDepthEstimation.from_pretrained(model_name)

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(DEVICE).eval()
print(f"    Device: {DEVICE}")

# =====================================================================
# STEP 2: Infer depth
# =====================================================================
print("[2/4] Running depth inference...")

img = Image.open(IMAGE_PATH).convert('RGB')
img_w, img_h = img.size
print(f"    Image: {img_w}x{img_h}")

inputs = processor(images=img, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

depth = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=(img_h, img_w),
    mode="bicubic",
    align_corners=False,
).squeeze().cpu().numpy()

# Normalize to metric range (1m near — 5m far)
d_min, d_max = depth.min(), depth.max()
depth_metric = 1.0 + (depth - d_min) / (d_max - d_min) * 4.0

print(f"    Raw depth range: [{d_min:.3f}, {d_max:.3f}]")
print(f"    Metric depth range: [{depth_metric.min():.2f}m, {depth_metric.max():.2f}m]")

# =====================================================================
# STEP 3: Build connected mesh — proportional XY, depth-only Z
# =====================================================================
print("[3/4] Building room mesh...")

img_np = np.array(img, dtype=np.uint8)

# Room physical size: map image proportionally to ~4.5m wide
room_width_m = 4.5
pixel_scale = room_width_m / img_w
room_height_m = img_h * pixel_scale
depth_range = depth_metric.max() - depth_metric.min()

print(f"    Physical size: {room_width_m:.1f}m x {room_height_m:.1f}m, depth relief: {depth_range:.2f}m")

step = 2
rows = np.arange(0, img_h, step)
cols = np.arange(0, img_w, step)
h_ds, w_ds = len(rows), len(cols)
print(f"    Grid: {w_ds}x{h_ds} = {w_ds * h_ds:,} vertices")

# X = pixel column mapped to meters (centered)
# Y = pixel row mapped to meters (centered, Y-up)
# Z = depth relief only (near objects pop out toward viewer)
vertices = np.zeros((h_ds * w_ds, 3), dtype=np.float32)
vertex_colors = np.zeros((h_ds * w_ds, 4), dtype=np.uint8)

d_max_val = depth_metric.max()

for ri, r in enumerate(rows):
    for ci, c in enumerate(cols):
        idx = ri * w_ds + ci
        d = depth_metric[r, c]
        vertices[idx] = [
            (c - img_w / 2.0) * pixel_scale,
            -(r - img_h / 2.0) * pixel_scale,
            -(d_max_val - d)  # near objects = negative Z (toward viewer), far = Z~0
        ]
        vertex_colors[idx] = [img_np[r, c, 0], img_np[r, c, 1], img_np[r, c, 2], 255]

# Triangulate — skip edges where depth jumps > 0.4m
faces = []
max_depth_jump = 0.4

for ri in range(h_ds - 1):
    for ci in range(w_ds - 1):
        i00 = ri * w_ds + ci
        i10 = ri * w_ds + (ci + 1)
        i01 = (ri + 1) * w_ds + ci
        i11 = (ri + 1) * w_ds + (ci + 1)

        d00 = depth_metric[rows[ri], cols[ci]]
        d10 = depth_metric[rows[ri], cols[ci + 1]]
        d01 = depth_metric[rows[ri + 1], cols[ci]]
        d11 = depth_metric[rows[ri + 1], cols[ci + 1]]

        if abs(d00 - d10) < max_depth_jump and abs(d00 - d01) < max_depth_jump:
            faces.append([i00, i10, i01])
        if abs(d10 - d11) < max_depth_jump and abs(d01 - d11) < max_depth_jump:
            faces.append([i10, i11, i01])

faces = np.array(faces, dtype=np.int32)
print(f"    Faces: {len(faces):,}")

# =====================================================================
# STEP 4: Export
# =====================================================================
print("[4/4] Exporting...")

mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    vertex_colors=vertex_colors,
    process=False
)

scene = trimesh.Scene()
scene.add_geometry(mesh, node_name='room')
scene.export(OUTPUT_GLB)
mesh.export(OUTPUT_OBJ)

print(f"    GLB: {OUTPUT_GLB}")
print(f"    OBJ: {OUTPUT_OBJ}")
print(f"\n[DONE] Room mesh: {len(vertices):,} verts, {len(faces):,} faces")
print(f"  Size X: ~{room_width_m:.1f}m | Size Y: ~{room_height_m:.1f}m | Size Z (relief): ~{depth_range:.1f}m")
