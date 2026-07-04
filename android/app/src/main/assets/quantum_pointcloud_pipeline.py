#!/usr/bin/env python3
"""
Quantum Point Cloud Pipeline
==============================
Parses symbolic quantum metadata (fractional math expressions like 4.5/2),
evaluates them into real 3D floats, generates a depth-mapped point cloud
from the room photo, applies brightness + outlier filters, exports PLY + GLB.
"""

import os
import re
import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

IMAGE_PATH = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room.jpeg"
OUTPUT_PLY = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room_perfect.ply"
OUTPUT_GLB = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room_perfect.glb"

# Raw quantum text output from Qwen
qwen_quantum_output = """
1) GLOBAL TOPOLOGICAL STATE (ROOM SHELL):
- HEIGHT: 3.0 m
- WIDTH: 4.5 m
- DEPTH: 2.8 m
- CALIBRATION_OPERATOR: Ceiling fan

2) ENTANGLED SUBSYSTEMS:
- SUBSYSTEM_LABEL: Office Chair
- EIGENVALUES_SCALE: [1.0, 1.0, 1.0]
- STATE_COORDINATES: [4.5/2, 3.0/2 - 0.7, 2.8/2]
- CAUSAL_ANCHOR: floor

- SUBSYSTEM_LABEL: Curtains
- EIGENVALUES_SCALE: [1.0, 1.0, 1.0]
- STATE_COORDINATES: [4.5/2 - 0.75, 3.0/2 + 0.75, 2.8/2]
- CAUSAL_ANCHOR: wall
"""


def safe_eval_coordinate(coord_string):
    """Evaluates math expressions like '4.5/2 - 0.7' into floats."""
    clean = coord_string.replace("m", "").strip("[] ")
    try:
        return float(eval(clean))
    except Exception:
        return float(re.sub(r'[^0-9.-]', '', clean))


print("[1/3] Parsing symbolic quantum metadata vectors...")

width = safe_eval_coordinate(re.search(r"WIDTH:\s*(.*)", qwen_quantum_output).group(1))
height = safe_eval_coordinate(re.search(r"HEIGHT:\s*(.*)", qwen_quantum_output).group(1))
depth = safe_eval_coordinate(re.search(r"DEPTH:\s*(.*)", qwen_quantum_output).group(1))

print(f"    Room: Width={width}m, Height={height}m, Depth={depth}m")

# Extract chair coordinates
chair_block = re.search(r"Office Chair.*?STATE_COORDINATES:\s*\[(.*?)\]", qwen_quantum_output, re.DOTALL).group(1)
chair_xyz = [safe_eval_coordinate(p) for p in chair_block.split(",")]
print(f"    Chair coordinates: {chair_xyz}")

# Extract curtain coordinates
curtain_block = re.search(r"Curtains.*?STATE_COORDINATES:\s*\[(.*?)\]", qwen_quantum_output, re.DOTALL).group(1)
curtain_xyz = [safe_eval_coordinate(p) for p in curtain_block.split(",")]
print(f"    Curtain coordinates: {curtain_xyz}")

# =====================================================================
# STEP 2: PIXEL UNPROJECTION WITH SMOOTH DEPTH
# =====================================================================
print("[2/3] Executing high-density pixel projection...")
img = Image.open(IMAGE_PATH)
img_w, img_h = img.size
img_np = np.array(img)
gray_np = np.array(img.convert("L")).reshape(-1)

cx, cy = img_w / 2.0, img_h / 2.0
focal_length = max(img_w, img_h) * 0.8

u_coords, v_coords = np.meshgrid(np.arange(img_w), np.arange(img_h))
z_buffer = np.ones((img_h, img_w)) * depth

# Smooth floor slope
for v in range(img_h):
    normalized_v = (v - cy) / focal_length
    if normalized_v > 0.08:
        z_buffer[v, :] = np.clip((-height / 2) / normalized_v, 0.5, depth)

# Lock chair depth to its evaluated Z eigenvalue
chair_v_start = int(img_h * 0.6)
chair_u_start = int(img_w * 0.6)
z_buffer[chair_v_start:, chair_u_start:] = np.minimum(
    z_buffer[chair_v_start:, chair_u_start:], chair_xyz[2]
)

# Lock curtain region
cv_s, cv_e = int(img_h * 0.25), int(img_h * 0.85)
cu_s, cu_e = int(img_w * 0.15), int(img_w * 0.85)
z_buffer[cv_s:cv_e, cu_s:cu_e] = np.minimum(
    z_buffer[cv_s:cv_e, cu_s:cu_e], curtain_xyz[2]
)

# =====================================================================
# STEP 3: ASSEMBLE + FILTER + EXPORT
# =====================================================================
print("[3/3] Assembling vector space, pruning noise, exporting...")

x_world = (u_coords - cx) * z_buffer / focal_length
y_world = -(v_coords - cy) * z_buffer / focal_length
z_world = z_buffer

points_3d = np.stack((x_world, y_world, z_world), axis=-1).reshape(-1, 3)
colors_rgb = img_np.reshape(-1, 3)

# Filter 1: Remove bright washed-out pixels (kills barcode stripes)
valid_mask = (points_3d[:, 2] <= depth + 0.1) & (points_3d[:, 2] > 0.2) & (gray_np < 242)
points_cleaned = points_3d[valid_mask]
colors_cleaned = colors_rgb[valid_mask]

# Downsample (every 3rd point)
step = 3
points_ds = points_cleaned[::step]
colors_ds = colors_cleaned[::step]

# Filter 2: Statistical outlier removal (40 neighbors, 1.0 std — aggressive)
print(f"    Points before outlier filter: {len(points_ds):,}")
tree = cKDTree(points_ds)
dists, _ = tree.query(points_ds, k=40)
mean_dists = dists.mean(axis=1)
threshold = mean_dists.mean() + 1.0 * mean_dists.std()
inlier_mask = mean_dists < threshold
points_final = points_ds[inlier_mask]
colors_final = colors_ds[inlier_mask]
print(f"    Points after outlier filter: {len(points_final):,}")

# Export PLY (ASCII, uint8 colors)
vertex_data = np.zeros(len(points_final), dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
])
vertex_data['x'] = points_final[:, 0].astype(np.float32)
vertex_data['y'] = points_final[:, 1].astype(np.float32)
vertex_data['z'] = points_final[:, 2].astype(np.float32)
vertex_data['red'] = colors_final[:, 0].astype(np.uint8)
vertex_data['green'] = colors_final[:, 1].astype(np.uint8)
vertex_data['blue'] = colors_final[:, 2].astype(np.uint8)

el = PlyElement.describe(vertex_data, 'vertex')
PlyData([el], text=True).write(OUTPUT_PLY)
print(f"    PLY saved: {OUTPUT_PLY}")

# Export GLB (tiny quads for universal viewer support)
print("    Converting to GLB...")
n = len(points_final)
size = 0.004

all_verts = np.zeros((n * 4, 3))
all_faces = np.zeros((n * 2, 3), dtype=np.int32)
all_colors = np.zeros((n * 4, 4), dtype=np.uint8)

for i in range(n):
    base = i * 4
    p = points_final[i]
    c = colors_final[i]
    all_verts[base] = [p[0] - size, p[1] - size, p[2]]
    all_verts[base + 1] = [p[0] + size, p[1] - size, p[2]]
    all_verts[base + 2] = [p[0] + size, p[1] + size, p[2]]
    all_verts[base + 3] = [p[0] - size, p[1] + size, p[2]]
    all_faces[i * 2] = [base, base + 1, base + 2]
    all_faces[i * 2 + 1] = [base, base + 2, base + 3]
    all_colors[base:base + 4] = [c[0], c[1], c[2], 255]

mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, vertex_colors=all_colors)
mesh.export(OUTPUT_GLB)
print(f"    GLB saved: {OUTPUT_GLB}")

print(f"\n[PIPELINE SUCCESS] Clean 3D room model generated:")
print(f"  PLY: {OUTPUT_PLY}")
print(f"  GLB: {OUTPUT_GLB}")
print(f"  {len(points_final):,} points, math-locked boundaries, no distortion")
