#!/usr/bin/env python3
"""
Quantum Grounded Point Cloud — Full Photometric Pipeline
==========================================================
1. Sigmoid soft-max depth transition (no hard boundaries)
2. 2x2 bilinear super-sampling (anti-aliased colors)
3. Voxel grid + PCA normal estimation (crisp surfaces)
4. Chromatic decoupling mask (barcode stripe removal)
"""

import os
import re
import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter

IMAGE_PATH = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room.jpeg"
OUTPUT_PLY = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room_quantum.ply"
OUTPUT_GLB = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room_quantum.glb"

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
    clean = coord_string.replace("m", "").strip("[] ")
    try:
        return float(eval(clean))
    except Exception:
        return float(re.sub(r'[^0-9.-]', '', clean))


print("[1/5] Parsing quantum state coordinates...")
width = safe_eval_coordinate(re.search(r"WIDTH:\s*(.*)", qwen_quantum_output).group(1))
height = safe_eval_coordinate(re.search(r"HEIGHT:\s*(.*)", qwen_quantum_output).group(1))
depth = safe_eval_coordinate(re.search(r"DEPTH:\s*(.*)", qwen_quantum_output).group(1))

chair_block = re.search(r"Office Chair.*?STATE_COORDINATES:\s*\[(.*?)\]", qwen_quantum_output, re.DOTALL).group(1)
chair_xyz = [safe_eval_coordinate(p) for p in chair_block.split(",")]

curtain_block = re.search(r"Curtains.*?STATE_COORDINATES:\s*\[(.*?)\]", qwen_quantum_output, re.DOTALL).group(1)
curtain_xyz = [safe_eval_coordinate(p) for p in curtain_block.split(",")]

print(f"    Room: {width}m x {height}m x {depth}m")
print(f"    Chair Z={chair_xyz[2]}, Curtain Z={curtain_xyz[2]}")

# =====================================================================
# STEP 2: QUANTUM COHERENT DEPTH — SIGMOID SOFT-MAX TRANSITION
# =====================================================================
print("[2/5] Computing sigmoid depth wave function (no hard boundaries)...")
img = Image.open(IMAGE_PATH)
img_w, img_h = img.size
img_np = np.array(img, dtype=np.float32)

cx, cy = img_w / 2.0, img_h / 2.0
focal_length = max(img_w, img_h) * 0.8

u_coords, v_coords = np.meshgrid(np.arange(img_w), np.arange(img_h))

# Base depth: smooth floor slope + back wall
z_buffer = np.ones((img_h, img_w), dtype=np.float64) * depth

for v in range(img_h):
    normalized_v = (v - cy) / focal_length
    if normalized_v > 0.08:
        z_buffer[v, :] = np.clip((-height / 2) / normalized_v, 0.5, depth)

# Sigmoid soft-max transition: σ(x) = 1 / (1 + exp(-k*(x - x0)))
# Smoothly blends Z_chair → Z_curtain across horizontal axis
# No hard rectangular boundaries — continuous probability wave
z_chair = chair_xyz[2]
z_curtain = depth * 0.9
sigmoid_k = 8.0  # steepness of transition

# Horizontal sigmoid: left = curtain depth, right = chair depth
u_normalized = np.linspace(-1, 1, img_w).reshape(1, -1)
sigmoid_u = 1.0 / (1.0 + np.exp(-sigmoid_k * u_normalized))

# Vertical sigmoid: top = back wall, bottom = foreground
v_normalized = np.linspace(-1, 1, img_h).reshape(-1, 1)
sigmoid_v = 1.0 / (1.0 + np.exp(-sigmoid_k * (v_normalized - 0.2)))

# Blend depth field as continuous wave function
z_foreground = z_curtain * (1.0 - sigmoid_u) + z_chair * sigmoid_u
z_blended = depth * (1.0 - sigmoid_v) + z_foreground * sigmoid_v

# Take minimum with geometric floor slope
z_buffer = np.minimum(z_buffer, z_blended)

print(f"    Depth range: [{z_buffer.min():.2f}, {z_buffer.max():.2f}]m")

# =====================================================================
# STEP 3: BILINEAR SUPER-SAMPLING (2x2 ANTI-ALIASED COLORS)
# =====================================================================
print("[3/5] Applying 2x2 bilinear super-sampling for texture clarity...")

# Apply 2x2 averaging kernel to each color channel (anti-alias)
img_smooth = np.zeros_like(img_np)
for ch in range(3):
    img_smooth[:, :, ch] = uniform_filter(img_np[:, :, ch], size=2)

img_smooth = img_smooth.astype(np.uint8)

# =====================================================================
# STEP 4: UNPROJECT + CHROMATIC DECOUPLING MASK
# =====================================================================
print("[4/5] Unprojecting with chromatic decoupling filter...")

x_world = (u_coords - cx) * z_buffer / focal_length
y_world = -(v_coords - cy) * z_buffer / focal_length
z_world = z_buffer

points_3d = np.stack((x_world, y_world, z_world), axis=-1).reshape(-1, 3)
colors_rgb = img_smooth.reshape(-1, 3)

# Chromatic decoupling: luminosity filter destroys barcode stripes
gray = np.array(img.convert("L")).reshape(-1)
valid_mask = (
    (points_3d[:, 2] > 0.3) &
    (points_3d[:, 2] <= depth + 0.05) &
    (gray < 242) &  # Remove bright flash bleeding
    (gray > 15)     # Remove pure black artifacts
)
points_cleaned = points_3d[valid_mask]
colors_cleaned = colors_rgb[valid_mask]

# Downsample to manageable size
step = 2
points_ds = points_cleaned[::step]
colors_ds = colors_cleaned[::step]

# Statistical outlier removal (40 neighbors, 1.0 std)
print(f"    Points before outlier filter: {len(points_ds):,}")
tree = cKDTree(points_ds)
dists, _ = tree.query(points_ds, k=40)
mean_dists = dists.mean(axis=1)
threshold = mean_dists.mean() + 1.0 * mean_dists.std()
inlier_mask = mean_dists < threshold
points_filtered = points_ds[inlier_mask]
colors_filtered = colors_ds[inlier_mask]
print(f"    Points after outlier filter: {len(points_filtered):,}")

# Voxel downsampling (crisp uniform grid)
voxel_size = 0.008
voxel_indices = np.floor(points_filtered / voxel_size).astype(np.int32)
_, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
points_voxel = points_filtered[unique_idx]
colors_voxel = colors_filtered[unique_idx]
print(f"    Points after voxel grid: {len(points_voxel):,}")

# =====================================================================
# STEP 5: EIGENVECTOR NORMAL ESTIMATION + EXPORT
# =====================================================================
print("[5/5] Computing eigenvector normals and exporting...")

# PCA normal estimation on all points
normals = np.zeros_like(points_voxel)
tree_final = cKDTree(points_voxel)

# Batch normal estimation
for i in range(len(points_voxel)):
    _, neighbors = tree_final.query(points_voxel[i], k=min(35, len(points_voxel)))
    if len(neighbors) < 3:
        normals[i] = [0, 0, -1]
        continue
    local_pts = points_voxel[neighbors]
    centroid = local_pts.mean(axis=0)
    cov = np.cov((local_pts - centroid).T)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        # Orient toward camera (-Z direction)
        if normal[2] > 0:
            normal = -normal
        normals[i] = normal
    except Exception:
        normals[i] = [0, 0, -1]

    if i % 5000 == 0 and i > 0:
        print(f"    Normals computed: {i:,}/{len(points_voxel):,}")

# Export PLY with normals + colors
vertex_data = np.zeros(len(points_voxel), dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
])
vertex_data['x'] = points_voxel[:, 0].astype(np.float32)
vertex_data['y'] = points_voxel[:, 1].astype(np.float32)
vertex_data['z'] = points_voxel[:, 2].astype(np.float32)
vertex_data['nx'] = normals[:, 0].astype(np.float32)
vertex_data['ny'] = normals[:, 1].astype(np.float32)
vertex_data['nz'] = normals[:, 2].astype(np.float32)
vertex_data['red'] = colors_voxel[:, 0].astype(np.uint8)
vertex_data['green'] = colors_voxel[:, 1].astype(np.uint8)
vertex_data['blue'] = colors_voxel[:, 2].astype(np.uint8)

el = PlyElement.describe(vertex_data, 'vertex')
PlyData([el], text=True).write(OUTPUT_PLY)
print(f"    PLY saved: {OUTPUT_PLY}")

# Export GLB with larger filled quads (no grain)
print("    Building GLB mesh...")
n = len(points_voxel)
size = 0.007

all_verts = np.zeros((n * 4, 3))
all_faces = np.zeros((n * 2, 3), dtype=np.int32)
all_colors = np.zeros((n * 4, 4), dtype=np.uint8)

for i in range(n):
    base = i * 4
    p = points_voxel[i]
    c = colors_voxel[i]
    nm = normals[i]

    # Orient quad perpendicular to normal for proper billboard effect
    # Use a simplified tangent frame
    if abs(nm[1]) < 0.99:
        tangent = np.cross(nm, [0, 1, 0])
    else:
        tangent = np.cross(nm, [1, 0, 0])
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8) * size
    bitangent = np.cross(nm, tangent)
    bitangent = bitangent / (np.linalg.norm(bitangent) + 1e-8) * size

    all_verts[base] = p - tangent - bitangent
    all_verts[base + 1] = p + tangent - bitangent
    all_verts[base + 2] = p + tangent + bitangent
    all_verts[base + 3] = p - tangent + bitangent

    all_faces[i * 2] = [base, base + 1, base + 2]
    all_faces[i * 2 + 1] = [base, base + 2, base + 3]
    all_colors[base:base + 4] = [c[0], c[1], c[2], 255]

mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, vertex_colors=all_colors)
mesh.export(OUTPUT_GLB)
print(f"    GLB saved: {OUTPUT_GLB}")

print(f"\n[COMPLETE] Quantum-grounded photometric room:")
print(f"  PLY: {OUTPUT_PLY}")
print(f"  GLB: {OUTPUT_GLB}")
print(f"  {len(points_voxel):,} points | Sigmoid depth | Bilinear AA | PCA normals | Billboard quads")
