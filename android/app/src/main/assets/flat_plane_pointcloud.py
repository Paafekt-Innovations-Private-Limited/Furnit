#!/usr/bin/env python3
"""
Flat-Plane Point Cloud — Piecewise Depth + Soft Edges
======================================================
Fixes the warp: uses constant Z per region (flat planes) with sigmoid
transitions ONLY at the narrow boundaries between objects.
"""

import re
import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter, gaussian_filter

IMAGE_PATH = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room.jpeg"
OUTPUT_PLY = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room_quantum.ply"
OUTPUT_GLB = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room_quantum.glb"

# Parsed quantum coordinates
ROOM_W, ROOM_H, ROOM_D = 4.5, 3.0, 2.8
CHAIR_Z = 1.4   # 2.8/2
CURTAIN_Z = 2.5  # on the back wall plane (not 1.4 — curtains hang ON the wall)

print("[1/4] Loading image and building flat-plane depth map...")
img = Image.open(IMAGE_PATH)
img_w, img_h = img.size
img_np = np.array(img, dtype=np.float32)

cx, cy = img_w / 2.0, img_h / 2.0
focal_length = max(img_w, img_h) * 0.8

u_coords, v_coords = np.meshgrid(np.arange(img_w), np.arange(img_h))

# =====================================================================
# PIECEWISE FLAT DEPTH MAP — each region gets a constant Z
# =====================================================================

# Start with everything at back wall depth (flat)
z_buffer = np.ones((img_h, img_w), dtype=np.float64) * ROOM_D

# Floor plane: recedes from camera toward back wall
# The floor is at y=0, so pixels below center see the floor receding
for v in range(img_h):
    normalized_v = (v - cy) / focal_length
    if normalized_v > 0.1:
        # Floor recedes linearly (not curved)
        floor_z = np.clip((-ROOM_H / 2) / normalized_v, 0.5, ROOM_D)
        z_buffer[v, :] = floor_z

# Define object regions as MASKS with constant depth
# Chair region: bottom-right quadrant
chair_mask = np.zeros((img_h, img_w), dtype=bool)
chair_mask[int(img_h * 0.55):, int(img_w * 0.55):] = True

# Curtain region: central band
curtain_mask = np.zeros((img_h, img_w), dtype=bool)
curtain_mask[int(img_h * 0.15):int(img_h * 0.75), int(img_w * 0.1):int(img_w * 0.9)] = True

# Ceiling region: top portion stays at back wall depth (flat)
ceiling_mask = np.zeros((img_h, img_w), dtype=bool)
ceiling_mask[:int(img_h * 0.25), :] = True

# Apply FLAT depths to each region
z_buffer[ceiling_mask] = ROOM_D  # ceiling at back wall
z_buffer[curtain_mask & ~chair_mask] = CURTAIN_Z  # curtains on wall
z_buffer[chair_mask] = CHAIR_Z  # chair in foreground

# SOFT EDGES ONLY: apply a small Gaussian blur to the depth map
# This smooths ONLY the 5-10px transitions between flat regions
z_buffer = gaussian_filter(z_buffer, sigma=3.0)

print(f"    Depth range: [{z_buffer.min():.2f}, {z_buffer.max():.2f}]m")
print(f"    Back wall: flat at {ROOM_D}m")
print(f"    Curtain plane: flat at {CURTAIN_Z}m")
print(f"    Chair plane: flat at {CHAIR_Z}m")

# =====================================================================
# STEP 2: BILINEAR SUPER-SAMPLING + UNPROJECT
# =====================================================================
print("[2/4] Super-sampling colors and unprojecting...")

# 2x2 bilinear anti-alias
img_smooth = np.zeros_like(img_np)
for ch in range(3):
    img_smooth[:, :, ch] = uniform_filter(img_np[:, :, ch], size=2)
img_smooth = img_smooth.astype(np.uint8)

# Unproject all pixels to 3D
x_world = (u_coords - cx) * z_buffer / focal_length
y_world = -(v_coords - cy) * z_buffer / focal_length
z_world = z_buffer

points_3d = np.stack((x_world, y_world, z_world), axis=-1).reshape(-1, 3)
colors_rgb = img_smooth.reshape(-1, 3)
gray = np.array(img.convert("L")).reshape(-1)

# =====================================================================
# STEP 3: CHROMATIC FILTER + OUTLIER REMOVAL + VOXEL GRID
# =====================================================================
print("[3/4] Filtering and voxelizing...")

# Chromatic decoupling: remove bright/dark extremes
valid_mask = (
    (points_3d[:, 2] > 0.3) &
    (points_3d[:, 2] <= ROOM_D + 0.05) &
    (gray < 245) &
    (gray > 12)
)
points_cleaned = points_3d[valid_mask]
colors_cleaned = colors_rgb[valid_mask]

# Downsample
step = 2
points_ds = points_cleaned[::step]
colors_ds = colors_cleaned[::step]

# Outlier removal
print(f"    Points before filter: {len(points_ds):,}")
tree = cKDTree(points_ds)
dists, _ = tree.query(points_ds, k=30)
mean_dists = dists.mean(axis=1)
threshold = mean_dists.mean() + 1.2 * mean_dists.std()
inlier_mask = mean_dists < threshold
points_filtered = points_ds[inlier_mask]
colors_filtered = colors_ds[inlier_mask]

# Voxel grid (fine enough to preserve detail)
voxel_size = 0.006
voxel_indices = np.floor(points_filtered / voxel_size).astype(np.int32)
_, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
points_voxel = points_filtered[unique_idx]
colors_voxel = colors_filtered[unique_idx]
print(f"    Points after voxel grid: {len(points_voxel):,}")

# =====================================================================
# STEP 4: EXPORT PLY + GLB
# =====================================================================
print("[4/4] Exporting...")

# PLY
vertex_data = np.zeros(len(points_voxel), dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
])
vertex_data['x'] = points_voxel[:, 0].astype(np.float32)
vertex_data['y'] = points_voxel[:, 1].astype(np.float32)
vertex_data['z'] = points_voxel[:, 2].astype(np.float32)
vertex_data['red'] = colors_voxel[:, 0].astype(np.uint8)
vertex_data['green'] = colors_voxel[:, 1].astype(np.uint8)
vertex_data['blue'] = colors_voxel[:, 2].astype(np.uint8)

el = PlyElement.describe(vertex_data, 'vertex')
PlyData([el], text=True).write(OUTPUT_PLY)
print(f"    PLY: {OUTPUT_PLY}")

# GLB with flat billboard quads
print("    Building GLB...")
n = len(points_voxel)
size = 0.006

all_verts = np.zeros((n * 4, 3))
all_faces = np.zeros((n * 2, 3), dtype=np.int32)
all_colors = np.zeros((n * 4, 4), dtype=np.uint8)

for i in range(n):
    base = i * 4
    p = points_voxel[i]
    c = colors_voxel[i]
    all_verts[base] = [p[0] - size, p[1] - size, p[2]]
    all_verts[base + 1] = [p[0] + size, p[1] - size, p[2]]
    all_verts[base + 2] = [p[0] + size, p[1] + size, p[2]]
    all_verts[base + 3] = [p[0] - size, p[1] + size, p[2]]
    all_faces[i * 2] = [base, base + 1, base + 2]
    all_faces[i * 2 + 1] = [base, base + 2, base + 3]
    all_colors[base:base + 4] = [c[0], c[1], c[2], 255]

mesh = trimesh.Trimesh(
    vertices=all_verts,
    faces=all_faces,
    vertex_colors=all_colors,
    process=False
)

scene = trimesh.Scene()
scene.add_geometry(mesh, node_name='room')
scene.export(OUTPUT_GLB)
mesh.export(OUTPUT_GLB.replace('.glb', '.obj'))

print(f"    GLB: {OUTPUT_GLB}")
print(f"    OBJ: {OUTPUT_GLB.replace('.glb', '.obj')}")

print(f"\n[COMPLETE] Flat-plane room (no warp):")
print(f"  {len(points_voxel):,} points")
print(f"  Surfaces are FLAT — no cylindrical bending")
print(f"  Soft edges only at region transitions (3px Gaussian)")
