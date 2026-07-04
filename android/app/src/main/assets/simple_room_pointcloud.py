#!/usr/bin/env python3
"""
Simple Room Point Cloud — No Segmentation
============================================
Pure perspective unprojection: every pixel gets a depth based on its
vertical position (perspective geometry), nothing else. No chair/curtain
masks, no region identification. The whole room renders as one continuous
surface — exactly what the camera saw.
"""

import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter

IMAGE_PATH = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room.jpeg"
OUTPUT_PLY = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room_simple.ply"
OUTPUT_GLB = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room_simple.glb"
OUTPUT_OBJ = "/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room_simple.obj"

ROOM_DEPTH = 3.5  # how deep the room is from camera

print("[1/3] Loading image and computing pure perspective depth...")
img = Image.open(IMAGE_PATH)
img_w, img_h = img.size
img_np = np.array(img, dtype=np.float32)

cx, cy = img_w / 2.0, img_h / 2.0
focal_length = max(img_w, img_h) * 0.85

u_coords, v_coords = np.meshgrid(np.arange(img_w), np.arange(img_h))

# Pure flat depth — entire image at Z=0 (aligned to XY plane)
# Viewer's front view will show it perfectly straight
z_buffer = np.zeros((img_h, img_w), dtype=np.float64)

print(f"    Image: {img_w}x{img_h}")
print(f"    Depth range: [{z_buffer.min():.2f}, {z_buffer.max():.2f}]m")

# =====================================================================
# STEP 2: UNPROJECT ALL PIXELS (no filtering, no segmentation)
# =====================================================================
print("[2/3] Unprojecting entire image as one surface...")

# 2x2 bilinear anti-alias
img_smooth = np.zeros_like(img_np)
for ch in range(3):
    img_smooth[:, :, ch] = uniform_filter(img_np[:, :, ch], size=2)
img_smooth = img_smooth.astype(np.uint8)

# Direct pixel-to-meter mapping (no pinhole projection, no tilt)
# Each pixel maps to a fixed XY position on a flat plane
meters_per_pixel = 3.0 / max(img_w, img_h)  # 3m across the longest dimension

x_world = (u_coords - cx) * meters_per_pixel
y_world = -(v_coords - cy) * meters_per_pixel  # flip Y so top of image = +Y
z_world = z_buffer  # all zeros = perfectly flat

points_3d = np.stack((x_world, y_world, z_world), axis=-1).reshape(-1, 3)
colors_rgb = img_smooth.reshape(-1, 3)

# No filtering — keep every pixel
points_cleaned = points_3d
colors_cleaned = img_smooth.reshape(-1, 3)

# Downsample for reasonable file size
step = 2
points_ds = points_cleaned[::step]
colors_ds = colors_cleaned[::step]

# Voxel grid for uniform density
voxel_size = 0.005
voxel_indices = np.floor(points_ds / voxel_size).astype(np.int32)
_, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
points_final = points_ds[unique_idx]
colors_final = colors_ds[unique_idx]

print(f"    Total points: {len(points_final):,}")

# =====================================================================
# STEP 3: EXPORT
# =====================================================================
print("[3/3] Exporting PLY + GLB + OBJ...")

# PLY
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

# GLB + OBJ with quads
n = len(points_final)
size = 0.005

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

mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, vertex_colors=all_colors, process=False)

# GLB export: trimesh auto-handles Y-up conversion
scene = trimesh.Scene()
scene.add_geometry(mesh, node_name='room')
scene.export(OUTPUT_GLB)

# OBJ export: apply -90° X rotation to match GLB's Y-up convention
# Without this, OBJ stays Z-up and appears rotated differently
transform_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
obj_mesh = mesh.copy()
obj_mesh.apply_transform(transform_matrix)
obj_mesh.export(OUTPUT_OBJ)

print(f"\n[COMPLETE] Whole room as one surface:")
print(f"  PLY: {OUTPUT_PLY}")
print(f"  GLB: {OUTPUT_GLB}")
print(f"  OBJ: {OUTPUT_OBJ}")
print(f"  {len(points_final):,} points — no segmentation, no object masks")
print(f"  Just the photo projected into 3D with mild perspective curve")
