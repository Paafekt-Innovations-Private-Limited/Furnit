#!/usr/bin/env python3
"""
Point Cloud Pipeline — VLM Depth + Pixel Unprojection
=======================================================
Queries Qwen for depth constraints, generates a depth buffer from
perspective geometry, unprojects every pixel into 3D, exports as PLY.
No mesh faces = no texture stretching or mirroring.
"""

import json
import numpy as np
import ollama
from PIL import Image
from plyfile import PlyData, PlyElement

IMAGE_PATH = "/Users/al/Documents/tries01/Furnit/FurnitTests/room.jpeg"
OUTPUT_PLY = "/Users/al/Documents/tries01/Furnit/FurnitTests/room_pointcloud.ply"

# =====================================================================
# STEP 1: QUERY QWEN FOR DEPTH BOUNDARIES AND PERSPECTIVE SCALES
# =====================================================================
prompt = """
Act as an expert 3D scene reconstructor. Analyze this room image and output a clean, strict JSON schema representing the physical layout depth and structural constraints.

Your output must be raw JSON only, mapping these exact keys:
1. "camera_parameters": {
     "approx_focal_length_multiplier": float (typically between 0.7 and 1.2 based on lens field of view),
     "camera_height_meters": float
   }
2. "structural_depths": {
     "back_wall_distance_meters": float,
     "floor_level_y": float,
     "ceiling_level_y": float
   }
3. "major_assets": An array of objects containing only "label" (e.g., 'chair', 'curtains') and their calculated "center_distance_meters": float from the camera lens.

Do not include any conversational markdown wrapper text outside the raw JSON.
"""

print("[1/3] Extracting depth constraints from local Qwen-VL model...")
response = ollama.generate(model='qwen2.5vl:7b', prompt=prompt, images=[IMAGE_PATH])
raw_text = response['response'].strip()

if "```json" in raw_text:
    raw_text = raw_text.split("```json")[-1].split("```")[0].strip()
elif "```" in raw_text:
    raw_text = raw_text.split("```")[1].split("```")[0].strip()

data = json.loads(raw_text)
print("-> Depth constraints matrix parsed successfully.")
print(f"   Camera: {data.get('camera_parameters', {})}")
print(f"   Depths: {data.get('structural_depths', {})}")
print(f"   Assets: {len(data.get('major_assets', []))} detected")

# =====================================================================
# STEP 2: PIXEL UNPROJECTION & DEPTH MAP INFERENCE PIPELINE
# =====================================================================
print("[2/3] Generating depth buffer and unprojecting pixels...")
img = Image.open(IMAGE_PATH)
img_w, img_h = img.size
img_np = np.array(img)

cam_meta = data.get("camera_parameters", {})
depth_meta = data.get("structural_depths", {})

focal_length = max(img_w, img_h) * cam_meta.get("approx_focal_length_multiplier", 0.8)
cx, cy = img_w / 2.0, img_h / 2.0

# Generate pixel coordinate grids
u_coords, v_coords = np.meshgrid(np.arange(img_w), np.arange(img_h))

# Depth buffer from VLM constraints
max_depth = depth_meta.get("back_wall_distance_meters", 4.5)
floor_y = depth_meta.get("floor_level_y", -1.2)
ceiling_y = depth_meta.get("ceiling_level_y", 1.5)

z_buffer = np.zeros((img_h, img_w))

for v in range(img_h):
    normalized_v = (v - cy) / focal_length
    if normalized_v > 0.1:
        z_buffer[v, :] = np.clip(floor_y / normalized_v, 0.5, max_depth)
    elif normalized_v < -0.1:
        z_buffer[v, :] = np.clip(ceiling_y / normalized_v, 0.5, max_depth)
    else:
        z_buffer[v, :] = max_depth

# Override depth for foreground assets identified by VLM
for asset in data.get("major_assets", []):
    label = asset.get("label", "").lower()
    dist = asset.get("center_distance_meters", 2.0)
    if "chair" in label:
        z_buffer[int(img_h * 0.65):, int(img_w * 0.65):] = min(dist, 1.5)
    elif "curtain" in label:
        z_buffer[int(img_h * 0.3):int(img_h * 0.8), int(img_w * 0.2):int(img_w * 0.8)] = dist

# =====================================================================
# STEP 3: ASSEMBLE 3D POINT CLOUD
# =====================================================================
print("[3/3] Unprojecting to 3D point cloud...")

x_world = (u_coords - cx) * z_buffer / focal_length
y_world = -(v_coords - cy) * z_buffer / focal_length
z_world = z_buffer

points_3d = np.stack((x_world, y_world, z_world), axis=-1).reshape(-1, 3)
colors_rgb = img_np.reshape(-1, 3)

# Filter out invalid/edge points
valid_mask = (points_3d[:, 2] > 0.2) & (points_3d[:, 2] <= max_depth + 0.5)
points_cleaned = points_3d[valid_mask]
colors_cleaned = colors_rgb[valid_mask]

# Downsample for performance (every 2nd pixel)
step = 2
points_ds = points_cleaned[::step]
colors_ds = colors_cleaned[::step]

print(f"   Total points: {len(points_ds):,}")

# Statistical outlier removal (simple: remove points far from local mean)
from scipy.spatial import cKDTree

print("   Running outlier removal...")
tree = cKDTree(points_ds)
dists, _ = tree.query(points_ds, k=20)
mean_dists = dists.mean(axis=1)
threshold = mean_dists.mean() + 2.0 * mean_dists.std()
inlier_mask = mean_dists < threshold
points_final = points_ds[inlier_mask]
colors_final = colors_ds[inlier_mask]

print(f"   After cleanup: {len(points_final):,} points")

# Export as ASCII PLY with explicit uint8 color channels for max viewer compatibility
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

print(f"\n[COMPLETE] 3D Point Cloud exported:")
print(f"  {OUTPUT_PLY}")
print(f"  {len(points_final):,} colored points")
print(f"  No mesh faces = no texture stretching or mirroring")
