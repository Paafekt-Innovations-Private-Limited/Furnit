#!/usr/bin/env python3
"""
Perfect Pipeline — Projective Texture-Mapped Room
===================================================
Queries Qwen VLM for spatial layout, builds a pinhole camera model,
projects the real photo onto 3D box meshes with:
  - Corrected R_x rotation matrix
  - Inverted room normals (interior view)
  - Facing-angle back-face culling
  - Room envelope ONLY (no furniture boxes)
"""

import json
import numpy as np
import ollama
import trimesh
from PIL import Image

IMAGE_PATH = "/Users/al/Documents/tries01/Furnit/FurnitTests/room.jpeg"
OUTPUT_GLB = "/Users/al/Documents/tries01/Furnit/FurnitTests/perfect_room.glb"

# =====================================================================
# STEP 1: TALK TO QWEN VLM TO GET THE CALIBRATED COORDINATES
# =====================================================================
prompt = """
Act as a precise 3D spatial engineering parser. Analyze this room image and output a clean, strict JSON schema representing the physical layout. 

Specifications:
- All position [x,y,z] and scale [w,h,d] values must be in meters relative to the room center.
- Orientations [r_x, r_y, r_z] must be in degrees.

Your output must be raw JSON only, mapping these exact keys:
1. "room_structure": {"width": float, "height": float, "depth": float}
2. "assets": An array of objects, each containing: "id": string, "label": string, "position": [x,y,z], "scale": [w,h,d], "rotation": [rx,ry,rz].

Ensure values strictly conform to realistic physical proportions based on the photo perspective. Do not include any markdown syntax or chat wrappers outside the raw JSON.
"""

print("[1/3] Querying Qwen-VL via Ollama for camera-calibrated geometry...")
response = ollama.generate(model='qwen2.5vl:7b', prompt=prompt, images=[IMAGE_PATH])
raw_text = response['response'].strip()

if "```json" in raw_text:
    raw_text = raw_text.split("```json")[-1].split("```")[0].strip()
elif "```" in raw_text:
    raw_text = raw_text.split("```")[1].split("```")[0].strip()

data = json.loads(raw_text)
print("-> Successfully parsed spatial data matrix from Qwen.")

# =====================================================================
# STEP 2: DEFINE THE CAMERA PERSPECTIVE MATRIX (FIXED)
# =====================================================================
print("[2/3] Initializing virtual pinhole camera projection matrix...")
texture_img = Image.open(IMAGE_PATH)
img_w, img_h = texture_img.size

focal_length = max(img_w, img_h) * 0.8
K = np.array([
    [focal_length, 0, img_w / 2],
    [0, focal_length, img_h / 2],
    [0, 0, 1]
])

cam_pos = np.array([0.0, 0.2, 3.5])
cam_rot_x = np.radians(-10)

R_x = np.array([
    [1, 0, 0],
    [0, np.cos(cam_rot_x), -np.sin(cam_rot_x)],
    [0, np.sin(cam_rot_x), np.cos(cam_rot_x)]
])

camera_forward = np.dot(R_x.T, np.array([0, 0, -1]))


def project_texture_uv(mesh, transform_matrix):
    """Projects 3D vertices onto 2D image plane with facing-angle culling."""
    uvs = []

    world_normals = np.dot(transform_matrix[:3, :3], mesh.vertex_normals.T).T

    for i, vert in enumerate(mesh.vertices):
        v_homo = np.append(vert, 1.0)
        v_world = np.dot(transform_matrix, v_homo)[:3]

        normal = world_normals[i]
        dot_product = np.dot(normal, camera_forward)

        if dot_product > 0.0:
            uvs.append([0.0, 0.0])
            continue

        v_cam = np.dot(R_x, v_world - cam_pos)

        if v_cam[2] != 0:
            pixel = np.dot(K, v_cam)
            u = pixel[0] / pixel[2]
            v = pixel[1] / pixel[2]

            u_norm = np.clip(u / img_w, 0.0, 1.0)
            v_norm = np.clip(1.0 - (v / img_h), 0.0, 1.0)
            uvs.append([u_norm, v_norm])
        else:
            uvs.append([0.0, 0.0])

    return np.array(uvs)


# =====================================================================
# STEP 3: ROOM ENVELOPE ONLY (no furniture boxes)
# =====================================================================
print("[3/3] Building room envelope (no asset boxes)...")
scene = trimesh.Scene()
material = trimesh.visual.texture.SimpleMaterial(image=texture_img)

room = data["room_structure"]
room_mesh = trimesh.creation.box(extents=[room["width"], room["height"], room["depth"]])

room_mesh.invert()

room_transform = trimesh.transformations.identity_matrix()
room_uvs = project_texture_uv(room_mesh, room_transform)
room_mesh.visual = trimesh.visual.TextureVisuals(uv=room_uvs, material=material)
scene.add_geometry(room_mesh, node_name="room_envelope")

scene.export(OUTPUT_GLB)
print(f"\n[COMPLETE] Room envelope only (no boxes):")
print(f"  {OUTPUT_GLB}")
