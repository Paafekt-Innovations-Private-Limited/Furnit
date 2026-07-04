#!/usr/bin/env python3
"""
Perfect Pipeline v2 — Clean Room with VLM Color Space
======================================================
Single Qwen call for layout + structural colors, extended walls with
solid VLM-inferred colors, filtered assets (no small boxes).
"""

import json
import numpy as np
import ollama
import trimesh
from PIL import Image

IMAGE_PATH = "/Users/al/Documents/tries01/Furnit/FurnitTests/room.jpeg"
OUTPUT_GLB = "/Users/al/Documents/tries01/Furnit/FurnitTests/perfect_room_v2.glb"

# =====================================================================
# STEP 1: RUN THE EXPERT SPATIAL ANALYSIS VIA LOCAL QWEN VLM
# =====================================================================
prompt = """
Act as a precise 3D spatial engineering parser and color-space analyst. Analyze this room image and output a clean, strict JSON schema representing the physical layout.

Specifications:
- All position [x,y,z] and scale [w,h,d] values must be in meters relative to the room center.
- Orientations [r_x, r_y, r_z] must be in degrees.
- Extract the normalized RGB color values [R, G, B] (from 0.0 to 1.0) for the structural paint/surface tones.

Your output must be raw JSON only, mapping these exact keys:
1. "room_structure": {
     "width": float, 
     "height": float, 
     "depth": float,
     "ceiling_rgb": [r, g, b],
     "floor_rgb": [r, g, b],
     "wall_rgb": [r, g, b]
   }
2. "assets": An array of objects, each containing: "id": string, "label": string, "position": [x,y,z], "scale": [w,h,d], "rotation": [rx,ry,rz].

Isolate only major structural furniture elements (e.g., Office Chair, Curtains). Do not include any conversational markdown syntax outside the JSON block.
"""

print("[1/3] Querying local Qwen-VL model for 3D layout bounds and structural color space...")
response = ollama.generate(model='qwen2.5vl:7b', prompt=prompt, images=[IMAGE_PATH])
raw_text = response['response'].strip()

if "```json" in raw_text:
    raw_text = raw_text.split("```json")[-1].split("```")[0].strip()
elif "```" in raw_text:
    raw_text = raw_text.split("```")[1].split("```")[0].strip()

data = json.loads(raw_text)
print("-> JSON compiled successfully by Qwen.")

# =====================================================================
# STEP 2: CAM CAMERA TRANSFORMS & STRUCTURAL COLOR MAPPING
# =====================================================================
print("[2/3] Setting camera viewport matrix to room center coordinates...")
texture_img = Image.open(IMAGE_PATH)
img_w, img_h = texture_img.size

room_meta = data["room_structure"]
ceiling_color = room_meta.get("ceiling_rgb", [0.9, 0.9, 0.9])
floor_color = room_meta.get("floor_rgb", [0.7, 0.7, 0.7])
wall_color = room_meta.get("wall_rgb", [0.8, 0.8, 0.8])

focal_length = max(img_w, img_h) * 0.8
K = np.array([
    [focal_length, 0, img_w / 2],
    [0, focal_length, img_h / 2],
    [0, 0, 1]
])

cam_pos = np.array([0.0, 0.0, 0.0])
cam_rot_x = np.radians(0)

# Corrected R_x matrix (identity when rotation is 0)
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(cam_rot_x), -np.sin(cam_rot_x)],
    [0, np.sin(cam_rot_x), np.cos(cam_rot_x)]
])

camera_forward = np.dot(R_x.T, np.array([0, 0, -1]))


def project_texture_uv(mesh, transform_matrix):
    uvs = []
    world_normals = np.dot(transform_matrix[:3, :3], mesh.vertex_normals.T).T
    for i, vert in enumerate(mesh.vertices):
        v_homo = np.append(vert, 1.0)
        v_world = np.dot(transform_matrix, v_homo)[:3]

        normal = world_normals[i]
        if np.dot(normal, camera_forward) > 0.0:
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
# STEP 3: CONSOLIDATED MESH COMPILATION LAYER
# =====================================================================
print("[3/3] Assembling environment meshes and excluding micro-assets...")
scene = trimesh.Scene()
material = trimesh.visual.texture.SimpleMaterial(image=texture_img)

room_mesh = trimesh.creation.box(extents=[room_meta["width"], room_meta["height"], room_meta["depth"]])
room_mesh.invert()

# Paint floor/walls/ceiling using VLM-inferred colors
room_colors = np.zeros((len(room_mesh.faces), 4))
for i, face_normal in enumerate(room_mesh.face_normals):
    if face_normal[1] > 0.7:
        room_colors[i] = np.append(floor_color, 1.0)
    elif face_normal[1] < -0.7:
        room_colors[i] = np.append(ceiling_color, 1.0)
    else:
        room_colors[i] = np.append(wall_color, 1.0)

room_mesh.visual.face_colors = (room_colors * 255).astype(np.uint8)
scene.add_geometry(room_mesh, node_name="room_base_structure")

# Asset insertion with filter
for asset in data["assets"]:
    label = asset["label"].lower()

    if any(item in label for item in ["ventilation", "switch", "fixture", "fan", "electrical"]):
        print(f"    -> Ignoring minor asset: '{asset['label']}'")
        continue

    pos = asset["position"]
    if abs(pos[0]) > 6 or abs(pos[2]) > 6:
        pos = [p * 0.25 for p in pos]

    scale = [max(s, 0.05) for s in asset["scale"]]
    mesh = trimesh.creation.box(extents=scale)

    rot = asset.get("rotation", [0, 0, 0])
    rot_rad = np.radians(rot)
    rotation_matrix = trimesh.transformations.euler_matrix(rot_rad[0], rot_rad[1], rot_rad[2])
    translation_matrix = trimesh.transformations.translation_matrix(pos)
    transform_matrix = trimesh.transformations.concatenate_matrices(translation_matrix, rotation_matrix)

    uvs = project_texture_uv(mesh, transform_matrix)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)

    mesh.apply_transform(transform_matrix)
    scene.add_geometry(mesh, node_name=asset["id"])

scene.export(OUTPUT_GLB)
print(f"\n[COMPLETE] Clean environment scene compiled:\n{OUTPUT_GLB}")
