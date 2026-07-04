#!/usr/bin/env python3
"""
Extended Room Pipeline — Multi-Pass VLM
=========================================
Pass 1: Get spatial layout (same as before)
Pass 2: Ask Qwen about unseen surfaces (wall colors, floor material, etc.)
Pass 3: Build GLB with projected photo on visible faces + inferred materials on back faces
Also fixes the horizontal mirror issue in UV projection.
"""

import json
import numpy as np
import ollama
import trimesh
from PIL import Image

IMAGE_PATH = "/Users/al/Documents/tries01/Furnit/FurnitTests/room.jpeg"
OUTPUT_GLB = "/Users/al/Documents/tries01/Furnit/FurnitTests/extended_room.glb"

# =====================================================================
# PASS 1: GET SPATIAL LAYOUT FROM QWEN
# =====================================================================
spatial_prompt = """
Act as a precise 3D spatial engineering parser. Analyze this room image and output a clean, strict JSON schema representing the physical layout. 

Specifications:
- All position [x,y,z] and scale [w,h,d] values must be in meters relative to the room center.
- Orientations [r_x, r_y, r_z] must be in degrees.

Your output must be raw JSON only, mapping these exact keys:
1. "room_structure": {"width": float, "height": float, "depth": float}
2. "assets": An array of objects, each containing: "id": string, "label": string, "position": [x,y,z], "scale": [w,h,d], "rotation": [rx,ry,rz].

Ensure values strictly conform to realistic physical proportions. Do not include any markdown or chat text outside the raw JSON.
"""

print("[1/4] Pass 1: Querying Qwen-VL for spatial layout...")
resp1 = ollama.generate(model='qwen2.5vl:7b', prompt=spatial_prompt, images=[IMAGE_PATH])
raw1 = resp1['response'].strip()

if "```json" in raw1:
    raw1 = raw1.split("```json")[-1].split("```")[0].strip()
elif "```" in raw1:
    raw1 = raw1.split("```")[1].split("```")[0].strip()

data = json.loads(raw1)
print(f"    Room: {data['room_structure']}")
print(f"    Assets: {len(data['assets'])} objects detected")

# =====================================================================
# PASS 2: ASK ABOUT UNSEEN SURFACES
# =====================================================================
material_prompt = """
Look at this room image carefully. I need to know the colors and materials of surfaces that are partially visible or can be inferred from context.

Output strict JSON only with these keys:
{
  "floor": {"color_rgb": [r,g,b], "material": "string"},
  "ceiling": {"color_rgb": [r,g,b], "material": "string"},
  "wall_left": {"color_rgb": [r,g,b], "material": "string"},
  "wall_right": {"color_rgb": [r,g,b], "material": "string"},
  "wall_back": {"color_rgb": [r,g,b], "material": "string"},
  "wall_front": {"color_rgb": [r,g,b], "material": "string"}
}

RGB values must be 0-255 integers. Material should be one of: paint, tile, wood, fabric, concrete, glass.
Do not include any markdown or text outside the JSON.
"""

print("[2/4] Pass 2: Querying Qwen-VL for unseen surface materials...")
resp2 = ollama.generate(model='qwen2.5vl:7b', prompt=material_prompt, images=[IMAGE_PATH])
raw2 = resp2['response'].strip()

if "```json" in raw2:
    raw2 = raw2.split("```json")[-1].split("```")[0].strip()
elif "```" in raw2:
    raw2 = raw2.split("```")[1].split("```")[0].strip()

try:
    materials = json.loads(raw2)
    print(f"    Floor: {materials.get('floor', {})}")
    print(f"    Ceiling: {materials.get('ceiling', {})}")
    print(f"    Walls: {list(materials.keys())}")
except json.JSONDecodeError:
    print(f"    Warning: Could not parse materials, using defaults")
    materials = {
        "floor": {"color_rgb": [210, 190, 170], "material": "tile"},
        "ceiling": {"color_rgb": [245, 245, 245], "material": "paint"},
        "wall_left": {"color_rgb": [240, 235, 230], "material": "paint"},
        "wall_right": {"color_rgb": [240, 235, 230], "material": "paint"},
        "wall_back": {"color_rgb": [240, 235, 230], "material": "paint"},
        "wall_front": {"color_rgb": [240, 235, 230], "material": "paint"},
    }

# =====================================================================
# PASS 3: BUILD CAMERA + PROJECTION (WITH MIRROR FIX)
# =====================================================================
print("[3/4] Building camera model with mirror correction...")
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
    """Project vertices onto image plane with mirror fix and facing-angle culling."""
    uvs = []
    facing = []

    world_normals = np.dot(transform_matrix[:3, :3], mesh.vertex_normals.T).T

    for i, vert in enumerate(mesh.vertices):
        v_homo = np.append(vert, 1.0)
        v_world = np.dot(transform_matrix, v_homo)[:3]

        normal = world_normals[i]
        dot_product = np.dot(normal, camera_forward)

        if dot_product > 0.0:
            uvs.append([0.0, 0.0])
            facing.append(False)
            continue

        v_cam = np.dot(R_x, v_world - cam_pos)

        if v_cam[2] != 0:
            pixel = np.dot(K, v_cam)
            u = pixel[0] / pixel[2]
            v = pixel[1] / pixel[2]

            # MIRROR FIX: flip U to correct horizontal inversion
            u_norm = np.clip(1.0 - (u / img_w), 0.0, 1.0)
            v_norm = np.clip(1.0 - (v / img_h), 0.0, 1.0)
            uvs.append([u_norm, v_norm])
            facing.append(True)
        else:
            uvs.append([0.0, 0.0])
            facing.append(False)

    return np.array(uvs), facing


def make_solid_color_image(rgb, size=64):
    """Create a small solid-color image for material assignment."""
    img = Image.new("RGB", (size, size), tuple(rgb))
    return img


# =====================================================================
# PASS 4: COMPILE SCENE WITH EXTENDED WALLS
# =====================================================================
print("[4/4] Compiling extended room with inferred materials...")
scene = trimesh.Scene()
material = trimesh.visual.texture.SimpleMaterial(image=texture_img)

room = data["room_structure"]
rw, rh, rd = room["width"], room["height"], room["depth"]

# Instead of one big box, build 6 individual wall planes so each gets its own material
wall_specs = [
    ("floor",      [rw, 0.02, rd],  [0, -rh/2, 0],    "floor"),
    ("ceiling",    [rw, 0.02, rd],  [0, rh/2, 0],     "ceiling"),
    ("wall_back",  [rw, rh, 0.02],  [0, 0, -rd/2],    "wall_back"),
    ("wall_front", [rw, rh, 0.02],  [0, 0, rd/2],     "wall_front"),
    ("wall_left",  [0.02, rh, rd],  [-rw/2, 0, 0],    "wall_left"),
    ("wall_right", [0.02, rh, rd],  [rw/2, 0, 0],     "wall_right"),
]

for name, extents, pos, mat_key in wall_specs:
    wall_mesh = trimesh.creation.box(extents=extents)
    wall_mesh.invert()

    transform = trimesh.transformations.translation_matrix(pos)
    uvs, facing_mask = project_texture_uv(wall_mesh, transform)

    visible_count = sum(facing_mask)
    total = len(facing_mask)

    if visible_count > total * 0.3:
        # Mostly facing camera — use projected photo texture
        wall_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    else:
        # Mostly away from camera — use VLM-inferred solid color
        mat_info = materials.get(mat_key, {"color_rgb": [220, 220, 220]})
        color_rgb = mat_info.get("color_rgb", [220, 220, 220])
        solid_img = make_solid_color_image(color_rgb)
        solid_material = trimesh.visual.texture.SimpleMaterial(image=solid_img)
        neutral_uvs = np.full((len(wall_mesh.vertices), 2), 0.5)
        wall_mesh.visual = trimesh.visual.TextureVisuals(uv=neutral_uvs, material=solid_material)

    wall_mesh.apply_transform(transform)
    scene.add_geometry(wall_mesh, node_name=name)

# Furniture assets skipped — room envelope only

scene.export(OUTPUT_GLB)
print(f"\n[COMPLETE] Extended room with inferred materials:")
print(f"  {OUTPUT_GLB}")
print(f"\n  Visible faces -> photo texture projected (mirror-corrected)")
print(f"  Hidden faces  -> VLM-inferred solid colors (no black gaps)")
