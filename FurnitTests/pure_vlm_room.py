#!/usr/bin/env python3
"""
Pure VLM Oracle Pipeline
==========================
Single Qwen-VL call with a master prompt → strict JSON → 3D room build.
No pinhole camera model, no geometry guessing. The VLM provides everything:
room dimensions, camera pose, surface pixel quads, object placement.
Texturing uses per-surface homography from VLM-provided image_quad_px.
"""

import json
import numpy as np
import ollama
import trimesh
import cv2
from PIL import Image
from pathlib import Path

IMAGE_PATH = "/Users/al/Documents/tries01/Furnit/FurnitTests/room.jpeg"
OUTPUT_GLB = "/Users/al/Documents/tries01/Furnit/FurnitTests/pure_vlm_room.glb"
OUTPUT_JSON = "/Users/al/Documents/tries01/Furnit/FurnitTests/pure_vlm_response.json"

# Load image to get dimensions
img = cv2.imread(IMAGE_PATH)
IMG_H, IMG_W = img.shape[:2]

# =====================================================================
# THE MASTER PROMPT — one call, everything needed
# =====================================================================
PROMPT = f"""
You are a metric 3D scene analyst. Analyze the single room photo and return a
complete spatial reconstruction as STRICT JSON. This JSON is the ONLY input to a
3D room builder — there is no other source of truth, so every field must be filled
with your best numeric estimate. Do not return null. Output JSON only, no prose.

CONVENTIONS (obey exactly):
- Units: meters and degrees.
- Coordinate frame: right-handed, +X right, +Y up, +Z toward the viewer.
  The camera looks toward -Z.
- Room origin [0,0,0] is the center of the floor. Floor at y=0, ceiling at y=+height.
- Image pixel coords: origin top-left, +u right, +v down. Image is WIDTH x HEIGHT px.
- For each visible surface, give the pixel coordinates of its 4 corners in the photo,
  ordered [top-left, top-right, bottom-right, bottom-left] as seen on that surface.

IMAGE SIZE: WIDTH={IMG_W} HEIGHT={IMG_H}

Return this schema exactly:

{{
  "camera": {{
    "horizontal_fov_deg": <num>,
    "vertical_fov_deg": <num>,
    "focal_length_px": <num>,
    "principal_point_px": [<cx>, <cy>],
    "position_m": [<x>, <y>, <z>],
    "euler_deg": [<pitch>, <yaw>, <roll>],
    "height_from_floor_m": <num>,
    "distance_to_back_wall_m": <num>,
    "confidence_0_1": <num>
  }},
  "room": {{
    "width_m": <num>, "height_m": <num>, "depth_m": <num>,
    "confidence_0_1": <num>
  }},
  "surfaces": [
    {{
      "id": "floor|ceiling|wall_back|wall_front|wall_left|wall_right",
      "visible_in_photo": <bool>,
      "image_quad_px": [[u,v],[u,v],[u,v],[u,v]] or null,
      "base_color_rgb": [<0-255>,<0-255>,<0-255>],
      "material_type": "tile|paint|wood|fabric|concrete|carpet|other",
      "pattern_scale_m": <num>,
      "roughness_0_1": <num>,
      "confidence_0_1": <num>
    }}
    // exactly 6 entries, one per surface
  ],
  "objects": [
    {{
      "id": "<slug>",
      "label": "<name>",
      "position_m": [<x>, <y>, <z>],
      "scale_m": [<w>, <h>, <d>],
      "rotation_deg": [<rx>, <ry>, <rz>],
      "resting_on": "floor|wall|ceiling",
      "base_color_rgb": [<0-255>,<0-255>,<0-255>],
      "visible_in_photo": <bool>,
      "confidence_0_1": <num>
    }}
  ],
  "scale_anchor": {{
    "object": "<what you used as a size reference>",
    "assumed_real_size_m": <num>
  }}
}}
"""

# =====================================================================
# STEP 1: QUERY QWEN — SINGLE CALL
# =====================================================================
print(f"[1/3] Querying Qwen-VL (image {IMG_W}x{IMG_H})...")
print("      This is the ONLY call. VLM is the single source of truth.")
response = ollama.generate(model='qwen2.5vl:7b', prompt=PROMPT, images=[IMAGE_PATH])
raw_text = response['response'].strip()

if "```json" in raw_text:
    raw_text = raw_text.split("```json")[-1].split("```")[0].strip()
elif "```" in raw_text:
    raw_text = raw_text.split("```")[1].split("```")[0].strip()

data = json.loads(raw_text)

Path(OUTPUT_JSON).write_text(json.dumps(data, indent=2))
print(f"      Raw VLM JSON saved: {OUTPUT_JSON}")
print(f"      Room: {data['room']['width_m']}m x {data['room']['height_m']}m x {data['room']['depth_m']}m")
print(f"      Surfaces: {len(data['surfaces'])}")
print(f"      Objects: {len(data['objects'])}")
print(f"      Scale anchor: {data.get('scale_anchor', {})}")
print()

# =====================================================================
# STEP 2: BUILD ROOM GEOMETRY + HOMOGRAPHY TEXTURES
# =====================================================================
print("[2/3] Building room from VLM JSON (no camera model, homography only)...")

scene = trimesh.Scene()
texture_img = Image.open(IMAGE_PATH)
tex_np = np.array(texture_img)

rw = data["room"]["width_m"]
rh = data["room"]["height_m"]
rd = data["room"]["depth_m"]

# Define the 6 room surfaces as quads in 3D space
# Convention: origin at floor center, +X right, +Y up, +Z toward viewer
# Each quad: [top-left, top-right, bottom-right, bottom-left] in 3D
surface_3d_quads = {
    "wall_back":  [[-rw/2, rh, -rd/2], [rw/2, rh, -rd/2], [rw/2, 0, -rd/2], [-rw/2, 0, -rd/2]],
    "wall_front": [[rw/2, rh, rd/2], [-rw/2, rh, rd/2], [-rw/2, 0, rd/2], [rw/2, 0, rd/2]],
    "wall_left":  [[-rw/2, rh, rd/2], [-rw/2, rh, -rd/2], [-rw/2, 0, -rd/2], [-rw/2, 0, rd/2]],
    "wall_right": [[rw/2, rh, -rd/2], [rw/2, rh, rd/2], [rw/2, 0, rd/2], [rw/2, 0, -rd/2]],
    "floor":      [[-rw/2, 0, -rd/2], [rw/2, 0, -rd/2], [rw/2, 0, rd/2], [-rw/2, 0, rd/2]],
    "ceiling":    [[-rw/2, rh, rd/2], [rw/2, rh, rd/2], [rw/2, rh, -rd/2], [-rw/2, rh, -rd/2]],
}

# UV quad for a full-face texture: maps [0,0]->[1,0]->[1,1]->[0,1]
uv_quad = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

TEX_SIZE = 1024  # resolution for warped per-face textures

for surface in data["surfaces"]:
    sid = surface["id"]
    visible = surface.get("visible_in_photo", False)
    quad_px = surface.get("image_quad_px")

    if sid not in surface_3d_quads:
        continue

    corners_3d = np.array(surface_3d_quads[sid], dtype=np.float32)

    # Build a quad mesh (2 triangles)
    vertices = corners_3d
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    if visible and quad_px and len(quad_px) == 4:
        # HOMOGRAPHY: warp photo region onto this face
        src_pts = np.array(quad_px, dtype=np.float32)
        dst_pts = np.array([[0, 0], [TEX_SIZE, 0], [TEX_SIZE, TEX_SIZE], [0, TEX_SIZE]], dtype=np.float32)

        H, _ = cv2.findHomography(src_pts, dst_pts)
        if H is not None:
            warped = cv2.warpPerspective(tex_np, H, (TEX_SIZE, TEX_SIZE))
            face_texture = Image.fromarray(warped)
        else:
            rgb = surface.get("base_color_rgb", [200, 200, 200])
            face_texture = Image.new("RGB", (TEX_SIZE, TEX_SIZE), tuple(rgb))
    else:
        # Non-visible: solid color from VLM
        rgb = surface.get("base_color_rgb", [200, 200, 200])
        face_texture = Image.new("RGB", (TEX_SIZE, TEX_SIZE), tuple(rgb))

    # Assign UV coordinates for the 2-triangle quad
    face_uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)

    face_material = trimesh.visual.texture.SimpleMaterial(image=face_texture)
    mesh.visual = trimesh.visual.TextureVisuals(uv=face_uvs, material=face_material)

    scene.add_geometry(mesh, node_name=sid)

# =====================================================================
# STEP 3: PLACE OBJECTS
# =====================================================================
print("[3/3] Placing objects...")

for obj in data["objects"]:
    pos = obj["position_m"]
    scale = [max(s, 0.05) for s in obj["scale_m"]]
    rot = obj.get("rotation_deg", [0, 0, 0])

    mesh = trimesh.creation.box(extents=scale)

    rgb = obj.get("base_color_rgb", [150, 150, 150])
    mesh.visual.face_colors = [rgb[0], rgb[1], rgb[2], 255]

    rot_rad = np.radians(rot)
    rotation_matrix = trimesh.transformations.euler_matrix(rot_rad[0], rot_rad[1], rot_rad[2])
    translation_matrix = trimesh.transformations.translation_matrix(pos)
    transform = trimesh.transformations.concatenate_matrices(translation_matrix, rotation_matrix)

    mesh.apply_transform(transform)
    scene.add_geometry(mesh, node_name=obj["id"])

# =====================================================================
# EXPORT
# =====================================================================
scene.export(OUTPUT_GLB)
print(f"\n{'='*60}")
print(f"[COMPLETE] Pure VLM room exported:")
print(f"  GLB:  {OUTPUT_GLB}")
print(f"  JSON: {OUTPUT_JSON}")
print(f"{'='*60}")
print()
print("SCORECARD (check manually):")
print("  - Room proportions: does width:depth:height feel right?")
print("  - Texture registration: do photo textures land square on walls?")
print("  - Object placement: are boxes where real objects are?")
print("  - Confidence calibration: do low-confidence fields = wrong results?")
