#!/usr/bin/env python3
"""
Final Room Pipeline — Textured + Extended Walls
=================================================
- Homography-based texture warping from detected image regions (no pinhole)
- Mirror-free: quad corners define exact source→dest mapping
- Unseen walls EXTENDED from adjacent edge pixels (not mirrored, not flat color)
- Interior-facing normals for walkable room
"""

import json
import numpy as np
import ollama
import trimesh
import cv2
from PIL import Image
from pathlib import Path

IMAGE_PATH = "/Users/al/Documents/tries01/Furnit/FurnitTests/room.jpeg"
OUTPUT_GLB = "/Users/al/Documents/tries01/Furnit/FurnitTests/final_room.glb"

img = cv2.imread(IMAGE_PATH)
IMG_H, IMG_W = img.shape[:2]
texture_img = Image.open(IMAGE_PATH)

# =====================================================================
# STEP 1: GET ROOM LAYOUT FROM QWEN
# =====================================================================
PROMPT = f"""
You are a metric 3D scene analyst. Analyze this room photo and return spatial data as STRICT JSON.

IMAGE SIZE: WIDTH={IMG_W} HEIGHT={IMG_H}

Return this JSON exactly:
{{
  "room": {{"width_m": <num>, "height_m": <num>, "depth_m": <num>}},
  "vanishing_point_px": [<u>, <v>],
  "boundaries": {{
    "ceiling_bottom_y_px": <num>,
    "floor_top_y_px": <num>,
    "left_wall_right_x_px": <num>,
    "right_wall_left_x_px": <num>
  }},
  "surfaces": [
    {{
      "id": "floor|ceiling|wall_back|wall_left|wall_right",
      "visible": <bool>,
      "base_color_rgb": [<0-255>,<0-255>,<0-255>],
      "material": "tile|paint|wood|fabric|carpet"
    }}
  ]
}}

Output JSON only, no markdown, no prose.
"""

print(f"[1/4] Querying Qwen-VL ({IMG_W}x{IMG_H})...")
response = ollama.generate(model='qwen2.5vl:7b', prompt=PROMPT, images=[IMAGE_PATH])
raw_text = response['response'].strip()

if "```json" in raw_text:
    raw_text = raw_text.split("```json")[-1].split("```")[0].strip()
elif "```" in raw_text:
    raw_text = raw_text.split("```")[1].split("```")[0].strip()

try:
    data = json.loads(raw_text)
except json.JSONDecodeError:
    data = {
        "room": {"width_m": 5, "height_m": 3, "depth_m": 4},
        "vanishing_point_px": [IMG_W // 2, IMG_H // 2],
        "boundaries": {},
        "surfaces": [],
    }

print(f"    Room: {data.get('room', {})}")
bounds = data.get("boundaries", {})

# =====================================================================
# STEP 2: DETECT BOUNDARIES (with VLM inversion fallback)
# =====================================================================
print("[2/4] Detecting surface boundaries...")

raw_ceil_y = int(bounds.get("ceiling_bottom_y_px", IMG_H * 0.2))
raw_floor_y = int(bounds.get("floor_top_y_px", IMG_H * 0.75))
raw_left_x = int(bounds.get("left_wall_right_x_px", IMG_W * 0.15))
raw_right_x = int(bounds.get("right_wall_left_x_px", IMG_W * 0.85))

if raw_ceil_y > raw_floor_y or abs(raw_ceil_y - raw_floor_y) < IMG_H * 0.2:
    print("    VLM boundaries invalid, using edge detection...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    top_third = edges[:IMG_H // 3, :]
    bot_third = edges[2 * IMG_H // 3:, :]
    top_rows = np.sum(top_third, axis=1)
    bot_rows = np.sum(bot_third, axis=1)
    ceil_y = int(np.argmax(top_rows)) if top_rows.max() > 0 else int(IMG_H * 0.22)
    floor_y = int(2 * IMG_H // 3 + np.argmax(bot_rows)) if bot_rows.max() > 0 else int(IMG_H * 0.78)
    if floor_y - ceil_y < IMG_H * 0.3:
        ceil_y = int(IMG_H * 0.22)
        floor_y = int(IMG_H * 0.78)
    left_x = int(IMG_W * 0.12)
    right_x = int(IMG_W * 0.88)
else:
    ceil_y = min(max(raw_ceil_y, 0), IMG_H - 1)
    floor_y = min(max(raw_floor_y, ceil_y + 10), IMG_H - 1)
    left_x = min(max(raw_left_x, 0), IMG_W - 1)
    right_x = min(max(raw_right_x, left_x + 10), IMG_W - 1)

print(f"    Final: ceil_y={ceil_y}, floor_y={floor_y}, left_x={left_x}, right_x={right_x}")

# =====================================================================
# STEP 3: WARP TEXTURES + EXTEND UNSEEN WALLS
# =====================================================================
print("[3/4] Warping textures from image regions...")

TEX_SIZE = 1024


def warp_quad_to_square(img_bgr, quad_pts):
    """Warp a quadrilateral from image to a square texture."""
    src = np.array(quad_pts, dtype=np.float32)
    dst = np.array([[0, 0], [TEX_SIZE, 0], [TEX_SIZE, TEX_SIZE], [0, TEX_SIZE]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        return None
    warped = cv2.warpPerspective(img_bgr, H, (TEX_SIZE, TEX_SIZE))
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


def extend_edge(texture_rgb, edge="right", width=TEX_SIZE):
    """
    Create an extended texture by repeating the edge pixels of a source texture.
    This continues the wall naturally rather than mirroring or using flat color.
    """
    arr = np.array(texture_rgb)
    if edge == "right":
        strip = arr[:, -1:, :]  # rightmost column
        extended = np.tile(strip, (1, width, 1))
    elif edge == "left":
        strip = arr[:, :1, :]  # leftmost column
        extended = np.tile(strip, (1, width, 1))
    elif edge == "bottom":
        strip = arr[-1:, :, :]  # bottom row
        extended = np.tile(strip, (width, 1, 1))
    elif edge == "top":
        strip = arr[:1, :, :]  # top row
        extended = np.tile(strip, (width, 1, 1))
    else:
        extended = arr

    # Resize to TEX_SIZE x TEX_SIZE
    ext_img = Image.fromarray(extended.astype(np.uint8))
    ext_img = ext_img.resize((TEX_SIZE, TEX_SIZE), Image.LANCZOS)
    return ext_img


# Visible surface quads (perspective trapezoids from the photo)
back_wall_quad = [
    [left_x, ceil_y], [right_x, ceil_y],
    [right_x, floor_y], [left_x, floor_y],
]
ceiling_quad = [
    [0, 0], [IMG_W, 0],
    [right_x, ceil_y], [left_x, ceil_y],
]
floor_quad = [
    [left_x, floor_y], [right_x, floor_y],
    [IMG_W, IMG_H], [0, IMG_H],
]
left_wall_quad = [
    [0, 0], [left_x, ceil_y],
    [left_x, floor_y], [0, IMG_H],
]
right_wall_quad = [
    [right_x, ceil_y], [IMG_W, 0],
    [IMG_W, IMG_H], [right_x, floor_y],
]

face_textures = {}
for name, quad in [("wall_back", back_wall_quad), ("ceiling", ceiling_quad),
                   ("floor", floor_quad), ("wall_left", left_wall_quad),
                   ("wall_right", right_wall_quad)]:
    warped = warp_quad_to_square(img, quad)
    if warped is not None:
        face_textures[name] = Image.fromarray(warped)
        print(f"    {name}: textured from photo")
    else:
        face_textures[name] = Image.new("RGB", (TEX_SIZE, TEX_SIZE), (220, 220, 220))
        print(f"    {name}: homography failed, flat color")

# EXTEND the front wall (behind camera) from adjacent wall edges
# Take the back edges of left and right walls and blend them
print("    wall_front: EXTENDING from adjacent wall edges...")
left_tex = face_textures["wall_left"]
right_tex = face_textures["wall_right"]

# Left wall's right edge → extends into front wall left half
left_extended = extend_edge(left_tex, edge="right", width=TEX_SIZE)
# Right wall's left edge → extends into front wall right half
right_extended = extend_edge(right_tex, edge="left", width=TEX_SIZE)

# Blend the two halves together for front wall
left_arr = np.array(left_extended).astype(np.float32)
right_arr = np.array(right_extended).astype(np.float32)
# Gradient blend: left side uses left_extended, right side uses right_extended
blend_weights = np.linspace(0, 1, TEX_SIZE).reshape(1, -1, 1)
blended = left_arr * (1 - blend_weights) + right_arr * blend_weights
face_textures["wall_front"] = Image.fromarray(blended.astype(np.uint8))

print()

# =====================================================================
# STEP 4: BUILD INTERIOR ROOM MESH
# =====================================================================
print("[4/4] Building interior-facing GLB...")

room = data.get("room", {"width_m": 5, "height_m": 3, "depth_m": 4})
rw = room.get("width_m", 5)
rh = room.get("height_m", 3)
rd = room.get("depth_m", 4)

scene = trimesh.Scene()

# 6 interior-facing panels with correct winding
surface_verts = {
    "wall_back": [
        [-rw/2, rh, -rd/2], [rw/2, rh, -rd/2], [rw/2, 0, -rd/2], [-rw/2, 0, -rd/2]
    ],
    "wall_front": [
        [rw/2, rh, rd/2], [-rw/2, rh, rd/2], [-rw/2, 0, rd/2], [rw/2, 0, rd/2]
    ],
    "wall_left": [
        [-rw/2, rh, rd/2], [-rw/2, rh, -rd/2], [-rw/2, 0, -rd/2], [-rw/2, 0, rd/2]
    ],
    "wall_right": [
        [rw/2, rh, -rd/2], [rw/2, rh, rd/2], [rw/2, 0, rd/2], [rw/2, 0, -rd/2]
    ],
    "floor": [
        [-rw/2, 0, rd/2], [rw/2, 0, rd/2], [rw/2, 0, -rd/2], [-rw/2, 0, -rd/2]
    ],
    "ceiling": [
        [-rw/2, rh, -rd/2], [rw/2, rh, -rd/2], [rw/2, rh, rd/2], [-rw/2, rh, rd/2]
    ],
}

# UV corners — no mirror, straight mapping
face_uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)

for name, verts in surface_verts.items():
    vertices = np.array(verts, dtype=np.float64)
    faces = np.array([[0, 2, 1], [0, 3, 2]])  # inward-facing winding

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    tex = face_textures.get(name, Image.new("RGB", (TEX_SIZE, TEX_SIZE), (200, 200, 200)))
    mat = trimesh.visual.texture.SimpleMaterial(image=tex)
    mesh.visual = trimesh.visual.TextureVisuals(uv=face_uvs, material=mat)

    scene.add_geometry(mesh, node_name=name)

scene.export(OUTPUT_GLB)
print(f"\n[COMPLETE] {OUTPUT_GLB}")
print(f"  - 5 visible faces: photo texture via homography (no mirror)")
print(f"  - Front wall: extended from adjacent edges (not mirrored, not flat)")
print(f"  - All normals face inward (interior walkable)")
