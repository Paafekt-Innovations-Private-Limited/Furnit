#!/usr/bin/env python3
"""
Room-to-3D Pipeline
====================
Queries local Qwen2.5-VL (7B) via Ollama to spatially parse a room image,
then compiles the JSON output into a GLB 3D scene using trimesh.
"""

import json
import os
import subprocess

import numpy as np
import trimesh
import ollama

IMAGE_PATH = "/Users/al/Documents/tries01/Furnit/FurnitTests/room.jpeg"
OUTPUT_GLB = "/Users/al/Documents/tries01/Furnit/FurnitTests/room_output.glb"
OUTPUT_USDZ = "/Users/al/Documents/tries01/Furnit/FurnitTests/room_output.usdz"

PROMPT = """
Act as a precise 3D spatial serialization compiler. Analyze this room image and output a clean, strict JSON schema representing the physical environment bounds and its core assets.

Use the following exact specifications for cross-platform rendering (Android GLB/iOS USDZ):
- All position [x,y,z] and scale [w,h,d] values must be in meters.
- Orientations [r_x, r_y, r_z] must be in degrees.

Your output must be raw JSON only, mapping these exact keys:
1. "room_structure": {"width": float, "height": float, "depth": float}
2. "assets": An array of objects, each containing: "id": string, "label": string, "position": [x,y,z], "scale": [w,h,d], "rotation": [rx,ry,rz].

Ensure values strictly conform to realistic physical proportions. Do not include any conversational text or markdown code blocks outside the JSON.
"""


def query_local_vlm():
    print("[1/4] Querying local Qwen-VL model via Ollama...")
    response = ollama.generate(
        model="qwen2.5vl:7b",
        prompt=PROMPT,
        images=[IMAGE_PATH],
    )
    return response["response"].strip()


def clean_vlm_output(raw_text):
    print("[2/4] Parsing and cleaning JSON syntax...")
    if "```json" in raw_text:
        raw_text = raw_text.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_text:
        raw_text = raw_text.split("```")[1].split("```")[0].strip()
    return json.loads(raw_text)


def compile_3d_scene(data):
    print("[3/4] Compiling 3D mesh architecture into GLB...")
    scene = trimesh.Scene()

    for asset in data["assets"]:
        pos = asset["position"]
        scale = asset["scale"]
        rot = asset.get("rotation", [0, 0, 0])

        if abs(pos[0]) > 6 or abs(pos[2]) > 6:
            pos = [p * 0.25 for p in pos]

        scale = [max(s, 0.05) for s in scale]

        mesh = trimesh.creation.box(extents=scale)

        label = asset["label"].lower()
        if "chair" in label:
            mesh.visual.face_colors = [40, 44, 52, 255]
        elif "covering" in label or "curtain" in label:
            mesh.visual.face_colors = [139, 90, 43, 255]
        elif "desk" in label or "table" in label:
            mesh.visual.face_colors = [101, 67, 33, 255]
        elif "monitor" in label or "screen" in label:
            mesh.visual.face_colors = [20, 20, 20, 255]
        elif "shelf" in label or "cabinet" in label:
            mesh.visual.face_colors = [160, 130, 100, 255]
        else:
            mesh.visual.face_colors = [200, 200, 200, 255]

        rotation_matrix = trimesh.transformations.euler_matrix(
            np.radians(rot[0]), np.radians(rot[1]), np.radians(rot[2])
        )
        translation_matrix = trimesh.transformations.translation_matrix(pos)
        transform = trimesh.transformations.concatenate_matrices(
            translation_matrix, rotation_matrix
        )

        mesh.apply_transform(transform)
        scene.add_geometry(mesh, node_name=asset["id"])

    scene.export(OUTPUT_GLB)
    print(f"    -> GLB saved: {OUTPUT_GLB}")


def convert_for_ios():
    print("[4/4] Attempting USDZ conversion...")
    if os.path.exists("/usr/bin/xcrun"):
        try:
            subprocess.run(
                ["xcrun", "usdzconvert", OUTPUT_GLB, OUTPUT_USDZ],
                check=True,
            )
            print(f"    -> USDZ saved: {OUTPUT_USDZ}")
        except Exception:
            print("    -> Skipping USDZ: usdzconvert not configured.")
    else:
        print("    -> Skipping USDZ: xcrun not found.")


if __name__ == "__main__":
    try:
        raw_response = query_local_vlm()
        print(f"    Raw VLM response ({len(raw_response)} chars):")
        print(f"    {raw_response[:300]}...")
        print()

        json_matrix = clean_vlm_output(raw_response)
        print(f"    Parsed {len(json_matrix.get('assets', []))} assets")
        print(f"    Room: {json_matrix.get('room_structure', {})}")
        print()

        compile_3d_scene(json_matrix)
        convert_for_ios()

        print("\n[COMPLETE] 3D Room Assets ready.")
    except Exception as e:
        print(f"\n[PIPELINE ERROR] {e}")
        import traceback
        traceback.print_exc()
