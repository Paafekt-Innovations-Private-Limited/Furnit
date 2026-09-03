#!/usr/bin/env python3
"""Build matching lightweight sample-room assets for Android and iOS."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ANDROID_ROOT = SCRIPT_DIR.parents[1]
SOURCE_DIR = SCRIPT_DIR / "generated"
ANDROID_ASSET_DIR = ANDROID_ROOT / "app/src/main/assets/bundled_rooms"
IOS_STAGING_DIR = Path("/tmp/furnit-room-assets-v3")


@dataclass(frozen=True)
class Material:
    name: str
    color: tuple[float, float, float]
    roughness: float
    metallic: float = 0.0
    opacity: float = 1.0
    texture: str | None = None


def vec(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{value:.5g}" for value in values) + ")"


def material_usda(material: Material) -> str:
    diffuse = (
        f"color3f inputs:diffuseColor.connect = </Room/Looks/{material.name}/DiffuseTexture.outputs:rgb>"
        if material.texture else f"color3f inputs:diffuseColor = {vec(material.color)}"
    )
    opacity = f"\n                float inputs:opacity = {material.opacity:.4g}" if material.opacity < 0.999 else ""
    texture_nodes = ""
    if material.texture:
        texture_nodes = f'''
            def Shader "PrimvarReader_st" {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
            def Shader "DiffuseTexture" {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @textures/{material.texture}@
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float2 inputs:st.connect = </Room/Looks/{material.name}/PrimvarReader_st.outputs:result>
                color3f outputs:rgb
            }}'''
    return f'''
        def Material "{material.name}" {{
            token outputs:surface.connect = </Room/Looks/{material.name}/PreviewSurface.outputs:surface>
            def Shader "PreviewSurface" {{
                uniform token info:id = "UsdPreviewSurface"
                {diffuse}
                float inputs:metallic = {material.metallic:.4g}
                float inputs:roughness = {material.roughness:.4g}{opacity}
                token outputs:surface
            }}{texture_nodes}
        }}'''


def quad(name: str, points: list[tuple[float, float, float]], material: str,
         uv_width: float = 1.0, uv_height: float = 1.0, reverse: bool = False) -> str:
    indices = "0, 2, 1, 0, 3, 2" if reverse else "0, 1, 2, 0, 2, 3"
    return f'''
        def Mesh "{name}" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        ) {{
            uniform token subdivisionScheme = "none"
            bool doubleSided = false
            point3f[] points = [{", ".join(vec(point) for point in points)}]
            int[] faceVertexCounts = [3, 3]
            int[] faceVertexIndices = [{indices}]
            texCoord2f[] primvars:st = [(0, 0), ({uv_width:.5g}, 0), ({uv_width:.5g}, {uv_height:.5g}), (0, {uv_height:.5g})] (
                interpolation = "vertex"
            )
            rel material:binding = </Room/Looks/{material}>
        }}'''


def box(name: str, center: tuple[float, float, float], size: tuple[float, float, float], material: str) -> str:
    cx, cy, cz = center
    sx, sy, sz = (dimension / 2.0 for dimension in size)
    points = [
        (cx - sx, cy - sy, cz - sz), (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz), (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz), (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz), (cx - sx, cy + sy, cz + sz),
    ]
    indices = [
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
        0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
        0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
    ]
    return f'''
        def Mesh "{name}" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        ) {{
            uniform token subdivisionScheme = "none"
            bool doubleSided = false
            point3f[] points = [{", ".join(vec(point) for point in points)}]
            int[] faceVertexCounts = [{", ".join(["3"] * 12)}]
            int[] faceVertexIndices = [{", ".join(str(index) for index in indices)}]
            rel material:binding = </Room/Looks/{material}>
        }}'''


def shell(width: float, height: float, depth: float, wall: str, floor: str, ceiling: str) -> list[str]:
    x, z = width / 2.0, depth / 2.0
    return [
        quad("floor", [(-x, 0, -z), (x, 0, -z), (x, 0, z), (-x, 0, z)], floor, width, depth, True),
        quad("ceiling", [(-x, height, -z), (x, height, -z), (x, height, z), (-x, height, z)], ceiling, width, depth),
        quad("back_wall", [(-x, 0, z), (x, 0, z), (x, height, z), (-x, height, z)], wall, width, height, True),
        quad("left_wall", [(-x, 0, -z), (-x, 0, z), (-x, height, z), (-x, height, -z)], wall, depth, height, True),
        quad("right_wall", [(x, 0, -z), (x, 0, z), (x, height, z), (x, height, -z)], wall, depth, height),
    ]


def perimeter_window(width: float, height: float, depth: float, frame: str, glass: str, margin: float) -> list[str]:
    z = -depth / 2.0 + 0.02
    opening_width, opening_height = width - 2.0 * margin, height - 2.0 * margin
    thickness, frame_depth = 0.055, 0.07
    pieces = [
        box("window_frame_left", (-opening_width / 2, height / 2, z), (thickness, opening_height, frame_depth), frame),
        box("window_frame_right", (opening_width / 2, height / 2, z), (thickness, opening_height, frame_depth), frame),
        box("window_frame_bottom", (0, margin, z), (opening_width, thickness, frame_depth), frame),
        box("window_frame_top", (0, height - margin, z), (opening_width, thickness, frame_depth), frame),
    ]
    pieces.append(quad("window_glass_uninterrupted", [
        (-opening_width / 2, margin, z - 0.01), (opening_width / 2, margin, z - 0.01),
        (opening_width / 2, height - margin, z - 0.01), (-opening_width / 2, height - margin, z - 0.01),
    ], glass))
    return pieces


def nordic_gallery() -> tuple[list[Material], list[str]]:
    width, height, depth = 5.8, 2.8, 4.6
    materials = [
        Material("limewash", (0.88, 0.86, 0.80), 0.76, texture="nordic_limewash.png"),
        Material("white_oak", (0.76, 0.62, 0.43), 0.56, texture="nordic_white_oak.png"),
        Material("warm_white", (0.93, 0.91, 0.86), 0.62),
        Material("bronze", (0.12, 0.095, 0.072), 0.33, metallic=0.72),
        Material("clear_glass", (0.70, 0.82, 0.88), 0.12, opacity=0.24),
    ]
    geometry = shell(width, height, depth, "limewash", "white_oak", "warm_white")
    x, z = width / 2, depth / 2
    geometry += [
        box("back_baseboard", (0, 0.065, z - 0.025), (width, 0.13, 0.05), "warm_white"),
        box("left_baseboard", (-x + 0.025, 0.065, 0), (0.05, 0.13, depth), "warm_white"),
        box("right_baseboard", (x - 0.025, 0.065, 0), (0.05, 0.13, depth), "warm_white"),
        box("ceiling_cove_back", (0, height - 0.045, z - 0.05), (width, 0.09, 0.10), "warm_white"),
        box("ceiling_cove_left", (-x + 0.05, height - 0.045, 0), (0.10, 0.09, depth), "warm_white"),
        box("ceiling_cove_right", (x - 0.05, height - 0.045, 0), (0.10, 0.09, depth), "warm_white"),
    ]
    geometry += perimeter_window(width, height, depth, "bronze", "clear_glass", 0.22)
    return materials, geometry


def contemporary_gallery() -> tuple[list[Material], list[str]]:
    width, height, depth = 7.2, 3.2, 5.4
    materials = [
        Material("limestone", (0.66, 0.64, 0.60), 0.72, texture="gallery_limestone.png"),
        Material("microcement", (0.24, 0.25, 0.25), 0.56, texture="gallery_microcement.png"),
        Material("soft_ceiling", (0.77, 0.76, 0.72), 0.78),
        Material("graphite_steel", (0.035, 0.04, 0.043), 0.30, metallic=0.82),
        Material("smoked_glass", (0.40, 0.50, 0.55), 0.14, opacity=0.28),
    ]
    geometry = shell(width, height, depth, "limestone", "microcement", "soft_ceiling")
    x, z = width / 2, depth / 2
    geometry += [
        box("back_shadow_plinth", (0, 0.045, z - 0.03), (width, 0.09, 0.06), "graphite_steel"),
        box("left_shadow_plinth", (-x + 0.03, 0.045, 0), (0.06, 0.09, depth), "graphite_steel"),
        box("right_shadow_plinth", (x - 0.03, 0.045, 0), (0.06, 0.09, depth), "graphite_steel"),
        box("ceiling_edge_left", (-x + 0.11, height - 0.075, 0), (0.12, 0.15, depth), "graphite_steel"),
        box("ceiling_edge_right", (x - 0.11, height - 0.075, 0), (0.12, 0.15, depth), "graphite_steel"),
    ]
    geometry += perimeter_window(width, height, depth, "graphite_steel", "smoked_glass", 0.28)
    return materials, geometry


def write_usda(path: Path, materials: list[Material], geometry: list[str]) -> None:
    document = f'''#usda 1.0
(
    defaultPrim = "Room"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Room" {{
    def Scope "Looks" {{{"".join(material_usda(material) for material in materials)}
    }}{"".join(geometry)}
}}
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def blender_worker(source_usda: Path, output_glb: Path) -> None:
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.wm.usd_import(filepath=str(source_usda), import_cameras=False, import_curves=False,
                                   import_lights=False, import_materials=True, import_meshes=True,
                                   import_volumes=False, scale=1.0)
    if "FINISHED" not in result:
        raise RuntimeError(f"USD import failed: {result}")
    result = bpy.ops.export_scene.gltf(filepath=str(output_glb), export_format="GLB", use_selection=False,
                                       export_apply=True, export_yup=True, export_texcoords=True,
                                       export_normals=True, export_materials="EXPORT", export_image_format="AUTO",
                                       export_draco_mesh_compression_enable=False)
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB export failed: {result}")


def validate_glb(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"GLB is too small: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        raise ValueError(f"Invalid GLB header: {path}")
    offset, document = 12, None
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk = data[offset:offset + chunk_length]
        offset += chunk_length
        if chunk_type == b"JSON":
            document = json.loads(chunk.rstrip(b" \x00"))
    if document is None:
        raise ValueError(f"GLB has no JSON chunk: {path}")
    unsupported = {"KHR_texture_basisu", "EXT_texture_webp", "KHR_draco_mesh_compression"}
    extensions = set(document.get("extensionsUsed", []))
    if extensions & unsupported:
        raise ValueError(f"Unsupported GLB extensions: {sorted(extensions & unsupported)}")
    bounds = [(item["min"], item["max"]) for item in document.get("accessors", [])
              if item.get("type") == "VEC3" and "min" in item and "max" in item]
    if not bounds:
        raise ValueError(f"GLB has no position bounds: {path}")
    minimum = [min(item[0][axis] for item in bounds) for axis in range(3)]
    maximum = [max(item[1][axis] for item in bounds) for axis in range(3)]
    return {"bytes": len(data), "min": minimum, "max": maximum, "extensions": sorted(extensions)}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def build() -> None:
    blender = shutil.which("blender") or "/opt/homebrew/bin/blender"
    usdzip, usdchecker = shutil.which("usdzip"), shutil.which("usdchecker")
    if not Path(blender).is_file() or not usdzip or not usdchecker:
        raise RuntimeError("Blender, usdzip, and usdchecker are required")
    rooms = {"scandinavian_minimal": nordic_gallery(), "industrial_loft": contemporary_gallery()}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    generated_textures = SOURCE_DIR / "textures"
    generated_textures.mkdir(parents=True, exist_ok=True)
    for texture in (SCRIPT_DIR / "textures").glob("*.png"):
        shutil.copy2(texture, generated_textures / texture.name)
    ANDROID_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    IOS_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    for room_id, (materials, geometry) in rooms.items():
        source = SOURCE_DIR / f"{room_id}.usda"
        glb = ANDROID_ASSET_DIR / f"{room_id}.glb"
        usdz = IOS_STAGING_DIR / f"{room_id}.usdz"
        write_usda(source, materials, geometry)
        run([blender, "--background", "--python", str(Path(__file__).resolve()), "--",
             "--worker", str(source), str(glb)])
        if usdz.exists():
            usdz.unlink()
        run([usdzip, "--arkitAsset", str(source), str(usdz)])
        run([usdchecker, "--arkit", str(usdz)])
        print(room_id, json.dumps(validate_glb(glb), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("SOURCE_USDA", "OUTPUT_GLB"))
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    if args.worker:
        blender_worker(Path(args.worker[0]), Path(args.worker[1]))
    else:
        build()


if __name__ == "__main__":
    main()
