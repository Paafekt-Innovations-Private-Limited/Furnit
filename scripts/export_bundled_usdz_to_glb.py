#!/usr/bin/env python3
"""Export bundled iOS USDZ sample rooms to meter-scaled GLB for Android SceneView."""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def read_glb_mesh_bounds(glb_path: Path) -> tuple[list[float], list[float]]:
    with glb_path.open("rb") as handle:
        magic, _, length = struct.unpack("<4sII", handle.read(12))
        if magic != b"glTF":
            raise ValueError(f"Not a GLB: {glb_path}")
        chunks: list[tuple[bytes, bytes]] = []
        while handle.tell() < length:
            chunk_len, chunk_type = struct.unpack("<I4s", handle.read(8))
            chunks.append((chunk_type, handle.read(chunk_len)))
    import json

    gltf = json.loads(next(data for kind, data in chunks if kind == b"JSON"))
    accessors = gltf.get("accessors", [])
    buffer_views = gltf.get("bufferViews", [])
    bin_chunk = next((data for kind, data in chunks if kind == b"BIN\x00"), b"")
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            position_index = primitive.get("attributes", {}).get("POSITION")
            if position_index is None:
                continue
            accessor = accessors[position_index]
            if "min" in accessor and "max" in accessor:
                for axis in range(3):
                    mins[axis] = min(mins[axis], accessor["min"][axis])
                    maxs[axis] = max(maxs[axis], accessor["max"][axis])
                continue
            buffer_view_index = accessor.get("bufferView")
            if buffer_view_index is None:
                continue
            buffer_view = buffer_views[buffer_view_index]
            start = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
            count = accessor["count"]
            data = bin_chunk[start : start + count * 12]
            for index in range(count):
                offset = index * 12
                x, y, z = struct.unpack_from("<fff", data, offset)
                mins[0] = min(mins[0], x)
                mins[1] = min(mins[1], y)
                mins[2] = min(mins[2], z)
                maxs[0] = max(maxs[0], x)
                maxs[1] = max(maxs[1], y)
                maxs[2] = max(maxs[2], z)
    if mins[0] == float("inf"):
        raise ValueError(f"Could not determine mesh bounds for {glb_path}")
    return mins, maxs


def extract_usdz_archive(usdz_path: Path, work_dir: Path) -> Path:
    with zipfile.ZipFile(usdz_path) as archive:
        archive.extractall(work_dir)
        usdc_names = [
            name
            for name in archive.namelist()
            if name.endswith((".usdc", ".usd")) and not name.startswith("__MACOSX")
        ]
        if not usdc_names:
            raise RuntimeError(f"No USD root found in {usdz_path}")
        return work_dir / Path(usdc_names[0]).name


def apply_uniform_scale(scale_factor: float) -> None:
    import bpy

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("No mesh objects available for scaling")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.select_all(action="DESELECT")


def export_with_blender(
    usdc_path: Path,
    output_glb: Path,
    meters_per_unit: float,
    *,
    enable_draco: bool = False,
) -> None:
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)

    import_result = bpy.ops.wm.usd_import(
        filepath=str(usdc_path),
        import_cameras=False,
        import_curves=False,
        import_lights=False,
        import_materials=True,
        import_meshes=True,
        import_volumes=False,
        scale=1.0,
    )
    if "FINISHED" not in import_result:
        raise RuntimeError(f"USD import failed: {import_result}")

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("USD import produced no mesh objects")

    if abs(meters_per_unit - 1.0) > 1e-6:
        apply_uniform_scale(meters_per_unit)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bpy.ops.object.select_all(action="DESELECT")

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    export_result = bpy.ops.export_scene.gltf(
        filepath=str(output_glb),
        export_format="GLB",
        use_selection=False,
        export_apply=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_draco_mesh_compression_enable=enable_draco,
        export_draco_mesh_compression_level=6 if enable_draco else 0,
    )
    if "FINISHED" not in export_result:
        raise RuntimeError(f"GLB export failed: {export_result}")


def resolve_ktx_bin_dir() -> Path | None:
    local_tools = Path(__file__).resolve().parent / ".tools" / "ktx" / "bin"
    if (local_tools / "ktx").is_file():
        return local_tools
    if shutil.which("ktx"):
        return None
    return None


def ensure_ktx_on_path(env: dict[str, str]) -> None:
    ktx_bin_dir = resolve_ktx_bin_dir()
    if ktx_bin_dir is not None:
        env["PATH"] = f"{ktx_bin_dir}{os.pathsep}{env.get('PATH', '')}"
        ktx_lib_dir = ktx_bin_dir.parent / "lib"
        if ktx_lib_dir.is_dir():
            env["DYLD_LIBRARY_PATH"] = f"{ktx_lib_dir}{os.pathsep}{env.get('DYLD_LIBRARY_PATH', '')}"
    elif shutil.which("ktx", path=env.get("PATH")) is None:
        raise RuntimeError(
            "ktx CLI not found. Install Khronos KTX-Software or extract tools+library "
            "payloads from the Darwin arm64 .pkg into scripts/.tools/ktx/{bin,lib}/"
        )


def compress_glb_with_gltf_transform(
    source_glb: Path,
    output_glb: Path,
    texture_compress: str = "auto",
) -> None:
    """Post-process GLB for Android SceneView.

    SceneView/Filament gltfio does not decode KHR_texture_basisu (KTX2) in GLB textures —
    use webp/jpeg (default: webp). Draco meshes are supported.
    """
    npx = shutil.which("npx")
    if npx is None:
        raise RuntimeError("npx not found; install Node.js to run gltf-transform for KTX2 textures")

    env = os.environ.copy()
    ensure_ktx_on_path(env)

    with tempfile.TemporaryDirectory(prefix="furnit-glb-compress-") as temp_dir:
        intermediate = Path(temp_dir) / "intermediate.glb"
        shutil.copy2(source_glb, intermediate)
        command = [
            npx,
            "--yes",
            "@gltf-transform/cli",
            "optimize",
            str(intermediate),
            str(output_glb),
            "--compress",
            "draco",
            "--texture-compress",
            texture_compress,
            "--texture-size",
            "2048",
            "--flatten",
            "false",
            "--join",
            "false",
            "--instance",
            "false",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"gltf-transform optimize failed: {message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usdz", type=Path, required=True, help="Source USDZ path")
    parser.add_argument("--output", type=Path, required=True, help="Destination GLB path")
    parser.add_argument(
        "--meters-per-unit",
        type=float,
        default=0.01,
        help="USD metersPerUnit metadata (cozy_living_room_baked uses 0.01)",
    )
    parser.add_argument(
        "--draco",
        action="store_true",
        help="Enable Draco mesh compression (off by default; vintage/cozy load safer without it)",
    )
    parser.add_argument(
        "--skip-compress",
        action="store_true",
        help="Skip gltf-transform post-pass (Blender export only)",
    )
    parser.add_argument(
        "--skip-ktx2",
        action="store_true",
        help=argparse.SUPPRESS,  # legacy alias for --skip-compress
    )
    parser.add_argument(
        "--texture-compress",
        choices=("webp", "auto", "false"),
        default="auto",
        help="gltf-transform texture format. Use auto/jpeg path for Android SceneView (NOT ktx2/webp).",
    )
    args = parser.parse_args(argv)

    usdz_path = args.usdz.expanduser().resolve()
    output_glb = args.output.expanduser().resolve()
    if not usdz_path.is_file():
        print(f"Missing USDZ: {usdz_path}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="furnit-usdz-export-") as temp_dir:
        usdc_path = extract_usdz_archive(usdz_path, Path(temp_dir))
        raw_glb = Path(temp_dir) / "raw.glb"
        export_with_blender(usdc_path, raw_glb, args.meters_per_unit, enable_draco=args.draco)
        if args.skip_compress or args.skip_ktx2:
            shutil.copy2(raw_glb, output_glb)
        else:
            compress_glb_with_gltf_transform(raw_glb, output_glb, args.texture_compress)

    mins, maxs = read_glb_mesh_bounds(output_glb)
    size = [round(maxs[i] - mins[i], 3) for i in range(3)]
    print(f"Wrote {output_glb} ({output_glb.stat().st_size / 1e6:.1f} MB)")
    print(f"Bounds min={[round(v, 3) for v in mins]} max={[round(v, 3) for v in maxs]} size={size}")
    return 0


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401
    except ImportError:
        print(
            "Run inside Blender:\n"
            '  blender --background --python scripts/export_bundled_usdz_to_glb.py -- '
            "--usdz <path.usdz> --output <path.glb>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    cli_args = sys.argv
    if "--" in cli_args:
        cli_args = cli_args[cli_args.index("--") + 1 :]
    else:
        cli_args = []
    raise SystemExit(main(cli_args))
