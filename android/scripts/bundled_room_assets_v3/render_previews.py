#!/usr/bin/env python3
"""Blender worker that renders quick visual checks for the generated Android GLBs."""

import sys
from pathlib import Path

import bpy
from mathutils import Vector


def point_camera(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def render(glb: Path, output: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    for material in bpy.data.materials:
        material.use_backface_culling = True
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
    center = (minimum + maximum) / 2
    width = maximum.x - minimum.x
    depth = maximum.y - minimum.y
    height = maximum.z - minimum.z

    bpy.ops.object.camera_add(location=(width * 0.11, minimum.y - depth * 1.15, center.z + height * 0.14))
    camera = bpy.context.object
    camera.data.lens = 32
    point_camera(camera, Vector((0, center.y, center.z * 0.92)))
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type="AREA", location=(-width * 0.35, minimum.y - 0.8, maximum.z + 1.8))
    key = bpy.context.object
    key.data.energy = 1300
    key.data.shape = "DISK"
    key.data.size = max(width, depth) * 0.7
    point_camera(key, center)
    bpy.ops.object.light_add(type="AREA", location=(width * 0.35, center.y, maximum.z - 0.2))
    fill = bpy.context.object
    fill.data.energy = 800
    fill.data.size = max(width, depth) * 0.5
    point_camera(fill, center)

    world = bpy.context.scene.world or bpy.data.worlds.new("Preview World")
    bpy.context.scene.world = world
    world.color = (0.055, 0.06, 0.065)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output)
    scene.render.film_transparent = False
    scene.view_settings.look = "AgX - Medium High Contrast"
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {glb.name}: bounds={tuple(minimum)}..{tuple(maximum)} -> {output}")


args = sys.argv[sys.argv.index("--") + 1:]
render(Path(args[0]), Path(args[1]))
