#!/usr/bin/env python3
"""
Room Completion Pipeline
========================
Takes a single-image depth-reconstructed .obj mesh and:
1. Detects wall planes (RANSAC on normals)
2. Extrudes left/right walls by 3m
3. Fills the occlusion hole behind the chair
4. Exports completed .obj

Requires: pip install open3d numpy trimesh
"""

import numpy as np
import open3d as o3d
import copy
import sys
from pathlib import Path

INPUT_OBJ = sys.argv[1] if len(sys.argv) > 1 else "/Users/al/Documents/tries01/Furnit/FurnitTests/room.obj"
OUTPUT_OBJ = sys.argv[2] if len(sys.argv) > 2 else "/Users/al/Documents/tries01/Furnit/FurnitTests/room_completed.obj"


def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    print(f"Loaded: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    return mesh


def detect_wall_plane(mesh, target_normal, threshold=0.85):
    """Find vertices whose normals align with target_normal (dot > threshold)."""
    normals = np.asarray(mesh.vertex_normals)
    vertices = np.asarray(mesh.vertices)
    dots = normals @ target_normal
    mask = dots > threshold
    wall_indices = np.where(mask)[0]
    wall_points = vertices[wall_indices]
    return wall_indices, wall_points


def get_boundary_of_wall(mesh, wall_indices):
    """Get the outermost edge vertices of a wall segment."""
    vertices = np.asarray(mesh.vertices)
    wall_verts = vertices[wall_indices]
    if len(wall_verts) == 0:
        return np.array([])

    y_min = wall_verts[:, 1].min()
    y_max = wall_verts[:, 1].max()
    z_min = wall_verts[:, 2].min()
    z_max = wall_verts[:, 2].max()
    x_val = np.median(wall_verts[:, 0])

    return wall_verts, x_val, y_min, y_max, z_min, z_max


def extrude_wall(mesh, wall_indices, direction, distance, include_floor_ceiling=True):
    """
    Extrude a wall plane outward by `distance` meters.
    Creates new wall faces + floor/ceiling extension.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    wall_verts, x_val, y_min, y_max, z_min, z_max = get_boundary_of_wall(mesh, wall_indices)
    if len(wall_verts) == 0:
        print(f"  Warning: no wall vertices found for direction {direction}")
        return mesh

    offset = direction * distance
    new_x = x_val + offset[0]
    height = y_max - y_min
    depth = z_max - z_min

    n_subdiv_y = max(2, int(height / 0.5))
    n_subdiv_z = max(2, int(depth / 0.5))

    new_verts = []
    new_faces = []
    base_idx = len(vertices)

    # Generate the extruded wall panel (grid)
    for iy in range(n_subdiv_y + 1):
        for iz in range(n_subdiv_z + 1):
            y = y_min + (y_max - y_min) * iy / n_subdiv_y
            z = z_min + (z_max - z_min) * iz / n_subdiv_z
            new_verts.append([new_x, y, z])

    # Triangulate the wall grid
    for iy in range(n_subdiv_y):
        for iz in range(n_subdiv_z):
            i00 = base_idx + iy * (n_subdiv_z + 1) + iz
            i01 = i00 + 1
            i10 = i00 + (n_subdiv_z + 1)
            i11 = i10 + 1
            new_faces.append([i00, i01, i11])
            new_faces.append([i00, i11, i10])

    # Connecting strip: bridge old wall edge to new wall edge
    strip_base = base_idx + len(new_verts)
    old_edge_z = np.linspace(z_min, z_max, n_subdiv_z + 1)

    # Top bridge (ceiling level)
    for iz in range(n_subdiv_z + 1):
        new_verts.append([x_val, y_max, old_edge_z[iz]])
    for iz in range(n_subdiv_z + 1):
        new_verts.append([new_x, y_max, old_edge_z[iz]])

    ceil_old_base = strip_base
    ceil_new_base = strip_base + (n_subdiv_z + 1)
    for iz in range(n_subdiv_z):
        new_faces.append([ceil_old_base + iz, ceil_old_base + iz + 1, ceil_new_base + iz + 1])
        new_faces.append([ceil_old_base + iz, ceil_new_base + iz + 1, ceil_new_base + iz])

    # Bottom bridge (floor level)
    floor_base = strip_base + 2 * (n_subdiv_z + 1)
    for iz in range(n_subdiv_z + 1):
        new_verts.append([x_val, y_min, old_edge_z[iz]])
    for iz in range(n_subdiv_z + 1):
        new_verts.append([new_x, y_min, old_edge_z[iz]])

    floor_old_base = floor_base
    floor_new_base = floor_base + (n_subdiv_z + 1)
    for iz in range(n_subdiv_z):
        new_faces.append([floor_old_base + iz, floor_new_base + iz, floor_new_base + iz + 1])
        new_faces.append([floor_old_base + iz, floor_new_base + iz + 1, floor_old_base + iz + 1])

    # Merge
    all_verts = np.vstack([vertices, np.array(new_verts)])
    all_faces = np.vstack([triangles, np.array(new_faces)])

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(all_verts)
    new_mesh.triangles = o3d.utility.Vector3iVector(all_faces)
    new_mesh.compute_vertex_normals()

    print(f"  Extruded wall: +{len(new_verts)} verts, +{len(new_faces)} faces")
    return new_mesh


def fill_occlusion_hole(mesh):
    """
    Fill holes in the mesh (occlusion gaps from the chair blocking depth).
    Uses Open3D's hole-filling via Poisson reconstruction on the existing points.
    """
    # Strategy: detect boundary edges, create a point cloud from them,
    # then fill with a planar patch

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Find boundary edges (edges appearing in only one triangle)
    edge_count = {}
    for tri in triangles:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1)%3]]))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    boundary_verts_idx = list(set(v for e in boundary_edges for v in e))
    boundary_points = vertices[boundary_verts_idx]

    if len(boundary_points) < 3:
        print("  No significant holes found")
        return mesh

    # Find the hole region: cluster boundary points that are in the
    # lower-right area (where the chair occlusion is)
    # The chair is in the +X, lower Y region
    median_x = np.median(vertices[:, 0])
    median_z = np.median(vertices[:, 2])

    hole_mask = (boundary_points[:, 0] > median_x * 0.3)
    hole_points = boundary_points[hole_mask]

    if len(hole_points) < 4:
        print("  Occlusion hole too small to fill")
        return mesh

    # Fill with a simple planar triangulation
    # Project hole boundary onto floor plane and wall plane
    y_floor = vertices[:, 1].min()
    z_min = hole_points[:, 2].min()
    z_max = hole_points[:, 2].max()
    x_min = hole_points[:, 0].min()
    x_max = hole_points[:, 0].max()

    # Create a grid fill for the floor gap
    n_x = max(2, int((x_max - x_min) / 0.3))
    n_z = max(2, int((z_max - z_min) / 0.3))

    fill_verts = []
    fill_faces = []
    fill_base = len(vertices)

    for ix in range(n_x + 1):
        for iz in range(n_z + 1):
            x = x_min + (x_max - x_min) * ix / n_x
            z = z_min + (z_max - z_min) * iz / n_z
            fill_verts.append([x, y_floor, z])

    for ix in range(n_x):
        for iz in range(n_z):
            i00 = fill_base + ix * (n_z + 1) + iz
            i01 = i00 + 1
            i10 = i00 + (n_z + 1)
            i11 = i10 + 1
            fill_faces.append([i00, i01, i11])
            fill_faces.append([i00, i11, i10])

    if fill_verts:
        all_verts = np.vstack([vertices, np.array(fill_verts)])
        all_faces = np.vstack([triangles, np.array(fill_faces)])
        mesh.vertices = o3d.utility.Vector3dVector(all_verts)
        mesh.triangles = o3d.utility.Vector3iVector(all_faces)
        mesh.compute_vertex_normals()
        print(f"  Hole fill: +{len(fill_verts)} verts, +{len(fill_faces)} faces")

    return mesh


def weld_vertices(mesh, tolerance=0.001):
    """Merge vertices within tolerance to make mesh watertight."""
    mesh = mesh.merge_close_vertices(tolerance)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    print(f"  After weld: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")
    return mesh


def main():
    print("=" * 60)
    print("ROOM COMPLETION PIPELINE")
    print("=" * 60)
    print(f"Input:  {INPUT_OBJ}")
    print(f"Output: {OUTPUT_OBJ}")
    print()

    if not Path(INPUT_OBJ).exists():
        print(f"ERROR: {INPUT_OBJ} not found")
        sys.exit(1)

    # Load
    mesh = load_mesh(INPUT_OBJ)

    # Detect + extrude left wall (+X normal = faces into room)
    print("\n[1/4] Detecting + extruding LEFT wall (+3m)...")
    left_idx, _ = detect_wall_plane(mesh, np.array([1.0, 0.0, 0.0]), threshold=0.8)
    print(f"  Found {len(left_idx)} left-wall vertices")
    if len(left_idx) > 10:
        mesh = extrude_wall(mesh, left_idx, np.array([-1.0, 0.0, 0.0]), 3.0)

    # Detect + extrude right wall (-X normal = faces into room)
    print("\n[2/4] Detecting + extruding RIGHT wall (+3m)...")
    right_idx, _ = detect_wall_plane(mesh, np.array([-1.0, 0.0, 0.0]), threshold=0.8)
    print(f"  Found {len(right_idx)} right-wall vertices")
    if len(right_idx) > 10:
        mesh = extrude_wall(mesh, right_idx, np.array([1.0, 0.0, 0.0]), 3.0)

    # Fill occlusion hole
    print("\n[3/4] Filling occlusion hole behind chair...")
    mesh = fill_occlusion_hole(mesh)

    # Weld
    print("\n[4/4] Welding vertices (1mm tolerance)...")
    mesh = weld_vertices(mesh)

    # Export
    print(f"\nExporting -> {OUTPUT_OBJ}")
    o3d.io.write_triangle_mesh(OUTPUT_OBJ, mesh)

    print("\n" + "=" * 60)
    print("DONE. Open in Blender/MeshLab/3dviewer.net to verify.")
    print("=" * 60)


if __name__ == "__main__":
    main()
