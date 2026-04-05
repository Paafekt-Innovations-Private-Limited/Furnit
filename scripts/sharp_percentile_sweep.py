#!/usr/bin/env python3
"""
sharp_percentile_sweep.py

Reads a SHARP classic PLY, projects all splat positions onto floor-aligned axes,
and sweeps P1–P50 low / P50–P99 high to find which trim gives dimensions
closest to tape-measured ground truth.

Usage:
    python scripts/sharp_percentile_sweep.py <ply_path> [--height 2.9] [--depth 3.06] [--width 4.0]

For parity with iOS: pass ``--floor-normal`` from ``RoomGeometryEngine`` / extractRoomModel logs
(scene-space floor **before** room-space alignment). Built-in RANSAC uses world-Y bottom 20% only;
tilted SHARP rooms often disagree with Swift — wrong normal ⇒ wrong projections and PLY extents.
"""

import argparse
import struct
from pathlib import Path

import numpy as np


def read_classic_ply(path: str) -> np.ndarray:
    """Read vertex positions (x, y, z) from a binary little-endian PLY."""
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break

        vertex_count = 0
        properties = []
        in_vertex = False
        for line in header.decode("ascii", errors="replace").splitlines():
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element") and in_vertex:
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                parts = line.split()
                properties.append((parts[1], parts[2]))

        print(f"[PLY] {vertex_count} vertices, {len(properties)} properties per vertex")

        type_map = {
            "float": "f",
            "double": "d",
            "uchar": "B",
            "uint8": "B",
            "short": "h",
            "ushort": "H",
            "int": "i",
            "uint": "I",
        }
        fmt = "<" + "".join(type_map.get(p[0], "f") for p in properties)
        stride = struct.calcsize(fmt)

        names = [p[1] for p in properties]
        ix = names.index("x")
        iy = names.index("y")
        iz = names.index("z")

        data = f.read(stride * vertex_count)
        positions = np.zeros((vertex_count, 3), dtype=np.float32)
        for i in range(vertex_count):
            vals = struct.unpack_from(fmt, data, i * stride)
            positions[i] = [vals[ix], vals[iy], vals[iz]]

    return positions


def build_floor_basis(floor_normal: np.ndarray, _floor_origin: np.ndarray):
    """Build orthonormal basis: up=floor_normal, right/forward span the floor."""
    up = floor_normal / np.linalg.norm(floor_normal)
    world_z = np.array([0, 0, 1], dtype=np.float32)
    world_x = np.array([1, 0, 0], dtype=np.float32)
    candidate = world_z if abs(np.dot(up, world_z)) < 0.9 else world_x
    right = np.cross(candidate, up)
    right /= np.linalg.norm(right)
    forward = np.cross(up, right)
    return right, up, forward


def project_to_floor_space(positions, floor_normal, floor_origin):
    """Project all positions into floor-aligned (right, up, forward) space."""
    right, up, forward = build_floor_basis(floor_normal, floor_origin)
    d = positions - floor_origin[np.newaxis, :]
    xs = d @ right
    ys = d @ up
    zs = d @ forward
    return xs, ys, zs


def sweep_percentiles(
    xs,
    ys,
    zs,
    tape_width=None,
    tape_height=None,
    tape_depth=None,
    lo_range=range(1, 20),
    hi_range=range(80, 100),
):
    """Test every (lo, hi) percentile pair. Returns sorted results."""
    results = []

    for lo in lo_range:
        for hi in hi_range:
            if lo >= hi:
                continue

            w = np.percentile(xs, hi) - np.percentile(xs, lo)
            h = np.percentile(ys, hi) - np.percentile(ys, lo)
            depth = np.percentile(zs, hi) - np.percentile(zs, lo)

            errors = {}
            total_err = 0.0
            count = 0
            if tape_height is not None:
                err = abs(h - tape_height)
                errors["h_err"] = err
                total_err += err
                count += 1
            if tape_depth is not None:
                err = abs(depth - tape_depth)
                errors["d_err"] = err
                total_err += err
                count += 1
            if tape_width is not None:
                err = abs(w - tape_width)
                errors["w_err"] = err
                total_err += err
                count += 1

            avg_err = total_err / max(count, 1)

            results.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "W": w,
                    "H": h,
                    "D": depth,
                    "avg_err": avg_err,
                    **errors,
                }
            )

    results.sort(key=lambda r: r["avg_err"])
    return results


def ransac_floor(positions, iterations=500, threshold=0.04):
    """Simple RANSAC floor plane from bottom 20% of Y range."""
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    cutoff = y_min + (y_max - y_min) * 0.20
    candidates = positions[positions[:, 1] <= cutoff]

    if len(candidates) < 3:
        return np.array([0, 1, 0], dtype=np.float32), np.mean(candidates, axis=0)

    best_inliers = 0
    best_normal = np.array([0, 1, 0], dtype=np.float32)
    best_point = candidates[0]

    rng = np.random.default_rng(42)
    for _ in range(iterations):
        idx = rng.choice(len(candidates), 3, replace=False)
        p0, p1, p2 = candidates[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n /= norm
        if n[1] < 0:
            n = -n

        dists = np.abs((candidates - p0) @ n)
        inliers = int(np.sum(dists < threshold))
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = n
            best_point = p0

    print(f"[RANSAC] floor normal={best_normal}, inliers={best_inliers}/{len(candidates)}")
    return best_normal, best_point


def print_scale_free_analysis(ys, zs, tape_height, tape_depth):
    """Show what percentile gives the tape value WITHOUT any sceneToMeters."""
    print("\n" + "=" * 70)
    print("SCALE-FREE: At which percentile does raw SU match tape metres?")
    print("(Only valid if SHARP scene units ≈ metres for this scene)")
    print("=" * 70)

    for label, vals, tape in [("Height(Y)", ys, tape_height), ("Depth(Z)", zs, tape_depth)]:
        if tape is None:
            continue
        print(f"\n  {label} — looking for span ≈ {tape:.3f}")
        for hi_p in [95, 96, 97, 98, 99]:
            hi_val = np.percentile(vals, hi_p)
            for lo_p in [1, 2, 3, 4, 5]:
                lo_val = np.percentile(vals, lo_p)
                span = hi_val - lo_val
                err = abs(span - tape)
                if err < 0.15:
                    mark = "OK" if err < 0.05 else "~"
                    print(f"    P{lo_p}–P{hi_p}: span={span:.4f} err={err:.4f} {mark}")


def print_scaled_analysis(xs, ys, zs, tape_height, tape_depth, tape_width):
    """For each percentile pair: derive scale from height, check depth & width."""
    print("\n" + "=" * 70)
    print("SCALED: Derive sceneToMeters from height, check depth/width")
    print("=" * 70)

    if tape_height is None:
        print("  (skipped — no tape height provided)")
        return

    results = []
    for lo in range(1, 15):
        for hi in range(85, 100):
            h_su = np.percentile(ys, hi) - np.percentile(ys, lo)
            if h_su < 0.01:
                continue
            scale = tape_height / h_su
            d_m = (np.percentile(zs, hi) - np.percentile(zs, lo)) * scale
            w_m = (np.percentile(xs, hi) - np.percentile(xs, lo)) * scale

            errs = []
            if tape_depth is not None:
                errs.append(abs(d_m - tape_depth))
            if tape_width is not None:
                errs.append(abs(w_m - tape_width))
            avg = float(np.mean(errs)) if errs else 0.0

            results.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "scale": scale,
                    "h_su": h_su,
                    "W_m": w_m,
                    "H_m": tape_height,
                    "D_m": d_m,
                    "avg_err": avg,
                }
            )

    results.sort(key=lambda r: r["avg_err"])
    print("\n  Top 15 (sorted by avg error vs tape depth/width):\n")
    print(f"  {'P_lo':>4} {'P_hi':>4} | {'scale':>7} | {'W_m':>6} × {'H_m':>5} × {'D_m':>6} | {'err':>6}")
    print(f"  {'-'*4} {'-'*4} | {'-'*7} | {'-'*6}   {'-'*5}   {'-'*6} | {'-'*6}")
    for r in results[:15]:
        print(
            f"  P{r['lo']:>2}  P{r['hi']:>2} | {r['scale']:>7.4f} | "
            f"{r['W_m']:>6.3f} × {r['H_m']:>5.3f} × {r['D_m']:>6.3f} | {r['avg_err']:>6.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="SHARP PLY percentile sweep")
    parser.add_argument("ply", help="Path to classic PLY file")
    parser.add_argument("--height", type=float, default=None, help="Tape-measured ceiling height (m)")
    parser.add_argument("--depth", type=float, default=None, help="Tape-measured room depth (m)")
    parser.add_argument("--width", type=float, default=None, help="Tape-measured room width (m)")
    parser.add_argument(
        "--floor-normal",
        type=float,
        nargs=3,
        default=None,
        metavar=("NX", "NY", "NZ"),
        help="Scene-space floor normal (3 floats) — use Swift log value for apples-to-apples with measurePLYExtents.",
    )
    args = parser.parse_args()

    ply_path = Path(args.ply)
    if not ply_path.is_file():
        raise SystemExit(f"PLY not found: {ply_path}")

    print(f"\n{'='*70}")
    print("SHARP PLY Percentile Analyzer")
    print(f"{'='*70}")
    print(f"  File:   {ply_path}")
    print(f"  Tape:   W={args.width}  H={args.height}  D={args.depth}")
    print()

    positions = read_classic_ply(str(ply_path))
    print(f"[DATA] {len(positions)} splats loaded")
    print(
        f"[DATA] AABB: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] "
        f"Y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] "
        f"Z[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]"
    )

    if args.floor_normal:
        floor_normal = np.array(args.floor_normal, dtype=np.float32)
        y_cut = positions[:, 1].min() + (positions[:, 1].max() - positions[:, 1].min()) * 0.2
        floor_origin = positions[positions[:, 1] <= y_cut].mean(axis=0)
        print(f"[FLOOR] manual normal={floor_normal}")
    else:
        floor_normal, floor_origin = ransac_floor(positions)

    print(f"[FLOOR] origin={floor_origin}")

    xs, ys, zs = project_to_floor_space(positions, floor_normal, floor_origin)
    print("\n[PROJECTED] floor-aligned extents (full):")
    print(f"  Width  (right):   {xs.min():.3f} to {xs.max():.3f} = {xs.max()-xs.min():.3f} su")
    print(f"  Height (up):      {ys.min():.3f} to {ys.max():.3f} = {ys.max()-ys.min():.3f} su")
    print(f"  Depth  (forward): {zs.min():.3f} to {zs.max():.3f} = {zs.max()-zs.min():.3f} su")

    print("\n[PERCENTILES] per-axis spans:")
    print(f"  {'Plo-Phi':>8} | {'W_su':>8} {'H_su':>8} {'D_su':>8}")
    print(f"  {'-'*8} | {'-'*8} {'-'*8} {'-'*8}")
    for lo, hi in [(0, 100), (1, 99), (2, 98), (3, 97), (5, 95), (10, 90)]:
        w = np.percentile(xs, hi) - np.percentile(xs, lo)
        h = np.percentile(ys, hi) - np.percentile(ys, lo)
        d = np.percentile(zs, hi) - np.percentile(zs, lo)
        print(f"  P{lo:>2}–P{hi:<2} | {w:>8.4f} {h:>8.4f} {d:>8.4f}")

    if args.height or args.depth:
        print_scale_free_analysis(ys, zs, args.height, args.depth)

    print_scaled_analysis(xs, ys, zs, args.height, args.depth, args.width)

    if args.height or args.depth or args.width:
        print(f"\n{'='*70}")
        print("BRUTE FORCE: raw SU treated as metres (no scale)")
        print(f"{'='*70}")
        results = sweep_percentiles(xs, ys, zs, args.width, args.height, args.depth)
        print("\n  Top 10 closest to tape:\n")
        print(f"  {'P_lo':>4} {'P_hi':>4} | {'W':>7} × {'H':>7} × {'D':>7} | {'err':>6}")
        print(f"  {'-'*4} {'-'*4} | {'-'*7}   {'-'*7}   {'-'*7} | {'-'*6}")
        for r in results[:10]:
            print(
                f"  P{r['lo']:>2}  P{r['hi']:>2} | "
                f"{r['W']:>7.3f} × {r['H']:>7.3f} × {r['D']:>7.3f} | {r['avg_err']:>6.3f}"
            )


if __name__ == "__main__":
    main()
