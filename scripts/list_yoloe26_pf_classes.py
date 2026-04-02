#!/usr/bin/env python3
"""
List every class id (0-based) and name for the YOLOE-26L prompt-free (PF) vocabulary
used in this app with ``yoloe-26l-seg-pf_seg_o2m`` (Core ML + ONNX).

**Source of truth in-repo:** ``Furnit/Views/FurnitureFit/classes.json`` (mirrored at
``android/app/src/main/assets/classes.json``). Furniture Fit loads this bundle file on
iOS; Android uses the same JSON. The PF checkpoint embeds a fixed ~4585-class head
(lrpc); labels must match this ordering.

Examples:
  python3 scripts/list_yoloe26_pf_classes.py
  python3 scripts/list_yoloe26_pf_classes.py --format json --out /tmp/yoloe26_classes.json
  python3 scripts/list_yoloe26_pf_classes.py --onnx Furnit/Resources/yoloe-26l-seg-pf_seg_o2m.onnx
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSES_JSON = REPO_ROOT / "Furnit" / "Views" / "FurnitureFit" / "classes.json"
DEFAULT_ONNX_CANDIDATES = (
    REPO_ROOT / "Furnit" / "Resources" / "yoloe-26l-seg-pf_seg_o2m.onnx",
    REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "yoloe-26l-seg-pf_seg_o2m.onnx",
    REPO_ROOT / ".build" / "yoloe-26l-seg-pf_seg_o2m.onnx",
)


def load_ordered_class_names(classes_json: Path) -> list[str]:
    with open(classes_json, encoding="utf-8") as f:
        d = json.load(f)
    n = len(d)
    names: list[str] = []
    for i in range(n):
        key = str(i)
        if key not in d:
            raise ValueError(f"classes.json missing key {key!r} (expected contiguous 0..{n - 1})")
        names.append(d[key])
    return names


def num_classes_from_onnx(onnx_path: Path) -> int:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.get_outputs()
    if len(outs) < 1:
        raise ValueError("ONNX has no outputs")
    shape = outs[0].shape
    # det [1, 4+nc+nm, 8400] — second dim is channel count
    if len(shape) < 2 or shape[1] is None:
        raise ValueError(f"Unexpected det output shape: {shape}")
    channels = int(shape[1])
    nm = 32
    nc = channels - 4 - nm
    if nc <= 0:
        raise ValueError(f"Cannot infer class count from channels={channels} (expected 4+nc+{nm})")
    return nc


def resolve_default_onnx(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_file() else None
    for candidate in DEFAULT_ONNX_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print all YOLOE-26L PF class ids and names (bundled classes.json)."
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=DEFAULT_CLASSES_JSON,
        help=f"Path to classes.json (default: {DEFAULT_CLASSES_JSON})",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Optional: verify class count vs det output channels (4+nc+32). "
        "If omitted, uses first existing path among Resources, android assets, .build.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "tsv", "csv", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write to file instead of stdout.")
    args = parser.parse_args()

    if not args.classes.is_file():
        raise SystemExit(f"Missing classes.json: {args.classes}")

    names = load_ordered_class_names(args.classes)
    onnx_path = resolve_default_onnx(args.onnx)

    if args.onnx is not None and not args.onnx.is_file():
        print(f"Warning: --onnx file not found: {args.onnx}", file=sys.stderr)

    if onnx_path is not None:
        try:
            nc_onnx = num_classes_from_onnx(onnx_path)
        except ImportError:
            print(
                "Warning: onnxruntime not installed; skip ONNX count check "
                "(pip install onnxruntime).",
                file=sys.stderr,
            )
        else:
            if nc_onnx != len(names):
                raise SystemExit(
                    f"Mismatch: classes.json has {len(names)} entries but "
                    f"{onnx_path.name} implies nc={nc_onnx} (det channels 4+nc+32)."
                )
            print(f"OK: ONNX {onnx_path} nc={nc_onnx} matches classes.json", file=sys.stderr)

    rows = [{"id": i, "name": names[i]} for i in range(len(names))]

    stream = open(args.out, "w", encoding="utf-8", newline="") if args.out else sys.stdout
    try:
        if args.format == "json":
            json.dump(rows, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        elif args.format == "tsv":
            stream.write("id\tname\n")
            for r in rows:
                stream.write(f"{r['id']}\t{r['name']}\n")
        elif args.format == "csv":
            w = csv.DictWriter(stream, fieldnames=["id", "name"])
            w.writeheader()
            w.writerows(rows)
        else:
            for r in rows:
                stream.write(f"{r['id']}\t{r['name']}\n")
    finally:
        if args.out:
            stream.close()

    if not args.out:
        print(f"# total classes: {len(names)}", file=sys.stderr)


if __name__ == "__main__":
    main()
