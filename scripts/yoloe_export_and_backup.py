#!/usr/bin/env python3
"""
Load YOLOE, optionally run predict with retina_masks=True, export to CoreML or ONNX,
and back up the .pt and exported model to an external HD.

**YOLOE-11L-Seg-PF (prompt-free, this repo)**  
The checkpoint has `lrpc` (fixed vocabulary ~4585 classes fused in the graph). Ultralytics
does *not* support `YOLOE.set_classes()` on prompt-free models (see `YOLOEModel.set_classes`).
Do not use the minimal `set_classes(["office chair"])` recipe here — classes are already
locked in the PF weights.

**Text/prompt YOLOE checkpoints (no `lrpc`)**  
Use `--set-classes-from-json` only for those; that path needs the text encoder (see
Ultralytics `get_text_pe` / MobileCLIP) installed.

**Inference / export**  
- `model.fuse()` (Conv+BN fusion, and for `Detect(end2end=True)` removes the one-to-many
  branch). Safe to call before export; PF 11l seg heads are often already fused.
- Export with `nms=False` so post-processing matches end-to-end tensors (e.g. `(1, N, 6+nm)`
  style outputs) without embedding NMS in the graph — matches iOS `YoloEDetectionParser`
  (`dim2 < 100` branch).

Usage:
  python scripts/yoloe_export_and_backup.py
  python scripts/yoloe_export_and_backup.py --pt path/to/yoloe-11l-seg-pf.pt
  python scripts/yoloe_export_and_backup.py --backup-dir /Volumes/LaCie/Furnit_yoloe_backup
  python scripts/yoloe_export_and_backup.py --backup-only
  python scripts/yoloe_export_and_backup.py --format onnx

Environment:
  YOLOE_PT_PATH   Default path to .pt (default: android/yoloe-11l-seg-pf.pt or repo-relative)
  YOLOE_BACKUP_ROOT  Default backup root (default: /Volumes/LaCie)

iOS CoreML (avoid export failures):
  Use a conda env with torch≈2.7 coremltools 7–9 (e.g. ``coreml-py311``). If this repo is on
  ``PYTHONPATH``, local ``executorch/`` can shadow site-packages and break ``import coremltools``.
  Run from ``/tmp`` with a minimal path, e.g.::

    cd /tmp && env PYTHONPATH=/path/to/env/lib/python3.11/site-packages \\
      /path/to/env/bin/python /path/to/repo/scripts/yoloe_export_and_backup.py --format coreml ...
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Repo root (script lives in scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSES_JSON = REPO_ROOT / "Furnit" / "Views" / "FurnitureFit" / "classes.json"


def _patch_ultralytics_head_fuse_noop():
    """Disable fuse() on *head module classes* so some YOLOE-26L CoreML traces do not crash.
    Not needed for yoloe-11l-seg-pf. Use only with --legacy-26l-coreml-patch."""
    try:
        from ultralytics.nn.modules import head

        for name in dir(head):
            cls = getattr(head, name)
            if isinstance(cls, type) and hasattr(cls, "fuse"):
                cls.fuse = lambda self, *args, **kwargs: None
    except Exception:
        pass


def _head_has_lrpc(model) -> bool:
    try:
        head = model.model.model[-1]
        return hasattr(head, "lrpc") and head.lrpc is not None
    except Exception:
        return False


def load_ordered_class_names(classes_json: Path) -> list[str]:
    """Load classes.json with keys \"0\"..\"N-1\" into a list in index order."""
    with open(classes_json, encoding="utf-8") as f:
        d = json.load(f)
    n = len(d)
    names = []
    for i in range(n):
        key = str(i)
        if key not in d:
            raise ValueError(f"classes.json missing key {key!r} (expected contiguous 0..{n - 1})")
        names.append(d[key])
    return names


def main():
    parser = argparse.ArgumentParser(
        description="YOLOE: load, optional predict, export CoreML/ONNX (nms=False), backup"
    )
    parser.add_argument("--pt", type=str, default=os.environ.get("YOLOE_PT_PATH", ""),
                        help="Path to yoloe .pt file (default: YOLOE_PT_PATH or android/yoloe-11l-seg-pf.pt)")
    parser.add_argument("--backup-dir", type=str, default="",
                        help="Backup directory (default: YOLOE_BACKUP_ROOT/Furnit_yoloe_backup_YYYYMMDD_HHMM)")
    parser.add_argument("--backup-root", type=str, default=os.environ.get("YOLOE_BACKUP_ROOT", "/Volumes/LaCie"),
                        help="Backup root for default backup dir (default: /Volumes/LaCie)")
    parser.add_argument("--skip-export", action="store_true", help="Do not export to CoreML/ONNX")
    parser.add_argument("--backup-only", action="store_true", help="Only copy .pt to backup (no export, no predict)")
    parser.add_argument("--verify-image", type=str, default="",
                        help="Run predict with retina_masks=True on this image (optional)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Export image size (default: 1280)")
    parser.add_argument(
        "--format",
        type=str,
        choices=("coreml", "onnx"),
        default="coreml",
        help="Export format (default: coreml). ONNX uses nms=False for end-to-end tensors.",
    )
    parser.add_argument(
        "--legacy-26l-coreml-patch",
        action="store_true",
        help="Apply no-op fuse on head module classes (old 26L CoreML workaround). Off by default.",
    )
    parser.add_argument(
        "--set-classes-from-json",
        action="store_true",
        help="Call set_classes() using --classes-json (only for non-prompt-free YOLOE; requires text encoder deps).",
    )
    parser.add_argument(
        "--classes-json",
        type=str,
        default=str(DEFAULT_CLASSES_JSON),
        help=f"Ordered classes file (default: {DEFAULT_CLASSES_JSON})",
    )
    args = parser.parse_args()

    pt_path = args.pt or str(REPO_ROOT / "android" / "yoloe-11l-seg-pf.pt")
    pt_path = Path(pt_path)
    if not pt_path.is_absolute():
        pt_path = REPO_ROOT / pt_path
    if not pt_path.exists():
        print(f"Error: .pt not found: {pt_path}", file=sys.stderr)
        sys.exit(1)

    # Backup dir: explicit --backup-dir always wins; else timestamped folder under --backup-root when exporting/backup-only
    backup_dir = None
    if args.backup_dir:
        backup_dir = Path(args.backup_dir).resolve()
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"Backup directory: {backup_dir}")
    elif args.backup_only or not args.skip_export:
        from datetime import datetime

        backup_root = Path(args.backup_root)
        if not backup_root.exists():
            print(f"Error: Backup root not found (mount external HD?): {backup_root}", file=sys.stderr)
            sys.exit(1)
        stamp = datetime.now().strftime("%Y%m%d_%H")
        backup_dir = (backup_root / f"Furnit_yoloe_backup_{stamp}").resolve()
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"Backup directory: {backup_dir}")

    if args.backup_only:
        dest_pt = backup_dir / pt_path.name
        print(f"Copying {pt_path} -> {dest_pt}")
        shutil.copy2(pt_path, dest_pt)
        print("Backup done.")
        return

    # Import after possible env setup (disable Ultralytics auto-upgrade of coremltools)
    try:
        import ultralytics.hub.utils as _hub_utils

        _hub_utils.ONLINE = False
    except Exception:
        pass
    if args.legacy_26l_coreml_patch:
        _patch_ultralytics_head_fuse_noop()
        print("Applied --legacy-26l-coreml-patch (head fuse no-op).")
    from ultralytics import YOLOE

    print(f"Loading YOLOE: {pt_path}")
    model = YOLOE(str(pt_path))

    if _head_has_lrpc(model):
        print(
            "Prompt-free checkpoint: head has `lrpc` — class vocabulary is fixed in the graph. "
            "Skipping set_classes (Ultralytics does not support set_classes on PF models)."
        )
        if args.set_classes_from_json:
            print(
                "Warning: --set-classes-from-json ignored for prompt-free models. "
                "Use a text-prompt YOLOE checkpoint without lrpc if you need set_classes.",
                file=sys.stderr,
            )
    elif args.set_classes_from_json:
        classes_path = Path(args.classes_json)
        if not classes_path.is_absolute():
            classes_path = REPO_ROOT / classes_path
        names = load_ordered_class_names(classes_path)
        print(f"set_classes from {classes_path} ({len(names)} names)…")
        model.set_classes(names)

    # Conv+BN fusion; for end2end Detect heads also removes one-to-many branch when applicable.
    print("model.fuse() …")
    model.fuse()

    # Optional: predict with retina_masks=True
    if args.verify_image:
        verify_path = Path(args.verify_image)
        if not verify_path.is_absolute():
            verify_path = REPO_ROOT / verify_path
        if verify_path.exists():
            print(f"Running predict with retina_masks=True on {verify_path}")
            results = model.predict(str(verify_path), retina_masks=True, imgsz=args.imgsz)
            for r in results:
                if hasattr(r, "masks") and r.masks is not None:
                    print(f"  Masks shape: {r.masks.data.shape if hasattr(r.masks, 'data') else 'N/A'}")
        else:
            print(f"Warning: verify image not found: {verify_path}")

    exported_path = None
    if not args.skip_export:
        fmt = args.format
        print(f"Exporting to {fmt.upper()} (nms=False)…")
        exported_path = model.export(
            format=fmt,
            imgsz=args.imgsz,
            batch=1,
            nms=False,
            half=False,
            simplify=True,
        )
        if isinstance(exported_path, (list, tuple)):
            exported_path = exported_path[0] if exported_path else None
        if exported_path:
            print(f"Exported: {exported_path}")

    # Backup: .pt + exported artifact when backup_dir is set (export / backup-only / explicit --backup-dir)
    if backup_dir is not None:
        dest_pt = backup_dir / pt_path.name
        shutil.copy2(pt_path, dest_pt)
        print(f"Backed up .pt -> {dest_pt}")
        if exported_path:
            exported_path = Path(exported_path)
            if exported_path.exists():
                dest_export = backup_dir / exported_path.name
                if exported_path.is_dir():
                    if dest_export.exists():
                        shutil.rmtree(dest_export)
                    shutil.copytree(exported_path, dest_export)
                else:
                    shutil.copy2(exported_path, dest_export)
                print(f"Backed up export -> {dest_export}")
    print("Done.")


if __name__ == "__main__":
    main()
