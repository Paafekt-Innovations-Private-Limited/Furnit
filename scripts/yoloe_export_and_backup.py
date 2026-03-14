#!/usr/bin/env python3
"""
Load YOLOe, optionally run predict with retina_masks=True, export to CoreML,
and back up the .pt and exported model to an external HD.

Usage:
  python scripts/yoloe_export_and_backup.py
  python scripts/yoloe_export_and_backup.py --pt path/to/yoloe-11l-seg-pf.pt
  python scripts/yoloe_export_and_backup.py --backup-dir /Volumes/LaCie/Furnit_yoloe_backup
  python scripts/yoloe_export_and_backup.py --backup-only

Environment:
  YOLOE_PT_PATH   Default path to .pt (default: android/yoloe-11l-seg-pf.pt or repo-relative)
  YOLOE_BACKUP_ROOT  Default backup root (default: /Volumes/LaCie)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Repo root (script lives in scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent


def _patch_ultralytics_head_fuse():
    """Disable fuse on head modules so 26l export does not crash. No-op for 11l."""
    try:
        from ultralytics.nn.modules import head
        for name in dir(head):
            cls = getattr(head, name)
            if isinstance(cls, type) and hasattr(cls, "fuse"):
                cls.fuse = lambda self, *args, **kwargs: None
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="YOLOe: load, (optional) predict with retina_masks, export CoreML, backup to external HD")
    parser.add_argument("--pt", type=str, default=os.environ.get("YOLOE_PT_PATH", ""),
                        help="Path to yoloe .pt file (default: YOLOE_PT_PATH or android/yoloe-11l-seg-pf.pt)")
    parser.add_argument("--backup-dir", type=str, default="",
                        help="Backup directory (default: YOLOE_BACKUP_ROOT/Furnit_yoloe_backup_YYYYMMDD_HHMM)")
    parser.add_argument("--backup-root", type=str, default=os.environ.get("YOLOE_BACKUP_ROOT", "/Volumes/LaCie"),
                        help="Backup root for default backup dir (default: /Volumes/LaCie)")
    parser.add_argument("--skip-export", action="store_true", help="Do not export to CoreML")
    parser.add_argument("--backup-only", action="store_true", help="Only copy .pt to backup (no export, no predict)")
    parser.add_argument("--verify-image", type=str, default="",
                        help="Run predict with retina_masks=True on this image (optional)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Export image size (default: 1280)")
    args = parser.parse_args()

    pt_path = args.pt or str(REPO_ROOT / "android" / "yoloe-11l-seg-pf.pt")
    pt_path = Path(pt_path)
    if not pt_path.is_absolute():
        pt_path = REPO_ROOT / pt_path
    if not pt_path.exists():
        print(f"Error: .pt not found: {pt_path}", file=sys.stderr)
        sys.exit(1)

    # Backup dir
    backup_dir = None
    if args.backup_only or not args.skip_export:
        if args.backup_dir:
            backup_dir = Path(args.backup_dir).resolve()
        else:
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
    _patch_ultralytics_head_fuse()
    from ultralytics import YOLO

    print(f"Loading YOLOe: {pt_path}")
    model = YOLO(str(pt_path))

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
        print("Exporting to CoreML...")
        exported_path = model.export(
            format="coreml",
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

    # Backup: .pt + exported .mlpackage (ensure backup_dir set when we skipped export)
    if backup_dir is None:
        from datetime import datetime
        backup_root = Path(args.backup_root)
        if not backup_root.exists():
            print(f"Error: Backup root not found: {backup_root}", file=sys.stderr)
            sys.exit(1)
        stamp = datetime.now().strftime("%Y%m%d_%H")
        backup_dir = (backup_root / f"Furnit_yoloe_backup_{stamp}").resolve()
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"Backup directory: {backup_dir}")
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
