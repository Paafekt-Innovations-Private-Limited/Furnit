#!/usr/bin/env python3
"""
Export `yoloe-26l-seg-pf.pt` to Core ML `.mlpackage` using the standard Ultralytics route.

Flow:
- load `YOLO(...)`
- set `head.end2end = True` when present
- call `export(format="coreml", nms=False, end2end=True, ...)`
- do not manually call `model.fuse()`
- patch the Ultralytics YOLOE fuse path only to safely skip `_fuse_tp(...)` when
  a YOLOESegment26 head exposes `cv3`/`cv4` or `bn_head` as `None`

Inference / preprocessing note (iOS):
- Ultralytics letterbox uses padding RGB **(114, 114, 114)**. The app uses the same fill (`YoloUltralyticsLetterboxFill`).
- Export **must** use **`nms=False`** so the graph matches end-to-end tensors parsed in iOS.
- After changing export flags, replace the bundled `.mlpackage` in Xcode and rebuild the app.

11L workflow:
- see **`scripts/yoloe_export_and_backup.py`**
"""

from __future__ import annotations

import argparse
from pathlib import Path
DEFAULT_PT = Path("/Users/al/Downloads/yoloe-26l-seg-pf.pt")
DEFAULT_OUT = Path("/Users/al/Downloads/yoloe-26l-seg-pf.mlpackage")


def _patch_yoloe_none_safe_fuse() -> None:
    """Make Ultralytics YOLOE export skip text-prompt fusion for prompt-free `lrpc` heads."""
    from ultralytics.nn.modules import head as yolo_head

    detect_class = getattr(yolo_head, "YOLOEDetect", None)
    if detect_class is None:
        print("Warning: YOLOEDetect class not found; no fuse patch applied.")
        return

    original_fuse = detect_class.fuse
    original_fuse_tp = detect_class._fuse_tp
    if getattr(detect_class, "_furnit_none_safe_fuse_patch", False):
        return

    def _safe_fuse(self, txt_feats=None):
        if txt_feats is not None and hasattr(self, "lrpc"):
            print("Skipping YOLOE text-prompt fuse for prompt-free lrpc head.")
            self.is_fused = True
            return
        return original_fuse(self, txt_feats)

    def _safe_fuse_tp(self, txt_feats, cls_head, bn_head):
        if cls_head is None or bn_head is None:
            print("Skipping YOLOE _fuse_tp() for missing cv3/cv4 or bn head.")
            return
        return original_fuse_tp(self, txt_feats, cls_head, bn_head)

    detect_class.fuse = _safe_fuse
    detect_class._fuse_tp = _safe_fuse_tp
    detect_class._furnit_none_safe_fuse_patch = True
    print("Applied YOLOE prompt-free export fuse patch.")


def _load_model_with_end2end(pt_path: Path):
    from ultralytics import YOLO

    print(f"Loading: {pt_path}")
    model = YOLO(str(pt_path))
    head = model.model.model[-1]
    print(f"Head type: {type(head).__name__}")
    print(f"Head end2end before: {getattr(head, 'end2end', None)}")
    if hasattr(head, "end2end"):
        head.end2end = True
    print(f"Head end2end after: {getattr(head, 'end2end', None)}")
    return model


def _export_coreml(model, image_size: int):
    print("Exporting Core ML package (nms=False, end2end=True, batch=1, simplify=True) ...")
    return model.export(
        format="coreml",
        imgsz=image_size,
        batch=1,
        nms=False,
        half=False,
        simplify=True,
        end2end=True,
    )


def _print_coreml_metadata(package_path: Path) -> None:
    import coremltools as ct

    model = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()
    print("\nCore ML inputs:")
    for feature in spec.description.input:
        feature_type = feature.type.WhichOneof("Type")
        if feature_type == "multiArrayType":
            detail = list(feature.type.multiArrayType.shape)
        elif feature_type == "imageType":
            image_type = feature.type.imageType
            detail = {
                "width": image_type.width,
                "height": image_type.height,
                "colorSpace": image_type.colorSpace,
            }
        else:
            detail = feature_type
        print(f"  {feature.name}: {detail}")
    print("Core ML outputs:")
    for feature in spec.description.output:
        shape = list(feature.type.multiArrayType.shape)
        print(f"  {feature.name}: {shape}")


def _run_coreml_smoke_test(package_path: Path, image_size: int) -> None:
    from PIL import Image
    import numpy as np
    import coremltools as ct

    model = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    image = Image.new("RGB", (image_size, image_size), (114, 114, 114))
    outputs = model.predict({"image": image})
    print("\nCore ML smoke test:")
    for name, value in outputs.items():
        array = np.asarray(value)
        print(
            f"  {name}: shape={array.shape} "
            f"mean={float(array.mean()):.6f} "
            f"min={float(array.min()):.6f} "
            f"max={float(array.max()):.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLOE-26L-Seg-PF to Core ML.")
    parser.add_argument("--pt", type=Path, default=DEFAULT_PT, help="Path to .pt checkpoint")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output .mlpackage path",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Export image size")
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip MLModel.predict smoke test after export",
    )
    args = parser.parse_args()

    if not args.pt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.pt}")

    _patch_yoloe_none_safe_fuse()
    model = _load_model_with_end2end(args.pt)
    exported_path = _export_coreml(model, args.imgsz)
    print("Standard export path succeeded.")
    exported_path = Path(exported_path)

    if exported_path.resolve() != args.out.resolve():
        import shutil

        if args.out.exists():
            shutil.rmtree(args.out)
        shutil.move(str(exported_path), str(args.out))
        exported_path = args.out

    print(f"Saved: {exported_path}")
    _print_coreml_metadata(exported_path)

    if not args.skip_smoke_test:
        _run_coreml_smoke_test(exported_path, args.imgsz)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
