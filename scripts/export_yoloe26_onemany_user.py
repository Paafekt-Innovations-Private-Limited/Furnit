#!/usr/bin/env python3
"""
Export yoloe-26l-seg-pf.pt → ONNX + CoreML (.mlpackage) under repo `.build/`.

Goal: one-to-many head (raw boxes + class scores + mask coeffs + protos) for on-device NMS.

Handles:
  - fuse(): attempted for speed; PF heads may raise — skipped gracefully (ORT/Core ML still fuse at load).
    If fuse() succeeds but removes 0 BN layers, that is normal: checkpoints are often pre-fused (BN may
    still exist as identity), and YOLOE PF heads may use conv stacks without foldable Conv+BN pairs.
  - ONNX opset 18 — matches current PyTorch exporter, quiets down-conversion noise; ORT 1.15+ on
    Android/iOS supports it.
  - cv2/cv3/cv4/cv5 aliasing from one2one_* when missing.
  - end2end bypass + export wrapper if direct forward fails.

Environment:
  YOLOE_PT  — path to .pt (default: ~/Downloads/yoloe-26l-seg-pf.pt)

Outputs:
  .build/<stem>_seg_o2m.onnx
  .build/<stem>_seg_o2m.mlpackage
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / ".build"
BUILD.mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH = os.environ.get("YOLOE_PT", str(Path.home() / "Downloads" / "yoloe-26l-seg-pf.pt"))
IMG_SIZE = 640
OPSET = 18
CONF_THRESHOLD_HINT = 0.25
IOU_THRESHOLD_HINT = 0.5

STEM = Path(MODEL_PATH).stem
ONNX_PATH = BUILD / f"{STEM}_seg_o2m.onnx"
ML_PATH = BUILD / f"{STEM}_seg_o2m.mlpackage"
MLMODELC_PATH = BUILD / f"{STEM}_seg_o2m.mlmodelc"

# ─────────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────────

print(f"[1/6] Loading {MODEL_PATH}...")
if not Path(MODEL_PATH).is_file():
    print(f"  ✗ Missing checkpoint: {MODEL_PATH}", file=sys.stderr)
    sys.exit(1)

yolo = YOLO(MODEL_PATH)
model = yolo.model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"  → Parameters: {num_params:,}")
print(f"  → Head type: {type(model.model[-1]).__name__}")

# ─────────────────────────────────────────────
# 2. ATTEMPT FUSE (graceful skip for PF heads)
# ─────────────────────────────────────────────

print("[2/6] Attempting layer fusion (Conv2d + BatchNorm2d)...")

fuse_succeeded = False
bn_count_before = 0
bn_count_after = 0
try:
    bn_count_before = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))

    model.fuse()
    fuse_succeeded = True

    bn_count_after = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))

    folded = bn_count_before - bn_count_after
    print(f"  ✓ fuse() succeeded — BatchNorm2d count {bn_count_before} → {bn_count_after} (removed {folded})")
    if folded == 0:
        print(
            "  → (0 removed is OK: often pre-fused weights or PF head without foldable Conv+BN pairs.)"
        )

except AttributeError as e:
    print(f"  ⚠ fuse() failed (expected for PF checkpoint): {e}")
    print("  → Skipping — ONNX Runtime / Core ML / TensorRT fuse automatically at load time")

except Exception as e:
    print(f"  ⚠ fuse() failed (unexpected): {e}")
    print("  → Skipping — runtime will handle Conv+BN fusion")

model.eval()

# ─────────────────────────────────────────────
# 3. PATCH HEAD → one-to-many path
# ─────────────────────────────────────────────

print("[3/6] Patching head for one-to-many export...")
head = model.model[-1]

print(f"  → end2end (before): {getattr(head, 'end2end', 'N/A')}")
print(f"  → cv2: {'exists' if getattr(head, 'cv2', None) is not None else 'None'}")
print(f"  → cv3: {'exists' if getattr(head, 'cv3', None) is not None else 'None'}")
print(f"  → cv4: {'exists' if getattr(head, 'cv4', None) is not None else 'None'}")
print(f"  → cv5: {'exists' if getattr(head, 'cv5', None) is not None else 'None'}")
print(f"  → one2one_cv2: {'exists' if hasattr(head, 'one2one_cv2') else 'missing'}")
print(f"  → one2one_cv3: {'exists' if hasattr(head, 'one2one_cv3') else 'missing'}")
print(f"  → one2one_cv4: {'exists' if hasattr(head, 'one2one_cv4') else 'missing'}")
print(f"  → one2one_cv5: {'exists' if hasattr(head, 'one2one_cv5') else 'missing'}")

head.end2end = False
head.export = True

aliased: list[str] = []

if getattr(head, "cv2", None) is None and hasattr(head, "one2one_cv2"):
    head.cv2 = head.one2one_cv2
    aliased.append("cv2 ← one2one_cv2")

if getattr(head, "cv3", None) is None and hasattr(head, "one2one_cv3"):
    head.cv3 = head.one2one_cv3
    aliased.append("cv3 ← one2one_cv3")

if getattr(head, "cv4", None) is None and hasattr(head, "one2one_cv4"):
    head.cv4 = head.one2one_cv4
    aliased.append("cv4 ← one2one_cv4")

if getattr(head, "cv5", None) is None and hasattr(head, "one2one_cv5"):
    head.cv5 = head.one2one_cv5
    aliased.append("cv5 ← one2one_cv5")

if aliased:
    for line in aliased:
        print(f"  → Aliased: {line}")
else:
    print("  → All conv heads already present, no aliasing needed")

print(f"  → end2end (after): {head.end2end}")
print(f"  → export (after): {head.export}")

# ─────────────────────────────────────────────
# 4. TRACE VALIDATION
# ─────────────────────────────────────────────

print("[4/6] Validating forward pass with dummy input...")
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

with torch.no_grad():
    try:
        outputs = model(dummy)
        print("  ✓ Direct forward pass succeeded")

    except Exception as e:
        print(f"  ⚠ Direct forward failed: {e}")
        print("  → Building export wrapper...")

        class ExportWrapper(nn.Module):
            """Layer-by-layer forward; handles skip connections via layer.f."""

            def __init__(self, m: nn.Module) -> None:
                super().__init__()
                self.layers = m.model

            def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
                cache: list[torch.Tensor] = []
                for layer in self.layers:
                    f = getattr(layer, "f", -1)
                    if isinstance(f, int):
                        inp = cache[f] if f != -1 else x
                        x = layer(inp)
                    elif isinstance(f, list):
                        inp = [cache[j] if j != -1 else x for j in f]
                        x = layer(inp)
                    else:
                        x = layer(x)
                    cache.append(x)
                return x

        model = ExportWrapper(model)
        model.eval()
        outputs = model(dummy)
        print("  ✓ Wrapper forward pass succeeded")

print("  Output structure:")
if isinstance(outputs, (tuple, list)):
    for i, o in enumerate(outputs):
        if isinstance(o, torch.Tensor):
            print(f"    [{i}] shape: {o.shape}  dtype: {o.dtype}")
        else:
            print(f"    [{i}] type: {type(o)}")
else:
    print(f"    shape: {outputs.shape}  dtype: {outputs.dtype}")

# ─────────────────────────────────────────────
# 5. EXPORT → ONNX
# ─────────────────────────────────────────────

print(f"[5/6] Exporting ONNX → {ONNX_PATH}...")

if isinstance(outputs, (tuple, list)):
    num_tensor_outs = len([o for o in outputs if isinstance(o, torch.Tensor)])
    if num_tensor_outs >= 2:
        output_names = ["det_output", "proto_output"]
        dynamic_axes = {
            "images": {0: "batch"},
            "det_output": {0: "batch"},
            "proto_output": {0: "batch"},
        }
    else:
        output_names = ["output"]
        dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}}
else:
    output_names = ["output"]
    dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}}

try:
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"  ✓ ONNX exported: {ONNX_PATH}")

    import onnx
    from onnx import shape_inference

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX validation passed")

    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, str(ONNX_PATH))

    for out in onnx_model.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"    → {out.name}: {dims}")

    file_size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"    → File size: {file_size_mb:.1f} MB")

except Exception as e:
    print(f"  ✗ ONNX export failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────
# 6. EXPORT → COREML
# ─────────────────────────────────────────────

print(f"[6/6] Exporting CoreML → {ML_PATH}...")

try:
    import coremltools as ct

    print("  → Tracing model...")
    traced = torch.jit.trace(model, dummy)

    print("  → Converting to CoreML...")
    # `image` matches Furnit Core ML path (FurnitureFitView / YoloEImageInference).
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, IMG_SIZE, IMG_SIZE),
                scale=1.0 / 255.0,
                bias=[0, 0, 0],
                color_layout=ct.colorlayout.RGB,
            )
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )

    mlmodel.short_description = (
        "YOLOE-26L Seg PF — one-to-many head, raw output, no NMS. Apply NMS on-device."
    )
    mlmodel.author = "Custom Export"
    mlmodel.version = "1.0"

    if ML_PATH.exists():
        shutil.rmtree(ML_PATH)
    mlmodel.save(str(ML_PATH))
    print(f"  ✓ CoreML saved: {ML_PATH}")

    spec = mlmodel.get_spec()
    for out in spec.description.output:
        print(f"    → {out.name}: {out.type.WhichOneof('Type')}")

    try:
        try:
            compiled_path = ct.models.CompiledMLModel.compile(str(ML_PATH))
            if MLMODELC_PATH.exists():
                shutil.rmtree(MLMODELC_PATH)
            shutil.copytree(compiled_path, MLMODELC_PATH)
            print(f"  ✓ Compiled: {MLMODELC_PATH}")
        except AttributeError:
            result = subprocess.run(
                ["xcrun", "coremlcompiler", "compile", str(ML_PATH), str(BUILD)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("  ✓ Compiled via xcrun (see .build for .mlmodelc)")
            else:
                print(f"  ⚠ xcrun compile failed: {result.stderr}")
                print("  → Use .mlpackage in Xcode — it auto-compiles on build")
    except Exception as compile_err:
        print(f"  ℹ .mlmodelc compilation skipped: {compile_err}")
        print("  → Drag .mlpackage into Xcode — compiles automatically")

    pkg_bytes = sum(f.stat().st_size for f in ML_PATH.rglob("*") if f.is_file())
    print(f"    → Package size: {pkg_bytes / (1024 * 1024):.1f} MB")

except ImportError:
    print("  ⚠ coremltools not installed")
    print("  → Run: pip install coremltools")
    print("  → ONNX export is ready, skipping CoreML")

except Exception as e:
    print(f"  ✗ CoreML export failed: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print()
print("=" * 60)
print(" EXPORT COMPLETE")
print("=" * 60)
print()
print(f"  ONNX   → {ONNX_PATH}")
print(f"  CoreML → {ML_PATH}")
if fuse_succeeded:
    folded = bn_count_before - bn_count_after
    fuse_line = (
        f"Yes — torch.nn.Module.fuse() ran (BatchNorm2d {bn_count_before}→{bn_count_after}, "
        f"folded {folded} BN layer(s))"
    )
else:
    fuse_line = "No — fuse() skipped or failed; ONNX Runtime / Core ML may still apply Conv+BN fusion at load"
print(f"  Fused  → {fuse_line}")
print()
print(" OUTPUT TENSOR LAYOUT (one-to-many head):")
print(" ┌─────────────────────────────────────────────────┐")
print(" │ det_output: [1, 4+num_classes+mask_dim, anchors] │")
print(" │   channels 0-3     → x_center, y_center, w, h   │")
print(" │   channels 4..4+nc → class scores               │")
print(" │   channels 4+nc.. → mask coefficients (32)     │")
print(" │ proto_output: [1, 32, Hp, Wp]                    │")
print(" └─────────────────────────────────────────────────┘")
print()
print(" ON-DEVICE PIPELINE:")
print(f"  1. Preprocess: letterbox {IMG_SIZE}×{IMG_SIZE}, normalize /255")
print("  2. Inference")
print("  3. Parse det_output (feature-major × anchors)")
print(f"  4. Filter: confidence > {CONF_THRESHOLD_HINT}")
print(f"  5. NMS: IoU threshold {IOU_THRESHOLD_HINT}")
print("  6. Primary: highest confidence post-NMS")
print("  7. Mask: sigmoid(coeffs @ protos)")
print()
print(" BUNDLE FOR APPS:")
print(f"  • Copy mlpackage → repo root: yoloe-26l-seg-pf_seg_o2m.mlpackage (Xcode target already references it)")
print(f"  • Copy ONNX → android/app/src/main/assets/yoloe-26l-seg-pf_seg_o2m.onnx")
print(f"  • Copy ONNX → Furnit/Resources/yoloe-26l-seg-pf_seg_o2m.onnx (optional iOS ONNX toggle)")
print("=" * 60)
