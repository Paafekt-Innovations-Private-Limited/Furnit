# YOLOE Core ML vs PyTorch (confidence parity)

**Furnit iOS** ships **26L PF** (`yoloe-26l-seg-pf` / `_seg_o2m`) only. Smaller checkpoints (e.g. 11L) may still appear in Android or backup export docs.

## Why on-device “chair” can sit around 0.6 while you expect ~0.9

1. **Different checkpoint** — `yoloe-26l-seg-pf` ≠ `yoloe-11l-seg-pf`. Scores are not comparable across sizes.
2. **Export flags** — Core ML must be exported with **`nms=False`** (and usual `batch=1`, `simplify=True`) so the graph matches end-to-end tensors parsed in iOS (`YoloEDetectionParser`). Exporting with the default (NMS inside the graph) changes calibration vs `model.predict()`.
3. **Letterbox padding** — Training/Ultralytics use RGB **(114,114,114)**. The app uses `YoloUltralyticsLetterboxFill` (BGRA 114,114,114,255).

## Scripts (this repo)

| Script | Role |
|--------|------|
| **`yoloe_export_and_backup.py`** | **11L (and generic YOLOE)**: `model.fuse()`, export CoreML/ONNX with **`nms=False`**, optional `--verify-image`. |
| **`export_yoloe26_fixed.py`** | **26L only**: standard Ultralytics export (**no manual `fuse()`**, **`end2end=True`**, **`nms=False`**). |
| **`probe_yoloe26_coreml.py`** | Letterbox + `MLModel.predict`, dump `var_2346` rows. |
| **`compare_yoloe_coreml_ultralytics.py`** | Core ML raw `(1,N,38)` vs standard **`YOLOE.predict()`** on the **source** image with **`imgsz`** = model input (Ultralytics letterbox). |

## `room.jpg`

There is **no** committed `room.jpg` in this tree. Put your file anywhere and pass the path:

```bash
python3 scripts/compare_yoloe_coreml_ultralytics.py \
  --image /full/path/to/room.jpg \
  --pt /path/to/yoloe-26l-seg-pf.pt \
  --mlpackage /path/to/yoloe-26l-seg-pf.mlpackage
```

If Core ML best ≪ PyTorch best, **re-export** with `export_yoloe26_fixed.py` (or `yoloe_export_and_backup.py` for 11L), then **replace** the `.mlpackage` inside the Xcode target / bundle.

## Fuse

- **11L / backup path**: `model.fuse()` before export (Conv+BN; end2end head behavior per Ultralytics).
- **26L**: use the standard exporter path only.
