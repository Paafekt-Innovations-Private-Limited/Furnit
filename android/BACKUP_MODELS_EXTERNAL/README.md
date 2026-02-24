# SHARP / ExecuTorch model backup (external disk)

These paths contain large model files that are **not** in git. Back them up to an external hard disk.

Copy the following folders/files from this repo root to your external disk (e.g. `Volumes/YourDisk/Furnit_models_backup/`):

## Folders (copy entire directory)

| Path (relative to repo) | Approx size | Contents |
|-------------------------|-------------|----------|
| `android/executorch_int8_models/` | ~1.2 GB | INT8 split .pte (Part 1–4, chunked 4a/4b) |
| `android/executorch_fp16_models/` | ~2.3 GB | FP16 split .pte |
| `android/executorch_models/` | ~26 GB | FP32/vulkan/full .pte and .pt |
| `android/executorch_models_vulkan/` | varies | Vulkan .pte |
| `android/sharp_litert_models/` | ~4.2 GB | PyTorch .pt (e.g. sharp_2572gikvuh.pt) |
| `android/sharp_onnx_models/` | varies | ONNX split parts |
| `android/sharp_ncnn_models/` | (gitignored) | NCNN exports |

## Root-level model files (in `android/`)

- `*.pte` (e.g. sharp_int8.pte, sharp_split_part*.pte)
- `*.onnx`, `*.onnx.data` (sharp_part*.onnx, sharp_*_fp16.onnx, etc.)
- `*.pt` (e.g. sharp_2572gikvuh.pt, yoloe-11l-seg-pf.pt)

Example backup command (run from repo root, adjust external path):

```bash
EXTERNAL=/Volumes/YourDisk/Furnit_models_backup
mkdir -p "$EXTERNAL"
cp -R android/executorch_int8_models android/executorch_fp16_models android/sharp_litert_models "$EXTERNAL/"
# Optional: android/executorch_models (large)
cp android/*.pte android/*.onnx android/*.pt "$EXTERNAL/" 2>/dev/null || true
```

This folder (`BACKUP_MODELS_EXTERNAL/`) is in git only to hold this README; the actual model files stay out of version control.
