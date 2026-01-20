Docker ONNX -> TFLite conversion

This folder provides a reproducible Docker environment to convert ONNX models to TensorFlow SavedModel and then to TFLite.

Quick steps

1. Build the Docker image and run conversion (from repository root):

```bash
./scripts/build_and_run_convert.sh android/yoloe-11l-seg-pf.onnx android/app/src/main/assets/yoloe_11l_from_onnx.tflite
```

2. The script builds a docker image using `scripts/Dockerfile.convert` and runs the conversion inside the container. Output TFLite will be placed at the path you provide (relative to repo root).

Notes

- The Dockerfile pins specific package versions known to work together in a linux x86_64 environment (Python 3.10). This avoids macOS host wheel incompatibilities.
- If conversion fails due to model specifics (custom ops), consider using ONNX Runtime for Android instead — this repo already supports that path.
- The conversion may take several minutes and requires network access to pull pip wheels on first run.

If you want, I can tune the pinned versions or add optional quantization flags to the converter.
