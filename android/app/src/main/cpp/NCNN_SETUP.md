# NCNN Setup for YOLOE on Android

This guide explains how to set up NCNN for running YOLOE object detection/segmentation on Android.

## Prerequisites

1. Android NDK r21+ (preferably r25)
2. CMake 3.18.1+
3. Python 3.8+ (for model conversion)

## Step 1: Download NCNN Android Libraries

1. Go to [NCNN Releases](https://github.com/Tencent/ncnn/releases)
2. Download the latest `ncnn-YYYYMMDD-android-vulkan.zip`
3. Extract to `app/src/main/cpp/ncnn-android-vulkan/`

The folder structure should be:
```
app/src/main/cpp/
├── ncnn-android-vulkan/
│   ├── arm64-v8a/
│   │   └── lib/
│   │       └── libncnn.a
│   ├── armeabi-v7a/
│   │   └── lib/
│   │       └── libncnn.a
│   └── include/
│       └── ncnn/
│           ├── net.h
│           ├── layer.h
│           └── ...
├── CMakeLists.txt
└── yoloe_ncnn.cpp
```

## Step 2: Export YOLOE to NCNN Format

### Option A: Using ONNX as intermediate format

1. Export PyTorch model to ONNX:
```python
from ultralytics import YOLO

# Load your YOLOE model
model = YOLO('yoloe-11l-seg.pt')

# Export to ONNX (required intermediate format)
model.export(format='onnx', imgsz=640, simplify=True, opset=12)
```

2. Install NCNN tools:
```bash
# Clone NCNN repo
git clone https://github.com/Tencent/ncnn.git
cd ncnn

# Build tools
mkdir build && cd build
cmake ..
make -j$(nproc)

# The onnx2ncnn tool will be in build/tools/onnx/
```

3. Convert ONNX to NCNN:
```bash
# Convert ONNX model to NCNN format
./onnx2ncnn yoloe-11l-seg.onnx yoloe-11l-seg.param yoloe-11l-seg.bin

# Optional: Optimize the model (reduce size, improve speed)
./ncnnoptimize yoloe-11l-seg.param yoloe-11l-seg.bin yoloe-11l-seg-opt.param yoloe-11l-seg-opt.bin 65536
```

### Option B: Direct export from Ultralytics (if supported)

```python
from ultralytics import YOLO

model = YOLO('yoloe-11l-seg.pt')

# Try direct NCNN export (may require ultralytics version with NCNN support)
model.export(format='ncnn', imgsz=640)
```

## Step 3: Place Model Files in Assets

Copy the converted model files to your Android assets:

```bash
cp yoloe-11l-seg.param app/src/main/assets/
cp yoloe-11l-seg.bin app/src/main/assets/
```

## Step 4: Build the Project

The native library will be built automatically when you build the Android project:

```bash
cd android
./gradlew assembleDebug
```

## Usage in Code

```kotlin
// In your Activity or Fragment
val smartyPants = SmartyPantsManager(context)

// Initialize with NCNN (recommended for best performance)
if (smartyPants.initializeNcnn(
    paramAsset = "yoloe-11l-seg.param",
    binAsset = "yoloe-11l-seg.bin",
    useGpu = true  // Uses Vulkan GPU acceleration
)) {
    // NCNN initialized successfully
} else {
    // Fall back to ONNX Runtime
    smartyPants.initializeOnnx()
}

// Or use auto-initialization (tries NCNN -> ONNX -> TFLite)
smartyPants.initializeAuto()

// Run inference
smartyPants.segmentImageAsync(cameraBitmap) { maskBitmap ->
    // maskBitmap contains the segmentation mask
}

// Don't forget to release resources
smartyPants.close()
```

## Troubleshooting

### "NCNN native library not available"
- Ensure you downloaded the correct NCNN Android package (vulkan version)
- Check that libraries are in the correct directory structure
- Verify ABI filters include your device's architecture

### "Failed to load model"
- Check that .param and .bin files are in the assets folder
- Verify file names match what you're passing to `initializeNcnn()`
- Ensure the model was converted correctly

### GPU not available
- Some older devices don't support Vulkan
- The code will automatically fall back to CPU if GPU is unavailable
- You can force CPU-only by passing `useGpu = false`

### Performance tips
1. Use smaller input size (e.g., 640x640 instead of 1536x1536)
2. Enable GPU acceleration for 2-5x speedup
3. Use fp16 model if available for faster inference
4. Consider quantized model for CPU inference

## Model Input/Output Specification

### Input
- Name: `images` (or check your model's input name)
- Shape: `[1, 3, H, W]` (NCHW format)
- Type: float32, normalized to [0, 1]

### Output
- `output0`: Detection tensor `[1, num_features, num_anchors]`
  - Features: bbox (4) + class_scores (N) + mask_coeffs (32)
- `output1`: Prototype masks `[1, 32, proto_H, proto_W]`

## Resources

- [NCNN GitHub](https://github.com/Tencent/ncnn)
- [NCNN Wiki](https://github.com/Tencent/ncnn/wiki)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [ONNX to NCNN Conversion](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-onnx)
