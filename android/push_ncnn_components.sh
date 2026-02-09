#!/bin/bash
# Push NCNN component model files to Android device
# Usage: ./push_ncnn_components.sh

MODEL_DIR="sharp_ncnn_models"
DEST="/sdcard/Android/data/com.furnit.android/files/models"

# Check if adb is available
if ! command -v adb &> /dev/null; then
    echo "Error: adb not found in PATH"
    exit 1
fi

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "Error: No device connected. Connect device and enable USB debugging."
    exit 1
fi

echo "Pushing NCNN component files to device..."
echo "Destination: $DEST"
echo ""

# Create destination directory
adb shell "mkdir -p $DEST"

# Files to push (all component files including prediction heads)
FILES=(
    "sharp_single_patch_embed.ncnn.param"
    "sharp_single_patch_embed.ncnn.bin"
    "sharp_single_patch_encoder.ncnn.param"
    "sharp_single_patch_encoder.ncnn.bin"
    "sharp_image_encoder.ncnn.param"
    "sharp_image_encoder.ncnn.bin"
    "patch_cls_token.bin"
    "patch_pos_embed.bin"
    # Proper Gaussian prediction models (trained weights)
    "encoder_projection.ncnn.param"
    "encoder_projection.ncnn.bin"
    "geometry_model.ncnn.param"
    "geometry_model.ncnn.bin"
    "texture_model.ncnn.param"
    "texture_model.ncnn.bin"
    # Legacy fallback
    "gaussian_head.ncnn.param"
    "gaussian_head.ncnn.bin"
)

total_size=0
for file in "${FILES[@]}"; do
    if [ -f "$MODEL_DIR/$file" ]; then
        size=$(stat -f%z "$MODEL_DIR/$file" 2>/dev/null || stat -c%s "$MODEL_DIR/$file" 2>/dev/null)
        total_size=$((total_size + size))
    fi
done

echo "Total size to transfer: $(echo "scale=1; $total_size / 1048576" | bc) MB"
echo ""

for file in "${FILES[@]}"; do
    if [ -f "$MODEL_DIR/$file" ]; then
        size=$(ls -lh "$MODEL_DIR/$file" | awk '{print $5}')
        echo "Pushing $file ($size)..."
        adb push "$MODEL_DIR/$file" "$DEST/$file"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to push $file"
            exit 1
        fi
    else
        echo "Warning: $file not found in $MODEL_DIR"
    fi
done

echo ""
echo "Done! Component files pushed successfully."
echo ""
echo "To use NCNN backend:"
echo "1. Enable NCNN in code (BackendConfig.ENABLE_NCNN = true)"
echo "2. Select Settings > Developer > Inference Backend = NCNN"
echo "3. The app will automatically use component mode"
