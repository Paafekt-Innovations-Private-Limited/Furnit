# DeepLabV3 Model Setup

## Overview
The AR furniture placement feature uses a DeepLabV3 CoreML model for semantic segmentation of furniture objects from camera images.

## Model Requirements
- Model Name: `DeepLabV3.mlmodel`
- Location: `Furnit/DeepLabV3.mlmodel`
- Input: 513x513 RGB image
- Output: 513x513 segmentation map with class indices

## Supported Classes
The model should detect these furniture classes from PASCAL VOC dataset:
- Class 9: Chair
- Class 11: Dining Table  
- Class 15: Sofa

## Model Download
Download the official DeepLabV3 model from Apple's Machine Learning page:
https://docs-assets.developer.apple.com/ml-res/models/DeepLabV3.mlmodel

Or use the Core ML Model Gallery:
https://developer.apple.com/machine-learning/models/

## Alternative Models
The architecture supports easy model replacement. You can substitute with:
- SAM2 (Segment Anything Model 2) for better furniture detection
- Custom trained models for specific furniture types
- YOLOv8 segmentation models

## Integration Notes
- The `ObjectSegmentationProcessor` class handles model loading
- Model compilation happens automatically at runtime
- Error handling is in place for missing or corrupted models
- The Metal rendering pipeline processes segmentation outputs