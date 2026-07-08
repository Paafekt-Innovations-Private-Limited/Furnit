# Android Room Generation Assets

This folder mirrors the Swift photo-to-3D-room asset contract:

- Depth Anything V2 Metric Indoor for metric depth
- GeoCalib pinhole CNN for focal length and gravity hints
- RTMDet-Ins for furniture/object masking during room measurement and GLB room segmentation

Packaged Android assets:

- `depth_anything/depth_anything_v2_metric_indoor_small.onnx`
- `../rtmdet-ins-m-raw.onnx`

Expected Android export still missing:

- `geocalib/geocalib_pinhole_cnn.onnx`

The Swift tree currently has GeoCalib as a Core ML package and raw checkpoint files only. Do not copy those Core ML files into Android assets; export an ONNX or TFLite version before wiring Android to the full Swift-parity reconstruction path.

AI photo rooms use `GlbGenerator.generateFlatPhotoGlb` (single full-photo plane). Manual rooms use the five-plane cuboid exporter in `GlbGenerator.generateGlb`.
