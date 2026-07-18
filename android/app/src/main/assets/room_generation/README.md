# Android Room Generation Assets

This folder mirrors the Swift photo-to-3D-room asset contract:

- Depth Anything V2 Metric Indoor for metric depth
- GeoCalib pinhole CNN for focal length and gravity hints
- RTMDet-Ins for furniture segmentation in GLB room viewers; the Swift room-measurement path also uses RTMDet as an object anchor.

Packaged Android assets:

- `depth_anything/depth_anything_v2_metric_indoor_small.onnx`
- `geocalib/geocalib_pinhole_cnn.onnx`
- `../rtmdet-ins-m-raw.onnx`

The Android GeoCalib ONNX and iOS GeoCalib ML package are exported from the same
pretrained pinhole checkpoint. Their platform-native inference wrappers feed the same
perspective-field outputs into matching native LM solvers.

Do not copy Core ML packages into Android assets.

AI photo rooms use `GlbGenerator.generateFlatPhotoGlb` (single full-photo plane with JPEG texture). Manual rooms use the five-plane cuboid exporter in `GlbGenerator.generateGlb`.
