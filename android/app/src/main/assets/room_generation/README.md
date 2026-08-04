# Android Room Generation Assets

This folder mirrors the Swift photo-to-3D-room asset contract:

- Depth Anything V2 Metric Indoor for metric depth
- GeoCalib pinhole CNN for focal length and gravity hints
- RTMDet-Ins for furniture segmentation in GLB room viewers; the Swift room-measurement path also uses RTMDet as an object anchor.

Packaged Android assets (shipped via install-time Play Asset Delivery packs; this
folder in the app module is kept only for the asset contract documentation):

- `room_generation_models/src/main/assets/room_generation/depth_anything/depth_anything_v2_metric_indoor_small.onnx`
- `room_generation_models/src/main/assets/room_generation/geocalib/geocalib_pinhole_cnn.onnx`
- `rtmdet_models/src/main/assets/rtmdet-ins-m-raw-fp16.tflite`

The Android GeoCalib ONNX and iOS GeoCalib ML package are exported from the same
pretrained pinhole checkpoint. Their platform-native inference wrappers feed the same
perspective-field outputs into matching native LM solvers.

Do not copy Core ML packages into Android assets.

AI photo rooms use `GlbGenerator.generateFlatPhotoGlb` (single full-photo plane with JPEG texture). Manual rooms use the five-plane cuboid exporter in `GlbGenerator.generateGlb`.
