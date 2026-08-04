# Furnit Android Diagrams

Open these SVG diagrams in a browser or Android Studio.

| Diagram | Flow |
|---|---|
| [`photo-room-generation-android-flow.svg`](photo-room-generation-android-flow.svg) | Photo-to-room flow: photo input, immediate AI/manual choice, off-thread sampled decode, flat-photo or cuboid GLB generation, metadata, preview, save, and `GLBRoomActivity` with Swift-style floating controls. |
| [`rtmdet-android-flow.svg`](rtmdet-android-flow.svg) | RTMDet furniture segmentation in `GLBRoomActivity`: CameraX analysis/preview, shared LiteRT backend, fast boxes-only identify, full segmentation modes, and transparent overlay on the 3D room. |

Written Android docs live in [`android/docs/`](../docs/).
