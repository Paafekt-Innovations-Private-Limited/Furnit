# Test And Settings

## Build Test

```bash
./gradlew :app:assembleDebug
```

## Room Creation Smoke Test

1. Launch the app on an arm64 device.
2. Open the photo-to-room flow.
3. Select or capture a room photo.
4. Choose AI generation.
5. Confirm a GLB preview opens.
6. Save the room.
7. Confirm the saved room appears in the home room list and opens in `GLBRoomActivity`.

## Settings

The old backend selection settings have been removed. Current developer settings focus on debug logging, furniture segmentation, room viewer behavior, and default manual/demo room dimensions.

## Asset Check

At startup, `RoomGenerationAssets.logAvailability` logs packaged room-generation assets and any missing expected assets.

Expected today:

- Depth Anything ONNX: present.
- RTMDet ONNX: present.
- GeoCalib Android export: missing until an ONNX or TFLite version is produced.
