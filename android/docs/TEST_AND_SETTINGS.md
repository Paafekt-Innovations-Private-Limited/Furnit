# Test And Settings

## Build Test

```bash
./gradlew :app:assembleDebug
```

## Room Creation Smoke Test

1. Launch the app on an arm64 device.
2. Open the photo-to-room flow.
3. Select or capture a room photo.
4. Confirm the AI/manual method picker appears immediately, then choose AI generation.
5. Confirm a GLB preview opens with a **flat full-photo** wall (no dragged/stretch artifacts on the front texture).
6. Confirm `GLBRoomActivity` uses floating top controls (back, ruler/pinch/tap helpers, recenter/save, AR), not a full-width band.
7. Save the room.
8. Confirm the saved room appears in the home room list and opens in `GLBRoomActivity`.

## GLB Room Full-Video Segmentation Smoke Test

1. Open a saved or preview room in `GLBRoomActivity`.
2. Tap the **brain** button (bottom-left). Default mode should auto-segment one primary item over the 3D room.
3. Tap the **viewfinder** button. Live camera preview should appear with fast detection boxes.
4. Tap two or more detected furniture boxes.
5. Tap **Segment**. Camera preview should hide; transparent furniture cutouts should composite over the **3D room** (not the live feed).
6. Tap **Stop** to return to live identification boxes, or tap brain again to exit.

Useful log filter:

```bash
adb logcat -s GLBRoomActivity:D FurnitureFitManager:D FurnitureFitOverlay:I -v time
```

## Settings

The old backend selection settings have been removed. Current developer settings focus on debug logging, furniture segmentation, room viewer behavior, and default manual/demo room dimensions.

**Full video with identifications** — legacy Settings switch under Furniture segmentation. `GLBRoomActivity` uses the in-room **viewfinder** toggle instead (Swift parity). The Settings preference is not read by the GLB room inline brain path.

## Asset Check

At startup, `RoomGenerationAssets.logAvailability` logs packaged room-generation assets and any missing expected assets.

Expected today:

- Depth Anything ONNX: present.
- RTMDet ONNX: present.
- GeoCalib Android export: missing until an ONNX or TFLite version is produced.
