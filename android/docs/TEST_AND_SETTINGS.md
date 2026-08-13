# Test And Settings

## Build Test

```bash
python3 scripts/verify_i18n.py
./gradlew :app:assembleDebug
```

`preBuild` depends on `verifyLegalAssets`, so normal debug and release builds fail if
the required Apache, LiteRT/Caffe, or model-modification notice is missing or
truncated. A release handoff should also run `./gradlew :app:bundleRelease --no-daemon`
and verify the signed AAB.

For Android Studio device testing, set **Run > Edit Configurations > app > Deploy** to
**APK from app bundle**. The default APK-only deployment omits the install-time
`rtmdet_models` and `room_generation_models` asset packs.

## Room Creation Smoke Test

1. Launch the app on an arm64 device.
2. Open the photo-to-room flow.
3. Select or capture a room photo.
4. Confirm the AI/manual method picker appears immediately, then choose AI generation.
5. Confirm choosing AI generation requests RTMDet for the upcoming preview; selecting a photo or
   launching the app must not request it.
6. Confirm a GLB preview opens with a **flat full-photo** wall (no dragged/stretch artifacts on the front texture).
7. Confirm `GLBRoomActivity` uses floating top controls (back, ruler/pinch/tap helpers, recenter/save, AR), not a full-width band.
8. Save the room.
9. Confirm the saved room appears in the home room list and opens in `GLBRoomActivity` as a projective depth surface.
10. Recenter, pinch in/out, rotate through the full horizontal range, and look nearly straight up/down. Confirm the eye remains at the authored optical center and foreground objects do not move out of place, stretch, or leave duplicate gray/photo traces. This candidate still requires manual device confirmation as of 2026-08-14.

## GLB Room Full-Video Segmentation Smoke Test

1. Open a saved or preview room in `GLBRoomActivity`.
2. Confirm RTMDet begins loading only after the room opens, not during application/login startup.
3. Tap the **brain** button (bottom-left). The Fit activation should ensure/reuse the loaded model,
   and default mode should auto-segment one primary item over the 3D room.
4. Tap the **viewfinder** button. Live camera preview should appear with fast detection boxes.
5. Tap two or more detected furniture boxes.
6. Tap **Segment**. Camera preview should hide; transparent furniture cutouts should composite over the **3D room** (not the live feed).
7. Tap **Stop** to return to live identification boxes, or tap brain again to exit.
8. Leave the room and confirm logs report shared RTMDet resource release; reopening a room should
   load it again.

Useful log filter:

```bash
adb logcat -s GLBRoomActivity:D RTMDetLiteRt:I FurnitureFitManager:D \
  FurnitureFitOverlay:I tflite:I -v time
```

For camera smoothness, color/orientation, AR sizing, thermal cadence, and frame-timing checks, follow
[`FURNITURE_FIT_PERFORMANCE.md`](FURNITURE_FIT_PERFORMANCE.md).

## Settings

The old backend selection and FurnitureFit tuning switches have been removed. Current settings focus on debug logging, the single-image furniture scan, room viewer behavior, and default manual/demo room dimensions.

Full-video identification is controlled in `GLBRoomActivity` by the in-room **viewfinder** button, not by Settings.

### Licenses smoke test

1. Disable network access.
2. Open Settings → Licenses.
3. Open **View bundled notices** and confirm it names Depth Anything V2 Metric Indoor
   Small, GeoCalib, RTMDet/COCO, and Paafekt's converted model formats.
4. Open an Apache-licensed component's full license and confirm the complete Apache
   text is readable.
5. Under **Other Android runtime libraries**, open the LiteRT license and confirm the
   Caffe/BSD attribution follows the Apache text.

## Asset Check

At startup, `RoomGenerationAssets.logAvailability` logs packaged room-generation assets and any missing expected assets.

Expected today:

- Depth Anything ONNX: present.
- RTMDet FP16 LiteRT: present (GPU when supported, XNNPACK CPU otherwise).
- RTMDet ONNX: absent; the retired fallback model must not be packaged.
- GeoCalib pinhole CNN ONNX: present and loaded by `GeoCalibCalibrationService`; focal/gravity fallbacks remain available if inference cannot run.
- Legal assets: `APACHE-2.0.txt`, `LITERT-LICENSE.txt`, and
  `THIRD_PARTY_NOTICES.txt` present under `app/src/main/assets/legal/`.

## Authentication smoke test

Use a Google Play install when validating production Phone Auth because a local APK
uses a different signing certificate. Follow [`AUTHENTICATION.md`](AUTHENTICATION.md),
including automatic and manual OTP entry checks.

## Localization smoke test

Follow [`LOCALIZATION.md`](LOCALIZATION.md) after any catalog, visible copy, country
preselection, or RTL change. At minimum, switch between English, Arabic, one Indic
language, and one Chinese locale; then exercise login, OTP, room creation, sharing,
Settings, Help, and Furniture Fit.
