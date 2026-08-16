# Test And Settings

## Build Test

```bash
python3 scripts/verify_i18n.py
./gradlew :app:assembleDebug
```

`preBuild` depends on `verifyLegalAssets`, so normal debug and release builds fail if
the required Apache, LiteRT/Caffe, ONNX Runtime, Three.js, reCAPTCHA, or
model-modification notice is missing or truncated. The reCAPTCHA document is copied
from the exact resolved 18.6.1 AAR into generated app assets so it cannot drift from
the dependency. A release handoff should also run
`./gradlew :app:bundleRelease --no-daemon` and verify the signed AAB.

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
6. Confirm generation completes before a version-5 continuous-depth GLB preview opens. The preview must contain one opaque photo surface without a duplicate foreground/background layer.
7. Confirm `GLBRoomActivity` uses floating top controls (back, ruler/pinch/tap helpers, recenter/save, AR), not a full-width band.
8. Save the room.
9. Confirm the saved room appears in the home room list and reopens with the same framing, geometry, texture, and initial camera position as the preview. Save must promote the preview GLB rather than rerun inference.
10. Recenter, pinch within the bounded forward/backward envelope, and use drag plus D-pad toward each coverage limit. Confirm foreground objects stay in place without duplicated layers, edge stretching, fog shells, or renderer-gray holes. Repeat with fresh portrait and landscape inputs. This candidate still requires manual device confirmation as of 2026-08-15.

The large `SavedRoomNavigationE2ETest` exercises the same create/save/reopen/navigation sequence with
a synthetic image generated at runtime. Compile it with `:app:assembleAndroidTest`; run it only on a
suitable arm64 device with the install-time model packs available. Its pixel checks are regression
signals, not a substitute for the room-photo smoke test above.

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

The old backend selection and FurnitureFit tuning switches have been removed. Current settings focus on debug logging, the single-image furniture scan, room viewer behavior, and default manual/demo room dimensions. In debug builds, the Developer section is visible only while the authenticated Firebase identity is the test number `+1 650-555-3434`; leaving that identity also disables any persisted debug-logging state.

**Infinite Zoom** defaults off. With the switch off, GLB/photo navigation and the legacy image
viewer retain their existing bounded behavior. After enabling it, reopen the room viewer and confirm
pinch can use the wider `0.05...1000` range; turn it off and reopen once more to confirm the normal
flow is restored. The enabled Android behavior was manually confirmed on 2026-08-16.

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
6. Open both ONNX Runtime documents and confirm the MIT license and upstream
   third-party notices are readable offline.
7. Open the Three.js MIT license and the reCAPTCHA SDK/third-party license bundle
   while the device remains offline.

## Asset Check

At startup, `RoomGenerationAssets.logAvailability` logs packaged room-generation assets and any missing expected assets.

Expected today:

- Depth Anything ONNX: present.
- RTMDet FP16 LiteRT: present (GPU when supported, XNNPACK CPU otherwise).
- RTMDet ONNX: absent; the retired fallback model must not be packaged.
- GeoCalib pinhole CNN ONNX: present and loaded by `GeoCalibCalibrationService`; focal/gravity fallbacks remain available if inference cannot run.
- Source legal assets: `APACHE-2.0.txt`, `LITERT-LICENSE.txt`,
  `ONNXRUNTIME-MIT.txt`, `ONNXRUNTIME-THIRD-PARTY-NOTICES.txt`, `THREEJS-MIT.txt`,
  and `THIRD_PARTY_NOTICES.txt` present under `app/src/main/assets/legal/`.
- Generated legal asset: `RECAPTCHA-THIRD-PARTY-LICENSES.txt` is extracted from the
  resolved reCAPTCHA AAR and packaged under the same `legal/` asset path.

## Authentication smoke test

Use a Google Play install when validating production Phone Auth because a local APK
uses a different signing certificate. Follow [`AUTHENTICATION.md`](AUTHENTICATION.md),
including automatic and manual OTP entry checks.

## Localization smoke test

Follow [`LOCALIZATION.md`](LOCALIZATION.md) after any catalog, visible copy, country
preselection, or RTL change. At minimum, switch between English, Arabic, one Indic
language, and one Chinese locale; then exercise login, OTP, room creation, sharing,
Settings, Help, and Furniture Fit.
