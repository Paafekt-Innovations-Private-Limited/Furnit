# Google Play Review Checklist

## Reviewer path
- Fresh install → phone OTP sign-in with the Firebase test number.
- Settings → confirm **Delete Account**, Privacy Policy, Terms, Credits, and Licenses.
- With networking disabled, open the bundled third-party notice, Apache license,
  LiteRT combined license/Caffe attribution, ONNX Runtime license/notices, Three.js
  license, and reCAPTCHA license bundle from Settings → Licenses.
- Create a room on device, open room viewers, capture/save a view, sign out, sign back in, delete account.

## Play Console
- Data safety: phone number, name, user ID / device installation ID for app functionality; no advertising ID; no photo upload for room ML.
- Privacy policy: `https://paafekt.com/privacy`
- Support: `support@paafekt.com`
- Do not describe room generation as a cloud upload feature. Depth Anything and
  GeoCalib use ONNX Runtime on-device; RTMDet uses FP16 LiteRT with GPU delegation or
  the same model through XNNPACK CPU. Filament/SceneView and Three.js render GLB rooms.
- Firebase Phone OTP: allowlist SMS regions (including **France / FR**) and use a test phone for review — see [firebase-sms-regions.md](firebase-sms-regions.md).

## Version 1.2 handoff (2026-08-05)

- Release metadata: version name **1.2**, version code **5**, package
  `com.paafekt.android`.
- Signed AAB build and signing verification passed. The bundle embeds its R8 mapping
  and the three required legal assets.
- Play Console upload/publication was not re-confirmed in this document. Keep 1.1 / code
  4 as the last confirmed live build until Play shows 1.2 published.
- The native debug-symbol warning from dependency `.so` files is non-blocking; it is a
  crash-diagnostics recommendation, not a license or device-support error.

## Reviewer notes (paste)

```
Demo access: [country code + Firebase test phone]. OTP: [fixed test code].

Delete Account: Settings → Delete Account.

Core features run on-device after sign-in. Room generation uses Depth Anything, GeoCalib, and RTMDet on device. Network is used for Firebase phone authentication.

Settings → Licenses contains offline third-party notices, the Apache License 2.0 text, LiteRT's combined license/Caffe attribution, ONNX Runtime and Three.js MIT licenses, and the reCAPTCHA SDK license bundle.

Permissions: Camera (room photos + AR furniture). Photos are picked via the system picker; screenshots save via MediaStore on modern Android.

No ads, no advertising ID, no in-app purchases. Support: support@paafekt.com
```
