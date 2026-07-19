# Google Play Review Checklist

## Reviewer path
- Fresh install → phone OTP sign-in with the Firebase test number.
- Settings → confirm **Delete Account**, Privacy Policy, Terms, Credits, Licenses.
- Create a room on device, open room viewers, capture/save a view, sign out, sign back in, delete account.

## Play Console
- Data safety: phone number, name, user ID / device installation ID for app functionality; no advertising ID; no photo upload for room ML.
- Privacy policy: `https://paafekt.com/privacy`
- Support: `support@paafekt.com`
- Do not describe room generation as a cloud upload feature. Android room ML is on-device (Depth Anything + GeoCalib + RTMDet via ONNX; Filament/SceneView for GLB).

## Reviewer notes (paste)

```
Demo access: [country code + Firebase test phone]. OTP: [fixed test code].

Delete Account: Settings → Delete Account.

Core features run on-device after sign-in. Room generation uses Depth Anything, GeoCalib, and RTMDet on device. Network is used for Firebase phone authentication.

Permissions: Camera (room photos + AR furniture). Photos are picked via the system picker; screenshots save via MediaStore on modern Android.

No ads, no advertising ID, no in-app purchases. Support: support@paafekt.com
```
