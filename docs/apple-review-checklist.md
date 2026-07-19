# Apple Review Checklist

## In-App Reviewer Path
- Launch the app from a fresh install.
- Complete phone sign-in with the reviewer test account.
- Open `Settings` and verify `Delete Account` is available under `Account`.
- Confirm the room viewers load without network-fetched JavaScript assets.
- Verify camera, photo-library, and motion permission prompts use accurate wording.

## App Store Connect
- App Privacy answers must cover phone-number authentication, local account name storage, user ID, Firebase device installation identifier (app functionality / abuse prevention), on-device room/furniture processing, and confirm advertising analytics / tracking are **not** enabled in the release build.
- Do not describe iOS room generation as a photo-upload/backend feature unless that backend path is explicitly enabled in the submitted build. Default iOS room creation is on-device: **instant preview (no ML)** then **GeoCalib + Depth Anything + RTMDet object anchor → USDZ on first save**.
- Confirm the export compliance answer remains correct for the shipped build (`ITSAppUsesNonExemptEncryption = NO`).
- Ensure screenshots and app description do not claim unfinished or hidden functionality, and do not imply room photos are uploaded for generation.
- Privacy Policy URL: `https://paafekt.com/privacy` — Terms: `https://paafekt.com/terms` — Support: `support@paafekt.com`.

## Reviewer Notes (paste into App Review Information)

```
Demo access: [country code + Firebase test phone number]. OTP: [code from Firebase Console / SMS path].

Delete Account: Settings → Account → Delete Account.

Core features run on-device after sign-in. Room preview opens instantly; metric room generation (GeoCalib + Depth Anything + RTMDet) runs on first save. Furniture detection may download RTMDet via On-Demand Resources (needs network once).

Permissions: Camera (room photos + AR furniture), Photos (pick/save), Motion (AR viewing).

No ads, no App Tracking, no in-app purchases. Support: support@paafekt.com
```

## Pre-Submission QA
- Fresh install on device.
- Sign in, generate a room, open each room viewer, capture/save a view to Photos, log out, sign back in, and delete the account.
- After models have downloaded once, repeat the viewer flow offline long enough to verify graceful operation or graceful failures instead of infinite loading states.
- Confirm Release archive push entitlement uses production APNs (source entitlements may show development; Xcode rewrites for App Store distribution).
