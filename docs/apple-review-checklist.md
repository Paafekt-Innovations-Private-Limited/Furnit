# Apple Review Checklist

## In-App Reviewer Path
- Launch the app from a fresh install.
- Complete phone sign-in with the reviewer test account.
- Open `Settings` and verify `Delete Account` is available under `Account`.
- Confirm the room viewers load without network-fetched JavaScript assets.
- Verify camera, photo-library, and motion permission prompts use accurate wording.

## App Store Connect
- App Privacy answers must cover phone-number authentication, local account storage, on-device room/furniture processing, and any Firebase analytics/crash usage that is enabled in the release build.
- Do not describe iOS room generation as a photo-upload/backend feature unless that backend path is explicitly enabled in the submitted build. Default iOS room creation is on-device: **instant preview (no ML)** then **GeoCalib + Depth Anything + RTMDet object anchor → USDZ on first save**.
- Confirm the export compliance answer remains correct for the shipped build.
- Ensure screenshots and app description do not claim unfinished or hidden functionality.

## Reviewer Notes
- Provide the reviewer phone-login test path and any OTP instructions.
- Mention that `Delete Account` is available in `Settings > Account`.
- Mention that iOS room preview opens instantly on device; metric room generation (GeoCalib + Depth Anything + RTMDet) runs on first save. Furniture detection in viewers may download RTMDet via On-Demand Resources. Network access is used for account authentication and ODR model download.
- Call out the motion permission only where the reviewer will actually encounter it.

## Pre-Submission QA
- Fresh install on device.
- Sign in, generate a room, open each room viewer, share a room, log out, sign back in, and delete the account.
- After models have downloaded once, repeat the viewer flow offline long enough to verify graceful operation or graceful failures instead of infinite loading states.
