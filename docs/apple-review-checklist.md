# Apple Review Checklist

## In-App Reviewer Path
- Launch the app from a fresh install.
- Complete phone sign-in with the reviewer test account.
- Open `Settings` and verify `Delete Account` is available under `Account`.
- Confirm the room viewers load without network-fetched JavaScript assets.
- Verify camera, photo-library, and motion permission prompts use accurate wording.

## App Store Connect
- App Privacy answers must cover phone-number authentication, local account storage, room-photo uploads for server generation, and any Firebase analytics/crash usage that is enabled in the release build.
- Confirm the export compliance answer remains correct for the shipped build.
- Ensure screenshots and app description do not claim unfinished or hidden functionality.

## Reviewer Notes
- Provide the reviewer phone-login test path and any OTP instructions.
- Mention that `Delete Account` is available in `Settings > Account`.
- Mention that room generation depends on the configured backend service and summarize the expected network behavior.
- Call out the motion permission only where the reviewer will actually encounter it.

## Pre-Submission QA
- Fresh install on device.
- Sign in, generate a room, open each room viewer, share a room, log out, sign back in, and delete the account.
- Repeat the viewer flow offline long enough to verify graceful failures instead of infinite loading states.
