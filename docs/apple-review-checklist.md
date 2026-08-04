# Apple Review Checklist

## In-App Reviewer Path
- Launch the app from a fresh install.
- Complete phone sign-in with the reviewer test account.
- Open `Settings` and verify `Delete Account` is available under `Account`.
- Open `Settings` → `Licenses`; open the bundled third-party notices and Apache text
  with the device offline.
- Confirm the room viewers load without network-fetched JavaScript assets.
- Verify camera, photo-library, and motion permission prompts use accurate wording.

## App Store Connect
- Do not describe iOS room generation as a photo-upload/backend feature unless that backend path is explicitly enabled in the submitted build. Default iOS room creation is on-device: **instant preview (no ML)** then **GeoCalib + Depth Anything + RTMDet object anchor → USDZ on first save**.
- Confirm the export compliance answer remains correct for the shipped build (`ITSAppUsesNonExemptEncryption = NO`).
- Ensure screenshots and app description do not claim unfinished or hidden functionality, and do not imply room photos are uploaded for generation.
- Describe generated dimensions as estimates and avoid unsupported “meter-accurate” or
  walkaround-scanner claims; the current default is a single-photo flow.
- Privacy Policy URL: `https://paafekt.com/privacy` — Terms: `https://paafekt.com/terms` — Support: `support@paafekt.com`.
- Firebase Phone OTP: allowlist SMS regions (including **France / FR**) and use a test phone for review — see [firebase-sms-regions.md](firebase-sms-regions.md).
- Copyright metadata should name the exclusive rights owner. For the current app
  record use `2026 Paafekt Innovations Private Limited` without a copyright symbol;
  App Store Connect adds the symbol.

### Version 1.2 handoff (2026-08-05)

- Application target: marketing version **1.2**, build **87**.
- App Store Connect Build Upload status was owner-reported as **Complete** at 00:14 IST.
- The version page was completed, but review/publication status was not separately
  captured; Build Upload `Complete` must not be treated as App Review approval.
- Select build 87, choose the intended release option, add the version to a submission,
  and confirm the final submission status becomes `Waiting for Review`.

## App Privacy (declare here — copy into App Store Connect)

**Where:** [App Store Connect](https://appstoreconnect.apple.com) → your app **Paafekt** → **App Privacy** (left sidebar) → **Get Started** / **Edit** → answer the questionnaire → **Publish**.

There is no Paafekt cloud DB for rooms or photos. Network is Firebase Authentication only (sign-in). Delete Account removes the Firebase Auth user. Rooms/photos stay on device only.

### First questions
| Question | Answer |
|----------|--------|
| Do you or your third-party partners collect data from this app? | **Yes** |
| Do you or your third-party partners use data from this app for tracking purposes? | **No** |

### Data types to add (only these)

**1. Contact Info → Phone Number**
- Collected: **Yes**
- Linked to the user’s identity: **Yes**
- Used for tracking: **No**
- Purposes: **App Functionality** only  
- Third party: Firebase Authentication (Google) processes phone OTP; not used for advertising.

**2. Identifiers → User ID**
- Collected: **Yes**
- Linked to the user’s identity: **Yes**
- Used for tracking: **No**
- Purposes: **App Functionality** only  
- Note: Firebase Auth UID; deleted when the user deletes their account in the app.

**3. Identifiers → Device ID**
- Collected: **Yes**
- Linked to the user’s identity: **Yes**
- Used for tracking: **No**
- Purposes: **App Functionality** and/or **Fraud Prevention** (Firebase abuse prevention for OTP)  
- Not used for advertising or cross-app tracking.

**4. Contact Info → Name** (optional but accurate)
- Collected: **Yes** (account display name)
- Linked to the user’s identity: **Yes**
- Used for tracking: **No**
- Purposes: **App Functionality** only  
- Note: Stored **on device only**; not uploaded to a Paafekt server.

### Do **not** add
- Photos or Videos (processed/stored on device; not collected by Paafekt servers)
- Product Interaction / Advertising Data / Usage Data for ads
- Precise Location / Coarse Location (unless you later add those features)
- Crash Data / Performance Data (unless you ship those SDKs)
- Any “Used for Tracking” = Yes

### After saving
Publish the privacy nutrition labels, then set Privacy Policy URL on the app version / App Information to `https://paafekt.com/privacy`.

## Reviewer Notes (paste into App Review Information)

```
Demo access: Use the Firebase test phone number and fixed OTP entered in the Sign-In Information fields. These are non-expiring Firebase test credentials, and no SMS is required.

Sign-in is required using Firebase phone authentication.

Suggested review flow:
1. Sign in using the supplied test credentials.
2. From Home, choose Photo → 3D and capture or select a room photo.
3. An immediate preview opens. Saving the room runs on-device metric reconstruction using GeoCalib, Depth Anything, and RTMDet.
4. Open the saved room and use Furniture Fit to identify and position furniture.
5. Open Settings → Licenses to view the bundled third-party notices and Apache license offline.

Furniture detection may download the RTMDet model through Apple On-Demand Resources when first needed.

Delete Account: Settings → Account → Delete Account removes the Firebase Authentication user. Locally stored rooms remain on the device and can be removed separately.

Permissions: Camera (room photos and AR furniture), Photos (select/save), Motion (AR viewing).

No ads, no App Tracking, no in-app purchases, and no subscriptions.
Privacy: https://paafekt.com/privacy
Support: support@paafekt.com
```

## Pre-Submission QA
- Fresh install on device.
- Sign in, generate a room, open each room viewer, capture/save a view to Photos, log out, sign back in, and delete the account.
- After models have downloaded once, repeat the viewer flow offline long enough to verify graceful operation or graceful failures instead of infinite loading states.
- With networking disabled, open both documents under Settings → Licenses and confirm
  the model-conversion notices name Depth Anything V2 Metric Indoor Small.
- Confirm Release archive push entitlement uses production APNs (source entitlements may show development; Xcode rewrites for App Store distribution).
