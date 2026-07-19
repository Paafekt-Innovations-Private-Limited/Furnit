# Apple Review Checklist

## In-App Reviewer Path
- Launch the app from a fresh install.
- Complete phone sign-in with the reviewer test account.
- Open `Settings` and verify `Delete Account` is available under `Account`.
- Confirm the room viewers load without network-fetched JavaScript assets.
- Verify camera, photo-library, and motion permission prompts use accurate wording.

## App Store Connect
- Do not describe iOS room generation as a photo-upload/backend feature unless that backend path is explicitly enabled in the submitted build. Default iOS room creation is on-device: **instant preview (no ML)** then **GeoCalib + Depth Anything + RTMDet object anchor → USDZ on first save**.
- Confirm the export compliance answer remains correct for the shipped build (`ITSAppUsesNonExemptEncryption = NO`).
- Ensure screenshots and app description do not claim unfinished or hidden functionality, and do not imply room photos are uploaded for generation.
- Privacy Policy URL: `https://paafekt.com/privacy` — Terms: `https://paafekt.com/terms` — Support: `support@paafekt.com`.
- Firebase Phone OTP: allowlist SMS regions (including **France / FR**) and use a test phone for review — see [firebase-sms-regions.md](firebase-sms-regions.md).

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
