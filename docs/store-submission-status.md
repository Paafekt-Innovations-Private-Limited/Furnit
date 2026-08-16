# Store submission status (Paafekt)

Last verified: 17 August 2026

This file records only state verified from the repository, the Play-installed Android
artifact, live Firebase configuration, or the owner's explicit account status. A
processed upload is not described as reviewed, approved, or published without a
separate confirmation.

## Android — Google Play

| Item | Current state |
|---|---|
| Publisher/legal entity | **Paafekt Innovations Private Limited (India)** |
| Release | **1.3 / code 6 confirmed at 100% production**; signed 1.4 / code 8 replacement prepared after correcting the code-7 16 KB page-size rejection |
| Package | `com.paafekt.android` |
| Kotlin namespace | `com.furnit.android` (internal; intentionally different) |
| Play signing | Play App Signing active; production-installed certificate extracted and verified |
| Firebase Phone Auth | Play signing SHA-1/SHA-256 registered live on 2026-08-03; original production authorization error resolved |
| OTP autofill | Current source preserves manual six-field entry and contains an on-device-unconfirmed SMS User Consent candidate |
| Legal resources | Current source bundles offline Apache, LiteRT/Caffe, ONNX Runtime, Three.js, reCAPTCHA, and model-conversion notices; enforced by `verifyLegalAssets` |
| 16 KB page sizes | Code 8 AAB has `0x4000`-aligned 64-bit ELF segments; arm64 APK passes `zipalign -P 16` |

The published-build failure shown on 2026-08-03 was not caused by the developer
account's legal name or by the upload key. Firebase received a Play Integrity token but
had no matching Play App Signing SHA-256. Both Play fingerprints are now registered on
the active Firebase app. See
[`../android/docs/AUTHENTICATION.md`](../android/docs/AUTHENTICATION.md).

### Android 1.4 handoff

- `android/app/build.gradle` resolves to version name **1.4**, version code **8**.
- Code 7 was uploaded but Play blocked it because app-owned `libfurnit_rtmdet.so` used
  4 KB ELF segments. NDK r27 now receives
  `ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON`; every 64-bit library in the rebuilt code-8
  AAB reports `0x4000` alignment and the arm64 APK passes `zipalign -P 16`.
- The signed `app/build/outputs/bundle/release/app-release.aab` was rebuilt and
  signature-verified on 2026-08-17. Its package is `com.paafekt.android`; its R8
  mapping is embedded in bundle metadata and also exists at
  `app/build/outputs/mapping/release/mapping.txt`.
- Do not mark 1.4 published until Play Console or a Play-installed update confirms it.
  After publication, smoke-test Firebase verification, manual OTP entry, offline
  Settings → Licenses documents, and the room/Furniture Fit flows.

Do not recreate the upload key because its certificate organization metadata names
Paafekt Inc. That metadata does not change the Play publisher or Firebase app identity.

## Firebase (`paafektprod`)

| Item | Current state |
|---|---|
| Phone Auth | Enabled |
| Active Android app | `com.paafekt.android` — `1:613415224058:android:8d0a97fe4990e559a13f43` |
| Legacy Android client | `com.furnit.android` retained in `google-services.json` for compatibility |
| Upload SHA-1/SHA-256 | Registered |
| Play App Signing SHA-1/SHA-256 | Registered and live-verified 2026-08-03 |
| SMS region policy | Last verified global/allow-by-default on 2026-07-25; see [`firebase-sms-regions.md`](firebase-sms-regions.md) |

The maintenance script is
[`../android/scripts/fix_firebase_android_auth.sh`](../android/scripts/fix_firebase_android_auth.sh).
It performs live writes and should be used only when the Firebase app or signing keys
change.

## iOS — App Store Connect

| Item | Current state |
|---|---|
| Publisher/legal entity | **Paafekt Innovations Private Limited (India)** |
| Live App Store version | **1.2**, observed on the public App Store listing on 2026-08-16 |
| Source candidate | **1.4 / build 88** for the iOS application target |
| Candidate validation | Fresh unsigned iPhoneOS Release build succeeded; it contains RTMDet Core ML and no `.tflite` resource |
| Build upload | Archive/upload of 1.4 / build 88 is not confirmed |
| Review/publication | Infinite Zoom and preview/save appearance remain device-unconfirmed; do not call the candidate release-ready |
| Legal resources | Current source bundles offline `APACHE-2.0.txt` and `THIRD_PARTY_NOTICES.txt` in Settings → Licenses; no LiteRT license ships on iOS |

Apple's upload status `Complete` means processing succeeded; it does not by itself mean
the version was added to an App Review submission or released. Historical build details
remain in [`release-status-2026-07-20.md`](release-status-2026-07-20.md); the reusable
procedure is [`apple-review-checklist.md`](apple-review-checklist.md).

## Legal and public URLs

| Item | Value |
|---|---|
| App operator/controller | Paafekt Innovations Private Limited (India) |
| Affiliate | Paafekt Inc. (United States), affiliate only |
| Privacy | `https://paafekt.com/privacy` |
| Terms | `https://paafekt.com/terms` |
| Support | `https://paafekt.com/support` / `support@paafekt.com` |
| Account deletion | `https://paafekt.com/delete-account` |

The repository source for the privacy page is [`privacy.html`](privacy.html). Committing
that file does not deploy the public website; deploy and visually verify the live page
through the website's own release process.
