# Store submission status (Paafekt)

Last verified: 5 August 2026

This file records only state verified from the repository, the Play-installed Android
artifact, live Firebase configuration, or the owner's explicit account status. A
processed upload is not described as reviewed, approved, or published without a
separate confirmation.

## Android — Google Play

| Item | Current state |
|---|---|
| Publisher/legal entity | **Paafekt Innovations Private Limited (India)** |
| Release | **1.1 / code 4 confirmed published**; signed 1.2 / code 5 artifact prepared, Play upload/publication not re-confirmed |
| Package | `com.paafekt.android` |
| Kotlin namespace | `com.furnit.android` (internal; intentionally different) |
| Play signing | Play App Signing active; production-installed certificate extracted and verified |
| Firebase Phone Auth | Play signing SHA-1/SHA-256 registered live on 2026-08-03; original production authorization error resolved |
| OTP autofill | Positional split-field hints are included in signed 1.2 / code 5; production delivery awaits Play publication confirmation |
| Legal resources | 1.2 bundles offline Apache, LiteRT/Caffe, and third-party/model-conversion notices; enforced by `verifyLegalAssets` |

The published-build failure shown on 2026-08-03 was not caused by the developer
account's legal name or by the upload key. Firebase received a Play Integrity token but
had no matching Play App Signing SHA-256. Both Play fingerprints are now registered on
the active Firebase app. See
[`../android/docs/AUTHENTICATION.md`](../android/docs/AUTHENTICATION.md).

### Android 1.2 handoff

- `android/app/build.gradle` resolves to version name **1.2**, version code **5**.
- The signed `app/build/outputs/bundle/release/app-release.aab` was rebuilt and verified
  on 2026-08-04. Its package is `com.paafekt.android`; its R8 mapping is embedded in
  bundle metadata and also exists at `app/build/outputs/mapping/release/mapping.txt`.
- Bundle inspection confirmed the three legal assets and the intended Depth Anything
  V2 Metric Indoor Small, GeoCalib pinhole, and RTMDet FP16 model files.
- Do not mark 1.2 published here until Play Console or a Play-installed update confirms
  it. After publication, smoke-test Firebase automatic verification/autofill, manual
  OTP entry, offline Settings → Licenses documents, and the room/Furniture Fit flows.

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
| Source version | **1.2 / build 87** for the iOS application target |
| Build upload | Owner-reported App Store Connect status **Complete**, created 2026-08-05 00:14 IST |
| Previous upload | 1.1 / build 86 shown **Complete**, created 2026-08-01 04:26 IST |
| Version metadata | Owner reported the 1.2 version page completed on 2026-08-05 |
| Review/publication | Not separately confirmed; do not infer it from Build Upload status `Complete` |
| Legal resources | 1.2 bundles offline `APACHE-2.0.txt` and `THIRD_PARTY_NOTICES.txt` in Settings → Licenses |

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
