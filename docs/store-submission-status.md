# Store submission status (Paafekt)

Last verified: 3 August 2026

This file records only state verified from the repository, the Play-installed Android
artifact, live Firebase configuration, or the owner's explicit account status. Console
state that was not re-opened on this date is labeled as a prior snapshot.

## Android — Google Play

| Item | Current state |
|---|---|
| Publisher/legal entity | **Paafekt Innovations Private Limited (India)** |
| Release | **Published on Google Play** |
| Package | `com.paafekt.android` |
| Kotlin namespace | `com.furnit.android` (internal; intentionally different) |
| Play signing | Play App Signing active; production-installed certificate extracted and verified |
| Firebase Phone Auth | Play signing SHA-1/SHA-256 registered live on 2026-08-03; original production authorization error resolved |
| OTP autofill | Positional split-field hints are in source as of 2026-08-03; requires the next Play release |

The published-build failure shown on 2026-08-03 was not caused by the developer
account's legal name or by the upload key. Firebase received a Play Integrity token but
had no matching Play App Signing SHA-256. Both Play fingerprints are now registered on
the active Firebase app. See
[`../android/docs/AUTHENTICATION.md`](../android/docs/AUTHENTICATION.md).

### Next Android release

1. Verify the active Play version code, then bump `versionCode` in
   `android/app/build.gradle` to a higher value.
2. Build and upload a signed AAB; archive the R8 `mapping.txt` for that release.
3. Install the update from Google Play and smoke-test automatic verification/autofill
   plus manual OTP entry.
4. Recheck country preselection under an English (India) locale and an `en-GB` locale.

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

The last repository snapshot, dated 25 July 2026, recorded build **1.0 (86)** as
**Waiting for Review**, submitted on 20 July 2026. App Store Connect was not re-opened
during the 2026-08-03 update, so that review state must not be presented as current.

The Apple developer account/publisher is Paafekt Innovations Private Limited (India).
Recheck App Store Connect before changing release status. Historical build details are
in [`release-status-2026-07-20.md`](release-status-2026-07-20.md); the reusable review
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
