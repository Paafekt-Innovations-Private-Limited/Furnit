# Android Submission Policy Audit

Date: 2026-07-25

This is a pragmatic release checklist, not legal advice.

> **Historical submission-preparation audit.** Android is now released on Google
> Play. Current authentication and signing state lives in
> [`docs/AUTHENTICATION.md`](docs/AUTHENTICATION.md); current store state lives in
> [`../docs/store-submission-status.md`](../docs/store-submission-status.md).

## Production addendum — 2026-08-03

- Google Play distributes `com.paafekt.android` using the Play App Signing key.
- The production-installed APK's Play signing SHA-1 and SHA-256 were extracted and
  added to the matching Firebase Android app. The live certificate list was verified.
- This resolves the Firebase error reporting that a Play Integrity token was present
  but no matching SHA-256 was registered.
- Split-field OTP autofill now uses Android's positional `smsOTPCode1` through
  `smsOTPCode6` hints in source. A new Play release is required to deliver that change.

## Production addendum — 2026-08-05

- Signed version **1.2 / code 5** was built and inspected; Play publication is not
  inferred until separately confirmed.
- Settings → Licenses now opens the full Apache License 2.0, LiteRT 1.4.2's combined
  license/Caffe attribution, and the third-party/model-conversion notice offline.
- `verifyLegalAssets` runs before `preBuild` and blocks missing or truncated legal
  resources.
- Current code chooses the default login country from mobile network, then SIM, then
  app/device locale. This supersedes the locale-only state recorded in the historical
  `Checked` list below; the user can always override the picker manually.

## Checked

- Privacy Policy link exists in app: Settings > Legal > Privacy Policy opens `https://paafekt.com/privacy`.
- Terms link exists in app: Settings > Legal > Terms of Service opens `https://paafekt.com/terms`.
- In-app account deletion exists: Settings > Account > Delete Account deletes the Firebase Auth account and clears local auth state.
- Camera permission is the only sensitive Android runtime permission declared.
- Advertising ID permission is explicitly removed from the merged manifest if a dependency contributes it.
- Rooms and auth/session shared preferences are excluded from Android cloud backup and device transfer.
- Room processing, furniture detection, and model inference are disclosed as on-device in Settings, FAQ, and privacy copy.
- Licenses and credits screens exist in Settings and cover shipped Android runtime libraries, model assets, datasets, AI/tool credits, full Apache text, and LiteRT/Caffe attribution.
- SIM/network country lookup was removed from login country preselection; Android now uses locale only and lets the user choose manually.

## Build and signing maintenance

- Build the bundle with the `PAAFEKT_UPLOAD_*` signing properties set; an unsigned `bundleRelease` output is rejected by Play. The upload keystore and credentials already exist on this machine (`~/.gradle/paafekt/paafekt-upload-key.jks`, values in `~/.gradle/gradle.properties`). Run the signed build from a normal terminal — sandboxed agent shells override `GRADLE_USER_HOME` and produce an unsigned AAB. See "Release Build and Signing" in `README_ANDROID.md`.
- Release builds are R8-minified. Archive `app/build/outputs/mapping/release/mapping.txt` per release; AGP also embeds it in the AAB for Play ingestion. Emailed crash reports from `CrashReportActivity` still need the archived copy for local retracing.
- Keep both Play App Signing fingerprints registered on the Firebase Android app; they were added and live-verified on 2026-08-03. Re-run `scripts/fix_firebase_android_auth.sh` only if the signing key or Firebase app changes.
- The application ID is `com.paafekt.android` (permanent after first publish; Kotlin namespace stays `com.furnit.android`). The matching Firebase Android app (`1:613415224058:android:8d0a97fe4990e559a13f43`) is registered in `paafektprod`, and `google-services.json` contains both the active and legacy clients.

## Play Console / website items to verify before submission

- Privacy Policy URL must be live, public, non-geofenced, non-PDF, and match the Play Console Data Safety form.
- Terms URL should be live and public.
- Because Paafekt allows account creation, Play requires an external web resource where users can request account deletion without reinstalling the app. Use `https://paafekt.com/delete-account` in Play Console's account deletion field rather than relying only on a mailto link buried in the privacy policy.
- Data Safety should disclose:
  - Phone number and authentication/user ID data, collected for account management and app functionality.
  - Device or other identifiers used by Firebase/Play Integrity/reCAPTCHA for authentication and abuse prevention.
  - Camera/photos/video frames are accessed for app functionality; core room/furniture processing stays on device unless the user explicitly shares/exports content.
  - Support email/message data if the user contacts support.
  - No advertising ID use, no ads, no sale of personal/sensitive data.
- Target audience should not be child-directed unless the product and SDKs are reviewed under Google Play Families policy.

## Known license diligence risk

- Depth Anything V2 Metric Indoor Small is documented upstream as Apache-2.0 for Small weights, but the metric indoor model was fine-tuned on Apple's Hypersim dataset under CC BY-SA 3.0. The repo audit treats this as a lawyer-priority tail risk for commercial redistribution of exported weights.
