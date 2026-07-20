# Paafekt Release Status - 2026-07-20

This document records the local release state for the iOS/Swift and Android builds prepared on July 20, 2026. It is a handoff checklist for App Store Connect and Google Play Console submission.

## Overall Status

| Platform | Status | Upload artifact | Notes |
|----------|--------|-----------------|-------|
| iOS / Swift | Ready for App Store Connect upload | `/Volumes/XcodeLocalCaches/Exports/Paafekt-1.0-85/Furnit.ipa` | Archive/export completed and Xcode validation passed. |
| Android | Ready for Play Console upload, with size watch | `/Users/al/Documents/tries01/Furnit/android/app/build/outputs/bundle/release/app-release.aab` | Signed AAB built successfully. Play Console is the final size gate. |
| Web compliance pages | Ready | `https://paafekt.com/privacy`, `https://paafekt.com/terms`, `https://paafekt.com/support`, `https://paafekt.com/delete-account` | Account deletion page exists for Play/App Review declarations. |

Generated binaries and archives are intentionally not source-control artifacts.

## iOS / Swift Release

### Build Identity

- App: `Paafekt` / target output name `Furnit`
- Bundle ID: `com.paafektinnovations.Paafekt`
- Version: `1.0`
- Build: `85`
- Team ID: `W38QXPUZWC`
- Xcode: `26.4.1`
- SDK: iPhoneOS `26.4`

### Artifact Locations

- App Store IPA: `/Volumes/XcodeLocalCaches/Exports/Paafekt-1.0-85/Furnit.ipa`
- Export folder: `/Volumes/XcodeLocalCaches/Exports/Paafekt-1.0-85`
- Archive: `/Volumes/XcodeLocalCaches/Archives/Paafekt-1.0-85-20260720.xcarchive`
- Validation export folder: `/Volumes/XcodeLocalCaches/Exports/Paafekt-1.0-85-validation`
- Distribution summary: `/Volumes/XcodeLocalCaches/Exports/Paafekt-1.0-85/DistributionSummary.plist`

### Validation Results

- Release archive completed successfully.
- App Store Connect export completed successfully.
- Xcode validation completed successfully: `Validated Furnit`.
- Distribution signing: `Cloud Managed Apple Distribution`.
- Provisioning profile: `iOS Team Store Provisioning Profile: com.paafektinnovations.Paafekt`.
- Entitlements confirmed from export summary:
  - `get-task-allow = false`
  - `aps-environment = production`
  - `beta-reports-active = true`

### Screenshot Assets

- iPhone App Store-size screenshot: `/Users/al/Desktop/paafekt-appstore-screenshots/paafekt-home-iphone-17-pro-max.png`
- iPad App Store-size screenshot: `/Users/al/Desktop/paafekt-appstore-screenshots/paafekt-home-ipad-pro-13.png`
- Restored simulator proof screenshots:
  - `/Users/al/Desktop/paafekt-appstore-screenshots/paafekt-home-iphone-16-pro-restored-sim.png`
  - `/Users/al/Desktop/paafekt-appstore-screenshots/paafekt-home-ipad-pro-13-restored-sim.png`

### iOS Submission Checklist

- [x] Release archive created.
- [x] App Store export created.
- [x] Xcode validation passed.
- [x] Build number raised above prior uploaded build `84`; current build is `85`.
- [x] Production APNs entitlement confirmed in exported build.
- [x] Account deletion exists in app and on web.
- [ ] Upload IPA/archive to App Store Connect.
- [ ] Select build `1.0 (85)` for the app version.
- [ ] Complete App Privacy labels using `docs/apple-review-checklist.md`.
- [ ] Set Privacy Policy URL: `https://paafekt.com/privacy`.
- [ ] Set Support URL/contact: `support@paafekt.com`.
- [ ] Add reviewer demo phone/test OTP details.
- [ ] Fill export compliance. Current checklist says `ITSAppUsesNonExemptEncryption = NO`.
- [ ] Upload screenshots and app metadata.
- [ ] Submit for App Review.

## Android Release

### Build Identity

- Application ID: `com.furnit.android`
- Version name: `1.0`
- Version code: `1`
- Min SDK: `24`
- Compile SDK: `36`
- Target SDK: `36`
- ABI split: `arm64-v8a`

### Artifact Locations

- Play upload AAB: `/Users/al/Documents/tries01/Furnit/android/app/build/outputs/bundle/release/app-release.aab`
- AAB size: about `359M`
- Release APK proxy/test artifact: `/Users/al/Documents/tries01/Furnit/android/app/build/outputs/apk/release/app-arm64-v8a-release.apk`
- APK size: about `340M`
- Unit test report: `/Users/al/Documents/tries01/Furnit/android/app/build/reports/tests/testReleaseUnitTest/index.html`
- Lint vital report: `/Users/al/Documents/tries01/Furnit/android/app/build/intermediates/lint_vital_intermediate_text_report/release/lintVitalReportRelease/lint-results-release.txt`

### Android Signing

- Upload keystore: `/Users/al/.gradle/paafekt/paafekt-upload-key.jks`
- Local signing credentials: `/Users/al/.gradle/gradle.properties`
- Do not commit the keystore or Gradle signing credentials.
- Upload certificate SHA-256:
  `5F:0D:73:33:18:7F:86:1A:5A:33:A6:02:0B:C2:24:78:53:1C:25:CF:13:68:23:87:ED:C2:F3:3B:C8:E0:BE:19`
- Upload certificate SHA-1:
  `E5:1A:A3:F3:B8:01:15:9E:AD:5A:86:C8:CE:F8:F5:1E:21:5B:2B:7A`

Important: after Play App Signing is configured, Google Play may distribute the app with a different app-signing certificate. Add the Play app-signing SHA-1/SHA-256 certificate from Play Console to Firebase for Android phone authentication. The local upload-key certificate is not necessarily the certificate installed on users' devices from Play.

### Validation Results

- `./gradlew :app:testReleaseUnitTest :app:lintVitalRelease :app:bundleRelease` passed.
- `./gradlew :app:assembleRelease` passed.
- Release unit tests passed:
  - `AestheticAdvisorTest`: 4 tests, 0 failures
  - `RoomIntelligenceEngineTest`: 8 tests, 0 failures
  - `RoomMathParityTest`: 3 tests, 0 failures
  - `RoomPaletteSamplerTest`: 3 tests, 0 failures
  - `FurnitureFitArMetricsTest`: 5 tests, 0 failures
- `lintVitalRelease`: no issues found.
- AAB `jarsigner -verify`: verified.
- APK `apksigner verify`: verified with APK Signature Scheme v2.

### Android Submission Checklist

- [x] Target SDK updated to `36`.
- [x] Signed release AAB created.
- [x] Release unit tests passed.
- [x] `lintVitalRelease` passed.
- [x] Account deletion exists in app and on web.
- [x] Advertising ID permission is removed from merged manifest.
- [ ] Upload AAB to Play Console.
- [ ] Configure Play App Signing.
- [ ] Copy Play app-signing SHA-1/SHA-256 into Firebase Android app settings.
- [ ] Use Firebase reviewer test phone number and fixed OTP.
- [ ] Complete Play Data Safety using `docs/android-play-review-checklist.md`.
- [ ] Set Privacy Policy URL: `https://paafekt.com/privacy`.
- [ ] Set Account deletion URL: `https://paafekt.com/delete-account`.
- [ ] Add app access instructions/test login details.
- [ ] Complete content rating, target audience, countries/regions, and store listing.
- [ ] Upload screenshots and graphics.
- [ ] Submit to internal testing first, then production.

## Android Size Watch

The Android release bundle includes large on-device model assets. Current artifact sizes:

- AAB: about `359M`
- arm64 APK proxy: about `340M`
- `android/app/src/main/assets`: about `319M`

Play Console is the final size authority. If Play rejects the release or flags download size, move the largest model assets to Play Asset Delivery, Play Feature Delivery, a first-run download flow, or ship smaller/quantized model variants.

## Firebase / Reviewer Access

- Firebase project in Android config: `paafektprod`.
- Phone OTP must be available for reviewer countries; include France (`FR`) because review traffic often comes from France.
- See `docs/firebase-sms-regions.md`.
- Suggested reviewer note format is in:
  - `docs/apple-review-checklist.md`
  - `docs/android-play-review-checklist.md`

## Web URLs To Use In Store Consoles

- Privacy: `https://paafekt.com/privacy`
- Terms: `https://paafekt.com/terms`
- Support: `https://paafekt.com/support`
- Account deletion: `https://paafekt.com/delete-account`
- Support email: `support@paafekt.com`

## Local Xcode Storage Note

The working Xcode/simulator storage is the APFS sparsebundle-backed volume:

- Mounted volume: `/Volumes/XcodeLocalCaches`
- Sparsebundle source: `/Volumes/ExtremeSSD/xcode/XcodeLocalCaches.sparsebundle`

The direct `/Volumes/ExtremeSSD/xcode/DerivedData/...` archive path created AppleDouble `._*` files in Swift Package checkouts and caused Clang to compile a binary sidecar header. Use `/Volumes/XcodeLocalCaches` for future simulator, DerivedData, archive, and export work. If the machine reboots, mount the sparsebundle again before using those paths.

## Final Upload Order

1. Upload iOS IPA/archive to App Store Connect.
2. Upload Android AAB to Play Console internal testing.
3. Configure Play App Signing and copy Play app-signing SHA fingerprints into Firebase.
4. Install store-distributed test builds from TestFlight and Play internal testing.
5. Run fresh-install QA on both platforms:
   - Phone sign-in
   - Create/open room
   - Camera/photo permissions
   - Save/capture flow
   - Sign out/sign in
   - Delete account
6. Complete store metadata, privacy/data-safety forms, screenshots, and reviewer notes.
7. Submit both stores for review.

