# Runbooks

Operational procedures for building, testing, authenticating, reviewing, and
diagnosing Furnit.

| Runbook | Covers |
|---|---|
| [`../../android/docs/AUTHENTICATION.md`](../../android/docs/AUTHENTICATION.md) | Android Firebase identity, signing certificates, locale country defaults, OTP autofill, and Play-installed smoke test |
| [`../firebase-sms-regions.md`](../firebase-sms-regions.md) | Firebase SMS region policy, CLI script, console route, and reviewer test numbers |
| [`../../android/docs/ANDROID_STUDIO_RUN.md`](../../android/docs/ANDROID_STUDIO_RUN.md) | Android Studio/device run procedure and logs |
| [`../../android/docs/TEST_AND_SETTINGS.md`](../../android/docs/TEST_AND_SETTINGS.md) | Android room/Furniture Fit/assets/auth smoke tests |
| [`../../android/README_ANDROID.md`](../../android/README_ANDROID.md) | Android release signing, AAB, R8 mapping, and model packs |
| [`../CHECK_APP_MEMORY.md`](../CHECK_APP_MEMORY.md) | iOS resident-memory, Xcode, Instruments, and device-log checks |
| [`../apple-review-checklist.md`](../apple-review-checklist.md) | App Store reviewer path and privacy declarations |
| [`../android-play-review-checklist.md`](../android-play-review-checklist.md) | Play reviewer path and Data Safety declarations |
| [`../store-submission-status.md`](../store-submission-status.md) | Current verified status and next-release work |
| [`../dev/UNUSED_CODE_PERIPHERY.md`](../dev/UNUSED_CODE_PERIPHERY.md) | iOS Periphery scan and triage |

## Common build checks

```bash
# Android, from android/
./gradlew :app:assembleDebug --no-daemon

# Android auth-focused compile/tests
./gradlew :app:compileDebugKotlin :app:testDebugUnitTest

# iOS, from repository root
xcodebuild -project "Furnit.xcodeproj" -scheme "Furnit" -configuration Debug \
  -sdk iphoneos -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO COMPILER_INDEX_STORE_ENABLE=NO ENABLE_PREVIEWS=NO \
  -jobs 2 build
```

Follow `CLAUDE.md` and `.cursor/rules/` before executing builds on this machine. Signed
Android release builds require the existing local Gradle signing properties and must
never expose or commit those values.
