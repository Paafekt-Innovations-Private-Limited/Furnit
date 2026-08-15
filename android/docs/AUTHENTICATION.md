# Android authentication

This is the operational source of truth for Paafekt Android phone-number sign-in.
The implementation is under
[`app/src/main/java/com/furnit/android/auth/`](../app/src/main/java/com/furnit/android/auth/).

## Production identity

| Item | Value |
|---|---|
| Google Play package / Gradle `applicationId` | `com.paafekt.android` |
| Kotlin namespace | `com.furnit.android` (intentionally unchanged) |
| Firebase project | `paafektprod` |
| Firebase Android app ID | `1:613415224058:android:8d0a97fe4990e559a13f43` |
| App operator and publisher | Paafekt Innovations Private Limited (India) |
| Affiliate | Paafekt Inc. (United States); not the App operator/controller |

`app/google-services.json` also retains the legacy `com.furnit.android` client for
compatibility. New production work must target the `com.paafekt.android` entry above.

## Signing certificates required by Firebase

Firebase Phone Auth must recognize the certificate that signs the installed APK. A
local or direct-upload build is signed by the upload key; a Google Play install is
signed by the Play App Signing key. Both identities are intentionally registered.

| Certificate | SHA-1 | SHA-256 |
|---|---|---|
| Upload key | `E5:1A:A3:F3:B8:01:15:9E:AD:5A:86:C8:CE:F8:F5:1E:21:5B:2B:7A` | `5F:0D:73:33:18:7F:86:1A:5A:33:A6:02:0B:C2:24:78:53:1C:25:CF:13:68:23:87:ED:C2:F3:3B:C8:E0:BE:19` |
| Play App Signing key | `6C:62:DC:4B:C6:53:CB:4B:0C:23:A6:F8:52:E7:FC:C8:D0:7E:70:20` | `DE:AD:21:9D:82:71:DA:FC:34:1E:0B:30:AB:02:C6:08:86:45:01:07:50:2E:53:C6:10:76:3E:34:56:9A:59:BF` |

On 2026-08-03, the Play-distributed app failed Phone Auth because the Play App
Signing SHA-256 was absent from Firebase. The Play SHA-1 and SHA-256 were added to the
live `com.paafekt.android` Firebase entry and verified through the Firebase Management
API. The repository repair script now carries debug, upload, and Play certificates:

```bash
bash scripts/fix_firebase_android_auth.sh
```

Run that script only from an authenticated terminal when a Firebase app or signing key
changes. It writes live Firebase configuration and refreshes
`app/google-services.json`; it is not a routine build step.

## Country-code preselection

[`CountryCode.getDefaultCountry(context)`](../app/src/main/java/com/furnit/android/auth/CountryCode.kt)
prefers the current mobile-network country, then the SIM country, then the app/device
locale. It does not use IP geolocation or request phone/SMS permissions. This prevents
an Indian phone or SIM using English (United Kingdom) from incorrectly defaulting to
United Kingdom (`+44`) when Android exposes India through the network or SIM.

The user can always choose India (`+91`) or another country in the localized picker.
If both network and SIM country are unavailable (for example, a Wi-Fi-only device),
the fallback is the effective app/device locale. Android app-language and locale
behavior are documented in [`LOCALIZATION.md`](LOCALIZATION.md).

## OTP autofill

The verification screen uses six visible one-character `EditText` fields. On Android
8.0 and newer, each field declares its positional autofill hint:

```text
smsOTPCode1, smsOTPCode2, smsOTPCode3, smsOTPCode4, smsOTPCode5, smsOTPCode6
```

This lets a compatible Android autofill service or keyboard distribute a received
six-digit code across the split fields. Firebase may also complete verification
directly through `onVerificationCompleted`. The source change was added on 2026-08-03
and is included in the signed version 1.2 / code 5 artifact. Production delivery must
still be confirmed from Play Console or a Play-installed update.

Manual entry remains supported and is the required fallback. The app does not start
the SMS User Consent API in parallel with Firebase Auth because Firebase already owns
the SMS Retriever path. Device services, keyboard settings, message format, and Google
Play services can still affect whether a suggestion appears; the app cannot force the
user's keyboard to show one.

## Debug Settings test identity

Developer Settings are compiled only into debug builds and are additionally visible
only when the authenticated Firebase phone number normalizes to `+1 650-555-3434`.
Formatting characters are ignored, but the country code is required. If a different
user opens Settings, the app hides the Developer section and clears any debug mode
previously persisted by the test identity. This gate does not bypass Firebase Auth or
change manual OTP entry.

## Production smoke test

Use a fresh Google Play install, not a locally signed APK:

1. Confirm the package is `com.paafekt.android` and the build came from Play.
2. Open login and verify the expected country code, or choose it manually.
3. Send a verification code to a real number in an enabled Firebase SMS region.
4. Confirm Firebase accepts the Play-signed build (no package/SHA authorization error).
5. On the OTP screen, wait for automatic verification or the keyboard/autofill
   suggestion; then verify that manual six-digit entry still succeeds.
6. Test resend/rate-limit messaging and sign-out/sign-in once more.

SMS region policy and reviewer test-number setup are documented in
[`../../docs/firebase-sms-regions.md`](../../docs/firebase-sms-regions.md).
