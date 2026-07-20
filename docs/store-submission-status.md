# Store submission status (Paafekt)

Last updated: 20 July 2026

## iOS — App Store Connect

| Item | Status |
|------|--------|
| Build submitted | **1.0 (86)** |
| Review status | **Waiting for Review** |
| Submitted | 20 Jul 2026 ~6:49 AM (by Kishore Shivanna) |
| Screenshots | iPhone 6.5" + iPad 13" uploaded |
| Privacy Policy URL | `https://paafekt.com/privacy` |
| Terms | `https://paafekt.com/terms` |
| Support | `support@paafekt.com` |

### App Privacy (published)

Data types declared (linked to user, **not** used for tracking):

- Phone Number — App Functionality (Firebase OTP)
- User ID — App Functionality (Firebase Auth UID)
- Device ID — App Functionality (Firebase auth / install verification)

Tracking / IDFA: **No**

### App Review Information

Filled under **General → App Review** (Sign-in required + demo phone/OTP + notes + contact).  
Demo credentials: Firebase Console → Authentication → Phone → Phone numbers for testing.

### Release

Automatically release when approved (as selected in App Review / version settings — confirm if you change this).

### Related docs

- [apple-review-checklist.md](apple-review-checklist.md)
- [firebase-sms-regions.md](firebase-sms-regions.md)

---

## Android — Google Play Console

| Item | Status |
|------|--------|
| Developer account | **PaafektInnovations** (org) |
| Publish blocked by | Identity / org verification |
| Google verifying identity | **In progress** (documents uploaded; wait for email, often a few business days) |
| Phone number verification | **Blocked** until identity/org docs approved |
| App upload / production submit | **Not started** — finish account setup first |

When verification clears: create app → Data safety → store listing → AAB → (new personal accounts may also need closed testing rules; org path may differ) → see [android-play-review-checklist.md](android-play-review-checklist.md).

---

## Firebase (`paafektprod`)

| Item | Status |
|------|--------|
| Phone Auth | Enabled |
| SMS region policy | Allowlist of launch countries (includes **FR**, **US**, **DE**, **CN**, **IN**, etc.) — not worldwide |
| Apply/update allowlist | `./scripts/set_firebase_sms_regions.sh` (needs `gcloud auth`) |

---

## Website (paafekt.com)

| Item | Status |
|------|--------|
| Privacy / Terms / Support URLs | Live; used by the apps |
| Support FAQ (App Store / Play copy) | Updated when deployed from the correct Netlify account |

---

## Next actions (when you’re back)

1. **iOS:** Wait for Apple email (Waiting for Review → In Review → Approved / Rejected). Respond if they ask for demo access.
2. **Android:** Wait for Play identity verification email; then complete phone verify and continue Play Console setup.
3. Keep App Privacy / Data safety in sync if you add Crashlytics or other SDKs later.
