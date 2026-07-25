# Firebase SMS regions (France and launch countries)

Phone OTP in Paafekt uses **Firebase Authentication** only. There is no Paafekt SMS server or user DB for rooms.

Firebase does **not** send OTP SMS worldwide by default. Regions must be allowlisted under **SMS region policy**. If France (`FR`, dial `+33`) is missing, OTP to French numbers fails (region not enabled / operation not allowed / SMS unable to be sent until this region is enabled).

The in-app country picker listing France does **not** enable Firebase SMS to France by itself.

## CLI (preferred)

Firebase CLI **does not** have a built-in SMS-region command. This repo has:

1. **Local Firebase CLI** (for general Firebase commands):  
   `.tools/firebase-cli/node_modules/.bin/firebase`  
   (install: `npm install firebase-tools --prefix .tools/firebase-cli` — folder is gitignored)

2. **SMS allowlist script** (Identity Toolkit Admin API — includes **FR**):  
   [`scripts/set_firebase_sms_regions.sh`](../scripts/set_firebase_sms_regions.sh)

```bash
# Preview body (no network write)
./scripts/set_firebase_sms_regions.sh --dry-run

# Apply to paafektprod (needs gcloud auth as support@paafekt.com)
brew install --cask google-cloud-sdk   # once, if gcloud missing
gcloud auth login support@paafekt.com
gcloud config set project paafektprod
./scripts/set_firebase_sms_regions.sh --global      # worldwide OTP (current)
# ./scripts/set_firebase_sms_regions.sh --allowlist # tighter policy if SMS abuse appears
```

Optional system-wide Firebase CLI: `brew install firebase-cli`.

## Console steps (alternative)

1. Open [Firebase Console](https://console.firebase.google.com) → Paafekt project (`paafektprod`).
2. **Authentication** → **Sign-in method** → confirm **Phone** is enabled.
3. **Authentication** → **Settings** → **SMS region policy**.
4. Current production setting: **Allow by default** (global). For a tighter policy, switch to allowlist-only and add launch countries (include **France / FR** for review traffic).
5. Save.
7. Test a real `+33` number, or use a Firebase **test phone number** for App Review / Play review.

## Current policy (2026-07-25)

**Global:** `paafektprod` uses Firebase SMS **allow by default** (all regions). This matches a worldwide Play release. Watch Firebase Usage / billing for SMS abuse; switch back to allowlist with:

```bash
./scripts/set_firebase_sms_regions.sh --allowlist
```

## Regions for allowlist mode (match app country picker)

If you re-enable allowlist-only, use at least these codes (same set as iOS `LoginView` / Android `CountryCode`):

`IN`, `US`, `GB`, `CA`, `AU`, `DE`, `FR`, `IT`, `ES`, `BR`, `MX`, `JP`, `KR`, `CN`, `SG`, `MY`, `ID`, `TH`, `VN`, `PH`, `PK`, `BD`, `LK`, `NP`, `AE`, `SA`, `QA`, `KW`, `OM`, `BH`, `ZA`, `NG`, `KE`, `EG`, `RU`, `NL`, `BE`, `CH`, `AT`, `SE`, `NO`, `DK`, `FI`, `IE`, `PT`, `GR`, `TR`, `PL`, `NZ`, `AR`, `CL`, `CO`, `PE`, `IL`

```bash
./scripts/set_firebase_sms_regions.sh --global      # all countries (default)
./scripts/set_firebase_sms_regions.sh --allowlist   # limited list above
```

## “More details” form (copy-paste answers)

If Firebase / Google asks for more information when enabling Phone Auth or a region, they want **anti-abuse / use-case** answers — not room ML or cloud storage details.

| They ask | Answer for Paafekt |
|----------|--------------------|
| What is the SMS used for? | One-time password (OTP) for user sign-in only |
| App / product name | Paafekt |
| Website | `https://paafekt.com` |
| Privacy policy | `https://paafekt.com/privacy` |
| Do you store SMS content? | No |
| Backend / own SMS DB? | No Paafekt server DB; Firebase Authentication only |
| Expected volume | Low–moderate login OTPs for launch countries above |
| Company | Paafekt Inc. (United States) / Paafekt Innovations Private Limited (India) |
| Account deletion | User can delete account in-app (Settings → Account); Firebase Auth user is removed |

They are **not** asking you to declare room photos or on-device ML as cloud-collected data.

## Not the same as store privacy questionnaires

| Topic | Where |
|-------|--------|
| Firebase France / SMS regions | This doc + Firebase Console |
| Apple App Privacy labels | [apple-review-checklist.md](apple-review-checklist.md) |
| Google Play Data safety | [android-play-review-checklist.md](android-play-review-checklist.md) |

## App Review tip

For Apple / Google reviewers, add a **Firebase test phone number + fixed OTP** under Authentication → Sign-in method → Phone → Phone numbers for testing, and paste those credentials into review notes. That avoids live SMS region delivery issues during review.
