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

# Apply to paafektprod (needs gcloud auth)
brew install --cask google-cloud-sdk   # once, if gcloud missing
gcloud auth login
gcloud config set project paafektprod
./scripts/set_firebase_sms_regions.sh
```

Optional system-wide Firebase CLI: `brew install firebase-cli`.

## Console steps (alternative)

1. Open [Firebase Console](https://console.firebase.google.com) → Paafekt project (`paafektprod`).
2. **Authentication** → **Sign-in method** → confirm **Phone** is enabled.
3. **Authentication** → **Settings** → **SMS region policy**.
4. Choose **Allow** (allowlist-only).
5. Add every country you ship OTP for (at minimum include **France / FR**).
6. Save.
7. Test a real `+33` number, or use a Firebase **test phone number** for App Review / Play review.

## Regions to allowlist (match app country picker)

Allowlist at least these region codes (same set as iOS `LoginView` / Android `CountryCode`):

`IN`, `US`, `GB`, `CA`, `AU`, `DE`, `FR`, `IT`, `ES`, `BR`, `MX`, `JP`, `KR`, `CN`, `SG`, `MY`, `ID`, `TH`, `VN`, `PH`, `PK`, `BD`, `LK`, `NP`, `AE`, `SA`, `QA`, `KW`, `OM`, `BH`, `ZA`, `NG`, `KE`, `EG`, `RU`, `NL`, `BE`, `CH`, `AT`, `SE`, `NO`, `DK`, `FI`, `IE`, `PT`, `GR`, `TR`, `PL`, `NZ`, `AR`, `CL`, `CO`, `PE`, `IL`

Add any other dial-code countries you enable later. Prefer allowlist-only over “allow all” to reduce SMS abuse risk. The script above uses this full list.

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
