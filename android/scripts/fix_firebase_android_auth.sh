#!/usr/bin/env bash
#
# Fixes Firebase Phone Auth "missing a valid app identifier" for the ANDROID app only.
# Adds the app's signing SHA fingerprints, enables Play Integrity, and refreshes
# android/app/google-services.json. Does NOT touch the iOS app.
#
# Run this in your own interactive Terminal (so gcloud can prompt for reauth):
#   bash android/scripts/fix_firebase_android_auth.sh
#
set -euo pipefail

PROJECT="paafektprod"
APP_ID="1:613415224058:android:8d0a97fe4990e559a13f43"
GS_JSON="$(cd "$(dirname "$0")/.." && pwd)/app/google-services.json"
API="https://firebase.googleapis.com/v1beta1/projects/${PROJECT}/androidApps/${APP_ID}"

# --- Fingerprints (colons stripped by the script). ---
SHAS=(
  "EB:8B:FC:A4:45:FD:F8:33:4C:EE:9C:A3:67:CA:98:E4:E6:84:B6:93"                                              # debug SHA-1
  "B7:B1:05:B3:A0:B6:92:FD:F2:D2:67:7E:75:B6:DE:2C:29:63:5F:87:7B:7A:0A:18:2E:F6:31:36:39:35:6B:D0"          # debug SHA-256
  "E5:1A:A3:F3:B8:01:15:9E:AD:5A:86:C8:CE:F8:F5:1E:21:5B:2B:7A"                                              # release upload SHA-1
  "5F:0D:73:33:18:7F:86:1A:5A:33:A6:02:0B:C2:24:78:53:1C:25:CF:13:68:23:87:ED:C2:F3:3B:C8:E0:BE:19"          # release upload SHA-256
  "6C:62:DC:4B:C6:53:CB:4B:0C:23:A6:F8:52:E7:FC:C8:D0:7E:70:20"                                              # Play app-signing SHA-1
  "DE:AD:21:9D:82:71:DA:FC:34:1E:0B:30:AB:02:C6:08:86:45:01:07:50:2E:53:C6:10:76:3E:34:56:9A:59:BF"          # Play app-signing SHA-256
)

echo "==> Ensuring gcloud is authenticated on project ${PROJECT}"
gcloud config set project "${PROJECT}" >/dev/null
TOKEN="$(gcloud auth print-access-token)"

# User (ADC) credentials must name a quota/billing project for firebase.googleapis.com.
QUOTA_HEADER="X-Goog-User-Project: ${PROJECT}"

echo "==> Enabling required APIs (Firebase Management + Play Integrity)"
gcloud services enable firebase.googleapis.com playintegrity.googleapis.com --project "${PROJECT}"

echo "==> Adding SHA fingerprints"
for RAW in "${SHAS[@]}"; do
  HASH="$(echo "$RAW" | tr -d ':[:space:]' | tr 'A-F' 'a-f')"
  [ -z "$HASH" ] && continue
  case "${#HASH}" in
    40) CERT_TYPE="SHA_1" ;;
    64) CERT_TYPE="SHA_256" ;;
    *) echo "   skip (unexpected length ${#HASH}): $RAW"; continue ;;
  esac
  echo "   + ${CERT_TYPE} ${HASH}"
  RESP="$(curl -s -X POST "${API}/sha" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "${QUOTA_HEADER}" \
    -H "Content-Type: application/json" \
    -d "{\"shaHash\":\"${HASH}\",\"certType\":\"${CERT_TYPE}\"}")"
  if echo "$RESP" | grep -q '"error"'; then
    if echo "$RESP" | grep -qi "already exists"; then
      echo "     (already registered)"
    else
      echo "     WARNING: $RESP"
    fi
  fi
done

echo "==> Current SHA certificates on the app:"
curl -s -H "Authorization: Bearer ${TOKEN}" -H "${QUOTA_HEADER}" "${API}/sha" \
  | python3 -c 'import sys,json;[print("   ",c.get("certType"),c.get("shaHash")) for c in json.load(sys.stdin).get("certificates",[])]' || true

echo "==> Downloading refreshed google-services.json -> ${GS_JSON}"
curl -s -H "Authorization: Bearer ${TOKEN}" -H "${QUOTA_HEADER}" "${API}/config" \
  | python3 -c 'import sys,json,base64;d=json.load(sys.stdin);open("'"${GS_JSON}"'","wb").write(base64.b64decode(d["configFileContents"]));print("   wrote",d.get("configFilename"))'

echo "==> Done. Rebuild the app so it picks up the new config."
