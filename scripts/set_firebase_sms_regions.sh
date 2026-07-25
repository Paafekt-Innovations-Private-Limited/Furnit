#!/usr/bin/env bash
# Configure Firebase Phone Auth SMS regions for Paafekt.
#
# Firebase CLI has no first-class "sms regions" command. This script PATCHes
# Identity Toolkit Admin API (same API the Console uses).
#
# Prerequisites (one of):
#   1) gcloud:  brew install --cask google-cloud-sdk && gcloud auth login
#   2) Or set:  export ACCESS_TOKEN="$(gcloud auth print-access-token)"
#
# Usage (from repo root):
#   ./scripts/set_firebase_sms_regions.sh              # global (allow by default)
#   ./scripts/set_firebase_sms_regions.sh --allowlist  # limited launch countries
#   ./scripts/set_firebase_sms_regions.sh --dry-run
#   PROJECT_ID=paafektprod ./scripts/set_firebase_sms_regions.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-paafektprod}"
DRY_RUN=0
MODE="global"

for arg in "${@:-}"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --allowlist) MODE="allowlist" ;;
    --global) MODE="global" ;;
    "") ;;
    *)
      echo "usage: $0 [--global|--allowlist] [--dry-run]" >&2
      exit 1
      ;;
  esac
done

# Region codes matching Furnit/Authentication/LoginView.swift country picker
# (used only with --allowlist).
REGIONS=(
  IN US GB CA AU DE FR IT ES BR MX JP KR CN SG MY ID TH VN PH
  PK BD LK NP AE SA QA KW OM BH ZA NG KE EG RU NL BE CH AT SE
  NO DK FI IE PT GR TR PL NZ AR CL CO PE IL
)

if [[ ! " ${REGIONS[*]} " =~ " FR " ]]; then
  REGIONS+=(FR)
fi

if [[ "${MODE}" == "global" ]]; then
  BODY='{"sms_region_config":{"allow_by_default":{"disallowed_regions":[]}}}'
  echo "Project: ${PROJECT_ID}"
  echo "Mode: global (allow SMS to all regions by default)"
else
  json_regions=$(printf '"%s",' "${REGIONS[@]}")
  json_regions="[${json_regions%,}]"
  BODY=$(cat <<EOF
{"sms_region_config":{"allowlist_only":{"allowed_regions":${json_regions}}}}
EOF
)
  echo "Project: ${PROJECT_ID}"
  echo "Mode: allowlist (${#REGIONS[@]} regions): ${REGIONS[*]}"
  echo "Includes France (FR): yes"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo
  echo "Dry run — request body:"
  echo "${BODY}"
  exit 0
fi

if [[ -z "${ACCESS_TOKEN:-}" ]]; then
  if command -v gcloud >/dev/null 2>&1; then
    ACCESS_TOKEN="$(gcloud auth print-access-token --account=support@paafekt.com --project="${PROJECT_ID}" 2>/dev/null \
      || gcloud auth print-access-token --project="${PROJECT_ID}")"
  else
    echo "error: no ACCESS_TOKEN and gcloud not found." >&2
    echo "Install: brew install --cask google-cloud-sdk" >&2
    echo "Then:    gcloud auth login && gcloud config set project ${PROJECT_ID}" >&2
    exit 1
  fi
fi

URL="https://identitytoolkit.googleapis.com/admin/v2/projects/${PROJECT_ID}/config?updateMask=sms_region_config"

HTTP_BODY_FILE="$(mktemp)"
HTTP_CODE=$(curl -sS -o "${HTTP_BODY_FILE}" -w "%{http_code}" -X PATCH \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -H "x-goog-user-project: ${PROJECT_ID}" \
  -d "${BODY}" \
  "${URL}")

echo "HTTP ${HTTP_CODE}"
# Print only smsRegionConfig from the response to avoid dumping unrelated secrets.
python3 - <<'PY' "${HTTP_BODY_FILE}" 2>/dev/null || cat "${HTTP_BODY_FILE}"
import json,sys
with open(sys.argv[1]) as f:
    d=json.load(f)
print(json.dumps({"smsRegionConfig": d.get("smsRegionConfig")}, indent=2))
PY
echo
rm -f "${HTTP_BODY_FILE}"

if [[ "${HTTP_CODE}" != "200" ]]; then
  echo "error: SMS region update failed (HTTP ${HTTP_CODE})." >&2
  echo "If Google asks for more business details, use docs/firebase-sms-regions.md." >&2
  exit 1
fi

echo "OK — SMS region policy updated for ${PROJECT_ID} (${MODE})."
