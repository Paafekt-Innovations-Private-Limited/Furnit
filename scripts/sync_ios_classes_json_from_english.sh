#!/usr/bin/env bash
# Creates or overwrites Furnit/*.lproj/classes.json from en.lproj/classes.json.
# The current repository tracks only the English runtime map. Run this only to seed
# locale-specific class maps before translating and committing those files.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EN="$ROOT/Furnit/en.lproj/classes.json"
if [[ ! -f "$EN" ]]; then
  echo "Missing $EN" >&2
  exit 1
fi
for d in "$ROOT/Furnit"/*.lproj; do
  [[ -d "$d" ]] || continue
  base="$(basename "$d")"
  if [[ "$base" == "en.lproj" ]]; then
    continue
  fi
  cp "$EN" "$d/classes.json"
  echo "Updated $base/classes.json"
done
echo "Done. en.lproj left as source."
