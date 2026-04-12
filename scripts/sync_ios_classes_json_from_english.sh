#!/usr/bin/env bash
# Overwrites every Furnit/*.lproj/classes.json from en.lproj/classes.json.
# Use when English keys changed and localized files should be refreshed before re-translation.
# Do NOT run blindly if you maintain translated strings in other .lproj copies.
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
