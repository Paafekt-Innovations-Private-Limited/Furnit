#!/usr/bin/env python3
"""
Merge Furnit/Localization/classes_batches/<lang>/<NNN>.json fragments into
Furnit/<lang>.lproj/classes.json.

- English source: Furnit/en.lproj/classes.json (defines full key set).
- Each fragment is a JSON object mapping string keys to translated labels.
- Later fragments override earlier ones for the same key.
- Any key missing from all fragments keeps the English value from en.

Run from repo root:
  python3 scripts/merge_classes_i18n_batches.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FURNIT = REPO / "Furnit"
EN_PATH = FURNIT / "en.lproj" / "classes.json"
BATCH_ROOT = FURNIT / "Localization" / "classes_batches"

LOCALES = [
    "ar",
    "bn",
    "de",
    "es",
    "es-MX",
    "fr",
    "hi",
    "kn",
    "ml",
    "ta",
    "te",
    "zh-Hans",
    "zh-Hant",
]


def main() -> None:
    if not EN_PATH.is_file():
        print(f"Missing {EN_PATH}", file=sys.stderr)
        sys.exit(1)
    en_flat = json.loads(EN_PATH.read_text(encoding="utf-8"))
    for lang in LOCALES:
        lang_dir = BATCH_ROOT / lang
        merged: dict[str, str] = dict(en_flat)
        if lang_dir.is_dir():
            parts = sorted(lang_dir.glob("*.json"), key=lambda p: p.name)
            for p in parts:
                try:
                    frag = json.loads(p.read_text(encoding="utf-8"))
                except json.JSONDecodeError as e:
                    print(f"Skip bad JSON {p}: {e}", file=sys.stderr)
                    continue
                if not isinstance(frag, dict):
                    continue
                for k, v in frag.items():
                    if isinstance(v, str):
                        merged[str(k)] = v
        out = FURNIT / f"{lang}.lproj" / "classes.json"
        out.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        n = len(merged)
        translated = sum(
            1 for k in merged if k in en_flat and merged[k] != en_flat[k]
        )
        print(f"{lang}: wrote {n} keys ({translated} differ from English)")


if __name__ == "__main__":
    main()
