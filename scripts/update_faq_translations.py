#!/usr/bin/env python3
"""Update FAQ localization keys across iOS .lproj files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FURNIT = REPO / "Furnit"
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

from faq_translations_data import TRANSLATIONS  # noqa: E402


def escape_strings_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def apply_updates(lproj: str, updates: dict[str, str]) -> int:
    path = FURNIT / f"{lproj}.lproj" / "Localizable.strings"
    if not path.exists():
        print(f"skip missing {path}")
        return 0
    text = path.read_text(encoding="utf-8")
    changed = 0
    for key, value in updates.items():
        escaped = escape_strings_value(value)
        pattern = rf'^"{re.escape(key)}" = ".*";$'
        replacement = f'"{key}" = "{escaped}";'
        new_text, n = re.subn(pattern, replacement, text, count=0, flags=re.MULTILINE)
        if n:
            text = new_text
            changed += n
        else:
            print(f"  missing key {key} in {lproj}")
    path.write_text(text, encoding="utf-8")
    return changed


def main() -> None:
    target_keys = list(next(iter(TRANSLATIONS.values())).keys())
    total = 0
    for locale, bundle in TRANSLATIONS.items():
        n = apply_updates(locale, {k: bundle[k] for k in target_keys})
        print(f"{locale}: {n} lines updated")
        total += n
    print(f"total: {total}")


if __name__ == "__main__":
    main()
