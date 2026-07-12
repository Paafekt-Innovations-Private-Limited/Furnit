#!/usr/bin/env python3
"""Sync FAQ strings from iOS .lproj files into Android values-*/strings.xml."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FURNIT = REPO / "Furnit"
ANDROID_RES = REPO / "android" / "app" / "src" / "main" / "res"

IOS_TO_ANDROID_DIR = {
    "kn": "values-kn",
    "hi": "values-hi",
    "de": "values-de",
    "fr": "values-fr",
    "es": "values-es",
    "es-MX": "values-es-rMX",
    "zh-Hans": "values-zh-rCN",
    "zh-Hant": "values-zh-rTW",
    "ar": "values-ar",
    "bn": "values-bn",
    "ml": "values-ml",
    "ta": "values-ta",
    "te": "values-te",
}


def camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def ios_key_to_android(ios_key: str) -> str:
    return "faq_" + camel_to_snake(ios_key.split(".", 1)[1])


def parse_ios_faq(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for match in re.finditer(r'^"(faq\.[^"]+)" = "(.*)";$', text, flags=re.MULTILINE):
        key = match.group(1)
        value = (
            match.group(2)
            .replace("\\n", "\n")
            .replace('\\"', '"')
            .replace("\\\\", "\\")
        )
        found[key] = value
    return found


def escape_android_xml(value: str) -> str:
    value = value.replace("&", "&amp;")
    value = value.replace("<", "&lt;")
    value = value.replace(">", "&gt;")
    value = value.replace("'", "\\'")
    value = value.replace('"', '\\"')
    value = value.replace("\n", "\\n")
    return value


def apply_android_faq(path: Path, updates: dict[str, str]) -> tuple[int, list[str]]:
    text = path.read_text(encoding="utf-8")
    changed = 0
    missing: list[str] = []
    for android_key, value in sorted(updates.items()):
        escaped = escape_android_xml(value)
        pattern = rf'<string name="{re.escape(android_key)}">.*?</string>'

        def replacer(match: re.Match[str], inner: str = escaped) -> str:
            return f'<string name="{android_key}">{inner}</string>'

        new_text, count = re.subn(pattern, replacer, text, count=1, flags=re.DOTALL)
        if count:
            text = new_text
            changed += count
        else:
            missing.append(android_key)
    path.write_text(text, encoding="utf-8")
    return changed, missing


def main() -> None:
    en_ios = parse_ios_faq(FURNIT / "en.lproj" / "Localizable.strings")
    ios_keys = sorted(en_ios)
    total = 0
    for ios_locale, android_dir in IOS_TO_ANDROID_DIR.items():
        ios_path = FURNIT / f"{ios_locale}.lproj" / "Localizable.strings"
        android_path = ANDROID_RES / android_dir / "strings.xml"
        if not ios_path.exists():
            print(f"skip missing iOS {ios_path}")
            continue
        if not android_path.exists():
            print(f"skip missing Android {android_path}")
            continue
        ios_faq = parse_ios_faq(ios_path)
        updates = {
            ios_key_to_android(key): ios_faq[key]
            for key in ios_keys
            if key in ios_faq
        }
        changed, missing = apply_android_faq(android_path, updates)
        print(f"{android_dir}: {changed} updated, {len(missing)} missing")
        if missing:
            print(f"  missing: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")
        total += changed
    print(f"total: {total}")


if __name__ == "__main__":
    main()
