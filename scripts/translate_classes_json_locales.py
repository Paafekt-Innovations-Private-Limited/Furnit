#!/usr/bin/env python3
"""
Translate Furnit/en.lproj/classes.json into each locale's classes.json.

Uses deep_translator GoogleTranslator with:
- small batches (avoids huge GET URLs)
- per-batch timeout (translate_batch can hang on some networks)

  pip install deep-translator
  python3 scripts/translate_classes_json_locales.py
"""
from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path

try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("Install: pip install deep-translator", file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
FURNIT = REPO / "Furnit"
SOURCE = FURNIT / "en.lproj" / "classes.json"

LOCALES: list[tuple[str, str]] = [
    ("ar", "ar"),
    ("bn", "bn"),
    ("de", "de"),
    ("es", "es"),
    ("es-MX", "es"),
    ("fr", "fr"),
    ("hi", "hi"),
    ("kn", "kn"),
    ("ml", "ml"),
    ("ta", "ta"),
    ("te", "te"),
    ("zh-Hans", "zh-CN"),
    ("zh-Hant", "zh-TW"),
]

BATCH_SIZE = 5
SLEEP_SEC = 0.1
BATCH_TIMEOUT_SEC = 45


def load_ordered(path: Path) -> tuple[list[str], list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    keys = sorted(raw.keys(), key=lambda k: int(k))
    values = [raw[k] for k in keys]
    return keys, values


def translate_batch_safe(translator: GoogleTranslator, chunk: list[str]) -> list[str]:
    """Run translate_batch in a worker thread with a hard timeout."""

    def _run() -> list[str]:
        return translator.translate_batch(chunk)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_run)
        return list(fut.result(timeout=BATCH_TIMEOUT_SEC))


def translate_all_values(values: list[str], target: str) -> list[str]:
    translator = GoogleTranslator(source="en", target=target)
    out: list[str] = []
    i = 0
    n = len(values)
    done_batches = 0
    while i < n:
        batch_len = min(BATCH_SIZE, n - i)
        while batch_len >= 1:
            chunk = values[i : i + batch_len]
            try:
                translated = translate_batch_safe(translator, chunk)
                if len(translated) != len(chunk):
                    raise ValueError("batch length mismatch")
                out.extend(translated)
                i += batch_len
                done_batches += 1
                if done_batches % 100 == 0:
                    print(f"    ... {i}/{n}", flush=True)
                time.sleep(SLEEP_SEC)
                break
            except Exception:
                if batch_len > 1:
                    batch_len = max(1, batch_len // 2)
                    continue
                try:
                    out.append(translator.translate(values[i]))
                except Exception as e:
                    print(f"  WARN [{i}]: {e!s}; keeping English", file=sys.stderr)
                    out.append(values[i])
                i += 1
                time.sleep(SLEEP_SEC)
                break
    return out


def main() -> None:
    if not SOURCE.is_file():
        print(f"Missing {SOURCE}", file=sys.stderr)
        sys.exit(1)
    keys, values = load_ordered(SOURCE)
    print(f"Loaded {len(keys)} labels from {SOURCE.relative_to(REPO)}")
    for folder, gcode in LOCALES:
        dest_dir = FURNIT / f"{folder}.lproj"
        dest = dest_dir / "classes.json"
        if not dest_dir.is_dir():
            print(f"Skip (no dir): {dest_dir}", file=sys.stderr)
            continue
        print(f"Translating -> {folder} ({gcode}) ...", flush=True)
        translated = translate_all_values(values, gcode)
        data = {k: v for k, v in zip(keys, translated)}
        dest.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {dest.relative_to(REPO)}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
