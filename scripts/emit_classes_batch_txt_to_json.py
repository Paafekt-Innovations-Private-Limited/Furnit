#!/usr/bin/env python3
"""
Convert Localization/classes_batches_source/batchN/<lang>.txt
(100 lines, line i = label for key (N*100 + i)) into
Localization/classes_batches/<lang>/<NNN>.json where NNN == N (zero-padded).

Batch 0: keys "0".."99"  -> 000.json
Batch 1: keys "100".."199" -> 001.json
...

Run from repo root:
  python3 scripts/emit_classes_batch_txt_to_json.py
  python3 scripts/emit_classes_batch_txt_to_json.py --batch 1

Or: CLASSES_BATCH=1 python3 scripts/emit_classes_batch_txt_to_json.py

For batch 0, every locale file must exist. For batch >= 1, missing files are skipped
with a warning so you can translate locales incrementally.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "Localization" / "classes_batches_source"
DEST_ROOT = REPO / "Localization" / "classes_batches"

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
    ap = argparse.ArgumentParser(description="Emit classes batch JSON from line files.")
    ap.add_argument(
        "--batch",
        type=int,
        default=int(os.environ.get("CLASSES_BATCH", "0")),
        help="Batch index N (keys N*100 .. N*100+99). Default 0 or CLASSES_BATCH.",
    )
    args = ap.parse_args()
    batch = args.batch
    if batch < 0:
        print("--batch must be >= 0", file=sys.stderr)
        sys.exit(1)
    key_start = batch * 100
    src_dir = SOURCE_ROOT / f"batch{batch}"
    out_name = f"{batch:03d}.json"

    if not src_dir.is_dir():
        print(f"Missing source directory {src_dir}", file=sys.stderr)
        sys.exit(1)

    any_written = False
    for lang in LOCALES:
        txt_path = src_dir / f"{lang}.txt"
        if not txt_path.is_file():
            if batch == 0:
                print(f"Missing {txt_path}", file=sys.stderr)
                sys.exit(1)
            print(f"Skip (no file): {txt_path.relative_to(REPO)}", file=sys.stderr)
            continue
        lines = txt_path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 100:
            print(
                f"{lang}: expected 100 lines in {txt_path}, got {len(lines)}",
                file=sys.stderr,
            )
            sys.exit(1)
        data = {str(key_start + i): lines[i] for i in range(100)}
        out_dir = DEST_ROOT / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_name
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_path.relative_to(REPO)}")
        any_written = True

    if not any_written:
        print("No locale files written.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
