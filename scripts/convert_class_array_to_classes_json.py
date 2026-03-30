#!/usr/bin/env python3
"""
Convert a JSON array of {\"id\": int, \"name\": str} into Furnit ``classes.json`` shape:
{\"0\": \"...\", \"1\": \"...\", ...} with contiguous ids 0..N-1.

Usage:
  python3 scripts/convert_class_array_to_classes_json.py my_list.json -o classes_out.json
  pbpaste | python3 scripts/convert_class_array_to_classes_json.py -o Furnit/Views/FurnitureFit/classes.json

Then copy the output to both:
  Furnit/Views/FurnitureFit/classes.json
  android/app/src/main/assets/classes.json
if you want iOS and Android in sync.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def array_to_classes_object(rows: list) -> dict[str, str]:
    if not isinstance(rows, list):
        raise ValueError("Top-level JSON must be an array")
    out: dict[str, str] = {}
    for i, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not an object")
        if "id" not in item or "name" not in item:
            raise ValueError(f"Item {i} must have \"id\" and \"name\"")
        key = str(int(item["id"]))
        if key in out:
            raise ValueError(f"Duplicate id: {key}")
        out[key] = str(item["name"])
    if not out:
        raise ValueError("Empty array")
    n = len(out)
    max_id = max(int(k) for k in out)
    if max_id != n - 1:
        raise ValueError(f"Ids must be contiguous 0..{n - 1}; max id is {max_id}, count is {n}")
    for i in range(n):
        if str(i) not in out:
            raise ValueError(f"Missing id {i}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert [{id,name},...] to classes.json object.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="Input .json file (default: read stdin)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write here (default: stdout)",
    )
    args = parser.parse_args()

    if args.input is not None:
        text = args.input.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    rows = json.loads(text)
    obj = array_to_classes_object(rows)
    out_s = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"

    if args.output is not None:
        args.output.write_text(out_s, encoding="utf-8")
    else:
        sys.stdout.write(out_s)


if __name__ == "__main__":
    main()
