#!/usr/bin/env python3
"""
Update all 'DIN: <number>' occurrences in a .docx to Jayamma's DIN 11594139.

This edits only the text runs (not images) so existing signature lines keep their styling.

Example:
  python docs/update_din_in_docx.py \\
      --input "/Users/al/Downloads/NDA - Paafekt v02_0903 (1).docx"
      --output "/Users/al/Downloads/NDA - Paafekt v02_0903 (1) - DIN11594139.docx"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TARGET_DIN = "11594139"
DIN_PATTERN = re.compile(r"(DIN:\s*)\d+")


def update_din(input_path: Path, output_path: Path) -> None:
    from docx import Document

    if not input_path.exists():
        raise FileNotFoundError(f"Input .docx not found: {input_path}")

    doc = Document(str(input_path))

    def patch_runs(container) -> None:
        for p in container.paragraphs:
            for run in p.runs:
                if "DIN:" in run.text:
                    new_text = DIN_PATTERN.sub(rf"\1{TARGET_DIN}", run.text)
                    run.text = new_text

    # Body
    patch_runs(doc)

    # Headers and footers for each section
    for section in doc.sections:
        patch_runs(section.header)
        patch_runs(section.footer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace all 'DIN: <number>' text in a .docx with DIN: 11594139.")
    parser.add_argument("--input", required=True, help="Path to input .docx")
    parser.add_argument(
        "--output",
        help="Path to output .docx (default: '<name> - DIN11594139.docx' next to input)",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_path = in_path.with_name(in_path.stem + " - DIN11594139.docx")

    update_din(in_path, out_path)
    print(f"Wrote updated DIN .docx to: {out_path}")


if __name__ == "__main__":
    main()

