#!/usr/bin/env python3
"""
Convert .pages files to PDF using macOS Pages (AppleScript).
Requires Pages app installed. Run: python3 docs/pages_to_pdf.py <file1.pages> [file2.pages ...]
"""
import subprocess
import sys
from pathlib import Path


def pages_to_pdf(pages_path: Path) -> Path:
    """Convert one .pages file to PDF. Returns path to the created .pdf."""
    pages_path = pages_path.resolve()
    if not pages_path.exists():
        raise FileNotFoundError(pages_path)
    if pages_path.suffix.lower() != ".pages":
        raise ValueError(f"Not a .pages file: {pages_path}")

    out_path = pages_path.with_suffix(".pdf")
    pages_posix = str(pages_path).replace("\\", "\\\\")
    out_posix = str(out_path).replace("\\", "\\\\")

    script = f'''
tell application "Pages"
    set theDoc to open POSIX file "{pages_posix}"
    export theDoc to POSIX file "{out_posix}" as PDF
    close theDoc saving no
end tell
'''
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"AppleScript failed: {result.stderr or result.stdout}")
    return out_path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 pages_to_pdf.py <file1.pages> [file2.pages ...]", file=sys.stderr)
        sys.exit(1)

    for arg in sys.argv[1:]:
        path = Path(arg).expanduser()
        try:
            out = pages_to_pdf(path)
            print(f"Converted: {path.name} -> {out}")
        except Exception as e:
            print(f"Error converting {path}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
