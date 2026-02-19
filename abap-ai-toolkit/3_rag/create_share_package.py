#!/usr/bin/env python3
"""
Create shareable ZIP package for the ABAP AI Toolkit.

Packages: scripts, config, rag_export.json, .mdc rules, migrated code.
Excludes: .venv, vector_db, .env, large binaries.
Outputs to data/exports/abap-ai-toolkit-share.zip
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = SCRIPT_DIR.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPORT_DIR = TOOLKIT_ROOT / "data" / "exports"
OUTPUT_ZIP = EXPORT_DIR / "abap-ai-toolkit-share.zip"

EXCLUDE_DIRS = {".venv", "vector_db", "__pycache__", ".git", "node_modules"}
EXCLUDE_FILES = {".env", ".env.local", ".DS_Store"}
EXCLUDE_EXTENSIONS = {".pyc", ".pyo", ".so", ".dll", ".exe", ".bin"}


def should_include(path: Path, base: Path) -> bool:
    """Determine if path should be included in ZIP."""
    rel = path.relative_to(base)
    parts = rel.parts
    if any(d in parts for d in EXCLUDE_DIRS):
        return False
    if path.name in EXCLUDE_FILES:
        return False
    if path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return False
    return True


def collect_files() -> list[Path]:
    """Collect all files to include in the share package."""
    files = []
    for path in TOOLKIT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if not should_include(path, TOOLKIT_ROOT):
            continue
        rel = str(path.relative_to(TOOLKIT_ROOT))
        if "vector_db" in rel:
            continue
        if rel.startswith(".venv"):
            continue
        files.append(path)
    return files


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create shareable ZIP package."
    )
    parser.add_argument("-o", "--output", default=str(OUTPUT_ZIP), help="Output ZIP path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files()
    logger.info("Packaging %d files", len(files))

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            arcname = file_path.relative_to(TOOLKIT_ROOT)
            zf.write(file_path, arcname)
            logger.debug("Added %s", arcname)

    logger.info("Created %s", output_path)
    print(f"Share package: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
