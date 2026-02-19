#!/usr/bin/env python3
"""
Pass 1 migration: 14 simple regex transforms for ZSAMPLE ABAP code.

Reads from data/sample_pkg_original/, writes to data/sample_pkg_migrated/.
Applies 14 ordered regex rules from doc section 11.3. Counts transforms
per rule and prints summary.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = SCRIPT_DIR.parent
DATA_DIR = TOOLKIT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "sample_pkg_original"
TARGET_DIR = DATA_DIR / "sample_pkg_migrated"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from abap_syntax_modernizer import AbapSyntaxModernizer


def migrate_directory(
    source_dir: Path, target_dir: Path, copy_unchanged: bool = True
) -> dict:
    """
    Migrate all ABAP files from source to target using Pass 1 rules.

    Args:
        source_dir: Input directory.
        target_dir: Output directory.
        copy_unchanged: If True, copy files with no transforms to target.

    Returns:
        Summary dict with files_migrated, total_transforms, rule_counts.
    """
    modernizer = AbapSyntaxModernizer()
    abap_extensions = (".clas.abap", ".intf.abap", ".fugr.abap", ".prog.abap", ".incl.abap")
    files = [
        f
        for f in source_dir.rglob("*")
        if f.is_file() and f.name.endswith(".abap")
    ]
    total_counts: dict[str, int] = {}
    files_migrated = 0
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        rel = file_path.relative_to(source_dir)
        out_path = target_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("Could not read %s: %s", file_path, e)
            continue

        transformed, counts = modernizer.transform_text(content)
        total = sum(counts.values())

        if total > 0:
            try:
                out_path.write_text(transformed, encoding="utf-8")
                files_migrated += 1
                for k, v in counts.items():
                    total_counts[k] = total_counts.get(k, 0) + v
            except OSError as e:
                logger.warning("Could not write %s: %s", out_path, e)
        elif copy_unchanged:
            try:
                shutil.copy2(file_path, out_path)
            except OSError as e:
                logger.warning("Could not copy %s: %s", file_path, e)

    return {
        "files_migrated": files_migrated,
        "files_total": len(files),
        "total_transforms": sum(total_counts.values()),
        "rule_counts": total_counts,
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pass 1 migration: 14 simple regex transforms for ZSAMPLE ABAP."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=SOURCE_DIR,
        help="Input directory (default: data/sample_pkg_original)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=TARGET_DIR,
        help="Output directory (default: data/sample_pkg_migrated)",
    )
    parser.add_argument(
        "--no-copy-unchanged",
        action="store_true",
        help="Do not copy files with no transforms",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    source_dir = args.input.resolve()
    target_dir = args.output.resolve()

    if not source_dir.exists():
        logger.error("Source directory does not exist: %s", source_dir)
        return 1

    logger.info("Pass 1 migration: %s -> %s", source_dir, target_dir)
    summary = migrate_directory(
        source_dir, target_dir, copy_unchanged=not args.no_copy_unchanged
    )

    # Write transform summary for check_coverage.py
    summary_path = DATA_DIR / "migrate_transform_summary.json"
    try:
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass

    print("\n=== Pass 1 Migration Summary ===")
    print(f"Files processed: {summary['files_total']}")
    print(f"Files with transforms: {summary['files_migrated']}")
    print(f"Total transforms: {summary['total_transforms']}")
    print("\nTransforms per rule:")
    for rule, count in sorted(
        summary["rule_counts"].items(), key=lambda x: -x[1]
    ):
        if count > 0:
            print(f"  {rule}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
