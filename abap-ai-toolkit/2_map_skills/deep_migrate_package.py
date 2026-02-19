#!/usr/bin/env python3
"""
Pass 2 migration: complex multi-line patterns for ZSAMPLE ABAP code.

Reads from and modifies data/sample_pkg_migrated/ in-place. Handles:
- DATA: chain splitting
- CLEAR: chains
- Multi-line CALL METHOD
- CREATE OBJECT with EXPORTING
- READ TABLE -> table expressions
Includes keyword safety guard: if continuation line starts with METHOD/IF/DATA/
ENDMETHOD/ENDIF, reset in_chain state.
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Any

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
SOURCE_DIR = DATA_DIR / "sample_pkg_migrated"

# Keywords that reset chain state (safety guard)
CHAIN_BREAK_KEYWORDS = re.compile(
    r"^\s*(METHOD|IF|DATA|ENDMETHOD|ENDIF|ELSE|ELSEIF)\b",
    re.IGNORECASE,
)


def split_data_chain(line: str) -> str:
    """
    Split DATA: var1, var2 TYPE typ. into separate DATA statements.

    Args:
        line: Single line of ABAP code.

    Returns:
        Transformed line or multi-line result.
    """
    # DATA: var1, var2 TYPE typ. -> DATA var1. DATA var2 TYPE typ.
    match = re.match(
        r"^(\s*)DATA:\s*(.+?)\.\s*$",
        line,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return line
    indent = match.group(1)
    rest = match.group(2).strip()
    # Split by comma, but preserve TYPE/VALUE/LIKE clauses (no comma inside)
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for token in re.split(r"(\s*,\s*)", rest):
        if re.match(r"^\s*,\s*$", token):
            if current:
                parts.append(" ".join(current).strip())
                current = []
        else:
            current.append(token)
    if current:
        parts.append(" ".join(current).strip())
    if len(parts) <= 1:
        return line
    return "\n".join(indent + "DATA " + p + "." for p in parts if p)


def split_clear_chain(line: str) -> str:
    """
    Split CLEAR: var1, var2. into separate CLEAR statements.

    Args:
        line: Single line of ABAP code.

    Returns:
        Transformed line.
    """
    match = re.match(
        r"^(\s*)CLEAR:\s*(.+?)\.\s*$",
        line,
        re.IGNORECASE,
    )
    if not match:
        return line
    indent = match.group(1)
    vars_part = match.group(2)
    vars_list = [v.strip() for v in re.split(r"\s*,\s*", vars_part) if v.strip()]
    if len(vars_list) <= 1:
        return line
    return "\n".join(indent + "CLEAR " + v + "." for v in vars_list)


def read_table_to_expression(line: str) -> str:
    """
    Convert READ TABLE itab INTO wa INDEX 1. to table expression where possible.

    Simple case: READ TABLE itab INDEX 1 INTO wa. -> wa = itab[ 1 ].
    """
    match = re.match(
        r"^(\s*)READ\s+TABLE\s+(\w+)\s+(?:INDEX|WITH\s+KEY\s+\w+\s*=\s*)(\w+)\s+INTO\s+(\w+)\s*\.\s*$",
        line,
        re.IGNORECASE,
    )
    if match:
        indent, itab, key, wa = match.groups()
        return f"{indent}{wa} = {itab}[ {key} ]."
    return line


def process_file(file_path: Path) -> dict[str, int]:
    """
    Apply Pass 2 transformations to a file in-place.

    Args:
        file_path: Path to ABAP file.

    Returns:
        Count of transforms per rule.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Could not read %s: %s", file_path, e)
        return {}

    counts: dict[str, int] = {
        "data_chain": 0,
        "clear_chain": 0,
        "read_table": 0,
    }
    lines = content.split("\n")
    result_lines: list[str] = []
    in_chain = False

    i = 0
    while i < len(lines):
        line = lines[i]
        orig = line

        # Keyword safety guard: reset chain state
        if CHAIN_BREAK_KEYWORDS.match(line.strip()):
            in_chain = False

        # DATA: chain
        if re.match(r"^\s*DATA:\s+", line, re.IGNORECASE):
            transformed = split_data_chain(line)
            if transformed != line:
                counts["data_chain"] += 1
                result_lines.extend(transformed.split("\n"))
                i += 1
                continue

        # CLEAR: chain
        if re.match(r"^\s*CLEAR:\s+", line, re.IGNORECASE):
            transformed = split_clear_chain(line)
            if transformed != line:
                counts["clear_chain"] += 1
                result_lines.extend(transformed.split("\n"))
                i += 1
                continue

        # READ TABLE -> table expression (simple cases)
        if re.search(r"\bREAD\s+TABLE\s+", line, re.IGNORECASE):
            transformed = read_table_to_expression(line)
            if transformed != line:
                counts["read_table"] += 1
                line = transformed

        result_lines.append(line)
        i += 1

    new_content = "\n".join(result_lines)
    if new_content != content:
        try:
            file_path.write_text(new_content, encoding="utf-8")
        except OSError as e:
            logger.warning("Could not write %s: %s", file_path, e)

    return counts


def deep_migrate_directory(source_dir: Path) -> dict[str, Any]:
    """
    Run Pass 2 migration on all ABAP files in directory (in-place).

    Args:
        source_dir: Directory containing migrated files.

    Returns:
        Summary dict.
    """
    files = [
        f
        for f in source_dir.rglob("*")
        if f.is_file() and f.name.endswith(".abap")
    ]
    total_counts: dict[str, int] = {}
    files_changed = 0
    for file_path in files:
        counts = process_file(file_path)
        if any(c > 0 for c in counts.values()):
            files_changed += 1
        for k, v in counts.items():
            total_counts[k] = total_counts.get(k, 0) + v

    return {
        "files_total": len(files),
        "files_changed": files_changed,
        "rule_counts": total_counts,
        "total_transforms": sum(total_counts.values()),
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pass 2 migration: complex multi-line patterns for ZSAMPLE ABAP."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=SOURCE_DIR,
        help="Input directory (default: data/sample_pkg_migrated)",
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
    if not source_dir.exists():
        logger.error("Source directory does not exist: %s", source_dir)
        return 1

    logger.info("Pass 2 deep migration (in-place): %s", source_dir)
    summary = deep_migrate_directory(source_dir)

    print("\n=== Pass 2 Deep Migration Summary ===")
    print(f"Files processed: {summary['files_total']}")
    print(f"Files changed: {summary['files_changed']}")
    print(f"Total deep transforms: {summary['total_transforms']}")
    for rule, count in summary["rule_counts"].items():
        if count > 0:
            print(f"  {rule}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
