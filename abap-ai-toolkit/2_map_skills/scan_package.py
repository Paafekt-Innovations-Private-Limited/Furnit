#!/usr/bin/env python3
"""
Syntax scan and Function Module classification for ZSAMPLE ABAP code.

Scans data/sample_pkg_original/ for obsolete ABAP patterns and classifies
CALL FUNCTION statements by extracting FM names. Outputs scan results as
JSON and summary to console.
"""

import argparse
import json
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
SOURCE_DIR = DATA_DIR / "sample_pkg_original"
OUTPUT_JSON = DATA_DIR / "sample_pkg_scan_results.json"

# Obsolete patterns to scan for
OBSOLETE_PATTERNS = [
    ("MOVE", r"\bMOVE\s+"),
    ("ADD", r"\bADD\s+"),
    ("SUBTRACT", r"\bSUBTRACT\s+"),
    ("CONCATENATE", r"\bCONCATENATE\s+"),
    ("CALL METHOD", r"\bCALL\s+METHOD\s+"),
    ("CREATE OBJECT", r"\bCREATE\s+OBJECT\s+"),
    ("TRANSLATE", r"\bTRANSLATE\s+"),
    ("DESCRIBE TABLE", r"\bDESCRIBE\s+TABLE\s+"),
    ("MOVE-CORRESPONDING", r"\bMOVE-CORRESPONDING\s+"),
    ("CHECK", r"^\s*CHECK\s+"),
    ("READ TABLE", r"\bREAD\s+TABLE\s+"),
    ("SELECT *", r"\bSELECT\s+\*\s+"),
]

# CALL FUNCTION extraction
CALL_FUNCTION_PATTERN = re.compile(
    r"\bCALL\s+FUNCTION\s+['\"]?([\w/]+)['\"]?",
    re.IGNORECASE,
)


def scan_file(file_path: Path) -> dict[str, Any]:
    """
    Scan a single ABAP file for obsolete patterns and FM calls.

    Args:
        file_path: Path to the ABAP file.

    Returns:
        Dict with patterns_found, fm_calls, line_details.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Could not read %s: %s", file_path, e)
        return {"patterns_found": {}, "fm_calls": [], "line_details": [], "error": str(e)}

    patterns_found: dict[str, int] = {}
    fm_calls: list[str] = []
    line_details: list[dict[str, Any]] = []

    for pattern_name, pattern_regex in OBSOLETE_PATTERNS:
        regex = re.compile(pattern_regex, re.MULTILINE | re.IGNORECASE)
        for match in regex.finditer(content):
            patterns_found[pattern_name] = patterns_found.get(pattern_name, 0) + 1
            line_num = content[: match.start()].count("\n") + 1
            line_details.append(
                {
                    "file": str(file_path.name),
                    "line": line_num,
                    "pattern": pattern_name,
                    "snippet": content.split("\n")[line_num - 1][:80].strip(),
                }
            )

    for match in CALL_FUNCTION_PATTERN.finditer(content):
        fm_name = match.group(1)
        if fm_name not in fm_calls:
            fm_calls.append(fm_name)
        line_num = content[: match.start()].count("\n") + 1
        line_details.append(
            {
                "file": str(file_path.name),
                "line": line_num,
                "pattern": "CALL FUNCTION",
                "fm_name": fm_name,
                "snippet": content.split("\n")[line_num - 1][:80].strip(),
            }
        )

    return {
        "patterns_found": patterns_found,
        "fm_calls": fm_calls,
        "line_details": line_details,
    }


def scan_directory(source_dir: Path) -> dict[str, Any]:
    """
    Scan all ABAP files in directory.

    Args:
        source_dir: Directory to scan.

    Returns:
        Aggregated scan results.
    """
    abap_files = list(source_dir.rglob("*")) if source_dir.exists() else []
    abap_files = [f for f in abap_files if f.is_file() and f.name.endswith(".abap")]

    all_patterns: dict[str, int] = {}
    all_fm_calls: list[str] = []
    per_file: dict[str, dict[str, Any]] = {}
    all_line_details: list[dict[str, Any]] = []

    for file_path in abap_files:
        rel = str(file_path.relative_to(source_dir))
        result = scan_file(file_path)
        per_file[rel] = {
            "patterns_found": result["patterns_found"],
            "fm_calls": result["fm_calls"],
        }
        for p, c in result["patterns_found"].items():
            all_patterns[p] = all_patterns.get(p, 0) + c
        for fm in result["fm_calls"]:
            if fm not in all_fm_calls:
                all_fm_calls.append(fm)
        all_line_details.extend(result.get("line_details", []))

    return {
        "source_dir": str(source_dir),
        "files_scanned": len(abap_files),
        "total_patterns": sum(all_patterns.values()),
        "patterns_by_type": all_patterns,
        "fm_calls": sorted(all_fm_calls),
        "fm_count": len(all_fm_calls),
        "per_file": per_file,
        "line_details": all_line_details[:500],  # Cap for JSON size
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Syntax scan and FM classification for ZSAMPLE ABAP code."
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
        default=OUTPUT_JSON,
        help="Output JSON path (default: data/sample_pkg_scan_results.json)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write JSON output file",
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

    logger.info("Scanning %s", source_dir)
    results = scan_directory(source_dir)

    # Console summary
    print("\n=== Scan Summary ===")
    print(f"Files scanned: {results['files_scanned']}")
    print(f"Total obsolete patterns: {results['total_patterns']}")
    print("\nPatterns by type:")
    for name, count in sorted(results["patterns_by_type"].items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    print(f"\nFunction modules found: {results['fm_count']}")
    for fm in results["fm_calls"][:30]:
        print(f"  - {fm}")
    if len(results["fm_calls"]) > 30:
        print(f"  ... and {len(results['fm_calls']) - 30} more")

    if not args.no_json:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info("Results written to %s", args.output)
        except OSError as e:
            logger.error("Failed to write JSON: %s", e)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
