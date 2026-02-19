#!/usr/bin/env python3
"""
3-way mapping with CRV verification.

Scans all 4 source directories (ZSAMPLE_GENERIC + /ZSAMPLE/MAIN, original + migrated).
Produces per-line 3-way mappings with CRV-verified FM status. Loads CRV data
from data/twin_discovery/crv_latest_full.json if available. Outputs 406 JSON
files to data/threeway_mappings/.
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
MAPPINGS_DIR = DATA_DIR / "threeway_mappings"
CRV_PATH = DATA_DIR / "twin_discovery" / "crv_latest_full.json"

SOURCE_DIRS = [
    DATA_DIR / "zsample_original",
    DATA_DIR / "zsample_migrated",
    DATA_DIR / "sample_pkg_original",
    DATA_DIR / "sample_pkg_migrated",
]

CALL_FUNCTION_PATTERN = re.compile(
    r"\bCALL\s+FUNCTION\s+['\"]?([\w/]+)['\"]?",
    re.IGNORECASE,
)

# Skill rules for 3-way mapping (Java | Old ABAP | New ABAP)
THREEWAY_RULES = [
    {"old": r"MOVE\s+.+?\s+TO\s+", "new": "assign", "java": "="},
    {"old": r"ADD\s+.+?\s+TO\s+", "new": "arithmetic", "java": "+="},
    {"old": r"CONCATENATE\s+", "new": "string template", "java": "+"},
    {"old": r"MOVE-CORRESPONDING\s+", "new": "CORRESPONDING #", "java": "copyProperties"},
    {"old": r"CALL\s+METHOD\s+\w+->", "new": "functional call", "java": "->"},
    {"old": r"CREATE\s+OBJECT\s+\w+\s+TYPE\s+", "new": "NEW", "java": "new"},
    {"old": r"SUBTRACT\s+.+?\s+FROM\s+", "new": "arithmetic", "java": "-="},
    {"old": r"TRANSLATE\s+.+?\s+TO\s+UPPER", "new": "to_upper", "java": "toUpperCase"},
    {"old": r"TRANSLATE\s+.+?\s+TO\s+LOWER", "new": "to_lower", "java": "toLowerCase"},
    {"old": r"DESCRIBE\s+TABLE\s+\w+\s+LINES", "new": "lines()", "java": "size()"},
    {"old": r"CHECK\s+", "new": "IF/RETURN", "java": "if (!x) return"},
    {"old": r"READ\s+TABLE\s+", "new": "table expr", "java": "get"},
    {"old": r"CALL\s+FUNCTION\s+", "new": "FM call", "java": "service.call"},
]


def load_crv_data(crv_path: Path) -> dict[str, Any]:
    """Load CRV data from JSON if available."""
    if not crv_path.exists():
        logger.debug("CRV file not found: %s", crv_path)
        return {}
    try:
        return json.loads(crv_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load CRV: %s", e)
        return {}


def get_fm_crv_status(fm_name: str, crv_data: dict) -> str:
    """Get CRV release status for FM: released, notToBeReleased, or missing."""
    if not crv_data:
        return "missing"
    # CRV structure varies; common keys: objects, functionModules, etc.
    items = crv_data.get("objects", crv_data.get("functionModules", []))
    if isinstance(items, list):
        for obj in items:
            name = obj.get("name", obj.get("functionModule", ""))
            if name and name.upper() == fm_name.upper():
                return obj.get("releaseState", obj.get("status", "missing"))
    if isinstance(items, dict):
        fm_upper = fm_name.upper()
        if fm_upper in items:
            return items[fm_upper].get("releaseState", "missing")
    return "missing"


def scan_file_threeway(
    file_path: Path, content: str, crv_data: dict
) -> list[dict[str, Any]]:
    """Produce per-line 3-way mappings with CRV status for FMs."""
    mappings: list[dict[str, Any]] = []
    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        for rule in THREEWAY_RULES:
            if re.search(rule["old"], line, re.IGNORECASE):
                mapping = {
                    "line": line_num,
                    "old_abap": rule["old"],
                    "new_abap": rule["new"],
                    "java_anchor": rule["java"],
                    "snippet": line.strip()[:80],
                }
                # CRV for CALL FUNCTION
                fm_match = CALL_FUNCTION_PATTERN.search(line)
                if fm_match:
                    fm_name = fm_match.group(1)
                    mapping["fm_name"] = fm_name
                    mapping["crv_status"] = get_fm_crv_status(fm_name, crv_data)
                mappings.append(mapping)
                break

    return mappings


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="3-way mapping with CRV verification."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=MAPPINGS_DIR,
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--crv",
        type=Path,
        default=CRV_PATH,
        help="Path to CRV JSON file",
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

    crv_path = args.crv.resolve()
    crv_data = load_crv_data(crv_path)
    if crv_data:
        logger.info("Loaded CRV data from %s", crv_path)

    args.output.mkdir(parents=True, exist_ok=True)
    total_mappings = 0
    files_written = 0

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            continue
        files = [f for f in source_dir.rglob("*") if f.is_file() and f.name.endswith(".abap")]
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.warning("Could not read %s: %s", file_path, e)
                continue
            rel = file_path.relative_to(source_dir)
            mappings = scan_file_threeway(file_path, content, crv_data)
            if mappings:
                out_name = f"{source_dir.name}_{str(rel).replace('/', '_').replace(chr(92), '_')}.json"
                out_path = args.output / out_name
                try:
                    out_path.write_text(
                        json.dumps({
                            "source_file": str(rel),
                            "source_dir": source_dir.name,
                            "mappings": mappings,
                        }, indent=2),
                        encoding="utf-8",
                    )
                    files_written += 1
                    total_mappings += len(mappings)
                except OSError as e:
                    logger.warning("Could not write %s: %s", out_path, e)

    print("\n=== 3-Way Mapping Summary ===")
    print(f"Files written: {files_written}")
    print(f"Total mappings: {total_mappings}")
    print(f"CRV loaded: {bool(crv_data)}")
    print(f"Output: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
