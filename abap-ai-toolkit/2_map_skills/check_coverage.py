#!/usr/bin/env python3
"""
Coverage analysis for ABAP migration.

Analyzes how many patterns were transformed vs total found. Reports coverage
percentage per rule and overall.
"""

import argparse
import json
import logging
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
SCAN_RESULTS = DATA_DIR / "sample_pkg_scan_results.json"
MIGRATE_SUMMARY_PATH = DATA_DIR / "sample_pkg_migrated"  # No summary file by default
THREEWAY_DIR = DATA_DIR / "threeway_mappings"


def load_scan_results(path: Path) -> dict[str, int]:
    """Load pattern counts from scan results."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("patterns_by_type", {})
    except (json.JSONDecodeError, OSError):
        return {}


def infer_transform_counts() -> dict[str, int]:
    """
    Infer transform counts from migration output.

    Reads migrate_transform_summary.json written by migrate_sample_pkg.py.
    Falls back to threeway_mappings if that file does not exist.
    """
    transform_summary = DATA_DIR / "migrate_transform_summary.json"
    if transform_summary.exists():
        try:
            data = json.loads(transform_summary.read_text(encoding="utf-8"))
            return data.get("rule_counts", {})
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: check threeway_mappings for pattern evidence
    rule_counts: dict[str, int] = {}
    if THREEWAY_DIR.exists():
        for f in THREEWAY_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                for m in data.get("mappings", []):
                    rule = m.get("new_abap", m.get("old_abap", "unknown"))
                    rule_counts[rule] = rule_counts.get(rule, 0) + 1
            except (json.JSONDecodeError, OSError):
                continue
    return rule_counts


def compute_coverage(
    found: dict[str, int], transformed: dict[str, int]
) -> dict[str, Any]:
    """
    Compute coverage per rule and overall.

    Args:
        found: Pattern counts from scan (pattern_name -> count).
        transformed: Transform counts per rule (rule_name -> count).

    Returns:
        Coverage report.
    """
    # Map scan pattern names to transform rule names
    pattern_to_rule = {
        "MOVE": "move_to_assign",
        "ADD": "add_to_arithmetic",
        "SUBTRACT": "subtract_to_arithmetic",
        "CONCATENATE": "concatenate_to_template",
        "CALL METHOD": "call_method_to_functional",
        "CREATE OBJECT": "create_object_to_new",
        "TRANSLATE": "translate_upper",  # or translate_lower, combined
        "DESCRIBE TABLE": "describe_table_to_lines",
        "MOVE-CORRESPONDING": "move_corresponding",
        "CHECK": "check_to_if_return",
        "READ TABLE": "read_table",
        "SELECT *": "select_star",
    }

    per_rule: dict[str, dict] = {}
    total_found = 0
    total_transformed = 0

    for pattern_name, count in found.items():
        rule_name = pattern_to_rule.get(pattern_name, pattern_name.lower().replace(" ", "_"))
        transform_count = transformed.get(rule_name, 0)
        # TRANSLATE splits into upper/lower
        if pattern_name == "TRANSLATE":
            transform_count = (
                transformed.get("translate_upper", 0) + transformed.get("translate_lower", 0)
            )
        total_found += count
        total_transformed += transform_count
        pct = (transform_count / count * 100) if count > 0 else 0
        per_rule[pattern_name] = {
            "found": count,
            "transformed": transform_count,
            "coverage_pct": round(pct, 1),
        }

    overall_pct = (total_transformed / total_found * 100) if total_found > 0 else 0

    return {
        "per_rule": per_rule,
        "total_found": total_found,
        "total_transformed": total_transformed,
        "overall_coverage_pct": round(overall_pct, 1),
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Coverage analysis: transformed vs total patterns."
    )
    parser.add_argument(
        "-s",
        "--scan",
        type=Path,
        default=SCAN_RESULTS,
        help="Scan results JSON path",
    )
    parser.add_argument(
        "-t",
        "--transforms",
        type=Path,
        default=None,
        help="Transform summary JSON (optional)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON report path",
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

    found = load_scan_results(args.scan)
    if args.transforms and args.transforms.exists():
        try:
            data = json.loads(args.transforms.read_text(encoding="utf-8"))
            transformed = data.get("rule_counts", {})
        except (json.JSONDecodeError, OSError):
            transformed = {}
    else:
        transformed = infer_transform_counts()

    if not found:
        logger.warning("No scan results. Run scan_sample_pkg.py first.")
        found = {"MOVE": 0, "ADD": 0, "CONCATENATE": 0}  # Placeholder

    report = compute_coverage(found, transformed)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            logger.info("Report written to %s", args.output)
        except OSError as e:
            logger.error("Failed to write: %s", e)

    print("\n=== Coverage Analysis ===")
    print(f"Total patterns found: {report['total_found']}")
    print(f"Total transformed: {report['total_transformed']}")
    print(f"Overall coverage: {report['overall_coverage_pct']}%")
    print("\nPer rule:")
    for rule, data in sorted(report["per_rule"].items(), key=lambda x: -x[1]["found"]):
        if data["found"] > 0:
            print(f"  {rule}: {data['transformed']}/{data['found']} ({data['coverage_pct']}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
