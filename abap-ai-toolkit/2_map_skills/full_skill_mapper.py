#!/usr/bin/env python3
"""
Comprehensive skill mapping across all packages.

Runs java_abap_mapper logic across ALL packages (all 4 source directories).
Generates comprehensive report with per-package and aggregate statistics.
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
MAPPINGS_DIR = DATA_DIR / "threeway_mappings"

# Import mapper logic
import sys
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from java_abap_mapper import map_all_sources, SKILL_RULES


def generate_comprehensive_report() -> dict[str, Any]:
    """
    Run full skill mapping and produce comprehensive report.

    Returns:
        Report dict with per_package, aggregate, categories.
    """
    results = map_all_sources()
    per_package: dict[str, dict[str, Any]] = {}
    category_counts: dict[str, int] = {}

    for key, mappings in results["mappings"].items():
        pkg = key.split("/")[0] if "/" in key else "unknown"
        if pkg not in per_package:
            per_package[pkg] = {"files": 0, "mappings": 0, "rules_matched": set()}
        per_package[pkg]["files"] += 1
        per_package[pkg]["mappings"] += len(mappings)
        for m in mappings:
            cat = m.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1
            per_package[pkg]["rules_matched"].add(m.get("old_abap", ""))

    # Convert sets to counts for JSON
    report = {
        "per_package": {
            k: {
                "files": v["files"],
                "mappings": v["mappings"],
                "unique_rules": len(v["rules_matched"]),
            }
            for k, v in per_package.items()
        },
        "aggregate": {
            "total_files": sum(p["files"] for p in per_package.values()),
            "total_mappings": results["total_patterns"],
            "unique_rules_matched": results["unique_rules"],
            "skill_rules_available": len(SKILL_RULES),
        },
        "by_category": category_counts,
    }
    return report


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive skill mapping across all packages."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON report path (default: data/threeway_mappings/full_skill_report.json)",
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

    output_path = args.output or MAPPINGS_DIR / "full_skill_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Running comprehensive skill mapping")
    report = generate_comprehensive_report()

    try:
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Report written to %s", output_path)
    except OSError as e:
        logger.error("Failed to write report: %s", e)
        return 1

    print("\n=== Full Skill Mapping Report ===")
    print("Per package:")
    for pkg, data in report["per_package"].items():
        print(f"  {pkg}: {data['files']} files, {data['mappings']} mappings, {data['unique_rules']} unique rules")
    print("\nAggregate:")
    for k, v in report["aggregate"].items():
        print(f"  {k}: {v}")
    print("\nBy category:")
    for cat, count in sorted(report["by_category"].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
