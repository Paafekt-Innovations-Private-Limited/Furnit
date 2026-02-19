#!/usr/bin/env python3
"""
Check specific Function Modules against CRV.

Reads FM list from scan results. Cross-references with CRV data. Reports:
released, notToBeReleased, missing.
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
CRV_PATH = DATA_DIR / "twin_discovery" / "crv_latest_full.json"


def load_fm_list(input_path: Path | None) -> list[str]:
    """Load FM list from scan results or JSON file."""
    path = input_path or SCAN_RESULTS
    if not path.exists():
        logger.warning("Scan results not found: %s", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("fm_calls", [])
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Could not load FM list: %s", e)
        return []


def load_crv(input_path: Path) -> dict[str, Any]:
    """Load CRV data and build FM index."""
    if not input_path.exists():
        return {}
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load CRV: %s", e)
        return {}

    # Use pre-built fm_index if available
    fm_index = data.get("fm_index", {})
    if fm_index:
        return fm_index

    # Parse objects/findings
    result: dict[str, Any] = {}
    for key in ("objects", "functionModules", "findings"):
        items = data.get(key, [])
        if isinstance(items, list):
            for obj in items:
                name = obj.get("name", obj.get("objectName", obj.get("functionModule", "")))
                if name:
                    result[str(name).upper()] = {
                        "releaseState": obj.get("releaseState", obj.get("status", "unknown")),
                        "successor": obj.get("successor", ""),
                    }
    return result


def check_fms(fm_list: list[str], crv_index: dict) -> dict[str, Any]:
    """
    Cross-reference FM list with CRV.

    Returns:
        Dict with released, notToBeReleased, missing lists and counts.
    """
    released: list[dict] = []
    not_to_be_released: list[dict] = []
    missing: list[str] = []

    for fm in fm_list:
        fm_upper = fm.upper()
        info = crv_index.get(fm_upper)
        if info is None:
            missing.append(fm)
            continue
        state = info.get("releaseState", info.get("status", "unknown"))
        successor = info.get("successor", "")
        entry = {"fm": fm, "successor": successor}
        if str(state).lower() in ("released", "release"):
            released.append(entry)
        elif str(state).lower() in ("nottobereleased", "not to be released", "deprecated"):
            not_to_be_released.append(entry)
        else:
            missing.append(fm)

    return {
        "released": released,
        "notToBeReleased": not_to_be_released,
        "missing": missing,
        "counts": {
            "released": len(released),
            "notToBeReleased": len(not_to_be_released),
            "missing": len(missing),
            "total": len(fm_list),
        },
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Check FMs against CRV."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Input JSON with fm_calls (default: scan results)",
    )
    parser.add_argument(
        "--crv",
        type=Path,
        default=CRV_PATH,
        help="CRV JSON path",
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

    fm_list = load_fm_list(args.input)
    if not fm_list:
        logger.warning("No FM list found. Run scan_sample_pkg.py first. Reporting empty results.")

    crv_index = load_crv(args.crv)
    if not crv_index:
        logger.warning("CRV data not found or empty. Run download_crv.py first.")

    result = check_fms(fm_list, crv_index)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            logger.info("Report written to %s", args.output)
        except OSError as e:
            logger.error("Failed to write report: %s", e)

    print("\n=== FM vs CRV Report ===")
    c = result["counts"]
    print(f"Total FMs checked: {c['total']}")
    print(f"Released: {c['released']}")
    print(f"Not to be released: {c['notToBeReleased']}")
    print(f"Missing (not in CRV): {c['missing']}")
    if result["released"]:
        print("\nReleased:")
        for r in result["released"][:10]:
            print(f"  - {r['fm']}")
    if result["notToBeReleased"]:
        print("\nNot to be released:")
        for r in result["notToBeReleased"][:10]:
            print(f"  - {r['fm']} -> {r.get('successor', 'N/A')}")
    if result["missing"]:
        print("\nMissing (sample):")
        for m in result["missing"][:15]:
            print(f"  - {m}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
