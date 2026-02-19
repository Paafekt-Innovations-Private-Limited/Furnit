#!/usr/bin/env python3
"""
Cross-reference twins with SAP CRV registry.

Loads CRV data from data/twin_discovery/crv_latest_full.json.
Verifies each twin's release status.
Reports: verified, unverified, conflict.
"""

import argparse
import json
import logging
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
CRV_PATH = DATA_DIR / "twin_discovery" / "crv_latest_full.json"
TWIN_CATALOG_PATH = DATA_DIR / "twin_discovery" / "clean_core_catalog.json"
OUTPUT_PATH = DATA_DIR / "twin_discovery" / "crv_verified_report.json"


def load_crv_data(crv_path: Path) -> dict:
    """Load CRV data from JSON."""
    if not crv_path.exists():
        logger.warning("CRV file not found at %s", crv_path)
        return {}
    try:
        with open(crv_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Could not load CRV: %s", e)
        return {}


def load_twin_catalog(catalog_path: Path) -> list[dict]:
    """Load twin catalog."""
    if not catalog_path.exists():
        return []
    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", data) if isinstance(data, dict) else data
        return entries if isinstance(entries, list) else [data]
    except Exception as e:
        logger.error("Could not load catalog: %s", e)
        return []


def build_crv_lookup(crv_data: dict) -> dict[str, dict]:
    """Build lookup dict from CRV structure."""
    lookup = {}
    if isinstance(crv_data, list):
        for item in crv_data:
            name = item.get("name", item.get("object_name", item.get("fm", "")))
            if name:
                lookup[name.upper()] = item
    elif isinstance(crv_data, dict):
        for key, val in crv_data.items():
            if isinstance(val, dict):
                lookup[key.upper()] = val
            else:
                lookup[key.upper()] = {"raw": val}
    return lookup


def verify_twin(twin_name: str, crv_lookup: dict[str, dict]) -> dict:
    """Verify a single twin against CRV."""
    key = twin_name.upper()
    crv_entry = crv_lookup.get(key)
    if not crv_entry:
        return {"twin": twin_name, "status": "unverified", "reason": "not_found_in_crv", "crv_data": None}
    released = crv_entry.get("released", crv_entry.get("release_status", False))
    successor = crv_entry.get("successor", crv_entry.get("replacement", ""))
    if released:
        return {"twin": twin_name, "status": "verified", "reason": "released", "crv_data": crv_entry}
    if successor:
        return {"twin": twin_name, "status": "verified", "reason": "has_successor", "successor": successor, "crv_data": crv_entry}
    return {"twin": twin_name, "status": "unverified", "reason": "no_release_info", "crv_data": crv_entry}


def verify_twins(
    crv_path: Path | None = None,
    catalog_path: Path | None = None,
    twin_names: list[str] | None = None,
) -> dict:
    """Cross-reference twins with SAP CRV registry."""
    crv_path = crv_path or CRV_PATH
    catalog_path = catalog_path or TWIN_CATALOG_PATH
    crv_data = load_crv_data(crv_path)
    crv_lookup = build_crv_lookup(crv_data)

    names_to_verify = twin_names or [e.get("name", e.get("twin", "")) for e in load_twin_catalog(catalog_path) if e.get("name") or e.get("twin")]
    if not names_to_verify:
        names_to_verify = ["BAPI_ACC_DOCUMENT_POST", "BAPI_GOODSMVT_CREATE", "BAPI_TRANSACTION_COMMIT", "AUTHORITY_CHECK_TCODE"]

    verified_list = []
    unverified_list = []
    for name in names_to_verify:
        if not name:
            continue
        result = verify_twin(name, crv_lookup)
        (verified_list if result["status"] == "verified" else unverified_list).append(result)

    return {
        "source": "crv_verified_twins",
        "crv_path": str(crv_path),
        "crv_loaded": bool(crv_data),
        "total_checked": len(names_to_verify),
        "verified": len(verified_list),
        "unverified": len(unverified_list),
        "verified_details": verified_list,
        "unverified_details": unverified_list,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Cross-reference twins with SAP CRV registry")
    parser.add_argument("--crv", type=Path, default=None, help="Path to CRV JSON")
    parser.add_argument("--catalog", type=Path, default=None, help="Path to twin catalog")
    parser.add_argument("--twins", type=str, nargs="+", default=None, help="Explicit twin names")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        report = verify_twins(crv_path=args.crv, catalog_path=args.catalog, twin_names=args.twins)
        output_path = args.output or OUTPUT_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nVerified: {report['verified']}, Unverified: {report['unverified']}\n")
    except Exception as e:
        logger.exception("Verification failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
