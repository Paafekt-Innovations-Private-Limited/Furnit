#!/usr/bin/env python3
"""
Map ECC legacy code to Clean Core by semantic meaning.

Uses purpose-based matching against SAP_TWIN_DB.
Supports cardinality: 1:1, 1:N, N:1, N:M.
Outputs matched twins with confidence scores.
"""

import argparse
import json
import logging
import re
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = TOOLKIT_ROOT / "config.yaml"
DATA_DIR = TOOLKIT_ROOT / "data"
TWIN_DB_PATH = DATA_DIR / "twin_discovery" / "sap_twin_db.json"
OUTPUT_DIR = DATA_DIR / "twin_discovery"

# SAP_TWIN_DB: purpose -> Clean Core twin mappings (simplified)
SAP_TWIN_DB = {
    "BAPI_ACC_DOCUMENT_POST": {
        "purpose": "post FI journal entry",
        "twins": [{"name": "I_JournalEntryTP", "type": "RAP_BO", "confidence": 0.95}],
        "cardinality": "1:1",
    },
    "BAPI_GOODSMVT_CREATE": {
        "purpose": "create material document",
        "twins": [{"name": "I_MaterialDocumentTP", "type": "RAP_BO", "confidence": 0.92}],
        "cardinality": "1:1",
    },
    "BAPI_SALESORDER_CREATEFROMDAT2": {
        "purpose": "create sales order",
        "twins": [{"name": "I_SalesOrderTP", "type": "RAP_BO", "confidence": 0.90}],
        "cardinality": "1:1",
    },
    "BAPI_TRANSACTION_COMMIT": {
        "purpose": "commit database changes",
        "twins": [{"name": "COMMIT ENTITIES", "type": "ABAP_STMT", "confidence": 0.98}],
        "cardinality": "1:1",
    },
    "AUTHORITY_CHECK_TCODE": {
        "purpose": "check transaction authorization",
        "twins": [{"name": "CL_ABAP_AUTH", "type": "CLASS", "confidence": 0.85}],
        "cardinality": "1:1",
    },
}

CARDINALITIES = ("1:1", "1:N", "N:1", "N:M")


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config = {}
    if CONFIG_PATH.exists() and yaml:
        try:
            with open(CONFIG_PATH, "r") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Could not load config: %s", e)
    return config


def load_twin_db() -> dict:
    """Load SAP_TWIN_DB from JSON file or use built-in fallback."""
    if TWIN_DB_PATH.exists():
        try:
            with open(TWIN_DB_PATH, "r", encoding="utf-8") as f:
                db = json.load(f)
            logger.info("Loaded twin DB from %s (%d entries)", TWIN_DB_PATH, len(db))
            return db
        except Exception as e:
            logger.warning("Could not load twin DB: %s; using built-in", e)
    return SAP_TWIN_DB


def match_by_name(fm_name: str, twin_db: dict) -> dict | None:
    """Exact match by function module or class name."""
    return twin_db.get(fm_name)


def match_by_purpose(purpose_query: str, twin_db: dict) -> list[dict]:
    """Semantic match by purpose string."""
    purpose_lower = purpose_query.lower()
    matches = []
    for key, entry in twin_db.items():
        purpose = entry.get("purpose", "").lower()
        if purpose_lower in purpose or purpose in purpose_lower:
            matches.append({"source": key, **entry})
    return matches


def match_twins(
    fm: str | None = None,
    purpose: str | None = None,
    twin_db: dict | None = None,
) -> dict:
    """
    Map ECC legacy code to Clean Core twins.

    Args:
        fm: Function module or class name (e.g. BAPI_ACC_DOCUMENT_POST).
        purpose: Optional purpose string for semantic search.
        twin_db: Optional twin database dict (loads from file if None).

    Returns:
        Dict with matches, cardinality, confidence scores.
    """
    db = twin_db or load_twin_db()

    if fm:
        exact = match_by_name(fm, db)
        if exact:
            twins = exact.get("twins", [])
            return {
                "input": fm,
                "match_type": "exact",
                "purpose": exact.get("purpose", ""),
                "twins": twins,
                "cardinality": exact.get("cardinality", "1:1"),
                "confidence_avg": sum(t.get("confidence", 0) for t in twins) / max(1, len(twins)),
            }
        fm_norm = fm.replace("_", "").upper()
        for key in db:
            if key.replace("_", "").upper() in fm_norm or fm_norm in key.replace("_", "").upper():
                entry = db[key]
                twins = entry.get("twins", [])
                return {
                    "input": fm,
                    "match_type": "fuzzy",
                    "matched_key": key,
                    "purpose": entry.get("purpose", ""),
                    "twins": twins,
                    "cardinality": entry.get("cardinality", "1:1"),
                    "confidence_avg": sum(t.get("confidence", 0) for t in twins) / max(1, len(twins)),
                }
        return {
            "input": fm,
            "match_type": "none",
            "purpose": "",
            "twins": [],
            "cardinality": "1:1",
            "confidence_avg": 0.0,
            "message": f"No twin found for {fm}. Consider adding to SAP_TWIN_DB.",
        }

    if purpose:
        semantic_matches = match_by_purpose(purpose, db)
        return {
            "input": purpose,
            "match_type": "semantic",
            "matches": semantic_matches,
            "count": len(semantic_matches),
        }

    return {"error": "Provide --fm or --purpose"}


def main() -> None:
    """Main entry point for twin matching."""
    parser = argparse.ArgumentParser(
        description="Map ECC legacy code to Clean Core by semantic meaning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fm", type=str, default=None, help="Function module or class name")
    parser.add_argument("--purpose", type=str, default=None, help="Purpose string for semantic search")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path for results")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.fm and not args.purpose:
        parser.error("Provide --fm or --purpose")

    try:
        result = match_twins(fm=args.fm, purpose=args.purpose)
        output_json = json.dumps(result, indent=2)
        print(output_json)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output_json, encoding="utf-8")
            logger.info("Wrote results to %s", args.output)
    except Exception as e:
        logger.exception("Match failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
