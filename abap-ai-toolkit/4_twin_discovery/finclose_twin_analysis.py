#!/usr/bin/env python3
"""
Financial-close twin analysis.

Analyzes period-close related FMs.
Maps to S/4HANA equivalents.
Domain-specific twin discovery for financial close.
"""

import argparse
import json
import logging
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

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
OUTPUT_DIR = DATA_DIR / "twin_discovery"
OUTPUT_PATH = OUTPUT_DIR / "finclose_twin_analysis.json"

FINCLOSE_FM_MAP = {
    "BAPI_ACC_DOCUMENT_POST": {"purpose": "Post FI journal entry", "s4_twin": "I_JournalEntryTP", "twin_type": "RAP_BO", "confidence": 0.95, "close_relevance": "high"},
    "BAPI_ACC_PERIOD_CLOSE": {"purpose": "Close FI posting period", "s4_twin": "I_AccountingDocument", "twin_type": "CDS_VIEW", "confidence": 0.88, "close_relevance": "high"},
    "FAGL_GET_OPEN_ITEMS": {"purpose": "Get open items", "s4_twin": "I_GLAccountLineItem", "twin_type": "CDS_VIEW", "confidence": 0.90, "close_relevance": "high"},
    "BAPI_TRANSACTION_COMMIT": {"purpose": "Commit postings", "s4_twin": "COMMIT ENTITIES", "twin_type": "ABAP_STMT", "confidence": 0.98, "close_relevance": "high"},
}


def analyze_finclose_fms(custom_fms: list[str] | None = None) -> dict:
    """Analyze financial close FMs and map to S/4HANA."""
    all_fms = list(FINCLOSE_FM_MAP.keys())
    if custom_fms:
        all_fms = list(set(all_fms) | set(custom_fms))
    mappings = []
    by_relevance: dict[str, list] = {"high": [], "medium": [], "low": []}
    for fm in all_fms:
        entry = FINCLOSE_FM_MAP.get(fm)
        if entry:
            mappings.append({"fm": fm, **entry})
            by_relevance[entry.get("close_relevance", "low")].append(fm)
        else:
            mappings.append({"fm": fm, "purpose": "unknown", "s4_twin": None, "close_relevance": "low"})
    return {
        "source": "finclose_twin_analysis",
        "domain": "financial_close",
        "summary": {"total_fms": len(mappings), "mapped": sum(1 for m in mappings if m.get("s4_twin"))},
        "by_relevance": by_relevance,
        "mappings": mappings,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Financial close twin analysis")
    parser.add_argument("--fms", type=str, nargs="+", default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        analysis = analyze_finclose_fms(custom_fms=args.fms)
        output_path = args.output or OUTPUT_PATH
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nMapped: {analysis['summary']['mapped']}/{analysis['summary']['total_fms']}\n")
    except Exception as e:
        logger.exception("Analysis failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
