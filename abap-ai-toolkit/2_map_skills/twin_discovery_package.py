#!/usr/bin/env python3
"""
Semantic twin discovery (patent concept).

Contains SAP_TWIN_DB: dictionary mapping legacy FMs to modern APIs by purpose.
Twin types: DIRECT (1:1), WRAPPER (wrap unreleased), REFACTOR (syntax only).
Generates wrapper class templates for WRAPPER twins. Outputs twin catalog JSON
and generated wrapper .abap files.
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
TWIN_DIR = DATA_DIR / "twin_discovery"
OUTPUT_CATALOG = TWIN_DIR / "twin_catalog_sample_pkg.json"
WRAPPER_OUTPUT_DIR = DATA_DIR / "sample_pkg_migrated"

# SAP_TWIN_DB: legacy FM -> modern API by purpose
SAP_TWIN_DB: dict[str, dict[str, Any]] = {
    "BAPI_ACC_DOCUMENT_POST": {
        "purpose": "post FI journal entry",
        "twin_type": "WRAPPER",
        "modern_api": "I_JournalEntryTP",
        "api_type": "RAP BO",
    },
    "BAPI_ACC_DOCUMENT_CHECK": {
        "purpose": "check FI journal entry",
        "twin_type": "WRAPPER",
        "modern_api": "I_JournalEntryTP",
        "api_type": "RAP BO",
    },
    "BAPI_ACC_ACT_POSTINGS_REVERSE": {
        "purpose": "reverse FI postings",
        "twin_type": "WRAPPER",
        "modern_api": "I_JournalEntryTP",
        "api_type": "RAP BO",
    },
    "BAPI_SALESORDER_CREATEFROMDAT2": {
        "purpose": "create sales order",
        "twin_type": "WRAPPER",
        "modern_api": "I_SalesOrderTP",
        "api_type": "RAP BO",
    },
    "BAPI_PO_CREATE1": {
        "purpose": "create purchase order",
        "twin_type": "WRAPPER",
        "modern_api": "I_PurchaseOrderTP_2",
        "api_type": "RAP BO",
    },
    "BAPI_GOODSMVT_CREATE": {
        "purpose": "create material document",
        "twin_type": "WRAPPER",
        "modern_api": "I_MaterialDocumentTP",
        "api_type": "RAP BO",
    },
    "FI_COMPANY_CODE_DATA": {
        "purpose": "company code data",
        "twin_type": "WRAPPER",
        "modern_api": "I_CompanyCode",
        "api_type": "CDS view",
    },
    "BAPI_TRANSACTION_COMMIT": {
        "purpose": "commit work",
        "twin_type": "DIRECT",
        "modern_api": "COMMIT WORK",
        "api_type": "statement",
    },
    "BAPI_TRANSACTION_ROLLBACK": {
        "purpose": "rollback work",
        "twin_type": "DIRECT",
        "modern_api": "ROLLBACK WORK",
        "api_type": "statement",
    },
    "CONVERSION_EXIT_ALPHA_INPUT": {
        "purpose": "alpha conversion input",
        "twin_type": "WRAPPER",
        "modern_api": "CONVERSION_EXIT_ALPHA_INPUT",
        "api_type": "FM (wrap)",
    },
    "READ_TEXT": {
        "purpose": "read text",
        "twin_type": "WRAPPER",
        "modern_api": "I_*Text RAP",
        "api_type": "RAP BO",
    },
    "SAVE_TEXT": {
        "purpose": "save text",
        "twin_type": "WRAPPER",
        "modern_api": "I_*Text RAP",
        "api_type": "RAP BO",
    },
}


def generate_wrapper_template(fm_name: str, twin_info: dict[str, Any]) -> str:
    """
    Generate ABAP wrapper class template for WRAPPER twin.

    Args:
        fm_name: Legacy FM name.
        twin_info: Twin metadata from SAP_TWIN_DB.

    Returns:
        ABAP source code string.
    """
    safe_fm = fm_name.lower().replace("/", "_")[:35]
    class_name = f"zcl_zsample_{safe_fm}_wrapper"
    return f'''*"----------------------------------------------------------------------
*"* Wrapper for legacy FM {fm_name}
*"* Purpose: {twin_info.get('purpose', 'N/A')}
*"* Modern API: {twin_info.get('modern_api', 'N/A')} ({twin_info.get('api_type', 'N/A')})
*"----------------------------------------------------------------------
CLASS {class_name} DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS call
      IMPORTING
        iv_fm_name TYPE string DEFAULT '{fm_name}'
      RETURNING
        VALUE(rv_success) TYPE abap_bool.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS {class_name} IMPLEMENTATION.
  METHOD call.
    CALL FUNCTION iv_fm_name
      EXCEPTIONS
        OTHERS = 1.
    rv_success = xsdbool( sy-subrc = 0 ).
  ENDMETHOD.
ENDCLASS.
'''


def discover_twins(fm_list: list[str]) -> dict[str, Any]:
    """
    Match FM list against SAP_TWIN_DB.

    Args:
        fm_list: List of FM names found in codebase.

    Returns:
        Twin catalog with direct, wrapper, refactor, review.
    """
    direct: list[dict] = []
    wrapper: list[dict] = []
    refactor: list[dict] = []
    review: list[dict] = []

    for fm in fm_list:
        fm_upper = fm.upper()
        if fm_upper in SAP_TWIN_DB:
            info = SAP_TWIN_DB[fm_upper].copy()
            info["legacy_fm"] = fm
            twin_type = info.get("twin_type", "REFACTOR")
            if twin_type == "DIRECT":
                direct.append(info)
            elif twin_type == "WRAPPER":
                wrapper.append(info)
            else:
                refactor.append(info)
        else:
            review.append({"legacy_fm": fm, "twin_type": "UNKNOWN", "note": "Manual review needed"})

    return {
        "direct": direct,
        "wrapper": wrapper,
        "refactor": refactor,
        "review": review,
        "summary": {
            "direct_count": len(direct),
            "wrapper_count": len(wrapper),
            "refactor_count": len(refactor),
            "review_count": len(review),
        },
    }


def load_fm_list_from_scan() -> list[str]:
    """Load FM list from scan results if available."""
    scan_path = DATA_DIR / "sample_pkg_scan_results.json"
    if not scan_path.exists():
        return []
    try:
        data = json.loads(scan_path.read_text(encoding="utf-8"))
        return data.get("fm_calls", [])
    except (json.JSONDecodeError, OSError):
        return []


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic twin discovery for ZSAMPLE ABAP."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Input JSON with fm_calls list (default: from scan results)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_CATALOG,
        help="Output twin catalog JSON path",
    )
    parser.add_argument(
        "--wrapper-dir",
        type=Path,
        default=WRAPPER_OUTPUT_DIR,
        help="Directory for generated wrapper .abap files",
    )
    parser.add_argument(
        "--no-generate-wrappers",
        action="store_true",
        help="Do not generate wrapper class files",
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

    if args.input and args.input.exists():
        try:
            data = json.loads(args.input.read_text(encoding="utf-8"))
            fm_list = data.get("fm_calls", [])
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Could not load input: %s", e)
            return 1
    else:
        fm_list = load_fm_list_from_scan()
        if not fm_list:
            logger.warning("No FM list found. Using sample FMs from SAP_TWIN_DB.")
            fm_list = list(SAP_TWIN_DB.keys())

    catalog = discover_twins(fm_list)
    catalog["sap_twin_db_keys"] = list(SAP_TWIN_DB.keys())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        args.output.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        logger.info("Twin catalog written to %s", args.output)
    except OSError as e:
        logger.error("Failed to write catalog: %s", e)
        return 1

    if not args.no_generate_wrappers and catalog["wrapper"]:
        args.wrapper_dir.mkdir(parents=True, exist_ok=True)
        for item in catalog["wrapper"]:
            fm_name = item.get("legacy_fm", "UNKNOWN")
            code = generate_wrapper_template(fm_name, item)
            safe_name = f"zcl_zsample_{fm_name.lower()[:40]}_wrapper.clas.abap"
            out_path = args.wrapper_dir / safe_name
            try:
                out_path.write_text(code, encoding="utf-8")
                logger.info("Generated wrapper: %s", out_path.name)
            except OSError as e:
                logger.warning("Could not write %s: %s", out_path, e)

    print("\n=== Twin Discovery Summary ===")
    s = catalog["summary"]
    print(f"DIRECT (1:1): {s['direct_count']}")
    print(f"WRAPPER: {s['wrapper_count']}")
    print(f"REFACTOR: {s['refactor_count']}")
    print(f"REVIEW (manual): {s['review_count']}")
    print(f"Catalog: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
