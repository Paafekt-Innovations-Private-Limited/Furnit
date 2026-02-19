#!/usr/bin/env python3
"""
Discover released S/4HANA Clean Core APIs.

Queries SAP API catalog for released RAP BOs and CDS views.
Categorizes by module (FI, CO, MM, SD, etc.) and outputs catalog JSON.
"""

import argparse
import json
import logging
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

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
OUTPUT_DIR = DATA_DIR / "twin_discovery"
CATALOG_PATH = OUTPUT_DIR / "clean_core_catalog.json"

# SAP module categories for RAP BOs and CDS views
SAP_MODULE_MAP = {
    "FI": ["I_JournalEntry", "I_GLAccountLineItem", "I_AccountingDocument", "I_BankStatement"],
    "CO": ["I_CostCenter", "I_ProfitCenter", "I_InternalOrder", "I_ActivityAllocation"],
    "MM": ["I_MaterialDocument", "I_PurchaseOrder", "I_SupplierInvoice", "I_Stock"],
    "SD": ["I_SalesOrder", "I_Delivery", "I_BillingDocument", "I_Customer"],
    "AA": ["I_Asset", "I_AssetDepreciation", "I_AssetValue"],
    "AP": ["I_Supplier", "I_SupplierInvoice", "I_Payment"],
    "AR": ["I_Customer", "I_CustomerInvoice", "I_Receivable"],
}

# Fallback catalog when API is unavailable (released RAP BOs from SAP docs)
FALLBACK_RAP_CATALOG = [
    {"name": "I_JournalEntryTP", "type": "RAP_BO", "module": "FI", "released": True},
    {"name": "I_GLAccountLineItem", "type": "CDS_VIEW", "module": "FI", "released": True},
    {"name": "I_AccountingDocument", "type": "CDS_VIEW", "module": "FI", "released": True},
    {"name": "I_MaterialDocumentTP", "type": "RAP_BO", "module": "MM", "released": True},
    {"name": "I_PurchaseOrder", "type": "CDS_VIEW", "module": "MM", "released": True},
    {"name": "I_SalesOrderTP", "type": "RAP_BO", "module": "SD", "released": True},
    {"name": "I_Customer", "type": "CDS_VIEW", "module": "SD", "released": True},
    {"name": "I_Supplier", "type": "CDS_VIEW", "module": "AP", "released": True},
    {"name": "I_Asset", "type": "CDS_VIEW", "module": "AA", "released": True},
    {"name": "I_CostCenter", "type": "CDS_VIEW", "module": "CO", "released": True},
]


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config = {}
    if CONFIG_PATH.exists() and yaml:
        try:
            with open(CONFIG_PATH, "r") as f:
                config = yaml.safe_load(f) or {}
            logger.info("Loaded config from %s", CONFIG_PATH)
        except Exception as e:
            logger.warning("Could not load config: %s", e)
    return config


def query_sap_api_catalog(api_hub_url: str) -> list[dict]:
    """
    Query SAP API Business Hub for released RAP BOs and CDS views.

    Args:
        api_hub_url: Base URL for SAP API catalog (e.g. https://api.sap.com).

    Returns:
        List of API entries with name, type, module, released status.
    """
    if not requests:
        logger.warning("requests not installed; using fallback catalog")
        return FALLBACK_RAP_CATALOG

    catalog_entries = []
    try:
        # SAP API Business Hub discovery endpoint (simplified - real API may differ)
        discovery_url = f"{api_hub_url.rstrip('/')}/api/sap/public/odata/v4/API_BUSINESS_PARTNER"
        response = requests.get(discovery_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Parse OData metadata for entity sets
            entities = data.get("d", {}).get("EntitySets", []) or []
            for ent in entities[:50]:  # Limit for demo
                name = ent.get("Name", ent) if isinstance(ent, dict) else str(ent)
                module = _infer_module(name)
                catalog_entries.append({
                    "name": name,
                    "type": "RAP_BO" if "TP" in name or "API" in name else "CDS_VIEW",
                    "module": module,
                    "released": True,
                })
        else:
            logger.info("SAP API returned %s; using fallback catalog", response.status_code)
    except Exception as e:
        logger.warning("Could not query SAP API catalog: %s; using fallback", e)

    if not catalog_entries:
        catalog_entries = FALLBACK_RAP_CATALOG

    return catalog_entries


def _infer_module(name: str) -> str:
    """Infer SAP module from API/entity name."""
    name_upper = name.upper()
    if "JOURNAL" in name_upper or "GL" in name_upper or "ACCOUNTING" in name_upper:
        return "FI"
    if "COST" in name_upper or "PROFIT" in name_upper or "ORDER" in name_upper:
        return "CO"
    if "MATERIAL" in name_upper or "PURCHASE" in name_upper or "STOCK" in name_upper:
        return "MM"
    if "SALES" in name_upper or "DELIVERY" in name_upper or "BILLING" in name_upper:
        return "SD"
    if "SUPPLIER" in name_upper or "INVOICE" in name_upper:
        return "AP"
    if "CUSTOMER" in name_upper or "RECEIVABLE" in name_upper:
        return "AR"
    if "ASSET" in name_upper:
        return "AA"
    return "OTHER"


def categorize_by_module(catalog_entries: list[dict]) -> dict[str, list[dict]]:
    """Group catalog entries by SAP module."""
    by_module: dict[str, list[dict]] = {}
    for entry in catalog_entries:
        module = entry.get("module", "OTHER")
        by_module.setdefault(module, []).append(entry)
    return by_module


def discover_clean_core(output_path: Path | None = None) -> dict:
    """
    Discover released S/4HANA Clean Core APIs and output catalog JSON.

    Returns:
        Summary dict with total_apis, by_module counts, output_path.
    """
    config = load_config()
    twin_config = config.get("twin_discovery", {})
    api_hub_url = twin_config.get("api_hub_url", "https://api.sap.com")
    modules = twin_config.get("sap_modules", list(SAP_MODULE_MAP.keys()))

    logger.info("Discovering Clean Core APIs from %s", api_hub_url)
    catalog_entries = query_sap_api_catalog(api_hub_url)

    # Filter to requested modules if specified
    if modules:
        catalog_entries = [e for e in catalog_entries if e.get("module") in modules]

    by_module = categorize_by_module(catalog_entries)
    catalog = {
        "source": "discover_clean_core",
        "api_hub_url": api_hub_url,
        "total_apis": len(catalog_entries),
        "by_module": by_module,
        "entries": catalog_entries,
    }

    out_path = output_path or CATALOG_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    logger.info("Wrote catalog with %d APIs to %s", len(catalog_entries), out_path)
    return {
        "total_apis": len(catalog_entries),
        "by_module": {k: len(v) for k, v in by_module.items()},
        "output_path": str(out_path),
    }


def main() -> None:
    """Main entry point for Clean Core API discovery."""
    parser = argparse.ArgumentParser(
        description="Discover released S/4HANA Clean Core APIs (RAP BOs, CDS views)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: data/twin_discovery/clean_core_catalog.json)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        summary = discover_clean_core(args.output)
        print("\n--- Clean Core Catalog Summary ---")
        print(f"  Total APIs:   {summary['total_apis']}")
        print(f"  Output:       {summary['output_path']}")
        print("  By module:")
        for mod, cnt in sorted(summary["by_module"].items(), key=lambda x: -x[1]):
            print(f"    {mod}: {cnt}")
        print("-----------------------------------\n")
    except Exception as e:
        logger.exception("Discovery failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
