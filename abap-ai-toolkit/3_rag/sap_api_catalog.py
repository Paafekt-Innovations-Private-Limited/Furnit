#!/usr/bin/env python3
"""
Index SAP modern API catalog into the RAG vector database.

Contains catalog of 21 RAP Business Objects + CDS views. Each entry:
{name, type (RAP BO/CDS), module, purpose, replaces}. Indexes as chunks
in 'sap_api_catalog' layer.
"""

import hashlib
import logging
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from _common import get_chroma_client, get_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# SAP modern API catalog - 21 RAP BOs + CDS views
SAP_API_CATALOG = [
    {"name": "I_JournalEntryTP", "type": "RAP BO", "module": "FI", "purpose": "FI journal entry posting", "replaces": "BAPI_ACC_DOCUMENT_POST"},
    {"name": "I_JournalEntryItem", "type": "RAP BO", "module": "FI", "purpose": "Journal entry line items", "replaces": "BAPI_ACC_DOCUMENT_POST"},
    {"name": "I_SalesOrderTP", "type": "RAP BO", "module": "SD", "purpose": "Sales order creation", "replaces": "BAPI_SALESORDER_CREATEFROMDAT2"},
    {"name": "I_PurchaseOrderTP_2", "type": "RAP BO", "module": "MM", "purpose": "Purchase order", "replaces": "BAPI_PO_CREATE1"},
    {"name": "I_MaterialDocumentTP", "type": "RAP BO", "module": "MM", "purpose": "Goods movement", "replaces": "BAPI_GOODSMVT_CREATE"},
    {"name": "I_CompanyCode", "type": "CDS", "module": "FI", "purpose": "Company code master", "replaces": "FI_COMPANY_CODE_DATA"},
    {"name": "I_FiscCalendarDateForCompCode", "type": "CDS", "module": "FI", "purpose": "Fiscal period determination", "replaces": "FI_PERIOD_DETERMINE"},
    {"name": "I_Product", "type": "RAP BO", "module": "MM", "purpose": "Product master", "replaces": "BAPI_MATERIAL_SAVEDATA"},
    {"name": "I_Customer", "type": "RAP BO", "module": "SD", "purpose": "Customer master", "replaces": "BAPI_CUSTOMER_*"},
    {"name": "I_Supplier", "type": "RAP BO", "module": "MM", "purpose": "Supplier master", "replaces": "BAPI_VENDOR_*"},
    {"name": "I_BusinessPartner", "type": "RAP BO", "module": "S4", "purpose": "Unified business partner", "replaces": "Customer/Supplier"},
    {"name": "I_DeliveryTP", "type": "RAP BO", "module": "SD", "purpose": "Delivery document", "replaces": "BAPI_DELIVERY_*"},
    {"name": "I_InvoiceTP", "type": "RAP BO", "module": "FI", "purpose": "Customer invoice", "replaces": "BAPI_BILLING_*"},
    {"name": "I_BankAccount", "type": "RAP BO", "module": "FI", "purpose": "Bank account master", "replaces": "BAPI_BANK_*"},
    {"name": "I_GLAccountInChartOfAccounts", "type": "CDS", "module": "FI", "purpose": "GL account master", "replaces": "SKA1/SKAT"},
    {"name": "I_CostCenter", "type": "CDS", "module": "CO", "purpose": "Cost center master", "replaces": "RK_KOKRS_FIND"},
    {"name": "I_Currency", "type": "CDS", "module": "FI", "purpose": "Currency conversion", "replaces": "CONVERT_TO_LOCAL_CURRENCY"},
    {"name": "I_Text", "type": "RAP BO", "module": "S4", "purpose": "Text storage", "replaces": "READ_TEXT/SAVE_TEXT"},
    {"name": "I_Log", "type": "RAP BO", "module": "S4", "purpose": "Application log", "replaces": "BAL_*"},
    {"name": "I_Lock", "type": "RAP BO", "module": "S4", "purpose": "Lock management", "replaces": "ENQUEUE_*/DEQUEUE_*"},
    {"name": "I_ConversionExit", "type": "CDS", "module": "S4", "purpose": "Conversion exits", "replaces": "CONVERSION_EXIT_*"},
]


def main() -> int:
    """CLI entry point."""
    client = get_chroma_client()
    collection = get_collection(client)

    ids = []
    documents = []
    metadatas = []
    for i, entry in enumerate(SAP_API_CATALOG):
        content = (
            f"API: {entry['name']} ({entry['type']})\n"
            f"Module: {entry['module']}\n"
            f"Purpose: {entry['purpose']}\n"
            f"Replaces: {entry.get('replaces', 'N/A')}"
        )
        raw = f"sap_api_catalog:{entry['name']}:{i}"
        chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:24]
        ids.append(chunk_id)
        documents.append(content)
        metadatas.append({
            "layer": "sap_api_catalog",
            "name": entry["name"],
            "type": entry["type"],
            "module": entry["module"],
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Indexed %d SAP API catalog entries", len(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
