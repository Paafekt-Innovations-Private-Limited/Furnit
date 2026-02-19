#!/usr/bin/env python3
"""
Index SAP Fiori knowledge into the RAG vector database.

Contains hardcoded Fiori knowledge across 5 layers: fiori_elements(5),
fiori_cds(5), fiori_ui5(3), fiori_manifest(2), fiori_patterns(2), fiori_odata(1).
Each chunk: { content, metadata: {layer, topic} }.
Total: 18 chunks.
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

# Hardcoded Fiori knowledge - 18 chunks across 6 layers
FIORI_KNOWLEDGE = [
    # fiori_elements (5)
    {"layer": "fiori_elements", "topic": "floorplans", "content": "Fiori Elements floorplans: List Report, Object Page, Overview Page. Use @UI.lineItem and @UI.identification for list display. Use @Consumption.valueHelp for value help."},
    {"layer": "fiori_elements", "topic": "draft", "content": "Fiori Elements draft: Enable draft in behavior definition. Use %is_draft for draft handling. Draft UUID in instance buffer."},
    {"layer": "fiori_elements", "topic": "actions", "content": "Fiori Elements actions: @UI.lineItem with DataFieldForAction. Side effects via determination. Action in behavior definition."},
    {"layer": "fiori_elements", "topic": "side_effects", "content": "Fiori Elements side effects: Use determination with %control to detect changed fields. Update dependent fields in same transaction."},
    {"layer": "fiori_elements", "topic": "annotations", "content": "Fiori Elements annotations: @UI, @Consumption, @OData. Annotate CDS view entities for UI behavior."},
    # fiori_cds (5)
    {"layer": "fiori_cds", "topic": "ui_annotations", "content": "CDS UI annotations: @UI.lineItem, @UI.identification, @UI.headerInfo. Control list and object page layout."},
    {"layer": "fiori_cds", "topic": "value_help", "content": "CDS value help: @Consumption.valueHelp or @ObjectModel.valueHelpDefinition. Link to CDS value help view."},
    {"layer": "fiori_cds", "topic": "criticality", "content": "CDS criticality: @UI.criticality for traffic light. Use CASE in CDS or determination."},
    {"layer": "fiori_cds", "topic": "search", "content": "CDS search: @Search.searchable for Fiori Elements search. Add to relevant fields."},
    {"layer": "fiori_cds", "topic": "selection", "content": "CDS selection: @Consumption.filter for filter fields. Use parameters for mandatory filters."},
    # fiori_ui5 (3)
    {"layer": "fiori_ui5", "topic": "freestyle", "content": "SAPUI5 freestyle: Use when Fiori Elements not sufficient. MVC: View (XML), Controller (JS), Model (JSON/OData)."},
    {"layer": "fiori_ui5", "topic": "odata_binding", "content": "SAPUI5 OData V4 binding: Use sap.ui.model.odata.v4.ODataModel. List binding with $expand, $filter."},
    {"layer": "fiori_ui5", "topic": "views", "content": "SAPUI5 views: XML views preferred. Fragment for dialogs. Controller lifecycle: onInit, onBeforeRendering, onAfterRendering."},
    # fiori_manifest (2)
    {"layer": "fiori_manifest", "topic": "manifest", "content": "manifest.json: App descriptor. dataSources for OData. routing for navigation. sap.app, sap.ui5 sections."},
    {"layer": "fiori_manifest", "topic": "launchpad", "content": "Fiori Launchpad: Tiles in manifest. Intent navigation. Semantic objects and actions for deep linking."},
    # fiori_patterns (2)
    {"layer": "fiori_patterns", "topic": "golden_rules", "content": "Fiori golden rules: Responsive, role-based, actionable. Use Fiori Elements when possible. Follow SAP design guidelines."},
    {"layer": "fiori_patterns", "topic": "elements_vs_freestyle", "content": "Fiori Elements vs Freestyle: Elements for standard CRUD. Freestyle for custom UX, complex logic, or non-standard patterns."},
    # fiori_odata (1)
    {"layer": "fiori_odata", "topic": "rap_binding", "content": "RAP service binding for Fiori: OData V4 UI or Web API. Publish service definition. Fiori Elements consumes OData V4."},
]


def main() -> int:
    """CLI entry point."""
    client = get_chroma_client()
    collection = get_collection(client)

    ids = []
    documents = []
    metadatas = []
    for i, item in enumerate(FIORI_KNOWLEDGE):
        raw = f"{item['layer']}:{item['topic']}:{i}"
        chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:24]
        ids.append(chunk_id)
        documents.append(item["content"])
        metadatas.append({"layer": item["layer"], "topic": item["topic"]})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Indexed %d Fiori knowledge chunks", len(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
