#!/usr/bin/env python3
"""
Index SAP RAP knowledge into the RAG vector database.

Contains hardcoded RAP knowledge across 7 layers: rap_impl(6), rap_advanced(5),
rap_cds(3), rap_behavior(3), rap_architecture(2), rap_eml(2), rap_auth(2),
rap_service(1). Total: 24 chunks.
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

# Hardcoded RAP knowledge - 24 chunks across 8 layers
RAP_KNOWLEDGE = [
    # rap_impl (6)
    {"layer": "rap_impl", "topic": "validation", "content": "RAP validation: Implement validate in behavior implementation. Use %key for instance. Return failed for errors. Use %control to check changed fields."},
    {"layer": "rap_impl", "topic": "determination", "content": "RAP determination: Implement determine in behavior. Runs on save. Use MODIFY ENTITY to update. Side effects for dependent fields."},
    {"layer": "rap_impl", "topic": "action", "content": "RAP action: Define in behavior definition. Implement in behavior implementation. Use MODIFY ENTITY or READ ENTITY. Return keys."},
    {"layer": "rap_impl", "topic": "numbering", "content": "RAP numbering: Use managed numbering for auto-generated keys. Implement number_get in behavior implementation. Use NUMBERING(managed) in BDEF."},
    {"layer": "rap_impl", "topic": "side_effects", "content": "RAP side effects: Use determination with %control. Update related entities. Avoid external API calls in determination."},
    {"layer": "rap_impl", "topic": "read", "content": "RAP read: Implement read in behavior implementation. Use READ ENTITY. Support $expand for associations."},
    # rap_advanced (5)
    {"layer": "rap_advanced", "topic": "locking", "content": "RAP locking: Use lock master in managed scenario. Pessimistic locking via ETag. Implement lock for custom locking."},
    {"layer": "rap_advanced", "topic": "etag", "content": "RAP ETag: Optimistic concurrency. LastChangedDateTime in root. Set in determination or on save."},
    {"layer": "rap_advanced", "topic": "error_handling", "content": "RAP error handling: Use failed, reported in RETURN. MESSAGE with sy-msgty. Use append to add multiple messages."},
    {"layer": "rap_advanced", "topic": "unmanaged", "content": "RAP unmanaged: Save in custom implementation. Use SAVE MODIFIED. Implement save for full control."},
    {"layer": "rap_advanced", "topic": "testing", "content": "RAP testing: Use ABAP Unit. Test behavior implementation. Mock READ/MODIFY for dependencies."},
    # rap_cds (3)
    {"layer": "rap_cds", "topic": "view_entities", "content": "CDS view entities: Define @AccessControl. Use associations. Replace CDS views for RAP."},
    {"layer": "rap_cds", "topic": "composition", "content": "RAP composition: Child entities via composition. Use _child in projection. Cascade delete optional."},
    {"layer": "rap_cds", "topic": "projection", "content": "CDS projection: @AccessControl.authorizationCheck. Expose fields. Use for consumption view."},
    # rap_behavior (3)
    {"layer": "rap_behavior", "topic": "bdef", "content": "Behavior definition: managed, draft. Define create, update, delete. Actions, validations, determinations."},
    {"layer": "rap_behavior", "topic": "draft", "content": "RAP draft: draft enable. %is_draft. Activate draft in behavior definition. Draft instance in buffer."},
    {"layer": "rap_behavior", "topic": "projection_bdef", "content": "Projection behavior: Extend root. Add actions. Use for service-specific behavior."},
    # rap_architecture (2)
    {"layer": "rap_architecture", "topic": "layers", "content": "RAP 3-layer: Data model (CDS), Business logic (behavior), Service (service definition). Clean separation."},
    {"layer": "rap_architecture", "topic": "managed_unmanaged", "content": "Managed vs unmanaged: Managed = framework handles save. Unmanaged = custom save implementation."},
    # rap_eml (2)
    {"layer": "rap_eml", "topic": "internal", "content": "EML internal: MODIFY ENTITY, READ ENTITY, DELETE ENTITY. Use in behavior implementation. Transaction scope."},
    {"layer": "rap_eml", "topic": "external", "content": "EML external: Use in ABAP code. COMMIT ENTITIES. Response with failed, reported. For OData use service."},
    # rap_auth (2)
    {"layer": "rap_auth", "topic": "global", "content": "RAP authorization: @AccessControl.authorizationCheck. Implement I_* in DCL. Check in handler."},
    {"layer": "rap_auth", "topic": "instance", "content": "RAP instance auth: Check in read/update. Use %key. Return 403 for unauthorized."},
    # rap_service (1)
    {"layer": "rap_service", "topic": "binding", "content": "RAP service: Service definition exposes CDS. Binding: OData V4 UI, OData V4 Web API. Publish to Fiori."},
]


def main() -> int:
    """CLI entry point."""
    client = get_chroma_client()
    collection = get_collection(client)

    ids = []
    documents = []
    metadatas = []
    for i, item in enumerate(RAP_KNOWLEDGE):
        raw = f"{item['layer']}:{item['topic']}:{i}"
        chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:24]
        ids.append(chunk_id)
        documents.append(item["content"])
        metadatas.append({"layer": item["layer"], "topic": item["topic"]})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Indexed %d RAP knowledge chunks", len(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
