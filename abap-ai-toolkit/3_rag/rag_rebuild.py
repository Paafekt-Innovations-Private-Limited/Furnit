#!/usr/bin/env python3
"""
Rebuild ChromaDB from rag_export.json.

Reads data/rag_export.json and rebuilds the ChromaDB at data/vector_db/.
Handles empty or missing export gracefully. Prints chunk count per layer
after rebuild.
"""

import json
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

from _common import (
    RAG_EXPORT_PATH,
    get_chroma_client,
    get_collection,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def rebuild_chromadb() -> dict[str, int]:
    """
    Rebuild ChromaDB from rag_export.json.

    Returns:
        Dict mapping layer name to chunk count.
    """
    if not RAG_EXPORT_PATH.exists():
        logger.warning("Export file not found: %s. Creating empty ChromaDB.", RAG_EXPORT_PATH)
        client = get_chroma_client()
        collection = get_collection(client)
        return {}

    with open(RAG_EXPORT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if not ids:
        logger.info("Export is empty. Creating empty ChromaDB.")
        client = get_chroma_client()
        collection = get_collection(client)
        return {}

    # Ensure lists are same length
    n = len(ids)
    documents = documents[:n] if documents else [""] * n
    metadatas = metadatas[:n] if metadatas else [{}] * n

    client = get_chroma_client()
    collection = get_collection(client)

    # Clear existing and re-add (ChromaDB doesn't have clear, so we delete and recreate)
    try:
        client.delete_collection("abap_rag")
    except Exception:
        pass
    collection = get_collection(client)

    # Add in batches to avoid memory issues
    batch_size = 100
    for i in range(0, n, batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
        )

    # Count per layer
    layer_counts: dict[str, int] = {}
    for meta in metadatas:
        layer = (meta or {}).get("layer", "unknown")
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    return layer_counts


def main() -> int:
    """CLI entry point."""
    try:
        layer_counts = rebuild_chromadb()
        print("\nChunk count per layer after rebuild:")
        for layer in sorted(layer_counts.keys()):
            print(f"  {layer}: {layer_counts[layer]}")
        total = sum(layer_counts.values())
        print(f"\nTotal: {total} chunks")
        return 0
    except Exception as e:
        logger.exception("Rebuild failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
