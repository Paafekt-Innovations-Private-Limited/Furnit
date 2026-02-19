#!/usr/bin/env python3
"""
CLI/REST/Python query API for the ABAP AI Toolkit RAG.

The main tool for querying the ChromaDB vector database. Loads ChromaDB from
data/vector_db/, uses SentenceTransformer('all-MiniLM-L6-v2') for embeddings,
and returns top-K most relevant chunks with metadata.

Usage:
    python 3_rag/rag_query_api.py "query text"
    python 3_rag/rag_query_api.py "query" --top-k 5
    python 3_rag/rag_query_api.py --export
"""

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add toolkit root to path
SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from _common import (
    RAG_EXPORT_PATH,
    get_chroma_client,
    get_collection,
    get_top_k,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def query_rag(query_text: str, top_k: int = 10) -> list[dict]:
    """
    Query the RAG vector database for relevant chunks.

    Args:
        query_text: Natural language or code query.
        top_k: Number of top results to return.

    Returns:
        List of dicts with content, metadata, and distance.
    """
    client = get_chroma_client()
    collection = get_collection(client)

    results = collection.query(
        query_texts=[query_text],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = (results["metadatas"][0][i] or {}).copy()
            distance = results["distances"][0][i] if results["distances"] else None
            relevance = 1.0 - (distance or 0) if distance is not None else 1.0
            output.append({
                "content": doc,
                "metadata": metadata,
                "relevance_score": round(relevance, 4),
            })
    return output


def pretty_print_results(results: list[dict]) -> None:
    """Pretty print query results with layer, source file, and relevance score."""
    for i, item in enumerate(results, 1):
        meta = item.get("metadata", {})
        layer = meta.get("layer", "unknown")
        source = meta.get("source_file", meta.get("source", "N/A"))
        score = item.get("relevance_score", 0)
        content_preview = (item.get("content", "") or "")[:200]
        if len(item.get("content", "") or "") > 200:
            content_preview += "..."
        print(f"\n--- Result {i} (score: {score:.4f}) ---")
        print(f"  Layer: {layer}")
        print(f"  Source: {source}")
        print(f"  Content: {content_preview}")


def export_rag() -> dict:
    """
    Export entire ChromaDB to data/rag_export.json.

    Returns:
        Export dict with ids, documents, metadatas.
    """
    client = get_chroma_client()
    collection = get_collection(client)
    count = collection.count()
    if count == 0:
        logger.warning("ChromaDB is empty, exporting empty structure")
        return {"ids": [], "documents": [], "metadatas": []}

    data = collection.get(
        include=["documents", "metadatas"],
    )
    export = {
        "ids": data["ids"],
        "documents": data["documents"],
        "metadatas": data["metadatas"],
    }
    RAG_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAG_EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    logger.info("Exported %d chunks to %s", count, RAG_EXPORT_PATH)
    return export


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Query the ABAP AI Toolkit RAG or export ChromaDB."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query text for semantic search",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results (default from config)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export entire ChromaDB to data/rag_export.json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    top_k = args.top_k or get_top_k()

    if args.export:
        export_rag()
        return 0

    if not args.query:
        parser.print_help()
        return 1

    try:
        results = query_rag(args.query, top_k=top_k)
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            pretty_print_results(results)
        return 0
    except Exception as e:
        logger.exception("Query failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
