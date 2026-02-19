#!/usr/bin/env python3
"""
Append-only knowledge addition to the RAG vector database.

Appends new chunks to existing ChromaDB without rebuilding. Supports
text files and directories. Use --layer to specify the knowledge layer.

Usage:
    python 3_rag/rag_append.py --layer my_layer --file path/to/file.txt
    python 3_rag/rag_append.py --layer docs --file path/to/directory/
"""

import argparse
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

from _common import (
    get_chroma_client,
    get_collection,
    get_chunk_size,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ABAP_EXTENSIONS = {".abap", ".clas.abap", ".intf.abap", ".prog.abap", ".incl.abap"}
TEXT_EXTENSIONS = {".txt", ".md", ".json"} | ABAP_EXTENSIONS


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most max_chars characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        if end < len(text):
            # Try to break at newline
            last_newline = chunk.rfind("\n")
            if last_newline > max_chars // 2:
                end = start + last_newline + 1
                chunk = text[start:end]
        chunks.append(chunk)
        start = len(chunk) + start if chunk else end
    return chunks


def collect_content_from_path(path: Path, layer: str) -> list[tuple[str, dict]]:
    """
    Collect (content, metadata) from file or directory.

    Returns:
        List of (content, metadata) tuples.
    """
    results = []
    max_chars = get_chunk_size()

    if path.is_file():
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            logger.warning("Skipping unsupported file: %s", path)
            return results
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("Could not read %s: %s", path, e)
            return results
        for i, chunk in enumerate(chunk_text(content, max_chars)):
            results.append((chunk, {"layer": layer, "source_file": str(path.name)}))
    else:
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in TEXT_EXTENSIONS:
                try:
                    content = child.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    logger.warning("Could not read %s: %s", child, e)
                    continue
                rel_path = str(child.relative_to(path))
                for i, chunk in enumerate(chunk_text(content, max_chars)):
                    results.append((chunk, {"layer": layer, "source_file": rel_path}))

    return results


def make_id(layer: str, source: str, content: str, index: int) -> str:
    """Generate unique ID for a chunk."""
    raw = f"{layer}:{source}:{index}:{content[:100]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Append new knowledge to the RAG vector database."
    )
    parser.add_argument("--layer", required=True, help="Knowledge layer name")
    parser.add_argument("--file", required=True, help="File or directory path")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        logger.error("Path not found: %s", path)
        return 1

    items = collect_content_from_path(path, args.layer)
    if not items:
        logger.warning("No content collected from %s", path)
        return 0

    client = get_chroma_client()
    collection = get_collection(client)

    ids = []
    documents = []
    metadatas = []
    for i, (content, meta) in enumerate(items):
        chunk_id = make_id(args.layer, meta.get("source_file", ""), content, i)
        ids.append(chunk_id)
        documents.append(content)
        metadatas.append(meta)

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Appended %d chunks to layer '%s'", len(ids), args.layer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
