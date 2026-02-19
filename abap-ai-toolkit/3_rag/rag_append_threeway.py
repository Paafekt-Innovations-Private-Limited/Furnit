#!/usr/bin/env python3
"""
Append 3-way mappings to the RAG vector database.

Reads JSON files from data/threeway_mappings/ and indexes each mapping as a
chunk with metadata (source_file, rule, java_anchor).
"""

import hashlib
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

from _common import get_chroma_client, get_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

THREEWAY_MAPPINGS_DIR = TOOLKIT_ROOT / "data" / "threeway_mappings"


def load_threeway_mappings() -> list[tuple[str, dict]]:
    """
    Load all 3-way mapping JSON files and yield (content, metadata) for each mapping.

    Returns:
        List of (content, metadata) tuples.
    """
    results = []
    if not THREEWAY_MAPPINGS_DIR.exists():
        logger.warning("Threeway mappings dir not found: %s", THREEWAY_MAPPINGS_DIR)
        return results

    for json_path in THREEWAY_MAPPINGS_DIR.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read %s: %s", json_path, e)
            continue

        # Handle both single mapping and list of mappings
        mappings = data if isinstance(data, list) else [data]
        source_file = json_path.name

        for m in mappings:
            if not isinstance(m, dict):
                continue
            java_anchor = m.get("java_anchor", m.get("java", ""))
            rule = m.get("rule", m.get("rule_id", ""))
            old_abap = m.get("old_abap", m.get("legacy", ""))
            new_abap = m.get("new_abap", m.get("modern", ""))
            content = f"Java: {java_anchor}\nOld ABAP: {old_abap}\nNew ABAP: {new_abap}"
            metadata = {
                "layer": "threeway_mappings",
                "source_file": source_file,
                "rule": str(rule),
                "java_anchor": str(java_anchor),
            }
            results.append((content, metadata))

    return results


def main() -> int:
    """CLI entry point."""
    items = load_threeway_mappings()
    if not items:
        logger.info("No 3-way mappings found to index.")
        return 0

    client = get_chroma_client()
    collection = get_collection(client)

    ids = []
    documents = []
    metadatas = []
    for i, (content, meta) in enumerate(items):
        raw = f"threeway:{meta.get('source_file','')}:{i}:{content[:80]}"
        chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:24]
        ids.append(chunk_id)
        documents.append(content)
        metadatas.append(meta)

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Indexed %d 3-way mapping chunks", len(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
