#!/usr/bin/env python3
"""
Master indexer that runs all individual indexers.

Runs: index_zsample_all, index_fiori_knowledge, index_rap_knowledge,
sap_api_catalog. Reports total chunks after indexing.

Usage:
    python 3_rag/index_all_knowledge.py
"""

import logging
import subprocess
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

INDEXERS = [
    "index_zsample_all.py",
    "index_fiori_knowledge.py",
    "index_rap_knowledge.py",
    "sap_api_catalog.py",
]


def main() -> int:
    """CLI entry point."""
    # Clear existing collection so we don't duplicate on re-run
    try:
        client = get_chroma_client()
        client.delete_collection("abap_rag")
        logger.info("Cleared existing RAG collection")
    except Exception:
        pass

    for script in INDEXERS:
        script_path = SCRIPT_DIR / script
        if not script_path.exists():
            logger.warning("Indexer not found: %s", script_path)
            continue
        logger.info("Running %s...", script)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(TOOLKIT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("%s failed: %s", script, result.stderr or result.stdout)
        else:
            logger.info("%s completed", script)

    # Report total
    client = get_chroma_client()
    collection = get_collection(client)
    total = collection.count()
    logger.info("Total chunks after indexing: %d", total)
    print(f"\nTotal chunks: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
