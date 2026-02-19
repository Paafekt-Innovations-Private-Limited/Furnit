"""
Shared utilities for 3_rag scripts.

Provides config loading, ChromaDB setup, embedding model initialization,
and path resolution for the ABAP AI Toolkit RAG pipeline.
"""

import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import yaml

# Resolve paths relative to abap-ai-toolkit root
SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = SCRIPT_DIR.parent
DATA_DIR = TOOLKIT_ROOT / "data"
VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", str(DATA_DIR / "vector_db")))
RAG_EXPORT_PATH = DATA_DIR / "rag_export.json"
CONFIG_PATH = TOOLKIT_ROOT / "config.yaml"

# Default embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def load_config() -> dict:
    """Load config.yaml with environment variable substitution."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        content = f.read()
    # Simple ${VAR} substitution
    for key, val in os.environ.items():
        content = content.replace(f"${{{key}}}", str(val))
    return yaml.safe_load(content) or {}


def get_vector_db_path() -> Path:
    """Return vector DB path from config or env."""
    config = load_config()
    path = config.get("rag", {}).get("vector_db_path")
    if path:
        return TOOLKIT_ROOT / path
    return VECTOR_DB_PATH


def get_embedding_model_name() -> str:
    """Return embedding model from config or env."""
    config = load_config()
    return config.get("rag", {}).get("embedding_model") or EMBEDDING_MODEL


def get_chunk_size() -> int:
    """Return chunk size from config."""
    config = load_config()
    return config.get("rag", {}).get("chunk_size", 2000)


def get_top_k() -> int:
    """Return default top_k from config."""
    config = load_config()
    return config.get("rag", {}).get("top_k", 10)


def get_chroma_client():
    """Return ChromaDB PersistentClient."""
    import chromadb
    path = str(get_vector_db_path())
    path_dir = Path(path)
    path_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def get_embedding_function():
    """Return ChromaDB SentenceTransformer embedding function."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    model_name = get_embedding_model_name()
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_collection(client, collection_name: str = "abap_rag"):
    """Get or create ChromaDB collection with embedding function."""
    embedding_fn = get_embedding_function()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
