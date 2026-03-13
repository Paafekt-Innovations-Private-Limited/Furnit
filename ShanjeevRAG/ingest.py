#!/usr/bin/env python3
"""
ShanjeevRAG - Ingest documents into the insurance RAG.

Usage:
  python ingest.py                    # Ingest all files from data/
  python ingest.py path/to/doc.pdf   # Ingest specific file(s)
  python ingest.py --reset            # Clear vector DB and re-ingest from data/

Add new insurance documents to data/ and run ingest.py again to train iteratively.
"""
import argparse
import re
from pathlib import Path

import yaml

# Project root (parent of script)
PROJECT_ROOT = Path(__file__).resolve().parent


def load_config():
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_paths(config):
    data_dir = PROJECT_ROOT / config.get("data_dir", "data")
    vector_db_path = PROJECT_ROOT / config.get("vector_db_path", "vector_db")
    return data_dir, vector_db_path


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if not text or not text.strip():
        return []
    text = text.replace("\r\n", "\n").strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            # Try to break at sentence or newline
            last_break = max(
                chunk.rfind(". "),
                chunk.rfind("\n"),
                chunk.rfind("; "),
            )
            if last_break > chunk_size // 2:
                chunk = chunk[: last_break + 1]
                end = start + last_break + 1
        chunks.append(chunk.strip())
        start = end - overlap if end < len(text) else len(text)
    return [c for c in chunks if c]


def load_document(path: Path) -> str | None:
    """Load text from PDF, DOCX, or TXT."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            import pypdf
            reader = pypdf.PdfReader(path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        if suffix in (".docx", ".doc"):
            from docx import Document as DocxDocument
            doc = DocxDocument(path)
            return "\n".join(p.text for p in doc.paragraphs)
        if suffix == ".txt" or suffix == ".md":
            return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"Warning: could not load {path}: {e}")
    return None


def collect_files(data_dir: Path, extra_paths: list[Path]) -> list[Path]:
    """Gather all supported files from data_dir and extra_paths."""
    supported = {".pdf", ".docx", ".doc", ".txt", ".md"}
    files = []
    if data_dir.exists():
        for f in data_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in supported:
                files.append(f)
    for p in extra_paths:
        if p.is_file() and p.suffix.lower() in supported:
            files.append(p.resolve())
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in supported:
                    files.append(f)
    return list(dict.fromkeys(files))  # dedupe


def run_ingest(reset: bool = False, extra_paths: list[Path] | None = None):
    config = load_config()
    data_dir, vector_db_path = get_paths(config)
    chunk_size = config.get("chunk_size", 500)
    chunk_overlap = config.get("chunk_overlap", 50)
    embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
    collection_name = config.get("collection_name", "insurance")

    extra_paths = extra_paths or []
    files = collect_files(data_dir, extra_paths)
    if not files:
        print("No documents found. Add PDF/DOCX/TXT files to data/ or pass paths.")
        return 1

    print(f"Loading embedding model: {embedding_model_name}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedding_model_name)

    import chromadb
    from chromadb.config import Settings

    vector_db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vector_db_path), settings=Settings(anonymized_telemetry=False))

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"Reset collection: {collection_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name, metadata={"description": "Insurance RAG"})

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for path in files:
        text = load_document(path)
        if not text:
            continue
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        rel = path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in path.parents else path.name
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            all_metadatas.append({"source": str(rel), "chunk_id": i})
            all_ids.append(f"{rel}!{i}")

    if not all_chunks:
        print("No text extracted from documents.")
        return 1

    print(f"Embedding {len(all_chunks)} chunks ...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    # Add in batches (ChromaDB limit)
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=all_ids[i:end],
            embeddings=embeddings[i:end],
            documents=all_chunks[i:end],
            metadatas=all_metadatas[i:end],
        )
    print(f"Ingest complete. {len(all_chunks)} chunks in collection '{collection_name}'.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="ShanjeevRAG ingest: add documents to the insurance RAG.")
    parser.add_argument("paths", nargs="*", type=Path, help="Extra file or folder paths to ingest")
    parser.add_argument("--reset", action="store_true", help="Clear vector DB and re-ingest from data/")
    args = parser.parse_args()
    return run_ingest(reset=args.reset, extra_paths=args.paths)


if __name__ == "__main__":
    raise SystemExit(main())
