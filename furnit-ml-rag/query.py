#!/usr/bin/env python3
"""
Furnit ML RAG - Query the knowledge base.

Usage:
  python furnit-ml-rag/query.py "How to enable GPU for ViT on Android?"
  python furnit-ml-rag/query.py "ExecuTorch Vulkan INT8 quantization"
  python furnit-ml-rag/query.py --export
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("ERROR: pip install chromadb sentence-transformers")
    sys.exit(1)

DB_PATH = str(Path(__file__).parent / "data" / "vector_db")
EXPORT_PATH = str(Path(__file__).parent / "data" / "rag_export.json")

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name="furnit_ml",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

def get_log_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name="furnit_android_logs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

def query(text, top_k=5, include_logs=False, log_file_path=None, from_adb=False):
    if include_logs and (log_file_path or from_adb):
        from index_all import index_android_logs
        if from_adb:
            index_android_logs(from_adb=True)
        else:
            index_android_logs(log_file_path=log_file_path)

    collection = get_collection()
    results = collection.query(query_texts=[text], n_results=top_k)

    print(f"\n{'='*80}")
    print(f"Query: {text}")
    print(f"{'='*80}\n")

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        relevance = max(0, 1 - dist)
        print(f"--- Result {i+1} [{meta['layer']}] (relevance: {relevance:.2f}) ---")
        print(doc.strip())
        print()

    if include_logs:
        try:
            log_coll = get_log_collection()
            n_logs = log_coll.count()
            if n_logs == 0:
                print("--- Android logs: no log chunks indexed. Use --log-file PATH or --adb-logcat first run. ---\n")
            else:
                log_results = log_coll.query(query_texts=[text], n_results=min(3, n_logs))
                print("--- Android logs (relevant snippets) ---")
                for i, (doc, meta, dist) in enumerate(zip(
                    log_results["documents"][0],
                    log_results["metadatas"][0],
                    log_results["distances"][0]
                )):
                    relevance = max(0, 1 - dist)
                    print(f"[{meta['layer']}] (relevance: {relevance:.2f})")
                    print(doc.strip())
                    print()
        except Exception as e:
            print(f"--- Android logs: {e} ---\n")

def export():
    collection = get_collection()
    data = collection.get()
    export = {"chunks": []}
    for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        export["chunks"].append({"id": id_, "layer": meta["layer"], "content": doc})
    with open(EXPORT_PATH, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Exported {len(export['chunks'])} chunks to {EXPORT_PATH}")

def rebuild():
    """Rebuild ChromaDB from rag_export.json."""
    if not Path(EXPORT_PATH).exists():
        print(f"No export file at {EXPORT_PATH}")
        return
    with open(EXPORT_PATH) as f:
        data = json.load(f)

    collection = get_collection()
    try:
        existing = collection.count()
        if existing > 0:
            collection.delete(ids=[c for c in collection.get()["ids"]])
    except Exception:
        pass

    ids = [c["id"] for c in data["chunks"]]
    docs = [c["content"] for c in data["chunks"]]
    metas = [{"layer": c["layer"]} for c in data["chunks"]]
    collection.add(ids=ids, documents=docs, metadatas=metas)
    print(f"Rebuilt {len(ids)} chunks from export")

def main():
    parser = argparse.ArgumentParser(description="Furnit ML RAG Query")
    parser.add_argument("query_text", nargs="?", help="Query text")
    parser.add_argument("--export", action="store_true", help="Export DB to JSON")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild DB from JSON")
    parser.add_argument("-k", type=int, default=5, help="Top K results")
    parser.add_argument("--log-file", type=str, metavar="PATH", help="Index Android logs from file, then include in query")
    parser.add_argument("--adb-logcat", action="store_true", help="Index Android logs from adb logcat -d, then include in query")
    args = parser.parse_args()

    include_logs = bool(args.log_file or args.adb_logcat)

    if args.export:
        export()
    elif args.rebuild:
        rebuild()
    elif args.query_text:
        query(args.query_text, top_k=args.k, include_logs=include_logs, log_file_path=args.log_file, from_adb=args.adb_logcat)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
