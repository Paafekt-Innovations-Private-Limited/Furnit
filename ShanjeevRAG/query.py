#!/usr/bin/env python3
"""
ShanjeevRAG - Query the insurance RAG.

Usage:
  python query.py "What is covered under home insurance?"
  python query.py "Explain excess in motor insurance" --top-k 5
  python query.py "Solvency II capital requirements" --llm openai       # Answer via OpenAI (OPENAI_API_KEY)
  python query.py "Solvency II capital requirements" --llm perplexity   # Answer via Perplexity (PERPLEXITY_API_KEY)
"""
import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config():
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_paths(config):
    vector_db_path = PROJECT_ROOT / config.get("vector_db_path", "vector_db")
    return vector_db_path


def run_query(
    question: str,
    top_k: int = 5,
    use_llm: str | None = None,
    perplexity_model: str | None = None,
) -> int:
    config = load_config()
    vector_db_path = get_paths(config)
    collection_name = config.get("collection_name", "insurance")
    embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")

    if not vector_db_path.exists():
        print("Vector DB not found. Run: python ingest.py")
        return 1

    from sentence_transformers import SentenceTransformer
    import chromadb

    model = SentenceTransformer(embedding_model_name)
    client = chromadb.PersistentClient(path=str(vector_db_path))
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        print(f"Collection '{collection_name}' not found. Run: python ingest.py")
        return 1

    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k, include=["documents", "metadatas"])

    docs = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    if not docs:
        print("No relevant chunks found.")
        return 0

    context = "\n\n---\n\n".join(docs)
    print("Retrieved chunks:\n")
    for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
        src = meta.get("source", "?")
        print(f"[{i}] (source: {src})\n{doc}\n")
    print()

    if use_llm == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Set OPENAI_API_KEY in .env for LLM answers.")
            return 0
        try:
            from openai import OpenAI
            client_llm = OpenAI(api_key=api_key)
            response = client_llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an insurance domain expert. Answer based only on the provided context. If the context does not contain the answer, say so."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
                ],
                max_tokens=500,
            )
            answer = response.choices[0].message.content
            print("Generated answer (OpenAI):\n", answer)
        except Exception as e:
            print("LLM error:", e)
    elif use_llm == "perplexity":
        api_key = os.environ.get("PERPLEXITY_API_KEY")
        if not api_key:
            print("Set PERPLEXITY_API_KEY in .env for Perplexity answers.")
            return 0
        model_name = perplexity_model or config.get("perplexity_model", "sonar")
        try:
            from openai import OpenAI
            client_llm = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
            response = client_llm.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an insurance domain expert. Answer based only on the provided context. If the context does not contain the answer, say so."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
                ],
                max_tokens=500,
                extra_body={"disable_search": True},
            )
            answer = response.choices[0].message.content
            print("Generated answer (Perplexity):\n", answer)
        except Exception as e:
            print("LLM error:", e)
    elif use_llm:
        print(f"Unknown --llm: {use_llm}. Use 'openai', 'perplexity', or omit for retrieval-only.")

    return 0


def main():
    parser = argparse.ArgumentParser(description="ShanjeevRAG query: ask the insurance RAG.")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default 5)")
    parser.add_argument("--llm", type=str, default=None, help="Use LLM for answer: e.g. openai, perplexity")
    parser.add_argument("--perplexity-model", type=str, default=None, help="Perplexity model when --llm perplexity (default from config: sonar, or sonar-pro, sonar-deep-research, sonar-reasoning-pro)")
    args = parser.parse_args()
    return run_query(question=args.question, top_k=args.top_k, use_llm=args.llm, perplexity_model=args.perplexity_model)


if __name__ == "__main__":
    raise SystemExit(main())
