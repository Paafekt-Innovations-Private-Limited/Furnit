"""Query the RAG: retrieve relevant chunks and optionally generate an answer."""
from pathlib import Path

from shanjeev_rag.config import PROJECT_ROOT, get_path, load_config


def get_collection():
    """Return ChromaDB collection (requires vector_db to exist)."""
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    config = load_config()
    vector_db_path = get_path("vector_db_path")
    collection_name = config.get("collection_name", "insurance_rag")
    model_name = config.get("embedding_model", "all-MiniLM-L6-v2")

    if not vector_db_path.exists():
        raise FileNotFoundError(
            f"Vector DB not found at {vector_db_path}. Run 'python -m shanjeev_rag index' first."
        )

    client = chromadb.PersistentClient(
        path=str(vector_db_path),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=collection_name)
    model = SentenceTransformer(model_name)
    return collection, model, config


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """Return list of dicts with keys: document, source, chunk."""
    collection, model, config = get_collection()
    k = top_k or config.get("top_k", 5)
    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas"],
    )
    out = []
    if results["documents"] and results["documents"][0]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            out.append({
                "document": doc,
                "source": meta.get("source", ""),
                "chunk": meta.get("chunk", 0),
            })
    return out


def run_query(question: str, verbose: bool = True) -> str:
    """Retrieve chunks and optionally generate answer. Returns a string for display."""
    chunks = retrieve(question)
    if not chunks:
        return "No relevant chunks found. Have you run 'python -m shanjeev_rag index' and added documents to data/raw/?"

    lines = []
    if verbose:
        lines.append("--- Retrieved chunks ---")
        for i, c in enumerate(chunks, 1):
            lines.append(f"\n[{i}] Source: {c['source']} (chunk {c['chunk']})")
            lines.append(c["document"][:800] + ("..." if len(c["document"]) > 800 else ""))
        lines.append("\n--- End retrieved ---\n")

    # Optional: call an LLM to generate an answer from chunks (if OPENAI_API_KEY etc. set)
    context = "\n\n".join(c["document"] for c in chunks)
    answer = _generate_answer(question, context)
    if answer:
        lines.append("Answer:\n" + answer)
    else:
        lines.append("(No LLM configured; showing retrieved chunks only. Set OPENAI_API_KEY or similar to get generated answers.)")

    return "\n".join(lines)


def _generate_answer(question: str, context: str) -> str:
    """If env has an LLM API key, generate an answer; else return empty string."""
    import os
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are an insurance domain assistant. Answer only from the provided context. If the context does not contain the answer, say so."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                max_tokens=500,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            return f"[LLM error: {e}]"
    return ""
