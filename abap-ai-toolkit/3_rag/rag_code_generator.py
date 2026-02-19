#!/usr/bin/env python3
"""
RAG-powered ABAP code generator.

CLI: python 3_rag/rag_code_generator.py "Generate a RAP BO for sales orders"
Queries RAG, constructs prompt with examples, calls LLM. Outputs generated ABAP code.
"""

import argparse
import logging
import os
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

from _common import get_chroma_client, get_collection, get_top_k, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def query_rag(prompt_text: str, top_k: int = 10) -> str:
    """Query RAG for relevant examples to include in generation prompt."""
    client = get_chroma_client()
    collection = get_collection(client)
    results = collection.query(
        query_texts=[prompt_text],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas"],
    )
    if not results["documents"] or not results["documents"][0]:
        return ""
    context_parts = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] or {}
        layer = meta.get("layer", "unknown")
        context_parts.append(f"[{layer}]\n{doc}")
    return "\n\n---\n\n".join(context_parts)


def generate_code(prompt_text: str, top_k: int = 10) -> str:
    """Generate ABAP code using RAG context + LLM."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Add to .env for code generation.")

    context = query_rag(prompt_text, top_k)
    config = load_config()
    model = config.get("llm", {}).get("model", "claude-sonnet-4-20250514")
    max_tokens = config.get("llm", {}).get("max_tokens", 4096)
    temperature = config.get("llm", {}).get("temperature", 0.1)

    full_prompt = f"""Generate SAP ABAP code based on this request: {prompt_text}

Use these RAG-retrieved examples and patterns as reference:
{context if context else "(No RAG context - use SAP ABAP Cloud / RAP best practices)"}

Output only the ABAP code with brief comments. Use modern syntax and RAP where appropriate."""

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": full_prompt}],
    )
    return response.content[0].text


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG-powered ABAP code generator."
    )
    parser.add_argument("prompt", help="Natural language description of code to generate")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--top-k", type=int, default=None, help="RAG context chunks")
    args = parser.parse_args()

    top_k = args.top_k or get_top_k()

    try:
        result = generate_code(args.prompt, top_k=top_k)
        if args.output:
            Path(args.output).write_text(result, encoding="utf-8")
            logger.info("Wrote output to %s", args.output)
        else:
            print(result)
        return 0
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
