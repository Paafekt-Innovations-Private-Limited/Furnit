#!/usr/bin/env python3
"""
LLM+RAG clean core conversion engine.

Takes legacy ABAP code as input, queries RAG for relevant examples and rules,
calls LLM (Claude via anthropic SDK) to generate Clean Core version, and
applies post-processing validation.
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


def query_rag_for_context(legacy_code: str, top_k: int = 10) -> str:
    """Query RAG for relevant examples and rules to include in prompt."""
    client = get_chroma_client()
    collection = get_collection(client)
    results = collection.query(
        query_texts=[legacy_code[:2000]],
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


def call_llm(prompt: str) -> str:
    """Call Claude via anthropic SDK to generate Clean Core code."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Add to .env for LLM conversion.")

    config = load_config()
    model = config.get("llm", {}).get("model", "claude-sonnet-4-20250514")
    max_tokens = config.get("llm", {}).get("max_tokens", 4096)
    temperature = config.get("llm", {}).get("temperature", 0.1)

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def validate_output(code: str) -> tuple[bool, list[str]]:
    """Post-processing validation: basic pattern checks."""
    issues = []
    if "CALL FUNCTION" in code and "BAPI_" in code:
        issues.append("Contains legacy BAPI call - consider RAP BO replacement")
    if "MOVE " in code and " TO " in code:
        issues.append("Contains MOVE - use assignment = instead")
    if "CONCATENATE" in code:
        issues.append("Contains CONCATENATE - use string template |{ }|")
    return len(issues) == 0, issues


def convert_to_clean_core(legacy_code: str, top_k: int = 10) -> str:
    """
    Convert legacy ABAP to Clean Core using RAG + LLM.

    Args:
        legacy_code: Legacy ABAP source code.
        top_k: Number of RAG chunks for context.

    Returns:
        Generated Clean Core ABAP code.
    """
    context = query_rag_for_context(legacy_code, top_k)
    prompt = f"""Convert the following legacy ABAP code to SAP Clean Core / ABAP Cloud style.

Use these RAG-retrieved examples and rules as reference:
{context}

Legacy code to convert:
```
{legacy_code}
```

Output only the converted ABAP code, with brief comments for major changes. Use modern syntax: assignments, string templates, RAP BOs where applicable, and avoid obsolete statements."""

    output = call_llm(prompt)
    is_valid, issues = validate_output(output)
    if issues:
        logger.warning("Validation issues: %s", issues)
    return output


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert legacy ABAP to Clean Core using RAG + LLM."
    )
    parser.add_argument("input", help="Path to ABAP file or '-' for stdin")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--top-k", type=int, default=None, help="RAG context chunks")
    args = parser.parse_args()

    top_k = args.top_k or get_top_k()

    if args.input == "-":
        legacy_code = sys.stdin.read()
    else:
        path = Path(args.input)
        if not path.exists():
            logger.error("File not found: %s", path)
            return 1
        legacy_code = path.read_text(encoding="utf-8", errors="replace")

    try:
        result = convert_to_clean_core(legacy_code, top_k=top_k)
        if args.output:
            Path(args.output).write_text(result, encoding="utf-8")
            logger.info("Wrote output to %s", args.output)
        else:
            print(result)
        return 0
    except Exception as e:
        logger.exception("Conversion failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
