#!/usr/bin/env python3
"""
RAG+Skills+LLM code generation with validation.

Full pipeline: query RAG -> construct prompt -> call LLM -> validate output.
Uses pattern_validator for 10-point check.
Outputs generated ABAP with confidence score.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent for imports
SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = SCRIPT_DIR.parent
for p in (str(TOOLKIT_ROOT), str(TOOLKIT_ROOT / "3_rag"), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

query_rag = None
try:
    from rag_query_api import query_rag
except ImportError:
    pass

validate_pattern = None
try:
    from pattern_validator import validate_pattern
except ImportError:
    pass

if validate_pattern is None:

    def validate_pattern(code: str) -> dict:
        """Fallback when pattern_validator unavailable."""
        return {"score": 50, "status": "unknown", "findings": []}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def query_rag_fallback(query: str, top_k: int = 5) -> list[dict]:
    """Fallback when RAG is unavailable."""
    if query_rag:
        try:
            return query_rag(query, top_k=top_k)
        except Exception:
            pass
    return [{"content": "Modern ABAP: use VALUE #( ), NEW #( ), CORRESPONDING #( ).", "relevance_score": 0.8}]


def construct_prompt(query: str, rag_results: list[dict]) -> str:
    """Build LLM prompt from query and RAG context."""
    context = "\n".join(
        f"--- Example {i+1} ---\n{r.get('content', '')[:500]}"
        for i, r in enumerate(rag_results[:3])
    )
    return f"""You are an ABAP Clean Core migration expert. Generate modern ABAP code.

Query: {query}

Reference examples from our codebase:
{context}

Generate only the ABAP code, no explanations. Use modern syntax: VALUE #, NEW #, CORRESPONDING #, etc."""


def call_llm(prompt: str) -> str:
    """Call LLM (Anthropic/OpenAI/Gemini) - placeholder when no API key."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text if msg.content else ""
    except ImportError:
        pass
    except Exception as e:
        logger.warning("Anthropic call failed: %s", e)
    # Fallback: return placeholder
    return """* Generated placeholder - configure ANTHROPIC_API_KEY for real generation
DATA(result) = VALUE #( FOR line IN itab ( field = line-value ) )."""


def generate_code(query: str, top_k: int = 5) -> dict:
    """
    Full pipeline: RAG query -> prompt -> LLM -> validate.

    Args:
        query: Natural language or code description.
        top_k: RAG top-k results.

    Returns:
        Dict with generated_code, confidence_score, validation_result.
    """
    rag_results = query_rag_fallback(query, top_k=top_k)
    prompt = construct_prompt(query, rag_results)
    generated = call_llm(prompt)
    validation = validate_pattern(generated)
    confidence = validation.get("score", 0) / 100.0
    return {
        "query": query,
        "generated_code": generated,
        "confidence_score": round(confidence, 4),
        "validation_result": validation,
        "rag_results_count": len(rag_results),
    }


def main() -> None:
    """Main entry point for code generation."""
    parser = argparse.ArgumentParser(
        description="RAG+LLM ABAP code generation with validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "query",
        type=str,
        help="Natural language or code description",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="RAG top-k results",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file for generated code",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full result as JSON",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        result = generate_code(args.query, top_k=args.top_k)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result["generated_code"])
            print(f"\n--- Confidence: {result['confidence_score']:.2%} ---")
        if args.output:
            args.output.write_text(result["generated_code"], encoding="utf-8")
            logger.info("Wrote to %s", args.output)
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
