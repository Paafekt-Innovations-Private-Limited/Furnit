#!/usr/bin/env python3
"""
AbapBLBrain MCP server for Cursor IDE.

Implements MCP (Model Context Protocol) server.
Exposes tools: rag_query, pattern_validate, twin_search, syntax_modernize.
Runs as: python 7_abap_bl_brain/mcp_server.py
Uses stdin/stdout JSON-RPC protocol.
"""

import json
import logging
import sys
from pathlib import Path

# Add toolkit root for imports
SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(TOOLKIT_ROOT))

# Configure logging to stderr (stdout is for JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    logger.warning("mcp package not installed. pip install mcp")


def _rag_query_impl(query: str, top_k: int = 5) -> str:
    """Query RAG vector database."""
    try:
        sys.path.insert(0, str(TOOLKIT_ROOT / "3_rag"))
        from rag_query_api import query_rag
        results = query_rag(query, top_k=top_k)
        return json.dumps([{"content": r.get("content", ""), "score": r.get("relevance_score", 0)} for r in results])
    except Exception as e:
        return json.dumps({"error": str(e), "fallback": "RAG unavailable - check data/vector_db"})


def _pattern_validate_impl(code: str) -> str:
    """Run 10-point pattern validation on ABAP code."""
    try:
        sys.path.insert(0, str(TOOLKIT_ROOT / "6_agent"))
        from pattern_validator import validate_pattern
        result = validate_pattern(code)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e), "score": 0})


def _twin_search_impl(fm_name: str) -> str:
    """Search for ECC->Clean Core twin mapping."""
    try:
        sys.path.insert(0, str(TOOLKIT_ROOT / "4_twin_discovery"))
        from match_twins import match_twins
        result = match_twins(fm=fm_name)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _syntax_modernize_impl(code: str) -> str:
    """Apply ABAP syntax modernization rules."""
    try:
        sys.path.insert(0, str(TOOLKIT_ROOT / "2_map_skills"))
        from abap_syntax_modernizer import AbapSyntaxModernizer
        modernizer = AbapSyntaxModernizer()
        transformed, _ = modernizer.transform_text(code)
        return transformed
    except Exception as e:
        return json.dumps({"error": str(e)})


def run_mcp_server() -> None:
    """Run MCP server with FastMCP (stdio transport)."""
    if not HAS_MCP:
        logger.error("MCP package required. pip install mcp")
        sys.exit(1)

    mcp = FastMCP("AbapBLBrain", json_response=True)

    @mcp.tool()
    def rag_query(query: str, top_k: int = 5) -> str:
        """Query the ABAP RAG knowledge base for relevant code examples and documentation."""
        return _rag_query_impl(query, top_k)

    @mcp.tool()
    def pattern_validate(code: str) -> str:
        """Validate ABAP code against 10-point pattern checklist (syntax, naming, RAP, Clean Core, etc.)."""
        return _pattern_validate_impl(code)

    @mcp.tool()
    def twin_search(fm_name: str) -> str:
        """Search for ECC function module to S/4HANA Clean Core twin mapping."""
        return _twin_search_impl(fm_name)

    @mcp.tool()
    def syntax_modernize(code: str) -> str:
        """Apply ABAP syntax modernization rules (MOVE->assign, CONCATENATE->template, etc.)."""
        return _syntax_modernize_impl(code)

    # Run with stdio (default for Cursor IDE - stdin/stdout JSON-RPC)
    mcp.run()


def main() -> None:
    """Entry point. Run: python 7_abap_bl_brain/mcp_server.py"""
    run_mcp_server()


if __name__ == "__main__":
    main()
