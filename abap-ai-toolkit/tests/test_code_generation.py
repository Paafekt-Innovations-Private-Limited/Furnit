"""
Tests for code generation pipeline.

Tests: RAG query returns results, LLM prompt construction, validation scoring.
Uses pytest fixtures.
"""

import json
import sys
from pathlib import Path

import pytest

# Add toolkit root and agent/rag paths first (before any local imports)
TESTS_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = TESTS_DIR.parent
for path in (str(TOOLKIT_ROOT / "6_agent"), str(TOOLKIT_ROOT), str(TOOLKIT_ROOT / "3_rag")):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture
def sample_abap_code() -> str:
    """Sample ABAP code for validation tests."""
    return """
CLASS z_test_class DEFINITION PUBLIC.
  PUBLIC SECTION.
    METHODS process_data IMPORTING it_data TYPE table.
ENDCLASS.

CLASS z_test_class IMPLEMENTATION.
  METHOD process_data.
    DATA(result) = VALUE #( FOR line IN it_data ( field = line-value ) ).
  ENDMETHOD.
ENDCLASS.
"""


@pytest.fixture
def legacy_abap_code() -> str:
    """Legacy ABAP for modernization tests."""
    return """
    MOVE var1 TO var2.
    ADD 1 TO counter.
    CONCATENATE a b INTO result.
"""


class TestPatternValidator:
    """Tests for pattern_validator module."""

    def test_validate_returns_dict(self, sample_abap_code: str) -> None:
        """Validation returns dict with score and findings."""
        from pattern_validator import validate_pattern
        result = validate_pattern(sample_abap_code)
        assert isinstance(result, dict)
        assert "score" in result
        assert "findings" in result
        assert "status" in result

    def test_validate_score_range(self, sample_abap_code: str) -> None:
        """Score is between 0 and 100."""
        from pattern_validator import validate_pattern
        result = validate_pattern(sample_abap_code)
        assert 0 <= result["score"] <= 100

    def test_validate_empty_code(self) -> None:
        """Empty code returns score 0."""
        from pattern_validator import validate_pattern
        result = validate_pattern("")
        assert result["score"] == 0
        assert result["status"] == "empty"

    def test_validate_findings_structure(self, sample_abap_code: str) -> None:
        """Findings have expected structure."""
        from pattern_validator import validate_pattern
        result = validate_pattern(sample_abap_code)
        for f in result.get("findings", []):
            assert "id" in f or "name" in f


class TestCodeGenerator:
    """Tests for code_generator module."""

    def test_construct_prompt_contains_query(self) -> None:
        """Prompt construction includes query."""
        from code_generator import construct_prompt
        prompt = construct_prompt("post FI document", [{"content": "example"}])
        assert "post FI document" in prompt
        assert "example" in prompt

    def test_generate_code_returns_dict(self) -> None:
        """generate_code returns dict with expected keys."""
        from code_generator import generate_code
        result = generate_code("create internal table from loop", top_k=2)
        assert isinstance(result, dict)
        assert "generated_code" in result
        assert "confidence_score" in result
        assert "validation_result" in result

    def test_confidence_in_range(self) -> None:
        """Confidence score is between 0 and 1."""
        from code_generator import generate_code
        result = generate_code("VALUE #( )", top_k=1)
        assert 0 <= result["confidence_score"] <= 1


class TestRagQuery:
    """Tests for RAG query (when available)."""

    def test_rag_query_fallback_returns_list(self) -> None:
        """RAG fallback returns list of results."""
        from code_generator import query_rag_fallback
        results = query_rag_fallback("BAPI_ACC_DOCUMENT_POST", top_k=3)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert "content" in results[0] or "relevance_score" in results[0]
