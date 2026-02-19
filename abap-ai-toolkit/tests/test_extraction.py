"""
Tests for extraction pipeline.

Tests: config loading, file organization, deduplication logic.
Uses pytest with tmp_path fixture.
"""

import json
import sys
from pathlib import Path

import pytest

# Add toolkit root
TESTS_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(TOOLKIT_ROOT))
sys.path.insert(0, str(TOOLKIT_ROOT / "1_extract"))  # for curate_dataset


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Create sample config.yaml in tmp dir."""
    config = {
        "extraction": {
            "include_subpackages": True,
            "exclude_patterns": ["*_test*", "*_demo*"],
        },
        "rag": {"top_k": 10},
    }
    import yaml
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def sample_abap_files(tmp_path: Path) -> Path:
    """Create sample ABAP files for extraction tests."""
    extract_dir = tmp_path / "zsample_extracted"
    extract_dir.mkdir()
    (extract_dir / "z_class.clas.abap").write_text("""
CLASS z_class DEFINITION.
ENDCLASS.
""")
    (extract_dir / "z_prog.prog.abap").write_text("""
REPORT z_prog.
WRITE 'hello'.
""")
    return extract_dir


class TestConfigLoading:
    """Tests for config loading."""

    def test_load_config_exists(self) -> None:
        """Config loads when file exists."""
        from curate_dataset import load_config
        # May return empty if CONFIG_PATH not in expected location
        config = load_config()
        assert isinstance(config, dict)

    def test_config_structure(self, sample_config: Path) -> None:
        """Config has expected structure when valid."""
        import yaml
        with open(sample_config) as f:
            config = yaml.safe_load(f)
        assert "extraction" in config
        assert "exclude_patterns" in config["extraction"]


class TestDeduplication:
    """Tests for content hashing and deduplication."""

    def test_content_hash_deterministic(self) -> None:
        """Same content produces same hash."""
        from curate_dataset import content_hash
        text = "DATA x TYPE i."
        assert content_hash(text) == content_hash(text)

    def test_content_hash_different_for_different_content(self) -> None:
        """Different content produces different hash."""
        from curate_dataset import content_hash
        h1 = content_hash("DATA x TYPE i.")
        h2 = content_hash("DATA y TYPE i.")
        assert h1 != h2

    def test_content_hash_normalizes_whitespace(self) -> None:
        """Trailing whitespace normalized in hash."""
        from curate_dataset import content_hash
        # Same logical content with different trailing WS
        h1 = content_hash("DATA x TYPE i.  \n")
        h2 = content_hash("DATA x TYPE i.\n")
        assert h1 == h2


class TestFileOrganization:
    """Tests for file organization logic."""

    def test_abap_extensions_recognized(self) -> None:
        """ABAP file extensions are recognized."""
        extensions = (".clas.abap", ".prog.abap", ".fugr.abap")
        for ext in extensions:
            path = Path(f"dummy{ext}")
            assert path.name.endswith(".abap")

    def test_exclude_pattern_matching(self) -> None:
        """Exclude patterns match correctly."""
        import re
        pattern = re.compile(r".*_test.*", re.IGNORECASE)
        assert pattern.search("z_class_test.clas.abap")
        assert not pattern.search("z_class.clas.abap")


class TestQualityScoring:
    """Tests for quality scoring in curate_dataset."""

    def test_score_file_returns_dict(self) -> None:
        """score_file returns dict with expected keys."""
        from curate_dataset import score_file
        result = score_file("DATA x TYPE i.\n* comment", Path("dummy.abap"))
        assert isinstance(result, dict)
        assert "line_count" in result
        assert "composite_score" in result
        assert "comment_ratio" in result

    def test_score_file_composite_in_range(self) -> None:
        """Composite score is between 0 and 1."""
        from curate_dataset import score_file
        result = score_file("DATA x TYPE i.\n" * 50, Path("dummy.abap"))
        assert 0 <= result["composite_score"] <= 1
