#!/usr/bin/env python3
"""
Quality scoring and deduplication for extracted ABAP dataset.

Reads from data/zsample_extracted/, scores files by line count, comment ratio,
and complexity (nested IFs, LOOPs), deduplicates by content hash, and outputs
curated files to data/sample_pkg_original/.
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = TOOLKIT_ROOT / "config.yaml"
ENV_PATH = TOOLKIT_ROOT / ".env"
DATA_DIR = TOOLKIT_ROOT / "data"
INPUT_DIR = DATA_DIR / "zsample_extracted"
OUTPUT_DIR = DATA_DIR / "sample_pkg_original"
PIPELINE_STATE_PATH = DATA_DIR / "pipeline_state.json"

# ABAP comment patterns
ABAP_COMMENT_PATTERN = re.compile(r"^\s*\*|^\s*\"", re.MULTILINE)
# Nested IF/ENDIF, LOOP/ENDLOOP for complexity
ABAP_IF_PATTERN = re.compile(r"\bIF\b", re.IGNORECASE)
ABAP_LOOP_PATTERN = re.compile(r"\bLOOP\b", re.IGNORECASE)


def load_config() -> dict:
    """Load configuration from config.yaml and .env."""
    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", CONFIG_PATH)
    else:
        logger.warning("Config file not found at %s", CONFIG_PATH)

    if load_dotenv and ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    elif load_dotenv:
        load_dotenv()

    return config


def content_hash(content: str) -> str:
    """Compute SHA256 hash of normalized content for deduplication."""
    normalized = "\n".join(line.rstrip() for line in content.splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def count_comments(content: str) -> int:
    """Count ABAP comment lines."""
    return len(ABAP_COMMENT_PATTERN.findall(content))


def count_complexity(content: str) -> int:
    """Count IF and LOOP keywords as proxy for complexity."""
    if_count = len(ABAP_IF_PATTERN.findall(content))
    loop_count = len(ABAP_LOOP_PATTERN.findall(content))
    return if_count + loop_count


def score_file(content: str, path: Path) -> dict:
    """
    Compute quality score for an ABAP file.

    Returns dict with line_count, comment_ratio, complexity, and composite score.
    """
    lines = content.splitlines()
    line_count = len(lines)
    code_lines = [l for l in lines if l.strip() and not ABAP_COMMENT_PATTERN.match(l)]
    code_count = len(code_lines)
    comment_count = count_comments(content)
    comment_ratio = comment_count / line_count if line_count > 0 else 0.0
    complexity = count_complexity(content)

    # Composite score: favor moderate size, some comments, moderate complexity
    # Penalize very small (boilerplate) or very large (god objects)
    size_score = min(1.0, line_count / 50) if line_count > 0 else 0
    comment_score = min(1.0, comment_ratio * 10)  # 10% comments = full score
    complexity_score = min(1.0, complexity / 20)  # 20 IF/LOOP = full score
    composite = (size_score * 0.3 + comment_score * 0.3 + complexity_score * 0.4)

    return {
        "line_count": line_count,
        "code_lines": code_count,
        "comment_count": comment_count,
        "comment_ratio": round(comment_ratio, 4),
        "complexity": complexity,
        "composite_score": round(composite, 4),
        "path": str(path),
    }


def curate_dataset() -> dict:
    """
    Read extracted files, score, deduplicate, and write curated output.

    Returns summary statistics dict.
    """
    config = load_config()
    exclude_patterns = config.get("extraction", {}).get("exclude_patterns", [])

    if not INPUT_DIR.exists():
        logger.warning("Input directory does not exist: %s", INPUT_DIR)
        return {
            "files_input": 0,
            "files_curated": 0,
            "files_deduplicated": 0,
            "by_extension": {},
            "avg_line_count": 0.0,
            "avg_comment_ratio": 0.0,
            "avg_complexity": 0.0,
        }

    exclude_re = [
        re.compile(re.escape(p).replace("\\*", ".*"), re.IGNORECASE)
        for p in exclude_patterns
    ]

    def should_exclude(name: str) -> bool:
        return any(r.search(name) for r in exclude_re)

    seen_hashes: dict[str, Path] = {}
    scores: list[dict] = []
    files_to_curate: list[tuple[Path, dict]] = []
    total_scanned = 0

    for file_path in INPUT_DIR.rglob("*"):
        if not file_path.is_file() or not file_path.name.endswith(".abap"):
            continue
        total_scanned += 1
        if should_exclude(file_path.name):
            logger.debug("Excluded by pattern: %s", file_path.name)
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Could not read %s: %s", file_path, e)
            continue

        file_hash = content_hash(content)
        if file_hash in seen_hashes:
            logger.debug("Duplicate of %s: %s", seen_hashes[file_hash].name, file_path.name)
            continue
        seen_hashes[file_hash] = file_path

        sc = score_file(content, file_path)
        scores.append(sc)
        files_to_curate.append((file_path, sc))

    # Sort by composite score descending
    files_to_curate.sort(key=lambda x: -x[1]["composite_score"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_extension: dict[str, int] = {}
    files_written = 0

    for file_path, sc in files_to_curate:
        rel = file_path.relative_to(INPUT_DIR)
        dest = OUTPUT_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_text(file_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            # Use full extension (e.g. .clas.abap) for proper categorization
            ext = "".join(file_path.suffixes) if len(file_path.suffixes) > 1 else file_path.suffix
            by_extension[ext] = by_extension.get(ext, 0) + 1
            files_written += 1
        except Exception as e:
            logger.warning("Could not write %s: %s", dest, e)

    # Update pipeline state
    if PIPELINE_STATE_PATH.exists():
        with open(PIPELINE_STATE_PATH, "r") as f:
            state = json.load(f)
    else:
        state = {"pipeline_summary": {}, "stats": {}}
    state.setdefault("stats", {})["files_curated"] = files_written
    state["last_curation"] = {
        "files_curated": files_written,
        "files_deduplicated": len(scores) - len(files_to_curate) + (len(files_to_curate) - files_written),
    }
    state["last_updated"] = __import__("datetime").datetime.now().isoformat()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

    summary = {
        "files_input": total_scanned,
        "files_curated": files_written,
        "files_deduplicated": total_scanned - len(files_to_curate),
        "by_extension": by_extension,
        "avg_line_count": sum(s["line_count"] for s in scores) / len(scores) if scores else 0,
        "avg_comment_ratio": sum(s["comment_ratio"] for s in scores) / len(scores) if scores else 0,
        "avg_complexity": sum(s["complexity"] for s in scores) / len(scores) if scores else 0,
    }
    return summary


def main() -> None:
    """Main entry point for dataset curation."""
    parser = argparse.ArgumentParser(
        description="Curate ABAP dataset with quality scoring and deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        summary = curate_dataset()
        logger.info("Curated %d files to %s", summary["files_curated"], OUTPUT_DIR)

        print("\n--- Summary Statistics ---")
        print(f"  Input files:        {summary['files_input']}")
        print(f"  Curated (output):   {summary['files_curated']}")
        print(f"  Deduplicated:       {summary['files_deduplicated']}")
        print(f"  Avg line count:     {summary['avg_line_count']:.1f}")
        print(f"  Avg comment ratio:  {summary['avg_comment_ratio']:.2%}")
        print(f"  Avg complexity:     {summary['avg_complexity']:.1f}")
        print("  By extension:")
        for ext, cnt in sorted(summary["by_extension"].items(), key=lambda x: -x[1]):
            print(f"    {ext}: {cnt}")
        print("-------------------------\n")
    except Exception as e:
        logger.exception("Curation failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
