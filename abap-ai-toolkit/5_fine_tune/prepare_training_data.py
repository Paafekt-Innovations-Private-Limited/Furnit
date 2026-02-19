#!/usr/bin/env python3
"""
Convert 3-way mappings to JSONL training pairs.

Reads from data/threeway_mappings/.
Converts to JSONL format: {"input": "old ABAP", "output": "new ABAP"}.
Filters for high-quality pairs only.
"""

import argparse
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = SCRIPT_DIR.parent
DATA_DIR = TOOLKIT_ROOT / "data"
MAPPINGS_DIR = DATA_DIR / "threeway_mappings"
OUTPUT_PATH = DATA_DIR / "training_data.jsonl"

# Quality thresholds
MIN_INPUT_LENGTH = 10
MIN_OUTPUT_LENGTH = 5
MAX_INPUT_LENGTH = 8000
MAX_OUTPUT_LENGTH = 8000
MAX_PAIR_LENGTH = 16000


def load_threeway_mappings(mappings_dir: Path) -> list[dict]:
    """Load all JSON files from threeway_mappings directory."""
    if not mappings_dir.exists():
        logger.warning("Mappings directory not found: %s", mappings_dir)
        return []

    mappings = []
    for json_path in mappings_dir.rglob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                mappings.extend(data)
            elif isinstance(data, dict):
                mappings.append(data)
        except Exception as e:
            logger.debug("Could not load %s: %s", json_path, e)
    return mappings


def extract_pair(mapping: dict) -> dict | None:
    """
    Extract input/output pair from a 3-way mapping.

    Returns:
        {"input": "old ABAP", "output": "new ABAP"} or None if invalid.
    """
    # Handle various 3-way mapping formats
    old_abap = mapping.get("old_abap", mapping.get("source", mapping.get("java", "")))
    new_abap = mapping.get("new_abap", mapping.get("target", mapping.get("abap", "")))

    if isinstance(old_abap, dict):
        old_abap = old_abap.get("code", str(old_abap))
    if isinstance(new_abap, dict):
        new_abap = new_abap.get("code", str(new_abap))

    old_abap = str(old_abap).strip()
    new_abap = str(new_abap).strip()

    if not old_abap or not new_abap:
        return None
    if len(old_abap) < MIN_INPUT_LENGTH or len(new_abap) < MIN_OUTPUT_LENGTH:
        return None
    if len(old_abap) > MAX_INPUT_LENGTH or len(new_abap) > MAX_OUTPUT_LENGTH:
        return None
    if len(old_abap) + len(new_abap) > MAX_PAIR_LENGTH:
        return None
    if old_abap == new_abap:
        return None

    return {"input": old_abap, "output": new_abap}


def prepare_training_data(
    mappings_dir: Path | None = None,
    output_path: Path | None = None,
    min_quality_score: float = 0.0,
) -> dict:
    """
    Convert 3-way mappings to JSONL training pairs.

    Args:
        mappings_dir: Source directory for 3-way mapping JSONs.
        output_path: Output JSONL file path.
        min_quality_score: Minimum quality score to include (0.0-1.0).

    Returns:
        Summary dict with total_pairs, filtered, written.
    """
    mappings_dir = mappings_dir or MAPPINGS_DIR
    output_path = output_path or OUTPUT_PATH

    mappings = load_threeway_mappings(mappings_dir)
    pairs = []
    for m in mappings:
        pair = extract_pair(m)
        if pair:
            quality = m.get("quality", m.get("score", 1.0))
            if isinstance(quality, (int, float)) and quality >= min_quality_score:
                pairs.append(pair)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Wrote %d training pairs to %s", written, output_path)
    return {
        "total_mappings": len(mappings),
        "total_pairs": len(pairs),
        "written": written,
        "output_path": str(output_path),
    }


def main() -> None:
    """Main entry point for training data preparation."""
    parser = argparse.ArgumentParser(
        description="Convert 3-way mappings to JSONL training pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Input mappings directory (default: data/threeway_mappings/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: data/training_data.jsonl)",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="Minimum quality score (0.0-1.0) to include",
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
        summary = prepare_training_data(
            mappings_dir=args.input,
            output_path=args.output,
            min_quality_score=args.min_quality,
        )
        print("\n--- Training Data Summary ---")
        print(f"  Total mappings: {summary['total_mappings']}")
        print(f"  Total pairs:    {summary['total_pairs']}")
        print(f"  Written:        {summary['written']}")
        print(f"  Output:         {summary['output_path']}")
        print("--------------------------------\n")
    except Exception as e:
        logger.exception("Preparation failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
