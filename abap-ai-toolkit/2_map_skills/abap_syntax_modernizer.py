#!/usr/bin/env python3
"""
ABAP Syntax Modernization Engine.

Core transformation engine containing all 12 rules for migrating legacy ABAP
to modern syntax. Used by migrate_sample_pkg.py and deep_migrate_sample_pkg.py.
Can transform individual files or entire directories.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

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


class AbapSyntaxModernizer:
    """
    Reusable ABAP syntax modernization engine with 12 transformation rules.

    Rules follow section 11.3 of ABAP_AI_Toolkit_Complete_Context.md.
    """

    def __init__(self) -> None:
        """Initialize the modernizer with all transformation rules."""
        self.rule_counts: dict[str, int] = {}
        self._init_rules()

    def _init_rules(self) -> None:
        """Initialize rule counts to zero."""
        self.rule_counts = {
            "move_to_assign": 0,
            "add_to_arithmetic": 0,
            "concatenate_to_template": 0,
            "move_corresponding": 0,
            "call_method_to_functional": 0,
            "create_object_to_new": 0,
            "subtract_to_arithmetic": 0,
            "translate_upper": 0,
            "translate_lower": 0,
            "describe_table_to_lines": 0,
            "create_object_no_type": 0,
            "call_method_static": 0,
            "check_to_if_return": 0,
        }

    def get_rules(self) -> list[dict[str, Any]]:
        """
        Return the 14 ordered regex rules (Pass 1 migration).

        Returns:
            List of rule dicts with keys: name, pattern, replacement, flags
        """
        return [
            {
                "name": "move_to_assign",
                "pattern": r"\bMOVE\s+(.+?)\s+TO\s+(.+?)\.\s*$",
                "replacement": r"\2 = \1.",
                "flags": re.MULTILINE | re.IGNORECASE,
            },
            {
                "name": "add_to_arithmetic",
                "pattern": r"\bADD\s+(.+?)\s+TO\s+(.+?)\.\s*$",
                "replacement": r"\2 = \2 + \1.",
                "flags": re.MULTILINE | re.IGNORECASE,
            },
            {
                "name": "concatenate_to_template",
                "pattern": r"\bCONCATENATE\s+(.+?)\s+INTO\s+(\w+)\s*\.\s*$",
                "replacement": None,  # Use callback
                "flags": re.MULTILINE | re.IGNORECASE,
            },
            {
                "name": "move_corresponding",
                "pattern": r"\bMOVE-CORRESPONDING\s+(.+?)\s+TO\s+(.+?)\.\s*$",
                "replacement": r"\2 = CORRESPONDING #( \1 ).",
                "flags": re.MULTILINE | re.IGNORECASE,
            },
            {
                "name": "call_method_to_functional",
                "pattern": r"\bCALL\s+METHOD\s+(\w+)->(\w+)\s*\.\s*$",
                "replacement": r"\1->\2( ).",
                "flags": re.MULTILINE | re.IGNORECASE,
            },
            {
                "name": "create_object_to_new",
                "pattern": r"\bCREATE\s+OBJECT\s+(\w+)\s+TYPE\s+(\w+)\s*\.\s*$",
                "replacement": r"\1 = NEW \2( ).",
                "flags": re.MULTILINE | re.IGNORECASE,
            },
            {
                "name": "subtract_to_arithmetic",
                "pattern": r"\bSUBTRACT\s+(.+?)\s+FROM\s+(.+?)\.\s*$",
                "replacement": r"\2 = \2 - \1.",
                "flags": re.MULTILINE | re.IGNORECASE,
            },
            {
                "name": "translate_upper",
                "pattern": r"\bTRANSLATE\s+(.+?)\s+TO\s+UPPER\s+CASE\s*\.\s*$",
                "replacement": r"\1 = to_upper( \1 ).",
                "flags": re.MULTILINE | re.IGNORECASE,
            },
            {
                "name": "translate_lower",
                "pattern": r"\bTRANSLATE\s+(.+?)\s+TO\s+LOWER\s+CASE\s*\.\s*$",
                "replacement": r"\1 = to_lower( \1 ).",
                "flags": re.MULTILINE | re.IGNORECASE,
            },
            {
                "name": "describe_table_to_lines",
                "pattern": r"\bDESCRIBE\s+TABLE\s+(\w+)\s+LINES\s+(\w+)\s*\.\s*$",
                "replacement": r"\2 = lines( \1 ).",
                "flags": re.MULTILINE | re.IGNORECASE,
            },
            {
                "name": "create_object_no_type",
                "pattern": r"\bCREATE\s+OBJECT\s+(\w+)\s*\.\s*$",
                "replacement": r"\1 = NEW #( ).",
                "flags": re.MULTILINE | re.IGNORECASE,
            },
            {
                "name": "call_method_static",
                "pattern": r"\bCALL\s+METHOD\s+(\w+)=>(\w+)\s*\.\s*$",
                "replacement": r"\1=>\2( ).",
                "flags": re.MULTILINE | re.IGNORECASE,
            },
            {
                "name": "check_to_if_return",
                "pattern": r"^\s*CHECK\s+(.+?)\.\s*$",
                "replacement": r"IF \1.\n  RETURN.\nENDIF.",
                "flags": re.MULTILINE | re.IGNORECASE,
            },
        ]

    def _concatenate_repl(self, match: re.Match) -> str:
        """Replace CONCATENATE a b INTO c with c = |{ a }{ b }|."""
        operands = match.group(1).split()
        target = match.group(2)
        parts = "".join(f"{{ {op} }}" for op in operands)
        return f"{target} = |{parts}|."

    def transform_text(self, text: str) -> tuple[str, dict[str, int]]:
        """
        Apply all Pass 1 rules to text.

        Args:
            text: Input ABAP source code.

        Returns:
            Tuple of (transformed text, rule counts dict).
        """
        self._init_rules()
        result = text
        for rule in self.get_rules():
            pattern = re.compile(rule["pattern"], rule.get("flags", 0))
            replacement = rule.get("replacement")
            if replacement is None and rule["name"] == "concatenate_to_template":
                matches = list(pattern.finditer(result))
                count = len(matches)
                if count > 0:
                    self.rule_counts[rule["name"]] = count
                    result = pattern.sub(self._concatenate_repl, result)
            else:
                matches = pattern.findall(result)
                count = len(matches)
                if count > 0 and replacement:
                    self.rule_counts[rule["name"]] = count
                    result = pattern.sub(replacement, result)
        return result, dict(self.rule_counts)

    def transform_file(self, input_path: Path, output_path: Path | None = None) -> dict[str, int]:
        """
        Transform a single ABAP file.

        Args:
            input_path: Path to input file.
            output_path: Optional path for output. If None, overwrites input.

        Returns:
            Rule counts dict.
        """
        try:
            content = input_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.error("Failed to read %s: %s", input_path, e)
            raise
        transformed, counts = self.transform_text(content)
        out = output_path if output_path is not None else input_path
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            out.write_text(transformed, encoding="utf-8")
        except OSError as e:
            logger.error("Failed to write %s: %s", out, e)
            raise
        return counts

    def transform_directory(
        self, input_dir: Path, output_dir: Path | None = None
    ) -> dict[str, Any]:
        """
        Transform all ABAP files in a directory.

        Args:
            input_dir: Input directory.
            output_dir: Output directory. If None, overwrites in place.

        Returns:
            Summary dict with total_files, total_transforms, rule_counts.
        """
        abap_extensions = (".clas.abap", ".intf.abap", ".fugr.abap", ".prog.abap", ".incl.abap")
        files = [f for f in input_dir.rglob("*") if f.suffix and f.name.endswith(".abap")]
        total_counts: dict[str, int] = {}
        files_processed = 0
        for file_path in files:
            rel = file_path.relative_to(input_dir)
            out_path = (output_dir / rel) if output_dir else file_path
            try:
                counts = self.transform_file(file_path, out_path)
                files_processed += 1
                for k, v in counts.items():
                    total_counts[k] = total_counts.get(k, 0) + v
            except OSError:
                continue
        return {
            "total_files": files_processed,
            "total_transforms": sum(total_counts.values()),
            "rule_counts": total_counts,
        }


def main() -> int:
    """CLI entry point for syntax modernization."""
    parser = argparse.ArgumentParser(
        description="ABAP syntax modernization engine - transform legacy ABAP to modern syntax."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input file or directory path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file or directory (default: overwrite input)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    modernizer = AbapSyntaxModernizer()
    try:
        if args.input.is_file():
            counts = modernizer.transform_file(args.input, args.output)
            total = sum(counts.values())
            logger.info("Transformed %s: %d replacements", args.input, total)
            for rule, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                if cnt > 0:
                    logger.info("  %s: %d", rule, cnt)
        elif args.input.is_dir():
            summary = modernizer.transform_directory(args.input, args.output)
            logger.info(
                "Transformed %d files, %d total replacements",
                summary["total_files"],
                summary["total_transforms"],
            )
            for rule, cnt in sorted(
                summary["rule_counts"].items(), key=lambda x: -x[1]
            ):
                if cnt > 0:
                    logger.info("  %s: %d", rule, cnt)
        else:
            logger.error("Input path does not exist: %s", args.input)
            return 1
    except OSError as e:
        logger.error("Error: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
