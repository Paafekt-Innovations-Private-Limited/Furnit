#!/usr/bin/env python3
"""
10-point pattern compiler checklist for ABAP code validation.

Checks: syntax validity, naming conventions, exception handling,
authorization, RAP patterns, CDS annotations, Clean Core compliance,
documentation, test coverage hints, performance patterns.
Returns score (0-100) and detailed findings.
"""

import argparse
import json
import logging
import re
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

# 10-point checklist definitions
CHECKLIST = [
    {"id": "syntax", "name": "Syntax validity", "weight": 15, "pattern": r"(CLASS|METHOD|DATA|IF|LOOP|END)"},
    {"id": "naming", "name": "Naming conventions", "weight": 10, "pattern": r"\b([A-Z][a-zA-Z0-9_]*)\b"},
    {"id": "exceptions", "name": "Exception handling", "weight": 10, "pattern": r"(TRY|CATCH|CLEANUP|RESUME)"},
    {"id": "authorization", "name": "Authorization checks", "weight": 10, "pattern": r"(AUTHORITY-CHECK|cl_abap_auth)"},
    {"id": "rap", "name": "RAP patterns", "weight": 12, "pattern": r"(managed|draft|I_[A-Za-z]+TP|behavior)"},
    {"id": "cds", "name": "CDS annotations", "weight": 8, "pattern": r"(@|#[\w.]+|annotate)"},
    {"id": "clean_core", "name": "Clean Core compliance", "weight": 12, "pattern": r"(VALUE\s*#|NEW\s+#|CORRESPONDING|lines\s*\()"},
    {"id": "documentation", "name": "Documentation", "weight": 8, "pattern": r"(\*|/\*|\"|##)"},
    {"id": "test_hints", "name": "Test coverage hints", "weight": 5, "pattern": r"(test|mock|double)"},
    {"id": "performance", "name": "Performance patterns", "weight": 10, "pattern": r"(SELECT\s+.*\s+INTO|FOR\s+ALL|READ\s+TABLE)"},
]


def validate_syntax(code: str) -> tuple[bool, str]:
    """Basic syntax validity: balanced keywords, no obvious errors."""
    open_keywords = ["IF", "LOOP", "CASE", "DO", "WHILE", "TRY", "METHOD", "CLASS"]
    close_keywords = ["ENDIF", "ENDLOOP", "ENDCASE", "ENDDO", "ENDWHILE", "ENDTRY", "ENDMETHOD", "ENDCLASS"]
    for o, c in zip(open_keywords, close_keywords):
        o_count = len(re.findall(rf"\b{o}\b", code, re.IGNORECASE))
        c_count = len(re.findall(rf"\b{c}\b", code, re.IGNORECASE))
        if o_count != c_count:
            return False, f"Unbalanced {o}/{c}: {o_count} vs {c_count}"
    return True, "OK"


def run_checklist(code: str) -> list[dict]:
    """Run all 10 checklist items and return findings."""
    findings = []
    for item in CHECKLIST:
        pattern = re.compile(item["pattern"], re.IGNORECASE | re.MULTILINE)
        matches = pattern.findall(code)
        score = min(1.0, len(matches) * 0.2) if item["weight"] > 5 else (1.0 if matches else 0.0)
        passed = score > 0 or item["id"] in ("documentation", "test_hints")
        findings.append({
            "id": item["id"],
            "name": item["name"],
            "weight": item["weight"],
            "passed": passed,
            "score": round(score, 2),
            "matches": len(matches),
        })
    return findings


def validate_pattern(code: str) -> dict:
    """
    Run 10-point pattern validation on ABAP code.

    Args:
        code: ABAP source code string.

    Returns:
        Dict with score (0-100), findings, and overall status.
    """
    if not code or not code.strip():
        return {
            "score": 0,
            "status": "empty",
            "findings": [],
            "message": "Empty code",
        }

    syntax_ok, syntax_msg = validate_syntax(code)
    findings = run_checklist(code)

    total_weight = sum(c["weight"] for c in CHECKLIST)
    weighted_sum = sum(
        f["score"] * next(c["weight"] for c in CHECKLIST if c["id"] == f["id"])
        for f in findings
    )
    score = round(weighted_sum / total_weight * 100) if total_weight else 0

    if not syntax_ok:
        score = max(0, score - 20)
        findings.insert(0, {"id": "syntax", "name": "Syntax validity", "passed": False, "message": syntax_msg})

    status = "pass" if score >= 70 and syntax_ok else "fail"
    return {
        "score": min(100, score),
        "status": status,
        "findings": findings,
        "message": syntax_msg if not syntax_ok else f"Score: {score}/100",
    }


def main() -> None:
    """Main entry point for pattern validation."""
    parser = argparse.ArgumentParser(
        description="10-point pattern validation for ABAP code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Input ABAP file (or stdin)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
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

    if args.input and args.input.exists():
        code = args.input.read_text(encoding="utf-8", errors="replace")
    else:
        import sys
        code = sys.stdin.read()

    result = validate_pattern(code)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Score: {result['score']}/100")
        print(f"Status: {result['status']}")
        for f in result.get("findings", []):
            status = "PASS" if f.get("passed", True) else "FAIL"
            print(f"  [{status}] {f.get('name', f['id'])}: {f.get('matches', 0)} matches")


if __name__ == "__main__":
    main()
