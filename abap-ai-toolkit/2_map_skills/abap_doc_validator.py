#!/usr/bin/env python3
"""
Doc-driven ABAP syntax validation.

CLI: python 2_map_skills/abap_doc_validator.py [--cloud] FILE

20 obsolete/style rules + 28 ABAP Cloud rules (15 categories). Checks:
deprecated statements, naming conventions, restricted language, structural
patterns. Returns errors, warnings, info counts + average score.
"""

import argparse
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

# 20 obsolete/style rules
OBSOLETE_RULES = [
    {"id": "OBS001", "desc": "MOVE statement (use =)", "pattern": r"\bMOVE\s+.+?\s+TO\s+", "severity": "warning"},
    {"id": "OBS002", "desc": "ADD statement (use arithmetic)", "pattern": r"\bADD\s+.+?\s+TO\s+", "severity": "warning"},
    {"id": "OBS003", "desc": "SUBTRACT statement", "pattern": r"\bSUBTRACT\s+", "severity": "warning"},
    {"id": "OBS004", "desc": "CONCATENATE (use string template)", "pattern": r"\bCONCATENATE\s+", "severity": "warning"},
    {"id": "OBS005", "desc": "CALL METHOD (use functional)", "pattern": r"\bCALL\s+METHOD\s+", "severity": "warning"},
    {"id": "OBS006", "desc": "CREATE OBJECT (use NEW)", "pattern": r"\bCREATE\s+OBJECT\s+", "severity": "warning"},
    {"id": "OBS007", "desc": "TRANSLATE (use to_upper/to_lower)", "pattern": r"\bTRANSLATE\s+", "severity": "warning"},
    {"id": "OBS008", "desc": "DESCRIBE TABLE (use lines())", "pattern": r"\bDESCRIBE\s+TABLE\s+", "severity": "warning"},
    {"id": "OBS009", "desc": "MOVE-CORRESPONDING (use CORRESPONDING)", "pattern": r"\bMOVE-CORRESPONDING\s+", "severity": "warning"},
    {"id": "OBS010", "desc": "CHECK (use IF/RETURN)", "pattern": r"^\s*CHECK\s+", "severity": "info"},
    {"id": "OBS011", "desc": "READ TABLE (prefer table expr)", "pattern": r"\bREAD\s+TABLE\s+", "severity": "info"},
    {"id": "OBS012", "desc": "SELECT * (use explicit fields)", "pattern": r"\bSELECT\s+\*\s+", "severity": "error"},
    {"id": "OBS013", "desc": "SY-SUBRC (prefer exception)", "pattern": r"\bSY-SUBRC\b", "severity": "info"},
    {"id": "OBS014", "desc": "Obsolete type declaration", "pattern": r"\bLIKE\s+(SY|SPACE)\b", "severity": "warning"},
    {"id": "OBS015", "desc": "Internal table with header line", "pattern": r"\bDATA\s+:\s*\w+\s+OCCURS\s+", "severity": "error"},
    {"id": "OBS016", "desc": "COMMUNICATION (obsolete)", "pattern": r"\bCOMMUNICATION\s+", "severity": "error"},
    {"id": "OBS017", "desc": "EXTRACT (obsolete)", "pattern": r"\bEXTRACT\s+", "severity": "error"},
    {"id": "OBS018", "desc": "Naming: short variable", "pattern": r"\bDATA\s+(\w{1,2})\s+", "severity": "info"},
    {"id": "OBS019", "desc": "Magic number", "pattern": r"\b(IF|WHEN|CHECK)\s+.*\b(0|1|2)\s*[=<>]", "severity": "info"},
    {"id": "OBS020", "desc": "Comment style", "pattern": r"^\s*\*[^ ]", "severity": "info"},
]

# 28 ABAP Cloud rules (15 categories)
CLOUD_RULES = [
    {"id": "CLD001", "desc": "CALL FUNCTION (restricted)", "pattern": r"\bCALL\s+FUNCTION\s+", "category": "fm", "severity": "warning"},
    {"id": "CLD002", "desc": "SYSTEM-CALL (restricted)", "pattern": r"\bSYSTEM-CALL\s+", "category": "system", "severity": "error"},
    {"id": "CLD003", "desc": "COMMIT WORK (use RAP)", "pattern": r"\bCOMMIT\s+WORK\b", "category": "transaction", "severity": "info"},
    {"id": "CLD004", "desc": "AUTHORITY-CHECK (use CDS)", "pattern": r"\bAUTHORITY-CHECK\b", "category": "auth", "severity": "info"},
    {"id": "CLD005", "desc": "MESSAGE (use RAP)", "pattern": r"\bMESSAGE\s+", "category": "message", "severity": "info"},
    {"id": "CLD006", "desc": "Database hint", "pattern": r"\b%_HINTS\b", "category": "sql", "severity": "error"},
    {"id": "CLD007", "desc": "Native SQL", "pattern": r"\bEXEC\s+SQL\b", "category": "sql", "severity": "error"},
    {"id": "CLD008", "desc": "Dynamic program", "pattern": r"\bGENERATE\s+SUBPROGRAM\b", "category": "dynamic", "severity": "error"},
    {"id": "CLD009", "desc": "File operations", "pattern": r"\bOPEN\s+DATASET\b", "category": "file", "severity": "error"},
    {"id": "CLD010", "desc": "Enqueue", "pattern": r"\bCALL\s+FUNCTION\s+['\"]?ENQUEUE_", "category": "lock", "severity": "warning"},
    {"id": "CLD011", "desc": "GUI (restricted)", "pattern": r"\bCALL\s+METHOD\s+cl_gui_", "category": "gui", "severity": "error"},
    {"id": "CLD012", "desc": "RFC (restricted)", "pattern": r"\bDESTINATION\s+", "category": "rfc", "severity": "warning"},
    {"id": "CLD013", "desc": "Number range", "pattern": r"\bNUMBER_GET_NEXT\b", "category": "numbering", "severity": "info"},
    {"id": "CLD014", "desc": "Workflow", "pattern": r"\bSWE_\w+", "category": "workflow", "severity": "info"},
    {"id": "CLD015", "desc": "BADI (prefer RAP)", "pattern": r"\bGET\s+BADI\b", "category": "badi", "severity": "info"},
    {"id": "CLD016", "desc": "Class naming", "pattern": r"CLASS\s+[a-z]\w+", "category": "naming", "severity": "info"},
    {"id": "CLD017", "desc": "Interface naming", "pattern": r"INTERFACE\s+[a-z]\w+", "category": "naming", "severity": "info"},
    {"id": "CLD018", "desc": "Method visibility", "pattern": r"PUBLIC\s+SECTION", "category": "structure", "severity": "info"},
    {"id": "CLD019", "desc": "CDS usage", "pattern": r"\bSELECT\s+FROM\s+@", "category": "cds", "severity": "info"},
    {"id": "CLD020", "desc": "EML usage", "pattern": r"\bMODIFY\s+ENTITY\b", "category": "eml", "severity": "info"},
    {"id": "CLD021", "desc": "RAP BO", "pattern": r"\bREAD\s+ENTITY\b", "category": "rap", "severity": "info"},
    {"id": "CLD022", "desc": "Draft handling", "pattern": r"\bDRAFT\b", "category": "rap", "severity": "info"},
    {"id": "CLD023", "desc": "Behavior def", "pattern": r"\bBEHAVIOR\s+DEFINITION\b", "category": "rap", "severity": "info"},
    {"id": "CLD024", "desc": "Service binding", "pattern": r"\bSERVICE\s+BINDING\b", "category": "rap", "severity": "info"},
    {"id": "CLD025", "desc": "Abstract entity", "pattern": r"\bABSTRACT\s+ENTITY\b", "category": "cds", "severity": "info"},
    {"id": "CLD026", "desc": "Value help", "pattern": r"@Consumption\.valueHelpDefinition", "category": "cds", "severity": "info"},
    {"id": "CLD027", "desc": "OData V4", "pattern": r"odata-v4", "category": "odata", "severity": "info"},
    {"id": "CLD028", "desc": "Fiori Elements", "pattern": r"@UI\.lineItem", "category": "fiori", "severity": "info"},
]


def validate_file(file_path: Path, cloud_mode: bool) -> dict[str, Any]:
    """
    Validate ABAP file against rules.

    Args:
        file_path: Path to ABAP file.
        cloud_mode: If True, include ABAP Cloud rules.

    Returns:
        Dict with errors, warnings, info, score, findings.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"error": str(e), "errors": 1, "warnings": 0, "info": 0, "score": 0}

    errors = 0
    warnings = 0
    info_count = 0
    findings: list[dict] = []

    rules = OBSOLETE_RULES + (CLOUD_RULES if cloud_mode else [])

    for rule in rules:
        pattern = re.compile(rule["pattern"], re.IGNORECASE | re.MULTILINE)
        for match in pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            line_content = content.split("\n")[line_num - 1].strip()[:60]
            severity = rule.get("severity", "info")
            if severity == "error":
                errors += 1
            elif severity == "warning":
                warnings += 1
            else:
                info_count += 1
            findings.append({
                "line": line_num,
                "rule_id": rule["id"],
                "desc": rule["desc"],
                "severity": severity,
                "snippet": line_content,
            })

    total_issues = errors + warnings + info_count
    lines = len(content.split("\n")) or 1
    # Score: 100 - penalty (errors*10 + warnings*2 + info*0.5) / lines * 10
    penalty = (errors * 10 + warnings * 2 + info_count * 0.5) / lines * 10
    score = max(0, min(100, 100 - penalty))

    return {
        "errors": errors,
        "warnings": warnings,
        "info": info_count,
        "score": round(score, 1),
        "findings": findings[:100],  # Cap for output
        "total_issues": total_issues,
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Doc-driven ABAP syntax validation."
    )
    parser.add_argument(
        "file",
        type=Path,
        nargs="?",
        default=None,
        help="ABAP file to validate",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Enable ABAP Cloud rules (28 additional rules)",
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

    if not args.file or not args.file.exists():
        logger.error("File not specified or does not exist: %s", args.file)
        return 1

    result = validate_file(args.file.resolve(), args.cloud)

    if "error" in result:
        logger.error("Validation failed: %s", result["error"])
        return 1

    print("\n=== ABAP Validation Result ===")
    print(f"Mode: {'ABAP Cloud' if args.cloud else 'Standard (7.40+)'}")
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")
    print(f"Info: {result['info']}")
    print(f"Average score: {result['score']}%")
    if result.get("findings"):
        print("\nSample findings:")
        for f in result["findings"][:10]:
            print(f"  L{f['line']} [{f['rule_id']}] {f['severity']}: {f['desc']}")

    return 0 if result["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
