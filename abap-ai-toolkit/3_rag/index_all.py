#!/usr/bin/env python3
"""
Index all ZSAMPLE knowledge into the RAG vector database.

Indexes: code (original + migrated ABAP from data/zsample_*), syntax rules,
FM classification. Chunks ABAP files by class/method boundaries (max 2000
chars per chunk). Adds metadata: layer, source_file, package, object_type.
"""

import hashlib
import json
import logging
import re
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

from _common import get_chroma_client, get_collection, get_chunk_size

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = TOOLKIT_ROOT / "data"
ZSAMPLE_DIRS = [
    "sample_pkg_original",
    "sample_pkg_migrated",
    "zsample_original",
    "zsample_migrated",
    "zsample_extracted",
]
ABAP_EXTENSIONS = {".clas.abap", ".intf.abap", ".prog.abap", ".incl.abap", ".fugr.abap"}

SYNTAX_RULES = [
    ("MOVE", "target = source.", "target = source;"),
    ("ADD value TO target", "target = target + value.", "target += value;"),
    ("CONCATENATE a b INTO c", "c = |{ a }{ b }|.", "c = a + b;"),
    ("MOVE-CORRESPONDING", "CORRESPONDING #( src ).", "BeanUtils.copyProperties()"),
    ("CALL METHOD obj->m", "obj->m( ).", "obj.method();"),
    ("CREATE OBJECT lo TYPE cl", "lo = NEW cl( ).", "obj = new Cl();"),
    ("TRANSLATE TO UPPER CASE", "to_upper( str ).", "str.toUpperCase()"),
    ("TRANSLATE TO LOWER CASE", "to_lower( str ).", "str.toLowerCase()"),
    ("DESCRIBE TABLE itab LINES n", "lines( itab ).", "list.size()"),
]


def chunk_by_boundaries(text: str, max_chars: int) -> list[str]:
    """Chunk ABAP by class/method boundaries, max max_chars per chunk."""
    chunks = []
    pattern = re.compile(
        r"^\s*(CLASS|METHOD|INTERFACE|ENDCLASS|ENDMETHOD|ENDINTERFACE|FUNCTION|ENDFUNCTION)",
        re.MULTILINE | re.IGNORECASE,
    )
    last_end = 0
    current_chunk = []

    for match in pattern.finditer(text):
        segment = text[last_end : match.start()]
        if segment.strip():
            current_chunk.append(segment)
        current_chunk.append(match.group(0))
        combined = "".join(current_chunk)
        if len(combined) >= max_chars:
            chunks.append(combined)
            current_chunk = []
        last_end = match.end()

    remainder = text[last_end:]
    if remainder.strip() or current_chunk:
        current_chunk.append(remainder)
        chunks.append("".join(current_chunk))

    if not chunks:
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            start = end

    return chunks


def infer_object_type(file_path: Path) -> str:
    """Infer ABAP object type from filename."""
    name = file_path.name.lower()
    if ".clas.abap" in name:
        return "class"
    if ".intf.abap" in name:
        return "interface"
    if ".prog.abap" in name:
        return "program"
    if ".incl.abap" in name:
        return "include"
    return "unknown"


def infer_package(file_path: Path, base_dir: str) -> str:
    """Infer package from directory structure."""
    if "sample_pkg" in base_dir:
        return "/ZSAMPLE/MAIN"
    if "zsample_generic" in base_dir or "zsample_original" in base_dir or "zsample_migrated" in base_dir:
        return "/ZSAMPLE/SHARED"
    return base_dir


def index_zsample_code() -> list[tuple[str, dict]]:
    """Index all ZSAMPLE ABAP code."""
    results = []
    max_chars = get_chunk_size()

    for dir_name in ZSAMPLE_DIRS:
        base_path = DATA_DIR / dir_name
        if not base_path.exists():
            continue
        for abap_file in base_path.rglob("*"):
            if not abap_file.is_file() or abap_file.suffix.lower() not in ABAP_EXTENSIONS:
                continue
            try:
                content = abap_file.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.warning("Could not read %s: %s", abap_file, e)
                continue
            rel = str(abap_file.relative_to(base_path))
            for i, chunk in enumerate(chunk_by_boundaries(content, max_chars)):
                if not chunk.strip():
                    continue
                meta = {
                    "layer": "code",
                    "source_file": rel,
                    "package": infer_package(abap_file, dir_name),
                    "object_type": infer_object_type(abap_file),
                }
                results.append((chunk, meta))
    return results


def index_syntax_rules() -> list[tuple[str, dict]]:
    """Index syntax modernization rules."""
    results = []
    for old_syntax, new_syntax, java_anchor in SYNTAX_RULES:
        content = f"Old ABAP: {old_syntax}\nNew ABAP: {new_syntax}\nJava equivalent: {java_anchor}"
        meta = {"layer": "syntax", "source_file": "syntax_rules", "topic": "modernization"}
        results.append((content, meta))
    return results


def index_fm_classification() -> list[tuple[str, dict]]:
    """Index FM classification from scan results if available."""
    results = []
    scan_path = DATA_DIR / "sample_pkg_scan_results.json"
    if not scan_path.exists():
        return results
    try:
        data = json.loads(scan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return results
    fm_list = data.get("fm_calls_summary", data.get("fm_calls", []))
    if isinstance(fm_list, dict):
        fm_list = list(fm_list.keys())
    for fm in fm_list[:200]:
        content = f"Function Module: {fm}\nCheck CRV for release status and successor API."
        meta = {"layer": "fm_classification", "source_file": "scan_results", "fm": str(fm)}
        results.append((content, meta))
    return results


def main() -> int:
    """CLI entry point."""
    all_items = []
    all_items.extend(index_zsample_code())
    all_items.extend(index_syntax_rules())
    all_items.extend(index_fm_classification())

    if not all_items:
        logger.warning("No ZSAMPLE content to index.")
        return 0

    client = get_chroma_client()
    collection = get_collection(client)

    ids = []
    documents = []
    metadatas = []
    for i, (content, meta) in enumerate(all_items):
        raw = f"{meta.get('layer','')}:{meta.get('source_file','')}:{i}:{content[:50]}"
        chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:24]
        ids.append(chunk_id)
        documents.append(content)
        metadatas.append(meta)

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Indexed %d ZSAMPLE chunks", len(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
