#!/usr/bin/env python3
"""
ABAP AI Toolkit - Single-command pipeline.

Takes an abapGit ZIP export (or folder) of any SAP Z-package and runs
the full migration + RAG pipeline:

  1. Extract & organize ABAP objects from ZIP
  2. Syntax scan (obsolete patterns, FM classification)
  3. Pass 1 migration (14 regex transforms)
  4. Pass 2 migration (complex multi-line patterns)
  5. Semantic twin discovery (FM -> Clean Core API mapping)
  6. RAG indexing (ChromaDB vector database)

Usage:
  python run_pipeline.py /path/to/package.zip
  python run_pipeline.py /path/to/abap_folder/
  python run_pipeline.py --skip-rag /path/to/package.zip   # skip RAG (no chromadb needed)

After running, open this folder in Cursor to get AI-assisted migration queries.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

TOOLKIT_ROOT = Path(__file__).resolve().parent
DATA_DIR = TOOLKIT_ROOT / "data"
ORIGINAL_DIR = DATA_DIR / "sample_pkg_original"
MIGRATED_DIR = DATA_DIR / "sample_pkg_migrated"
SCAN_RESULTS = DATA_DIR / "sample_pkg_scan_results.json"

ABAP_EXTENSIONS = {
    ".clas.abap", ".intf.abap", ".fugr.abap", ".prog.abap", ".incl.abap",
    ".tabl.abap", ".dtel.abap", ".doma.abap", ".msag.abap", ".ttyp.abap",
    ".shlp.abap", ".enqu.abap", ".func.abap",
}


def is_abap_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in ABAP_EXTENSIONS)


def extract_zip(zip_path: Path) -> int:
    """Extract abapGit ZIP into data/sample_pkg_original/. Returns file count."""
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if is_abap_file(Path(name)):
                target = ORIGINAL_DIR / name
                with zf.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                count += 1
    return count


def copy_folder(folder_path: Path) -> int:
    """Copy ABAP files from a folder into data/sample_pkg_original/. Returns file count."""
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    if folder_path.resolve() == ORIGINAL_DIR.resolve():
        return len([f for f in ORIGINAL_DIR.iterdir() if f.is_file() and is_abap_file(f)])
    count = 0
    for f in folder_path.rglob("*"):
        if f.is_file() and is_abap_file(f):
            shutil.copy2(f, ORIGINAL_DIR / f.name)
            count += 1
    return count


def run_step(step_num: int, description: str, script: str, args: list[str] | None = None):
    """Run a pipeline step as a subprocess."""
    script_path = TOOLKIT_ROOT / script
    if not script_path.exists():
        logger.warning(f"  SKIP step {step_num}: {script} not found")
        return False

    logger.info(f"Step {step_num}: {description}")
    cmd = [sys.executable, str(script_path)] + (args or [])
    result = subprocess.run(cmd, cwd=str(TOOLKIT_ROOT), capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"  FAILED: {result.stderr[-500:] if result.stderr else 'no stderr'}")
        return False

    for line in result.stdout.strip().split("\n")[-3:]:
        if line.strip():
            logger.info(f"  {line.strip()}")
    return True


def print_summary():
    """Print pipeline results summary."""
    original_count = len(list(ORIGINAL_DIR.glob("*"))) if ORIGINAL_DIR.exists() else 0
    migrated_count = len(list(MIGRATED_DIR.glob("*"))) if MIGRATED_DIR.exists() else 0

    scan_summary = ""
    if SCAN_RESULTS.exists():
        try:
            data = json.loads(SCAN_RESULTS.read_text())
            total_obs = sum(r.get("count", 0) for r in data.get("obsolete_patterns", []))
            total_fms = len(data.get("function_modules", []))
            scan_summary = f"  Obsolete patterns: {total_obs}, Function modules: {total_fms}"
        except Exception:
            pass

    twin_catalog = DATA_DIR / "twin_discovery" / "twin_catalog_sample_pkg.json"
    twin_count = 0
    if twin_catalog.exists():
        try:
            twins = json.loads(twin_catalog.read_text())
            twin_count = len(twins.get("twins", []))
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("ABAP AI Toolkit - Pipeline Complete")
    print("=" * 60)
    print(f"  Original ABAP files:  {original_count}")
    print(f"  Migrated ABAP files:  {migrated_count}")
    if scan_summary:
        print(scan_summary)
    if twin_count:
        print(f"  Twin mappings found:  {twin_count}")
    print(f"\n  Data directory: {DATA_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Open this folder in Cursor")
    print(f"  2. The .cursor/rules/ will auto-load migration context")
    print(f"  3. Ask: 'How do I migrate BAPI_ACC_DOCUMENT_POST to Clean Core?'")
    print(f"\n  Query RAG: python 3_rag/rag_query_api.py \"your question\"")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="ABAP AI Toolkit - Run full migration pipeline from abapGit ZIP or folder.",
        epilog="After running, open this folder in Cursor for AI-assisted migration.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to abapGit ZIP file or folder containing ABAP source files",
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Skip RAG indexing (useful if chromadb is not installed)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean previous data before running (removes data/sample_pkg_*)",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return 1

    print("=" * 60)
    print("ABAP AI Toolkit - Migration Pipeline")
    print("=" * 60)
    print(f"  Input: {input_path}")
    print()

    if args.clean:
        for d in [ORIGINAL_DIR, MIGRATED_DIR]:
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"Cleaned {d.name}/")
        if SCAN_RESULTS.exists():
            SCAN_RESULTS.unlink()

    # Step 0: Extract input
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        logger.info(f"Step 0: Extracting ZIP -> data/sample_pkg_original/")
        count = extract_zip(input_path)
    elif input_path.is_dir():
        logger.info(f"Step 0: Copying folder -> data/sample_pkg_original/")
        count = copy_folder(input_path)
    else:
        logger.error(f"Input must be a .zip file or a directory: {input_path}")
        return 1

    logger.info(f"  {count} ABAP files extracted")
    if count == 0:
        logger.error("No ABAP files found in input. Expected files like *.clas.abap, *.fugr.abap, etc.")
        return 1

    # Step 1: Syntax scan
    run_step(1, "Syntax scan + FM classification", "2_map_skills/scan_package.py")

    # Step 2: Pass 1 migration (14 regex transforms)
    run_step(2, "Pass 1 migration (regex transforms)", "2_map_skills/migrate_package.py")

    # Step 3: Pass 2 migration (complex multi-line patterns)
    run_step(3, "Pass 2 migration (complex patterns)", "2_map_skills/deep_migrate_package.py")

    # Step 4: Semantic twin discovery
    run_step(4, "Semantic twin discovery", "2_map_skills/twin_discovery_package.py")

    # Step 5: 3-way mapping (Java-ABAP-CRV)
    run_step(5, "3-way mapping + CRV verification", "2_map_skills/threeway_mapper.py")

    # Step 6: RAG indexing
    if args.skip_rag:
        logger.info("Step 6: RAG indexing SKIPPED (--skip-rag)")
    else:
        if not run_step(6, "RAG indexing (ChromaDB)", "3_rag/index_all.py"):
            logger.warning("  RAG indexing failed. Install chromadb: pip install chromadb")
            logger.info("  You can still use Cursor rules for migration guidance.")

    print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
