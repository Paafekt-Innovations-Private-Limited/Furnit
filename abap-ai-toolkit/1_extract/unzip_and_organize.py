#!/usr/bin/env python3
"""
Unzip exported ABAP ZIPs and organize files by ABAP object type.

Reads ZIP archives from data/exports/, extracts contents, and organizes
by extension (e.g. .clas.abap, .intf.abap, .fugr.abap) into data/zsample_extracted/.
Uses abap_extensions from config.yaml.
"""

import argparse
import logging
import zipfile
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
EXPORTS_DIR = DATA_DIR / "exports"
OUTPUT_DIR = DATA_DIR / "zsample_extracted"

# Default extensions if config not available
DEFAULT_ABAP_EXTENSIONS = [
    ".clas.abap",
    ".intf.abap",
    ".fugr.abap",
    ".prog.abap",
    ".incl.abap",
    ".tabl.abap",
    ".dtel.abap",
    ".doma.abap",
    ".msag.abap",
    ".ttyp.abap",
    ".shlp.abap",
    ".enqu.abap",
    ".func.abap",
]


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


def get_abap_extensions(config: dict) -> list[str]:
    """Get ABAP file extensions from config, or use defaults."""
    exts = config.get("extraction", {}).get("abap_extensions", [])
    return exts if exts else DEFAULT_ABAP_EXTENSIONS


def is_abap_file(filename: str, extensions: list[str]) -> bool:
    """Check if filename has one of the ABAP extensions."""
    name_lower = filename.lower()
    return any(name_lower.endswith(ext) for ext in extensions)


def organize_extracted_files(
    extract_root: Path,
    output_dir: Path,
    extensions: list[str],
) -> dict[str, int]:
    """
    Organize extracted files by ABAP object type into subdirectories.

    Returns dict mapping extension to file count.
    """
    counts: dict[str, int] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in extract_root.rglob("*"):
        if not file_path.is_file():
            continue
        rel_name = file_path.name
        if not is_abap_file(rel_name, extensions):
            continue

        # Determine extension (e.g. .clas.abap)
        matched_ext = None
        for ext in extensions:
            if rel_name.lower().endswith(ext):
                matched_ext = ext
                break
        if not matched_ext:
            continue

        # Subdir by extension without leading dot (e.g. clas_abap)
        subdir_name = matched_ext.lstrip(".").replace(".", "_")
        subdir = output_dir / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)

        dest = subdir / rel_name
        if dest != file_path:
            # Handle duplicates by adding suffix
            if dest.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                counter = 1
                while dest.exists():
                    dest = subdir / f"{stem}_{counter}{suffix}"
                    counter += 1
            dest.write_bytes(file_path.read_bytes())
            logger.debug("Organized %s -> %s", rel_name, dest.relative_to(output_dir))

        counts[matched_ext] = counts.get(matched_ext, 0) + 1

    return counts


def unzip_and_organize() -> dict[str, int]:
    """
    Unzip all ZIPs from data/exports/ and organize by ABAP type.

    Returns dict mapping extension to file count.
    """
    config = load_config()
    extensions = get_abap_extensions(config)

    if not EXPORTS_DIR.exists():
        logger.warning("Exports directory does not exist: %s", EXPORTS_DIR)
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        return {}

    zip_files = list(EXPORTS_DIR.glob("*.zip"))
    if not zip_files:
        logger.warning("No ZIP files found in %s", EXPORTS_DIR)
        return {}

    # Temporary extraction root
    temp_extract = OUTPUT_DIR / "_temp_extract"
    temp_extract.mkdir(parents=True, exist_ok=True)

    try:
        for zip_path in zip_files:
            logger.info("Extracting %s", zip_path.name)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(temp_extract)
            except zipfile.BadZipFile as e:
                logger.error("Invalid ZIP %s: %s", zip_path, e)
                continue
            except Exception as e:
                logger.exception("Failed to extract %s: %s", zip_path, e)
                continue

        counts = organize_extracted_files(temp_extract, OUTPUT_DIR, extensions)
    finally:
        # Cleanup temp
        if temp_extract.exists():
            import shutil
            try:
                shutil.rmtree(temp_extract)
            except OSError as e:
                logger.warning("Could not remove temp dir: %s", e)

    return counts


def main() -> None:
    """Main entry point for unzip and organize."""
    parser = argparse.ArgumentParser(
        description="Unzip ABAP exports and organize by object type",
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
        counts = unzip_and_organize()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        total = sum(counts.values())
        logger.info("Organized %d ABAP files to %s", total, OUTPUT_DIR)
        for ext, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            logger.info("  %s: %d", ext, cnt)
    except Exception as e:
        logger.exception("Unzip and organize failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
