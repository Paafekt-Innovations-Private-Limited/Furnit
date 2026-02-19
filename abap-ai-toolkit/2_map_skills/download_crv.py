#!/usr/bin/env python3
"""
Download CRV JSON from SAP GitHub.

Downloads from sap.github.io/abap-atc-cr-cv-s4hc/. Saves to
data/twin_discovery/crv_latest_full.json. Parses FM release status.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

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
TWIN_DIR = DATA_DIR / "twin_discovery"
OUTPUT_PATH = TWIN_DIR / "crv_latest_full.json"

# SAP CRV base URL (Cloudification Repository Viewer for S/4HANA Cloud)
CRV_BASE_URL = "https://sap.github.io/abap-atc-cr-cv-s4hc/"
CRV_JSON_URLS = [
    "https://raw.githubusercontent.com/SAP/abap-atc-cr-cv-s4hc/main/cv/cv.json",
    "https://sap.github.io/abap-atc-cr-cv-s4hc/cv.json",
]


def fetch_url(url: str, timeout: int = 60) -> str:
    """
    Fetch URL content as string.

    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Response body as string.

    Raises:
        URLError, HTTPError on failure.
    """
    req = Request(url, headers={"User-Agent": "ABAP-AI-Toolkit/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_crv_for_fms(data: Any) -> dict[str, Any]:
    """
    Parse CRV JSON to extract FM release status.

    Args:
        data: Parsed JSON from CRV.

    Returns:
        Dict mapping FM name -> {releaseState, successor, ...}.
    """
    result: dict[str, Any] = {}
    if isinstance(data, dict):
        # Common structures: objects, functionModules, findings
        for key in ("objects", "functionModules", "findings", "data"):
            items = data.get(key)
            if items is None:
                continue
            if isinstance(items, list):
                for obj in items:
                    name = obj.get("name", obj.get("objectName", obj.get("functionModule", "")))
                    if name:
                        result[str(name).upper()] = {
                            "releaseState": obj.get("releaseState", obj.get("status", "unknown")),
                            "successor": obj.get("successor", obj.get("replacement", "")),
                            "objectType": obj.get("objectType", "FUNC"),
                        }
            elif isinstance(items, dict):
                for name, info in items.items():
                    if isinstance(info, dict):
                        result[str(name).upper()] = info
                    else:
                        result[str(name).upper()] = {"releaseState": str(info)}
    return result


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download CRV JSON from SAP GitHub."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output JSON path (default: data/twin_discovery/crv_latest_full.json)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Override CRV JSON URL",
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

    urls_to_try = [args.url] if args.url else CRV_JSON_URLS
    data: Any = None
    last_error: Exception | None = None

    for url in urls_to_try:
        try:
            logger.info("Fetching %s", url)
            content = fetch_url(url)
            data = json.loads(content)
            logger.info("Parsed CRV JSON successfully")
            break
        except (URLError, HTTPError, json.JSONDecodeError) as e:
            last_error = e
            logger.warning("Failed %s: %s", url, e)
            continue

    if data is None:
        logger.error("Could not download CRV from any URL. Last error: %s", last_error)
        # Write minimal placeholder for offline use
        args.output.parent.mkdir(parents=True, exist_ok=True)
        placeholder = {
            "source": "placeholder",
            "note": "CRV download failed. Run with network access.",
            "objects": [],
            "fm_index": {},
        }
        try:
            args.output.write_text(json.dumps(placeholder, indent=2), encoding="utf-8")
            logger.info("Wrote placeholder to %s", args.output)
        except OSError as e:
            logger.error("Could not write placeholder: %s", e)
        return 1

    # Enrich with parsed FM index
    fm_index = parse_crv_for_fms(data)
    if isinstance(data, dict):
        data["fm_index"] = fm_index
        data["fm_count"] = len(fm_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved CRV to %s (%d FMs indexed)", args.output, len(fm_index))
    except OSError as e:
        logger.error("Failed to write %s: %s", args.output, e)
        return 1

    print("\n=== CRV Download Summary ===")
    print(f"Output: {args.output}")
    print(f"Function modules indexed: {len(fm_index)}")
    if fm_index:
        sample = list(fm_index.items())[:5]
        for name, info in sample:
            print(f"  {name}: {info.get('releaseState', 'N/A')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
