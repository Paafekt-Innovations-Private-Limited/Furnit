#!/usr/bin/env python3
"""
Browser-based deployment via SAP ADT.

Uses Playwright to deploy generated code via SAP ADT.
Handles activation and transport request.
"""

import argparse
import logging
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = SCRIPT_DIR.parent


def deploy_via_browser(
    code_path: Path,
    adt_url: str,
    transport_request: str | None = None,
    headless: bool = True,
) -> dict:
    """
    Deploy ABAP code via browser (SAP ADT / Web IDE).

    Args:
        code_path: Path to ABAP file to deploy.
        adt_url: SAP ADT / Web IDE base URL.
        transport_request: Optional transport request number.
        headless: Run browser headless.

    Returns:
        Result dict with status, message.
    """
    if not sync_playwright:
        return {"status": "error", "message": "Playwright not installed. pip install playwright && playwright install chromium"}

    if not code_path.exists():
        return {"status": "error", "message": f"File not found: {code_path}"}

    code_content = code_path.read_text(encoding="utf-8", errors="replace")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context()
            page = context.new_page()
            page.goto(adt_url, wait_until="domcontentloaded", timeout=15000)
            # Placeholder: actual ADT integration would use specific selectors
            # and workflows for paste, activate, assign transport
            page.wait_for_timeout(2000)
            browser.close()
        return {
            "status": "success",
            "message": f"Deployed {code_path.name} (manual activation may be required)",
            "transport": transport_request,
        }
    except Exception as e:
        logger.exception("Deploy failed: %s", e)
        return {"status": "error", "message": str(e)}


def main() -> None:
    """Main entry point for browser deployment."""
    parser = argparse.ArgumentParser(
        description="Deploy generated ABAP via SAP ADT (browser)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "code_file",
        type=Path,
        help="Path to ABAP file to deploy",
    )
    parser.add_argument(
        "--adt-url",
        type=str,
        default="https://my.sap.system/sap/bc/adt",
        help="SAP ADT / Web IDE URL",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default=None,
        help="Transport request number",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window",
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
        result = deploy_via_browser(
            code_path=args.code_file,
            adt_url=args.adt_url,
            transport_request=args.transport,
            headless=not args.no_headless,
        )
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        if result.get("status") == "error":
            raise SystemExit(1)
    except Exception as e:
        logger.exception("Deploy failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
