#!/usr/bin/env python3
"""
Playwright-based abapGit ZIP export from SAP WebGUI.

Automates the abapGit ZIP download flow using Chromium browser.
Handles SAP login and persists session to sap_login_state.json for reuse.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

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
CONFIG_PATH = TOOLKIT_ROOT / "config.yaml"
ENV_PATH = TOOLKIT_ROOT / ".env"
DATA_DIR = TOOLKIT_ROOT / "data"
EXPORTS_DIR = DATA_DIR / "exports"
LOGIN_STATE_PATH = DATA_DIR / "sap_login_state.json"


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
        logger.info("Loaded .env from %s", ENV_PATH)
    elif load_dotenv:
        load_dotenv()
    else:
        logger.debug("python-dotenv not installed, using environment variables only")

    return config


def get_webgui_url(config: dict) -> str:
    """Get SAP WebGUI URL from config or environment."""
    webgui_url = config.get("sap", {}).get("webgui_url", "")
    if webgui_url and webgui_url.startswith("${") and webgui_url.endswith("}"):
        env_var = webgui_url[2:-1]
        webgui_url = os.environ.get(env_var, "")
    if not webgui_url:
        webgui_url = os.environ.get("SAP_WEBGUI_URL", "")
    if not webgui_url:
        raise ValueError("SAP_WEBGUI_URL not configured. Set in .env or config.yaml")
    return webgui_url.rstrip("/")


def perform_login(page, user: str, password: str, webgui_url: str) -> bool:
    """Perform SAP login and return True if successful."""
    try:
        page.goto(webgui_url, timeout=60000)
        page.wait_for_load_state("domcontentloaded", timeout=15000)

        # Common SAP login field selectors
        user_selector = 'input[name="sap-user"], input[id*="user"], input[placeholder*="User"]'
        pass_selector = 'input[name="sap-password"], input[id*="password"], input[type="password"]'
        submit_selector = 'input[type="submit"], button[type="submit"], input[name="sap-login"]'

        if page.locator(user_selector).count() > 0:
            page.locator(user_selector).first.fill(user)
            page.locator(pass_selector).first.fill(password)
            page.locator(submit_selector).first.click()
            page.wait_for_load_state("networkidle", timeout=30000)
            return True
    except Exception as e:
        logger.warning("Login flow error: %s", e)
    return False


def export_package_via_abapgit(
    page,
    package_name: str,
    downloads_dir: Path,
) -> Optional[Path]:
    """
    Navigate to abapGit and trigger ZIP export for the given package.

    Returns path to downloaded ZIP file, or None if failed.
    """
    # abapGit is typically accessed via transaction or URL in SAP
    # This is a simplified flow - real implementation depends on SAP system setup
    logger.info("Triggering abapGit export for package: %s", package_name)

    try:
        # Placeholder: real flow would navigate to abapGit UI and click export
        # Common patterns: /sap/bc/ui2/flp or custom abapGit transaction
        page.wait_for_timeout(2000)

        with page.expect_download(timeout=60000) as download_info:
            # Simulate export button click - selector depends on abapGit UI
            export_btn = page.locator('button:has-text("Export"), a:has-text("Export"), input[value*="Export"]')
            if export_btn.count() > 0:
                export_btn.first.click()
            else:
                logger.warning("Export button not found - abapGit UI may differ")
                return None

        download = download_info.value
        safe_name = package_name.replace("/", "_").strip("_")
        zip_path = downloads_dir / f"{safe_name}.zip"
        download.save_as(zip_path)
        logger.info("Saved ZIP to %s", zip_path)
        return zip_path
    except Exception as e:
        logger.warning("abapGit export failed for %s: %s", package_name, e)
        return None


def run_browser_export(packages: list[str], config: dict) -> list[Path]:
    """
    Run Playwright browser automation to export packages via abapGit.

    Returns list of paths to downloaded ZIP files.
    """
    if sync_playwright is None:
        raise ImportError("Playwright required. Run: pip install playwright && playwright install chromium")

    webgui_url = get_webgui_url(config)
    user = os.environ.get("SAP_USER", "")
    password = os.environ.get("SAP_PASS", "")

    if not user or not password:
        raise ValueError("SAP_USER and SAP_PASS must be set in .env")

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_zips: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context_options = {
            "accept_downloads": True,
            "viewport": {"width": 1280, "height": 720},
        }

        # Try to restore saved session
        if LOGIN_STATE_PATH.exists():
            try:
                with open(LOGIN_STATE_PATH, "r") as f:
                    state = json.load(f)
                context_options["storage_state"] = state
                logger.info("Restored session from %s", LOGIN_STATE_PATH)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Could not restore session: %s", e)

        context = browser.new_context(**context_options)
        page = context.new_page()

        try:
            page.goto(webgui_url, timeout=60000)
            page.wait_for_load_state("domcontentloaded", timeout=15000)

            # Check if we need to login (look for login form)
            if page.locator('input[name="sap-user"], input[type="password"]').count() > 0:
                if perform_login(page, user, password, webgui_url):
                    # Save session for next run
                    context.storage_state(path=str(LOGIN_STATE_PATH))
                    logger.info("Logged in and saved session to %s", LOGIN_STATE_PATH)
                else:
                    raise RuntimeError("SAP login failed")

            for package_name in packages:
                zip_path = export_package_via_abapgit(page, package_name, EXPORTS_DIR)
                if zip_path and zip_path.exists():
                    downloaded_zips.append(zip_path)

        finally:
            browser.close()

    return downloaded_zips


def main() -> None:
    """Main entry point for browser-based abapGit export."""
    parser = argparse.ArgumentParser(
        description="Export ABAP packages via abapGit ZIP using Playwright",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        default=[],
        help="Package names to export (e.g. /ZSAMPLE/SHARED /ZSAMPLE/MAIN)",
    )
    parser.add_argument(
        "--config-packages",
        action="store_true",
        help="Use packages from config.yaml if --packages not provided",
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

    config = load_config()

    packages = args.packages
    if not packages and args.config_packages:
        packages = config.get("sap", {}).get("packages", [])
    if not packages:
        parser.error("Provide --packages or use --config-packages to use config.yaml")

    try:
        zips = run_browser_export(packages, config)
        logger.info("Exported %d ZIP(s) to %s", len(zips), EXPORTS_DIR)
        for z in zips:
            print(z)
    except Exception as e:
        logger.exception("Browser export failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
