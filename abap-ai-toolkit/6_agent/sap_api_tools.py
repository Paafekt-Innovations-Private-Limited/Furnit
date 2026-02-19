#!/usr/bin/env python3
"""
SAP OData API tools for the agent.

Provides tools to query SAP OData services.
Uses requests with SAP authentication.
Supports: entity metadata, service catalog, entity data.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from urllib.parse import quote

try:
    import requests
except ImportError:
    requests = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = SCRIPT_DIR.parent


def get_sap_session(
    base_url: str,
    username: str | None = None,
    password: str | None = None,
) -> requests.Session | None:
    """Create authenticated session for SAP OData (basic auth or cookie)."""
    if not requests:
        logger.warning("requests not installed")
        return None
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
    if username and password:
        session.auth = (username, password)
    session.base_url = base_url.rstrip("/")
    return session


def get_entity_metadata(service_url: str, entity_set: str, session: requests.Session | None = None) -> dict:
    """
    Fetch OData entity set metadata.

    Args:
        service_url: OData service base URL.
        entity_set: Entity set name (e.g. I_JournalEntry).
        session: Optional authenticated session.

    Returns:
        Metadata dict or error.
    """
    if not requests:
        return {"error": "requests not installed"}
    url = f"{service_url.rstrip('/')}/{entity_set}/$metadata"
    try:
        resp = (session or requests).get(url, timeout=30)
        resp.raise_for_status()
        return {"metadata": resp.json() if "json" in resp.headers.get("Content-Type", "") else resp.text}
    except Exception as e:
        return {"error": str(e)}


def get_service_catalog(service_url: str, session: requests.Session | None = None) -> dict:
    """Fetch OData service catalog (entity sets)."""
    if not requests:
        return {"error": "requests not installed"}
    url = f"{service_url.rstrip('/')}/"
    try:
        resp = (session or requests).get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json() if "json" in resp.headers.get("Content-Type", "") else {}
        return {"value": data.get("value", []), "entity_sets": [e.get("name") for e in data.get("value", [])]}
    except Exception as e:
        return {"error": str(e)}


def get_entity_data(
    service_url: str,
    entity_set: str,
    top: int = 10,
    filter_query: str | None = None,
    session: requests.Session | None = None,
) -> dict:
    """
    Fetch entity data from OData service.

    Args:
        service_url: OData service base URL.
        entity_set: Entity set name.
        top: Max results.
        filter_query: OData $filter query.
        session: Optional authenticated session.

    Returns:
        Entity data dict.
    """
    if not requests:
        return {"error": "requests not installed"}
    url = f"{service_url.rstrip('/')}/{entity_set}?$top={top}"
    if filter_query:
        url += f"&$filter={quote(filter_query)}"
    try:
        resp = (session or requests).get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    """CLI entry point for SAP API tools."""
    parser = argparse.ArgumentParser(
        description="SAP OData API tools - entity metadata, catalog, data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("SAP_ODATA_URL", "https://my.sap.system/sap/opu/odata/sap/API_OPLAC"),
        help="OData service base URL",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Entity set name for metadata/data",
    )
    parser.add_argument(
        "--catalog",
        action="store_true",
        help="Fetch service catalog",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Max results for entity data",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=os.getenv("SAP_USER"),
        help="SAP username",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.getenv("SAP_PASSWORD"),
        help="SAP password",
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

    session = get_sap_session(args.url, args.user, args.password) if args.user else None

    try:
        if args.catalog:
            result = get_service_catalog(args.url, session)
        elif args.entity:
            result = get_entity_data(args.url, args.entity, top=args.top, session=session)
            if "error" in result:
                result = get_entity_metadata(args.url, args.entity, session)
        else:
            parser.print_help()
            return
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        logger.exception("Request failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
