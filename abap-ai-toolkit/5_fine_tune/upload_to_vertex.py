#!/usr/bin/env python3
"""
Upload training data to Google Cloud Storage.

Uploads JSONL to GCS bucket for Vertex AI fine-tuning.
Returns GCS URI for fine-tuning job.
"""

import argparse
import json
import logging
from pathlib import Path

try:
    from google.cloud import storage
except ImportError:
    storage = None

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
DEFAULT_JSONL_PATH = DATA_DIR / "training_data.jsonl"


def upload_to_gcs(
    jsonl_path: Path,
    bucket_name: str,
    blob_name: str | None = None,
) -> str:
    """
    Upload JSONL file to Google Cloud Storage.

    Args:
        jsonl_path: Path to JSONL training file.
        bucket_name: GCS bucket name.
        blob_name: Optional blob path (default: training_data/YYYYMMDD/training_data.jsonl).

    Returns:
        GCS URI (gs://bucket/path).
    """
    if not storage:
        raise ImportError("google-cloud-storage not installed. pip install google-cloud-storage")

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Training file not found: {jsonl_path}")

    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")
    blob_name = blob_name or f"training_data/{date_str}/training_data.jsonl"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    logger.info("Uploading %s to gs://%s/%s", jsonl_path, bucket_name, blob_name)
    blob.upload_from_filename(str(jsonl_path), content_type="application/jsonl")

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    logger.info("Upload complete: %s", gcs_uri)
    return gcs_uri


def main() -> None:
    """Main entry point for GCS upload."""
    parser = argparse.ArgumentParser(
        description="Upload training data to GCS for Vertex AI fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_JSONL_PATH,
        help="Input JSONL path (default: data/training_data.jsonl)",
    )
    parser.add_argument(
        "-b",
        "--bucket",
        type=str,
        required=True,
        help="GCS bucket name",
    )
    parser.add_argument(
        "-o",
        "--blob",
        type=str,
        default=None,
        help="GCS blob path (default: training_data/YYYYMMDD/training_data.jsonl)",
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
        gcs_uri = upload_to_gcs(
            jsonl_path=args.input,
            bucket_name=args.bucket,
            blob_name=args.blob,
        )
        print(f"\nGCS URI: {gcs_uri}\n")
        print("Use this URI in fine_tune_gemini.py --training-data-uri")
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        raise SystemExit(1)
    except Exception as e:
        logger.exception("Upload failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
