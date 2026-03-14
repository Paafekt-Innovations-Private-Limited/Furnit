"""Load config from config.yaml relative to project root."""
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

# Project root = parent of shanjeev_rag package
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


def load_config():
    path = PROJECT_ROOT / "config.yaml"
    if not path.exists():
        return _default_config()
    if yaml is None:
        return _default_config()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data else _default_config()


def _default_config():
    return {
        "data_dir": "data/raw",
        "vector_db_path": "vector_db",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 5,
        "collection_name": "insurance_rag",
    }


def get_path(key: str) -> Path:
    """Return project-root-relative path from config."""
    cfg = load_config()
    value = cfg.get(key, "")
    return PROJECT_ROOT / value if value else PROJECT_ROOT
