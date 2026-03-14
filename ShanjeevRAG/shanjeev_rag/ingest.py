"""Ingest documents from data/raw: chunk, embed, store in ChromaDB."""
import hashlib
import logging
from pathlib import Path

from shanjeev_rag.config import PROJECT_ROOT, load_config, get_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def load_document(file_path: Path) -> str:
    """Load a single document as text."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix in (".txt", ".md", ".markdown"):
        return file_path.read_text(encoding="utf-8", errors="replace")
    logger.warning("Unsupported format %s, skipping %s", suffix, file_path.name)
    return ""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    if not text or not text.strip():
        return []
    chunks = []
    start = 0
    text = text.replace("\r\n", "\n")
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            # Try to break at a sentence or newline
            last_newline = chunk.rfind("\n")
            last_period = chunk.rfind(". ")
            break_at = max(last_newline, last_period)
            if break_at > chunk_size // 2:
                chunk = text[start : start + break_at + 1]
                end = start + break_at + 1
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = end
    return chunks


def iter_documents(data_dir: Path):
    """Yield (content, source_path) for each supported file under data_dir."""
    if not data_dir.exists():
        logger.warning("Data dir does not exist: %s", data_dir)
        return
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".pdf", ".txt", ".md", ".markdown"):
            continue
        content = load_document(path)
        if content.strip():
            yield content, str(path.relative_to(PROJECT_ROOT))


def build_index(reset: bool = False) -> None:
    """Read documents from config data_dir, chunk, embed, and store in ChromaDB."""
    config = load_config()
    data_dir = get_path("data_dir")
    vector_db_path = get_path("vector_db_path")
    model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
    chunk_size = config.get("chunk_size", 1000)
    chunk_overlap = config.get("chunk_overlap", 200)
    collection_name = config.get("collection_name", "insurance_rag")

    vector_db_path.mkdir(parents=True, exist_ok=True)

    try:
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Install chromadb and sentence-transformers: pip install chromadb sentence-transformers"
        ) from e

    # Persistent client
    client = chromadb.PersistentClient(
        path=str(vector_db_path),
        settings=Settings(anonymized_telemetry=False),
    )

    if reset and collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection %s", collection_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "ShanjeevRAG insurance knowledge base"},
    )

    # Load embedding model
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    documents = []
    metadatas = []
    ids = []

    for content, source_path in iter_documents(data_dir):
        chunks = chunk_text(content, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            doc_id = hashlib.sha256(f"{source_path}:{i}:{chunk[:50]}".encode()).hexdigest()[:24]
            documents.append(chunk)
            metadatas.append({"source": source_path, "chunk": i})
            ids.append(doc_id)

    if not documents:
        logger.warning("No documents found in %s. Add PDF/txt/md files and run again.", data_dir)
        return

    logger.info("Embedding %d chunks...", len(documents))
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    # ChromaDB prefers batches for large inserts
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
    logger.info("Indexed %d chunks into %s", len(documents), collection_name)
