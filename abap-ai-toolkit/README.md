# ABAP AI Toolkit

Python-based system that teaches AI models to understand and generate SAP ABAP code for ECC-to-Clean-Core migration.

## Quick Start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
playwright install chromium
python 3_rag/rag_rebuild.py     # Build vector DB from rag_export.json
```

## Pipeline

1. `python 1_extract/sap_package_downloader.py /ZSAMPLE/MAIN` - Extract ABAP from SAP
2. `python 2_map_skills/scan_package.py` - Syntax scan
3. `python 2_map_skills/migrate_package.py` - Pass 1 migration
4. `python 2_map_skills/deep_migrate_package.py` - Pass 2 migration
5. `python 2_map_skills/java_abap_mapper.py` - Java-ABAP skill mapping
6. `python 2_map_skills/twin_discovery_package.py` - Semantic twin discovery
7. `python 2_map_skills/threeway_mapper.py` - 3-way mapping + CRV
8. `python 3_rag/index_all.py` - RAG indexing
9. `python 3_rag/rag_query_api.py "query"` - Query knowledge base

## Documentation

See `ABAP_AI_Toolkit_Complete_Context.md` for full project context and replication guide.

## Author

Kishore Shivanna
