# ABAP AI Toolkit

AI-powered system that migrates SAP ABAP code from ECC to S/4HANA Clean Core. Works with any SAP Z-package exported via abapGit.

## 3-Step Quickstart

### Step 1: Export your ABAP package from SAP

In your SAP system, use **abapGit** to export your custom Z-package as a ZIP:
- Open abapGit in SAP GUI or Fiori
- Select your package (e.g. `/YOUR_PKG/MAIN`)
- Export as ZIP file to your laptop

### Step 2: Run the migration pipeline

```bash
# One-time setup
git clone <this-repo>
cd abap-ai-toolkit
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt

# Run pipeline on your ZIP
python run_pipeline.py /path/to/your_package.zip
```

This runs 6 stages automatically:
1. Extract & organize ABAP objects from ZIP
2. Syntax scan (obsolete patterns, FM classification)
3. Pass 1 migration (14 regex transforms)
4. Pass 2 migration (complex multi-line patterns)
5. Semantic twin discovery (FM -> Clean Core API mapping)
6. RAG indexing (vector database for AI queries)

### Step 3: Open in Cursor and ask migration questions

1. Open this `abap-ai-toolkit/` folder in **Cursor**
2. The `.cursor/rules/` auto-loads migration context (CRV mappings, RAP patterns, syntax rules)
3. Ask questions like:
   - "How do I migrate BAPI_ACC_DOCUMENT_POST to Clean Core?"
   - "What is the RAP equivalent of this CALL FUNCTION?"
   - "Generate a wrapper class for BAPI_GOODSMVT_CREATE"

## What the Pipeline Does

| Stage | Script | What it does |
|-------|--------|-------------|
| Extract | `run_pipeline.py` | Unzips abapGit export, organizes by ABAP object type |
| Scan | `2_map_skills/scan_package.py` | Finds obsolete patterns (MOVE, ADD, CALL METHOD, etc.) |
| Migrate P1 | `2_map_skills/migrate_package.py` | 14 regex transforms (syntax modernization) |
| Migrate P2 | `2_map_skills/deep_migrate_package.py` | Complex patterns (DATA chains, multi-line CALL METHOD) |
| Twins | `2_map_skills/twin_discovery_package.py` | Maps legacy FMs to modern Clean Core APIs by purpose |
| RAG | `3_rag/index_all.py` | Indexes code + rules into ChromaDB for AI queries |

## Input Format

The toolkit accepts:
- **ZIP file** from abapGit (any SAP Z-package)
- **Folder** containing ABAP source files (`.clas.abap`, `.fugr.abap`, `.prog.abap`, etc.)

## Key Concepts

### Semantic Twin Discovery (Patent)
Instead of name-matching, the toolkit finds modern API replacements by **purpose**:
- `BAPI_ACC_DOCUMENT_POST` (purpose: "post FI journal entry") -> `I_JournalEntryTP` (RAP BO)
- `BAPI_GOODSMVT_CREATE` (purpose: "create material document") -> `I_MaterialDocumentTP` (RAP BO)

### Cross-Lingual Transfer Learning
Uses Java/Python skill anchoring to teach AI models ABAP vocabulary:
- Java `target = source;` -> ABAP `target = source.` (replaces `MOVE source TO target.`)
- Java `list.size()` -> ABAP `lines( itab )` (replaces `DESCRIBE TABLE itab LINES n.`)

### CRV (Custom Code Compatibility) Verification
Validates FM replacements against SAP's official Compatibility Rule Verification data.

## Advanced Usage

```bash
# Skip RAG (if chromadb not installed)
python run_pipeline.py --skip-rag /path/to/package.zip

# Clean previous data before running
python run_pipeline.py --clean /path/to/package.zip

# Query RAG directly
python 3_rag/rag_query_api.py "BAPI_ACC_DOCUMENT_POST successor"

# Rebuild RAG on a new machine (from exported JSON)
python 3_rag/rag_rebuild.py
```

## Project Structure

```
abap-ai-toolkit/
  run_pipeline.py              # Single entry point (start here)
  config.yaml                  # Configuration
  requirements.txt             # Python dependencies
  .cursor/rules/               # Auto-loaded Cursor AI rules
  1_extract/                   # ZIP extraction + SAP download
  2_map_skills/                # Scan, migrate, twin discovery
  3_rag/                       # RAG indexing + query
  4_twin_discovery/            # Domain-specific twin analysis
  5_fine_tune/                 # Model fine-tuning (advanced)
  6_agent/                     # Code generation agent (advanced)
  data/                        # All generated data (gitignored except exports)
```

## Documentation

See `ABAP_AI_Toolkit_Complete_Context.md` for full technical context and architecture details.

## Author

Kishore Shivanna
