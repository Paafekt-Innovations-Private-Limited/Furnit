# ABAP AI Toolkit

## Complete Project Context & Replication Guide

**Prepared by:** Kishore Shivanna
**Date:** February 15, 2026
**Author:** Kishore Shivanna

---

## Table of Contents

1. Executive Summary
2. What This Toolkit Does
3. Key Architectural Concepts (5 Pillars)
   - 3.1 Cross-Lingual Transfer Learning (Java to ABAP)
   - 3.2 Semantic Twin Discovery (Patent Concept)
   - 3.3 RAG as Model-Agnostic Knowledge Base
   - 3.4 Deterministic Syntax Rules + CRV Verification
   - 3.5 Doc-Driven ABAP Validation (No Kernel)
4. System Prerequisites
5. Project Directory Structure
6. Setup Instructions (New Laptop)
7. Pipeline Stages (Step-by-Step)
   - 7.1 Stage 1: Extract ABAP Code from SAP
   - 7.2 Stage 2a: Syntax Scan
   - 7.3 Stage 2b: Pass 1 Migration (Simple Transforms)
   - 7.4 Stage 2c: Pass 2 Migration (Complex Patterns)
   - 7.5 Stage 2d: Java-ABAP Skill Mapping
   - 7.6 Stage 2e: Semantic Twin Discovery
   - 7.7 Stage 2f: 3-Way Mapping + CRV Verification
   - 7.8 Stage 3: RAG Indexing
   - 7.9 Stage 4: Generated Clean Core Code
8. RAG Knowledge Base (1,044 Chunks)
   - 8.1 RAG Layers Breakdown
   - 8.2 How RAG Works in Cursor
   - 8.3 Real RAG vs Poor Man's RAG
   - 8.4 RAG Training Scripts
9. All Scripts Reference
   - 9.1 Phase 1: Extraction Scripts
   - 9.2 Phase 2: Migration & Mapping Scripts
   - 9.3 Phase 3: RAG Scripts
   - 9.4 Phase 4-7: Twin Discovery, Fine-Tune, Agent, MCP
10. Configuration Files
11. Migration Report Summary
    - 11.1 CRV Successor Mapping (FM Replacements)
    - 11.2 Class-Level Twin Mapping
    - 11.3 ABAP Syntax Modernization Rules
    - 11.4 Wrapper Classes Generated
    - 11.5 Validation Results
12. Data Artifacts & File Counts
13. Cursor IDE Integration
    - 13.1 Cursor Rules (.mdc file)
    - 13.2 Workspace Settings
    - 13.3 MCP Server (AbapBLBrain)
14. Security Considerations
15. Sharing & Collaboration via Git
16. Patentable Ideas
17. Known Issues & Lessons Learned
18. Session Recovery Checklist

---

## 1. Executive Summary

The ABAP AI Toolkit is a Python-based system that teaches AI models (Claude, GPT-4, Gemini, etc.) to understand and generate SAP ABAP code for ECC-to-Clean-Core migration. It combines:

- Browser automation (Playwright) to download ABAP code from SAP WebGUI/SE80/ADT
- Deterministic regex-based ABAP syntax modernization (12 transformation rules)
- Cross-lingual transfer learning: teaching ABAP through Java skill anchoring
- Semantic Twin Discovery: finding modern SAP APIs by understanding legacy code PURPOSE
- RAG (Retrieval-Augmented Generation) with ChromaDB vector database (1,044 chunks)
- CRV verification against SAP's public GitHub for FM release status
- Doc-driven ABAP validation without SAP kernel (pattern-based compilation)
- Cursor IDE integration via .mdc rules file for model-agnostic knowledge injection
- SAP Fiori and RAP knowledge training for end-to-end S/4HANA development

The toolkit has been used to migrate /ZSAMPLE/SHARED (25 files) and /ZSAMPLE/MAIN (174 files) packages, producing 199+ migrated files, 7 wrapper classes, 9,426 three-way mappings, and a comprehensive migration report. All knowledge is persisted in RAG and can be shared via Git.

---

## 2. What This Toolkit Does

The ABAP AI Toolkit automates the SAP ECC-to-S/4HANA Clean Core migration pipeline:

| Step | Description |
|------|-------------|
| **Download** | Extract ABAP source code from SAP systems (WebGUI, SE80, ADT, abapGit) |
| **Scan** | Identify legacy syntax patterns, FM calls, and modernization opportunities |
| **Transform** | Apply deterministic regex rules to modernize ABAP syntax (MOVE->assign, CONCATENATE->string template, etc.) |
| **Skill Map** | Create 3-way mappings: Java <-> Old ABAP <-> New ABAP for cross-lingual transfer learning |
| **Twin Discovery** | Find modern Clean Core "twins" of legacy code by semantic purpose matching |
| **CRV Verify** | Check FM release status against SAP's Cloudification Repository Viewer (sap.github.io) |
| **RAG Index** | Store all knowledge in ChromaDB vector database for semantic search |
| **Generate** | Use RAG + LLM to generate new Clean Core ABAP code from natural language prompts |
| **Validate** | Validate generated code against SAP documentation patterns (no kernel needed) |

---

## 3. Key Architectural Concepts (5 Pillars)

### 3.1 Cross-Lingual Transfer Learning (Java to ABAP)

Programming is a universal skill: if/else, for/while, try/catch, class/method. LLMs already know these skills deeply through Java (massive public training data). ABAP has almost zero public training data, so LLMs hallucinate ABAP syntax. The solution: don't teach the LLM new skills - just teach it the ABAP VOCABULARY for skills it already has.

The 3-way mapping is a TEACHING MECHANISM, not a runtime translation pipeline:

- Claude knows `list.stream().filter().findFirst()` perfectly (Java skill)
- Mapping says: that skill in ABAP = `VALUE #( itab[ key = v ] OPTIONAL )`
- Now Claude generates correct ABAP directly, anchored to existing knowledge

Stats: 68 skill rules, 8,016 pattern matches, 12 rules can auto-transform code.

Script: `2_map_skills/java_abap_mapper.py`

### 3.2 Semantic Twin Discovery (Patent Concept)

Find the modern Clean Core "twin" of legacy code by understanding its PURPOSE, not just matching names:

- OLD: `CALL FUNCTION 'BAPI_ACC_DOCUMENT_POST'` (purpose: "post FI journal entry")
- TWIN: `I_JournalEntryTP` (RAP BO - released in ABAP Cloud)

Twin types: DIRECT (1:1 API), WRAPPER (wrap unreleased FM), REFACTOR (syntax only)

Cardinality-aware (v2): Real-world migration is NOT always 1:1.

| Type | Description | Example |
|------|-------------|---------|
| **1:1** | Old FM -> New API | BAPI_GOODSMVT_CREATE -> I_MaterialDocumentTP |
| **1:N** | One old God-class -> Many new APIs | CL_SAP_CALLS -> 10+ RAP BOs + wrappers |
| **N:1** | Many old classes -> One new API | 7 FI doc classes -> I_JournalEntryTP |
| **N:M** | Complex recomposition | SO classes -> separate RAP BOs per doc type |

### 3.3 RAG as Model-Agnostic Knowledge Base

ChromaDB stores 1,044 chunks (ABAP code + Fiori + RAP knowledge). Any model retrieves correct examples at query time via In-Context Learning / Few-Shot Learning. The RAG grounds the LLM's output in verified, real production code from 400+ files - not generic examples.

- Embedding model: all-MiniLM-L6-v2 (SentenceTransformer)
- Database: ChromaDB PersistentClient at `data/vector_db/`
- Export: `data/rag_export.json` (2,802 KB, tracked in git)

### 3.4 Deterministic Syntax Rules + CRV Verification

The migration scripts use regex-based Old->New ABAP transformation (12 rules). These are deterministic (no hallucination possible). CRV data from SAP's public GitHub verifies FM release status and provides successor API mappings. Source: sap.github.io/abap-atc-cr-cv-s4hc/

### 3.5 Doc-Driven ABAP Validation (No Kernel)

ABAP syntax is validated against patterns from SAP official docs, NOT by SAP kernel:

- SAP ABAP Keyword Docs: help.sap.com/doc/abapdocu_latest_index_htm
- SAP RAP Docs: help.sap.com/docs/abap-cloud/abap-rap
- SAP code-pal-for-abap: github.com/SAP/code-pal-for-abap
- SAP ABAP Cloud restricted language: help.sap.com/doc/abapdocu_cp_index_htm/CLOUD

20 obsolete/style rules + 28 ABAP Cloud rules (15 categories) + structural checks.

Script: `2_map_skills/abap_doc_validator.py` (standard or --cloud mode)

---

## 4. System Prerequisites

| Component | Version/Details |
|-----------|----------------|
| **OS** | Windows 10/11 (tested on Win10 build 26100) |
| **Python** | 3.13.x (tested with 3.13.7) |
| **IDE** | Cursor IDE (VS Code fork with AI) |
| **Shell** | PowerShell |
| **Git** | Any recent version for repo management |
| **SAP Access** | WebGUI access to SAP system (for code extraction only) |
| **Disk Space** | ~500 MB for project + venv + vector_db |
| **Internet** | Required for pip install and CRV download from SAP GitHub |

### Python packages (from requirements.txt):

| Package | Version | Purpose |
|---------|---------|---------|
| **playwright** | 1.49.0 | Browser automation for SAP code extraction |
| **browser-use** | 0.4.0 | Browser automation helper |
| **anthropic** | 0.40.0 | Claude API (optional, for LLM code generation) |
| **google-cloud-aiplatform** | 1.74.0 | Vertex AI (optional, for fine-tuning) |
| **langchain** | 0.3.12 | LLM framework |
| **langchain-google-genai** | 2.0.0 | Google Gemini integration |
| **chromadb** | 0.5.20 | Vector database for RAG |
| **sentence-transformers** | 3.3.0 | Embedding model for semantic search |
| **pandas** | 2.2.0 | Data processing |
| **pyyaml** | 6.0.2 | YAML config parsing |
| **requests** | 2.32.0 | HTTP client for SAP ADT API |
| **pytest** | 8.3.0 | Testing framework |
| **pytest-asyncio** | 0.24.0 | Async test support |
| **python-docx** | 1.2.0 | Word document generation (this doc) |

---

## 5. Project Directory Structure

Workspace root: `c:\coding\ai\demo\demo\abap-ai-toolkit`

```
abap-ai-toolkit/
├── .cursor/rules/
│   └── abap-ai-toolkit.mdc          # Auto-loaded Cursor rule (model-agnostic, 213 lines)
├── 1_extract/                         # Phase 1: Get code out of SAP
│   ├── sap_package_downloader.py      # Multi-method SAP downloader (ADT/SE80/abapGit)
│   ├── browser_export.py              # Playwright-based abapGit ZIP export
│   ├── unzip_and_organize.py          # Organize ZIPs by ABAP object type
│   └── curate_dataset.py              # Quality scoring and deduplication
├── 2_map_skills/                      # Phase 2: Migration, mapping, twin discovery
│   ├── scan_sample_pkg.py              # Syntax scan + FM classification
│   ├── migrate_sample_pkg.py           # Pass 1: simple regex transforms (14 rules)
│   ├── deep_migrate_sample_pkg.py      # Pass 2: complex patterns (DATA chains, etc.)
│   ├── java_abap_mapper.py            # THE unified engine (detect+map+transform+RAG)
│   ├── threeway_mapper.py             # 3-way mapping + CRV verification
│   ├── full_skill_mapper.py           # Comprehensive skill mapping across packages
│   ├── twin_discovery_sample_pkg.py    # Semantic twin discovery (patent concept)
│   ├── abap_doc_validator.py          # Doc-driven syntax validation
│   ├── download_crv.py                # Download CRV JSON from SAP GitHub
│   ├── check_our_fms_in_crv.py        # Check specific FMs against CRV
│   ├── abap_syntax_modernizer.py      # Syntax modernization engine
│   └── check_coverage.py              # Coverage analysis
├── 3_rag/                             # Phase 3: RAG pipeline
│   ├── rag_query_api.py               # CLI/REST/Python query API (THE tool)
│   ├── rag_rebuild.py                 # Rebuild ChromaDB from rag_export.json
│   ├── rag_append.py                  # Append-only knowledge addition
│   ├── rag_append_threeway.py         # Append 3-way mappings
│   ├── index_zsample_all.py             # Index all ZSAMPLE knowledge
│   ├── index_all_knowledge.py         # Index all knowledge layers
│   ├── index_fiori_knowledge.py       # Index SAP Fiori knowledge (18 chunks)
│   ├── index_rap_knowledge.py         # Index SAP RAP knowledge (24 chunks)
│   ├── sap_api_catalog.py             # Index SAP modern APIs (21 RAP BOs + CDS)
│   ├── llm_clean_core_engine.py       # LLM+RAG clean core conversion engine
│   ├── rag_code_generator.py          # RAG-powered ABAP code generator
│   └── create_share_package.py        # Create shareable ZIP
├── 4_twin_discovery/                  # Semantic twin discovery engine
│   ├── discover_clean_core.py         # Discover released S/4HANA APIs
│   ├── match_twins.py                 # Map ECC -> Clean Core by meaning
│   ├── crv_verified_twins.py          # Cross-reference with CRV
│   └── FinClose_twin_analysis.py    # FinClose-specific analysis
├── 5_fine_tune/                       # Vertex AI fine-tuning (if RAG < 80%)
│   ├── prepare_training_data.py       # Convert to JSONL training pairs
│   ├── fine_tune_gemini.py            # Launch Gemini fine-tuning job
│   └── upload_to_vertex.py            # Upload training data
├── 6_agent/                           # Full AI agent with SAP tools
│   ├── code_generator.py              # RAG+Skills+LLM code generation
│   ├── pattern_validator.py           # 10-point validation checklist
│   ├── sap_api_tools.py               # SAP OData API tools
│   └── browser_deploy.py              # Browser-based deployment
├── 7_abap_bl_brain/                   # MCP server for Cursor
│   └── mcp_server.py                  # AbapBLBrain MCP server
├── tests/                             # Test suite
│   ├── test_code_generation.py
│   └── test_extraction.py
├── data/                              # All data artifacts
│   ├── sample_pkg_original/            # 174 original ECC ABAP files
│   ├── sample_pkg_migrated/            # 175 migrated files (174 + wrapper)
│   ├── sample_pkg_generated/           # 206 generated Clean Core files
│   ├── zsample_original/                # 25 ZSAMPLE_GENERIC original files
│   ├── zsample_migrated/                # 26 ZSAMPLE_GENERIC migrated files
│   ├── threeway_mappings/             # 406 JSON files (4 source dirs)
│   ├── twin_discovery/                # Twin catalogs + CRV data
│   ├── vector_db/                     # ChromaDB (NOT in git, rebuild)
│   ├── rag_export.json                # Full RAG export (2,802 KB, in git)
│   ├── rag_*.md                       # Markdown exports per layer
│   ├── pipeline_state.json            # Pipeline status tracking
│   └── sample_pkg_migrated.zip         # ZIP for review
├── config.yaml                        # Project configuration
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── .env.example                       # Environment variable template
├── .gitignore                         # Git exclusions
└── abap-ai-toolkit.code-workspace     # VS Code/Cursor workspace
```

---

## 6. Setup Instructions (New Laptop)

### Step 1: Install Prerequisites

1. Install Python 3.13.x from python.org (add to PATH)
2. Install Cursor IDE from cursor.sh
3. Install Git from git-scm.com

### Step 2: Clone or Copy the Project

```bash
# Option A: Clone from Git (if repo exists)
git clone <your-repo-url> abap-ai-toolkit
cd abap-ai-toolkit

# Option B: Copy entire folder from USB/network
# Copy c:\coding\ai\demo\demo\abap-ai-toolkit to new laptop
# Keep the SAME directory structure
```

### Step 3: Setup Python Virtual Environment

```bash
cd c:\coding\ai\demo\demo\abap-ai-toolkit

# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Install Playwright browsers (for SAP extraction)
playwright install chromium
```

### Step 4: Rebuild the RAG Vector Database

```bash
# This rebuilds ChromaDB from the tracked rag_export.json
python 3_rag/rag_rebuild.py

# Verify it worked
python 3_rag/rag_query_api.py "FI document posting"
# Should return relevant ABAP code chunks
```

### Step 5: Open in Cursor IDE

Open Cursor IDE and open the abap-ai-toolkit folder. The `.cursor/rules/abap-ai-toolkit.mdc` file will auto-load into ANY model you select (Claude, GPT-4, Gemini, etc.), giving it full project context including RAG layer details, CRV mappings, and modernization rules.

### Step 6: Configure SAP Access (Optional)

```bash
# Copy .env.example to .env and fill in your credentials
copy .env.example .env
# Edit .env with your SAP WebGUI URL, user, password

# Test SAP browser connection
python test_sap_browser.py
# This opens a browser, you log in once, session is saved
```

### Step 7: Verify Everything Works

```bash
# Query the RAG
python 3_rag/rag_query_api.py "BAPI_ACC_DOCUMENT_POST successor"

# Check pipeline state
python -c "import json; d=json.load(open('data/pipeline_state.json'));
print(json.dumps(d['pipeline_summary'], indent=2))"

# Run validation on a migrated file
python 2_map_skills/abap_doc_validator.py
data/sample_pkg_migrated/#zsample#cl_fi_doc_base.clas.abap
```

---

## 7. Pipeline Stages (Step-by-Step)

All stages are INCREMENTAL. Each stage reads from the previous stage's output. State is tracked in `data/pipeline_state.json`.

### Stage 1: Extract ABAP Code from SAP

**Script:** `1_extract/sap_package_downloader.py`

Downloads ABAP source from SAP via ADT REST API (headless), SE80 Playwright automation, or abapGit ZIP export. Requires SAP login credentials.

**Command:**
```bash
python 1_extract/sap_package_downloader.py /ZSAMPLE/MAIN --method auto --recursive
```

**Output:** `data/sample_pkg_original/` (174 files)

### Stage 2a: Syntax Scan

**Script:** `2_map_skills/scan_sample_pkg.py`

Scans all ABAP files for obsolete syntax patterns (MOVE, CONCATENATE, CREATE OBJECT, etc.) and classifies Function Module calls against CRV.

**Command:**
```bash
python 2_map_skills/scan_sample_pkg.py
```

**Output:** 4,766 syntax patterns found, 189 FM calls classified

### Stage 2b: Pass 1 Migration (Simple Transforms)

**Script:** `2_map_skills/migrate_sample_pkg.py`

Applies 14 ordered regex transformations: MOVE->assign, CONCATENATE->string template, CALL METHOD->functional, CREATE OBJECT->NEW, CHECK->IF/RETURN, TRANSLATE->to_upper/to_lower, etc.

**Command:**
```bash
python 2_map_skills/migrate_sample_pkg.py
```

**Output:** 175 files migrated, 147 transformations applied

### Stage 2c: Pass 2 Migration (Complex Patterns)

**Script:** `2_map_skills/deep_migrate_sample_pkg.py`

Handles complex multi-line patterns: DATA: chain splitting, CLEAR: chains, multi-line CALL METHOD, CREATE OBJECT with EXPORTING, READ TABLE->table expressions. Includes keyword safety guard for DATA chains.

**Command:**
```bash
python 2_map_skills/deep_migrate_sample_pkg.py
```

**Output:** 42 files changed, 77 deep fixes applied

### Stage 2d: Java-ABAP Skill Mapping

**Script:** `2_map_skills/java_abap_mapper.py`

THE unified engine. Detects patterns, maps to Java anchors (60+ skill rules, 12 categories), transforms code, verifies FMs against CRV, and indexes into RAG. Categories: class, method, data, control, table, exception, fm, string, auth, transaction, rap, sql.

**Command:**
```bash
python 2_map_skills/java_abap_mapper.py
```

**Output:** 8,081 patterns across 406 files, 54 unique rules, 12 transformable

### Stage 2e: Semantic Twin Discovery

**Script:** `2_map_skills/twin_discovery_sample_pkg.py`

Patent concept: matches legacy ABAP to modern SAP APIs by PURPOSE/MEANING. Uses SAP_TWIN_DB dictionary + CRV data. Generates wrapper classes when needed.

**Command:**
```bash
python 2_map_skills/twin_discovery_sample_pkg.py
```

**Output:** 15 direct twins, 11 wrapper twins, 133 refactor, 15 review. 7 wrapper classes generated.

### Stage 2f: 3-Way Mapping + CRV

**Script:** `2_map_skills/threeway_mapper.py`

Scans all 4 source directories (ZSAMPLE_GENERIC + /ZSAMPLE/MAIN, original + migrated). Produces per-line 3-way mappings with CRV-verified FM status and SAP doc references.

**Command:**
```bash
python 2_map_skills/threeway_mapper.py
```

**Output:** 400 files, 9,426 mappings, 52 unique rules. CRV: 2 released, 18 not released, 111 missing.

### Stage 3: RAG Indexing

**Script:** `3_rag/index_zsample_all.py` (and others)

Indexes all knowledge into ChromaDB: code, syntax rules, FM classification, twin discovery, Fiori patterns, RAP patterns. Then exports to `rag_export.json` for Git tracking.

**Command:**
```bash
python 3_rag/index_zsample_all.py
python 3_rag/index_fiori_knowledge.py
python 3_rag/index_rap_knowledge.py
```

**Output:** 1,044 total chunks across 17 layers

### Stage 4: Generated Clean Core Code

**Script:** Output of pipeline

Final migrated ABAP files with: modernized syntax, CRV successor annotations/replacements, wrapper class generation, inline comments for all changes.

**Command:**
```
Output at data/sample_pkg_generated/ and data/sample_pkg_migrated/
```

**Output:** 206 generated files, 175 migrated files, 7 new wrapper classes

---

## 8. RAG Knowledge Base (1,044 Chunks)

### 8.1 RAG Layers Breakdown

| Layer | What it Contains | Chunks |
|-------|-----------------|--------|
| **code** | Original + migrated ABAP source (ZSAMPLE packages) | 817 |
| **fm_classification** | Function Module release status + CRV actions | 131 |
| **syntax** | ABAP syntax modernization rules | 54 |
| **rap_impl** | RAP validations, determinations, actions, numbering, side effects | 6 |
| **rap_advanced** | Locking, ETag, error handling, unmanaged/additional save, testing | 5 |
| **fiori_elements** | Fiori Elements floorplans, draft, actions, side effects | 5 |
| **fiori_cds** | CDS UI annotations, value help, criticality, search | 5 |
| **rap_cds** | CDS view entities, composition trees, projections, abstract entities | 3 |
| **rap_behavior** | Behavior definitions, projections, draft handling | 3 |
| **fiori_ui5** | SAPUI5 freestyle: views, controllers, OData V4 binding | 3 |
| **rap_architecture** | RAP 3-layer architecture, managed/unmanaged scenarios | 2 |
| **rap_eml** | Entity Manipulation Language (internal + external consumption) | 2 |
| **rap_auth** | Authorization (global, instance), feature control | 2 |
| **fiori_manifest** | manifest.json, launchpad tiles, intent navigation | 2 |
| **fiori_patterns** | Golden rules, Elements vs Freestyle decision guide | 2 |
| **fiori_odata** | RAP service binding for Fiori (OData V4) | 1 |
| **rap_service** | Service definition, binding, OData V4 UI/API | 1 |

### 8.2 How RAG Works in Cursor

There are TWO ways the LLM accesses project knowledge in Cursor:

1. **"Poor Man's RAG" (Static Context Injection):**

The `.cursor/rules/abap-ai-toolkit.mdc` file is automatically injected into every LLM prompt. It contains ~213 lines of distilled knowledge: CRV mappings, modernization rules, directory paths, and quick references. Cap: ~200KB effective context. Works with ANY model, zero setup.

2. **"Real RAG" (ChromaDB Semantic Search):**

The LLM runs `python 3_rag/rag_query_api.py "query"` via shell tool to perform semantic vector search across 1,044 chunks. Returns the most relevant code examples, mappings, and patterns. Unlimited knowledge capacity. Requires ChromaDB to be built (`python 3_rag/rag_rebuild.py`).

### 8.3 Real RAG vs Poor Man's RAG Comparison

| Aspect | Poor Man's RAG | Real RAG (ChromaDB) |
|--------|---------------|-------------------|
| **Mechanism** | Static text in .mdc file | Vector similarity search |
| **Capacity** | ~200KB (Cursor context limit) | Unlimited (1,044+ chunks, 2.8MB+) |
| **Search** | LLM reads file, uses Grep/Read | Semantic embedding search (cosine similarity) |
| **Setup** | Zero - auto-loads in Cursor | Requires pip install + rag_rebuild.py |
| **Speed** | Instant (already in context) | ~30-60 seconds per query |
| **Precision** | LLM must find relevant section | Returns top-K most relevant chunks |
| **Works with** | Any model in Cursor | Any model that can run shell commands |
| **Sharing** | Just copy .mdc file | Share rag_export.json, rebuild on each machine |

### 8.4 RAG Training Scripts

To add new knowledge to the RAG:

| Script | What it Indexes | Chunks Added |
|--------|----------------|-------------|
| **3_rag/index_zsample_all.py** | All ZSAMPLE code + syntax + FM classification | ~1,002 |
| **3_rag/index_fiori_knowledge.py** | SAP Fiori Elements, UI5, CDS annotations | 18 |
| **3_rag/index_rap_knowledge.py** | SAP RAP architecture, EML, behavior, actions | 24 |
| **3_rag/sap_api_catalog.py** | SAP modern API catalog (21 RAP BOs + CDS) | varies |
| **3_rag/rag_append.py** | Append any new knowledge (append-only) | varies |

After indexing, always export for Git:
```bash
python 3_rag/rag_query_api.py --export
```

---

## 9. All Scripts Reference

### 9.1 Phase 1: Extraction Scripts

| Script | Purpose | CLI Usage |
|--------|---------|-----------|
| **sap_package_downloader.py** | Multi-method SAP downloader (ADT/SE80/abapGit) | `python 1_extract/sap_package_downloader.py /ZSAMPLE/MAIN` |
| **browser_export.py** | Playwright abapGit ZIP export | `python 1_extract/browser_export.py --packages /ZSAMPLE/SHARED` |
| **unzip_and_organize.py** | Organize ZIPs by ABAP object type | `python 1_extract/unzip_and_organize.py` |
| **curate_dataset.py** | Quality scoring and deduplication | `python 1_extract/curate_dataset.py` |

### 9.2 Phase 2: Migration & Mapping Scripts

| Script | Purpose | CLI Usage |
|--------|---------|-----------|
| **scan_sample_pkg.py** | Syntax scan + FM classification | `python 2_map_skills/scan_sample_pkg.py` |
| **migrate_sample_pkg.py** | Pass 1: 14 simple regex transforms | `python 2_map_skills/migrate_sample_pkg.py` |
| **deep_migrate_sample_pkg.py** | Pass 2: complex patterns (DATA chains) | `python 2_map_skills/deep_migrate_sample_pkg.py` |
| **java_abap_mapper.py** | Unified engine: detect+map+transform+RAG | `python 2_map_skills/java_abap_mapper.py` |
| **threeway_mapper.py** | 3-way mapping + CRV verification | `python 2_map_skills/threeway_mapper.py` |
| **full_skill_mapper.py** | Comprehensive skill mapping | `python 2_map_skills/full_skill_mapper.py` |
| **twin_discovery_sample_pkg.py** | Semantic twin discovery (patent) | `python 2_map_skills/twin_discovery_sample_pkg.py` |
| **abap_doc_validator.py** | Doc-driven syntax validation | `python 2_map_skills/abap_doc_validator.py --cloud <file>` |
| **download_crv.py** | Download CRV JSON from SAP GitHub | `python 2_map_skills/download_crv.py` |
| **check_our_fms_in_crv.py** | Check FMs against CRV | `python 2_map_skills/check_our_fms_in_crv.py` |

### 9.3 Phase 3: RAG Scripts

| Script | Purpose | CLI Usage |
|--------|---------|-----------|
| **rag_query_api.py** | CLI/REST/Python query API | `python 3_rag/rag_query_api.py "query"` |
| **rag_rebuild.py** | Rebuild ChromaDB from export | `python 3_rag/rag_rebuild.py` |
| **rag_append.py** | Append-only knowledge addition | `python 3_rag/rag_append.py --layer twins` |
| **index_zsample_all.py** | Index all ZSAMPLE knowledge | `python 3_rag/index_zsample_all.py` |
| **index_fiori_knowledge.py** | Index Fiori knowledge (18 chunks) | `python 3_rag/index_fiori_knowledge.py` |
| **index_rap_knowledge.py** | Index RAP knowledge (24 chunks) | `python 3_rag/index_rap_knowledge.py` |
| **sap_api_catalog.py** | Index SAP modern APIs | `python 3_rag/sap_api_catalog.py` |
| **llm_clean_core_engine.py** | LLM+RAG conversion engine | `python 3_rag/llm_clean_core_engine.py` |
| **rag_code_generator.py** | RAG-powered code generator | `python 3_rag/rag_code_generator.py` |
| **create_share_package.py** | Create shareable ZIP | `python 3_rag/create_share_package.py` |

### 9.4 Phase 4-7: Twin Discovery, Fine-Tune, Agent, MCP

| Script | Purpose |
|--------|---------|
| **4_twin_discovery/discover_clean_core.py** | Discover released S/4HANA Clean Core APIs |
| **4_twin_discovery/match_twins.py** | Map ECC -> Clean Core by semantic meaning |
| **4_twin_discovery/crv_verified_twins.py** | Cross-reference with SAP CRV registry |
| **5_fine_tune/prepare_training_data.py** | Prepare JSONL training pairs (if RAG < 80%) |
| **5_fine_tune/fine_tune_gemini.py** | Launch Vertex AI Gemini fine-tuning |
| **6_agent/code_generator.py** | RAG+Skills+LLM code generation with validation |
| **6_agent/pattern_validator.py** | 10-point pattern compiler checklist |
| **6_agent/sap_api_tools.py** | SAP OData API tools for agent |
| **7_abap_bl_brain/mcp_server.py** | MCP server for Cursor IDE integration |

---

## 10. Configuration Files

### config.yaml

Main project configuration:

| Section | Key Settings |
|---------|-------------|
| **sap** | webgui_url, packages list |
| **extraction** | include_subpackages, export_dir, abap_extensions (13 types), exclude_patterns |
| **rag** | vector_db_path, embedding_model (all-MiniLM-L6-v2), chunk_size (2000), top_k (10) |
| **llm** | provider (anthropic), model (claude-sonnet-4-20250514), max_tokens (4096), temperature (0.1) |
| **twin_discovery** | sap_modules (FI, CO, GL, AP, AR, AA), api_hub_url |

### .env.example

Environment variables template:

| Variable | Purpose |
|----------|---------|
| **SAP_WEBGUI_URL** | SAP WebGUI URL for browser automation |
| **SAP_USER / SAP_PASS** | SAP login credentials |
| **ANTHROPIC_API_KEY** | Claude API key (optional) |
| **GOOGLE_CLOUD_PROJECT** | GCP project for Vertex AI (optional) |
| **VECTOR_DB_PATH** | ChromaDB location (default: ./data/vector_db) |
| **EMBEDDING_MODEL** | SentenceTransformer model (default: all-MiniLM-L6-v2) |

### .gitignore

What is excluded from Git:

- `.venv/`
- `data/vector_db/` (rebuild from rag_export.json)
- `__pycache__/`
- `.env` (secrets)
- `sap_login_state.json`
- `sap_screenshot*.png`
- `data/twin_discovery/crv_latest_full.json` (re-downloadable)
- `data/exports/`
- `data/migrated/*.zip`

---

## 11. Migration Report Summary

Full report at: `data/migrated/ZSAMPLE_MIGRATION_REPORT.md` (263 lines)

### 11.1 CRV Successor Mapping (FM Replacements)

| Legacy FM | CRV Status | Modern Successor |
|-----------|-----------|-----------------|
| **BAPI_ACC_DOCUMENT_POST** | notToBeReleased | I_JournalEntryTP (RAP BO) |
| **BAPI_ACC_DOCUMENT_CHECK** | notToBeReleased | I_JournalEntryTP (RAP BO) |
| **BAPI_ACC_ACT_POSTINGS_REVERSE** | notToBeReleased | I_JournalEntryTP (RAP BO) |
| **BAPI_SALESORDER_CREATEFROMDAT2** | notToBeReleased | I_SalesOrderTP (RAP BO) |
| **BAPI_PO_CREATE1** | notToBeReleased | I_PurchaseOrderTP_2 (RAP BO) |
| **BAPI_GOODSMVT_CREATE** | notToBeReleased | I_MaterialDocumentTP (RAP BO) |
| **FI_COMPANY_CODE_DATA** | notToBeReleased | I_CompanyCode (CDS view) |
| **RK_KOKRS_FIND** | notToBeReleased | I_CompanyCode (CDS view) |
| **BAL_DB_SAVE/LOAD/SEARCH** | notToBeReleased | CL_BALI_LOG_DB (released class) |
| **BAL_LOG_MSG_ADD** | notToBeReleased | CL_BALI_LOG + CL_BALI_MESSAGE_SETTER |
| **FI_PERIOD_DETERMINE** | notToBeReleased | I_FiscCalendarDateForCompCode (CDS) |
| **BAPI_TRANSACTION_COMMIT** | RELEASED | Keep as-is |
| **BAPI_TRANSACTION_ROLLBACK** | RELEASED | Keep as-is |

### 11.2 Class-Level Twin Mapping (N:1 Example)

12 old FI document classes -> 1 new RAP BO (I_JournalEntryTP):

- /zsample/cl_fi_doc_base -> I_JournalEntryTP
- /zsample/cl_fi_doc_standard -> I_JournalEntryTP
- /ZSAMPLE/CL_FI_DOC_REVERSAL -> I_JournalEntryTP
- /ZSAMPLE/CL_FI_DOC_ACCRUAL -> I_JournalEntryTP
- /zsample/cl_fi_doc_copa -> I_JournalEntryTP
- /zsample/cl_fi_doc_copa_s4h -> I_JournalEntryTP
- /zsample/cl_fi_doc_tax -> I_JournalEntryTP
- /zsample/cl_fi_doc_tax_s4h -> I_JournalEntryTP
- /zsample/cl_fi_doc_tax_clear -> I_JournalEntryTP
- /zsample/cl_fi_doc_tax_clear_s4h -> I_JournalEntryTP
- /zsample/cl_fi_doc_discount_s4h -> I_JournalEntryTP
- /zsample/cl_ac_2_fi_conversion -> I_JournalEntryTP

### 11.3 ABAP Syntax Modernization Rules (12 Rules)

| # | Old Syntax | New Syntax | Count | Java Anchor |
|---|-----------|-----------|-------|-------------|
| 1 | MOVE source TO target. | target = source. | 138 | target = source; |
| 2 | ADD value TO target. | target = target + value. | 48 | target += value; |
| 3 | CONCATENATE a b INTO c. | c = \|{ a }{ b }\|. | 47 | c = a + b; |
| 4 | MOVE-CORRESPONDING | CORRESPONDING #( src ). | 28 | BeanUtils.copyProperties() |
| 5 | CALL METHOD obj->m. | obj->m( ). | 15 | obj.method(); |
| 6 | CREATE OBJECT lo TYPE cl. | lo = NEW cl( ). | 14 | obj = new Cl(); |
| 7 | SUBTRACT val FROM target. | target = target - val. | 8 | target -= value; |
| 8 | TRANSLATE TO UPPER CASE. | to_upper( str ). | 5 | str.toUpperCase() |
| 9 | TRANSLATE TO LOWER CASE. | to_lower( str ). | 4 | str.toLowerCase() |
| 10 | DESCRIBE TABLE itab LINES n. | lines( itab ). | 4 | list.size() |
| 11 | CREATE OBJECT lo. | lo = NEW #( ). | 3 | new Cl() |
| 12 | CALL METHOD cl=>method. | cl=>method( ). | 1 | Cl.staticMethod() |

Total syntax transforms: 315

### 11.4 Wrapper Classes Generated (7 New Classes)

| Wrapper Class | Wraps | Domain |
|---------------|-------|--------|
| **zcl_zsample_conv_exit_wrapper** | CONVERSION_EXIT_ALPHA/EXCRT_INPUT/OUTPUT | Conversion |
| **zcl_zsample_co_posting_wrapper** | BAPI_ACC_*_CHECK/POST (CO variants) | CO (N:1 merge) |
| **zcl_zsample_currency_wrapper** | CONVERT_TO_LOCAL_CURRENCY, READ_EXCHANGE_RATE | Currency |
| **zcl_zsample_tax_wrapper** | Tax-related BAPIs | Tax |
| **zcl_zsample_sd_wrapper** | SD BAPIs (delivery, pricing) | SD |
| **zcl_zsample_text_wrapper** | READ_TEXT, SAVE_TEXT, DELETE_TEXT | Text |
| **zcl_zsample_lock_wrapper** | ENQUEUE_*/DEQUEUE_* | Locking |

### 11.5 Validation Results

| Mode | Errors | Warnings | Info | Avg Score |
|------|--------|----------|------|-----------|
| **Standard (7.40+)** | 82 | 287 | 57 | 97% |
| **ABAP Cloud (strict)** | 313 | 356 | 63 | 92% |

---

## 12. Data Artifacts & File Counts

| Directory / File | Contents | Count/Size |
|------------------|----------|-----------|
| **data/sample_pkg_original/** | Original ECC ABAP files (/ZSAMPLE/MAIN) | 174 files |
| **data/sample_pkg_migrated/** | Migrated ABAP files + wrappers | 175 files |
| **data/sample_pkg_generated/** | Generated Clean Core with CRV annotations | 206 files |
| **data/zsample_original/** | Original ZSAMPLE_GENERIC files | 25 files |
| **data/zsample_migrated/** | Migrated ZSAMPLE_GENERIC files | 26 files |
| **data/zsample_extracted/** | Raw extracted files | 25 files |
| **data/threeway_mappings/** | 3-way mapping JSONs (4 source dirs) | 406 JSON files |
| **data/twin_discovery/** | Twin catalogs + CRV data | ~10 JSON files |
| **data/vector_db/** | ChromaDB (rebuild from export) | varies |
| **data/rag_export.json** | Full RAG database export | 2,802 KB |
| **data/pipeline_state.json** | Pipeline status tracking | ~5 KB |
| **data/sample_pkg_migrated.zip** | ZIP for developer review | ~200 KB |
| **data/rag_*.md** | Markdown exports per RAG layer | 8 files |

---

## 13. Cursor IDE Integration

### 13.1 Cursor Rules (.mdc file)

File: `.cursor/rules/abap-ai-toolkit.mdc` (213 lines, alwaysApply: true). This file is automatically injected into EVERY LLM prompt in Cursor. It contains:

- 5 architectural concepts (cross-lingual learning, twin discovery, RAG, CRV, doc validation)
- Complete CRV successor mapping table (13 FMs)
- ABAP modernization rules table (9 old->new mappings with Java anchors)
- RAP quick reference (20 bullet points)
- Fiori quick reference (10 bullet points)
- RAG layer table (17 layers, 1,044 chunks)
- Pipeline scripts list (9 stages)
- Key directory paths
- CLI commands for RAG query, twin search, purpose storage
- Git sharing instructions

### 13.2 Workspace Settings

File: `abap-ai-toolkit.code-workspace`

- Excludes `__pycache__`, `.venv`, `node_modules` from file explorer.
- Excludes `vector_db` and `crv_latest_full.json` from search.
- Associates `.abap` -> ABAP language, `.mdc` -> Markdown.
- Enables word wrap.

### 13.3 MCP Server (AbapBLBrain)

File: `7_abap_bl_brain/mcp_server.py`. Exposes RAG, pattern validator, twin discovery, and syntax modernizer as Cursor-native tools via MCP (Model Context Protocol). Run: `python 7_abap_bl_brain/mcp_server.py`

---

## 14. Security Considerations

Key security points discussed during development:

**RAG is LOCAL:** The ChromaDB vector database and all ABAP code live on your local machine. Nothing is uploaded to any cloud service unless you explicitly call an LLM API.

**Cursor + LLM Context:** When using Cursor with Claude/GPT-4, the .mdc rules file and any code you open/reference IS sent to the LLM provider as part of the prompt context. This means proprietary ABAP code snippets in the rules file and opened files DO go to the LLM API.

**GRC Approval:** For production use with external LLMs, GRC team approval may be needed. Google is a cloud partner, so Vertex AI/Gemini may have faster approval path.

**.env Secrets:** .env file contains SAP credentials and API keys. It is in .gitignore and should NEVER be committed to Git.

**SAP Login State:** `sap_login_state.json` contains browser session cookies. Also in .gitignore.

**Offline Mode:** The RAG + deterministic regex pipeline works 100% offline. Only the LLM code generation step (optional) requires internet/API access.

---

## 15. Sharing & Collaboration via Git

The RAG and all knowledge are designed to be shared via Git:

```bash
# On the source machine (after adding new knowledge):
python 3_rag/rag_query_api.py --export     # Export ChromaDB to rag_export.json
git add data/rag_export.json data/rag_*.md data/pipeline_state.json
git add .cursor/rules/abap-ai-toolkit.mdc
git commit -m "Update RAG with new knowledge"
git push

# On the target machine (new laptop):
git pull
python 3_rag/rag_rebuild.py     # Rebuild ChromaDB from rag_export.json
# Done! Full RAG available locally.
```

**What IS tracked in Git:**

- `data/rag_export.json` (2.8 MB) - full RAG database as JSON
- `data/rag_*.md` - Markdown exports readable without ChromaDB
- `data/pipeline_state.json` - pipeline status
- `.cursor/rules/abap-ai-toolkit.mdc` - auto-loaded context
- All Python scripts
- `config.yaml`, `requirements.txt`, `README.md`
- `data/sample_pkg_original/`, `data/sample_pkg_migrated/`, `data/sample_pkg_generated/`
- `data/threeway_mappings/`, `data/twin_discovery/`

**What is NOT tracked (rebuild locally):**

- `data/vector_db/` (rebuild with `rag_rebuild.py`)
- `.venv/` (rebuild with `pip install -r requirements.txt`)
- `.env` (create from `.env.example`)
- `sap_login_state.json` (re-login)
- `data/twin_discovery/crv_latest_full.json` (re-download with `download_crv.py`)

---

## 16. Patentable Ideas

### Cross-Lingual Skill Anchoring for Low-Resource Languages

Using a high-resource language (Java) as a "skill anchor" to teach an LLM correct syntax in a low-resource language (ABAP). The 3-way mapping (Java | Old ABAP | New ABAP) leverages the LLM's existing Java knowledge to generate accurate ABAP without hallucination. This is a novel application of transfer learning applied to code generation.

### Semantic Twin Discovery with Cardinality

Finding modern API replacements for legacy code based on PURPOSE/MEANING rather than name matching. Includes cardinality awareness (1:1, 1:N, N:1, N:M mappings) and automatic wrapper generation. Uses LLM-based purpose extraction + vector similarity search against a pre-indexed SAP API catalog.

### Doc-Driven Compilation Without a Kernel

Validating generated code against patterns extracted from official documentation rather than requiring an actual language runtime/compiler. Enables code validation in environments where the SAP kernel is not available (CI/CD, cloud development).

---

## 17. Known Issues & Lessons Learned

**DATA DATA bug:** The `deep_replace_data_chain` function had a chain-state leak causing "DATA DATA" syntax errors. Fixed by adding a keyword safety guard: if a continuation line starts with a known ABAP keyword (METHOD, IF, DATA), the `in_chain` state is reset to False.

**SAP WebGUI iframe limitation:** Playwright cannot interact with content inside SAP WebGUI iframes. Workaround: use ADT REST API method or manual abapGit ZIP export.

**PowerShell quoting issues:** Complex Python inline commands with f-strings fail in PowerShell. Workaround: write logic to a temporary .py file and execute it.

**Console encoding (cp1252):** Python `print()` fails on Windows with Unicode box-drawing characters. Not a data issue - only affects console output display.

**Custom FMs not in CRV:** 111 FMs are not in SAP's public CRV registry (custom /ZSAMPLE/ FMs or unclassified SAP FMs). These need manual review.

**Wrapper vs Direct replacement:** Initially, CRV successors were only annotated with comments. Later, user requested actual FM replacement with RAP BO calls in the same class files.

**SELECT * replacement:** SELECT * was replaced with explicit field lists, which can cause type mismatch errors (e.g., field ACTYPE). These need manual verification by ABAP developers.

**AUTHORITY_CHECK_TCODE:** FM AUTHORITY_CHECK_TCODE was assumed released but needs verification. Lesson: never assume release status - always check CRV.

---

## 18. Session Recovery Checklist

When starting a new Cursor session or on a new laptop:

1. Open the abap-ai-toolkit folder in Cursor
2. The `.cursor/rules/abap-ai-toolkit.mdc` auto-loads (verify: mention "RAG" and see if model understands)
3. Check `pipeline_state.json`: `python -c "import json; print(json.load(open('data/pipeline_state.json'))['pipeline_summary'])"`
4. Rebuild ChromaDB if needed: `python 3_rag/rag_rebuild.py`
5. Test RAG query: `python 3_rag/rag_query_api.py "FI document posting"`
6. All data is on disk - NOTHING is lost on restart or session change
7. The .mdc file provides FULL project context to any model in any session
8. For SAP access: run `python test_sap_browser.py` to re-establish session

---

**--- END OF DOCUMENT ---**
