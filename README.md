# BookForest2

A fully local, RAG-based bookmark organizer. Drop in a Chrome/Brave bookmarks HTML file and get back a clean, multi-level semantic taxonomy — all offline using Ollama and ChromaDB.

---

## Prerequisites

| Tool | Install |
|---|---|
| Python 3.11+ | [python.org](https://www.python.org/) |
| Ollama | [ollama.com](https://ollama.com/) |
| nomic-embed-text | `ollama pull nomic-embed-text` |
| llama3.2 | `ollama pull llama3.2` |

---

## Setup

```bash
# 1. Clone / open the project
cd bookforest2

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure Ollama is running
ollama serve
```

---

## Usage

### Process bookmarks

```bash
python main.py run data/input/bookmarks.html
```

With page scraping enabled:

```bash
python main.py run data/input/bookmarks.html --scrape
```

### Launch the visualization GUI

```bash
python main.py gui
# Opens http://localhost:8080
```

---

## Configuration

All settings live in `config.yaml`. Key options:

| Key | Default | Description |
|---|---|---|
| `ollama.embedding_model` | `nomic-embed-text` | Embedding model |
| `ollama.llm_model` | `llama3.2` | LLM for category naming |
| `processing.scrape_page` | `false` | Fetch page content |
| `clustering.algorithm` | `hdbscan` | `hdbscan` or `kmeans` |
| `validation.max_categories` | `50` | Hard cap on total categories |
| `validation.min_confidence` | `0.75` | Min LLM confidence to create a category |
| `gui.port` | `8080` | GUI server port |

---

## Outputs

After running, `data/output/` contains:

| File | Description |
|---|---|
| `bookmarks_organized.html` | Clean Netscape bookmark file, reimportable into any browser |
| `taxonomy.json` | Full structured taxonomy + bookmark list |
| `graph.json` | Graph data for the Sigma.js GUI |
| `report.txt` | Human-readable stats and category tree |

---

## GUI Views

| View | Description |
|---|---|
| **Galaxy** | Force-directed cloud of all bookmarks, colored by root bucket |
| **Tree** | Radial hierarchical layout (L1 → L2 → L3 → L4) |
| **Focus** | Click any node to zoom into its branch; breadcrumb navigation |

- **Search**: filter nodes by title or domain
- **Sidebar**: click any node to see full metadata
- **Drag & drop**: reassign bookmarks to different categories (patches `graph.json`)

---

## Architecture

```
Phase 1   Parse HTML → flat Bookmark list
Phase 2   Embed with Ollama → store in ChromaDB
Phase 3   Assign to preset root buckets (similarity ≥ 0.75)
Phase 4   HDBSCAN cluster the UNSORTED pool (UMAP → HDBSCAN)
Phase 5   LLM labels each cluster (Level 4: Topics)
Phase 6   Bottom-up aggregation: Topics → Subdomains → Domains → Roots
Phase 7   Validate (size ≥ 10, confidence ≥ 0.75, merge ≥ 0.85, cap ≤ 50)
Phase 8   Construct final TaxonomyTree
Phase 9   Semantic deduplication (cosine > 0.95)
Phase 10  Export: HTML + JSON + report
Phase 11  GUI: FastAPI + Sigma.js (optional)
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Root Buckets (Level 1)

```
AI  ·  DEV  ·  FINANCE  ·  FOOD  ·  GPS  ·  LEARNING  ·  MISC
```

These are always present. MISC catches everything that doesn't meet the confidence/size thresholds.
