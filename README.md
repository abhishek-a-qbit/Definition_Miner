# Definition Miner (No-LLM)

A Python CLI that mines repeated/canonical definitions for a category (e.g., `CRM`, `ABM`) by:

1. Searching the web for definition-focused queries.
2. Scraping pages with an open-source scraper (`trafilatura`) and extracting candidate definition sentences.
3. Clustering similar sentences with Hugging Face sentence embeddings + cosine similarity.
4. Printing top verbatim definitions ranked by multi-site agreement.

No LLM rewriting is used; output sentences are verbatim extracted text.

## Features

- **No LLMs** in extraction, clustering, or ranking
- **Open-source scraper backend** (`trafilatura`) with BeautifulSoup fallback parsing
- **Verbatim output** (never rewritten)
- **Domain-level dedup** (`same sentence + same domain` counted once)
- **Embedding similarity clustering** (Agglomerative + cosine threshold)
- **SQLite cache** for fetched HTML
- Configurable thresholds and ranking depth

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Search provider options

The miner supports:

- `brave` (requires `BRAVE_API_KEY`)
- `serpapi` (requires `SERPAPI_API_KEY`)
- `duckduckgo` (no key, HTML parsing fallback)
- `auto` (default): Brave -> SerpAPI -> DuckDuckGo

Set keys (PowerShell):

```powershell
$env:BRAVE_API_KEY="your_key_here"
# or
$env:SERPAPI_API_KEY="your_key_here"
```

## Usage

```bash
python definition_miner.py CRM
```

Output includes tracked source URLs per result under `sources`.

With custom settings:

```bash
python definition_miner.py ABM \
  --search-provider auto \
  --max-results-per-query 10 \
  --max-urls 40 \
  --similarity-threshold 0.78 \
  --min-cluster-size 2 \
  --top-k 5 \
  --output-json outputs/abm.json
```

## Example output

```text
=== Top Definitions for: CRM ===

#1 — Found on 8 sites
"Customer relationship management (CRM) is a technology for managing ..."
  └─ salesforce.com, hubspot.com, ...
```

## FastAPI service

Start API server:

```bash
uvicorn api_app:app --reload
```

Endpoints:

- `GET /health`
- `POST /mine`

Example request:

```json
{
  "category": "CRM",
  "top_k": 5,
  "similarity_threshold": 0.78
}
```

Each result includes:

- `definition` (verbatim sentence)
- `sources` (list of `{domain, url}` for traceable references)

## Streamlit app

Run:

```bash
streamlit run streamlit_app.py
```

The app shows top definitions and an expandable **Sources** section with clickable links for each cluster.

## Notes

- First run downloads the embedding model from Hugging Face (`sentence-transformers/all-MiniLM-L6-v2` by default).
- You can change the embedding model via `--embedding-model`.
- If results are too strict, lower:
  - `--similarity-threshold` (e.g., `0.74`)
  - `--min-cluster-size` (e.g., `1`)
  - `--min-words` (e.g., `12`)
