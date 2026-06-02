# Week 3 — MediAssist AI: Ingestion + RAG Q&A

**MediAssist AI** is a healthcare Q&A system for **MediInsight Health
Solutions**, giving clinicians evidence-based summaries on **intermittent fasting
(IF)** for obesity / Type 2 Diabetes / metabolic disorders, drawn from
peer-reviewed literature.

This week has two parts:

1. **Ingestion pipeline** — pull research articles from PubMed and store them in
   a **ChromaDB** vector collection.
2. **RAG Q&A** — retrieve relevant articles for a user's question and generate a
   grounded, cited answer with **Groq + Llama 3**.

## What it does

```
INGESTION:  PubMed ── PMIDs ──▶ fetch articles ──▶ embed + store in ChromaDB
RAG Q&A:    question ──▶ retrieve top-k from ChromaDB ──▶ prompt (context+query)
                    ──▶ Groq Llama 3 ──▶ evidence-based answer with PMID citations
```

## Files

| File | Purpose |
|---|---|
| `pubmed.py` | *(provided)* `PubMedRetriever` — search PMIDs + fetch abstracts |
| `chroma_store.py` | `ChromaStore` — create/manage a Chroma collection; ingest + retrieve |
| `ingest.py` | Orchestrates the full PubMed → Chroma pipeline (CLI) |
| `rag.py` | RAG Q&A: retrieve context → build prompt → generate with Groq Llama 3 |
| `.env.example` | Template for the `GROQ_API_KEY` (copy to `.env`) |
| `requirements.txt` | Dependencies |

### `chroma_store.py` — collection management

`ChromaStore` wraps a **persistent** Chroma client and one collection:

- `add_documents(ids, documents, metadatas)` — batched **upsert** (re-ingesting
  the same PMID updates, never duplicates)
- `query(text, n_results, where)` — semantic search, returns
  `[{id, document, metadata, distance}, ...]`
- `get(ids)`, `count()`, `peek()`, `reset()` — management helpers

Embeddings use Chroma's default **all-MiniLM-L6-v2** (384-dim, local, free).
Distance metric is **cosine**.

### `ingest.py` — the pipeline

- `abstract_to_text()` — flattens the structured abstract (a dict of
  `{section: text}`) into one string
- `article_to_document()` — builds `(pmid, document, metadata)`; the embedded
  text is *title + abstract*, structured fields go into metadata
- `run_ingestion(term, max_articles)` — end-to-end search → fetch → ingest

## Run

```bash
pip install -r requirements.txt

# default: 300 articles on "intermittent fasting"
python ingest.py

# custom term / count
python ingest.py --term "time-restricted eating" --max 200
```

This creates a persistent store in `./chroma_db/` (collection
`pubmed_intermittent_fasting`). The run finishes with a sample query.

## Example retrieval

```python
from chroma_store import ChromaStore

store = ChromaStore()                      # reopens the persisted collection
print(store.count())                       # 300

for hit in store.query("alternate day fasting for type 2 diabetes", n_results=3):
    md = hit["metadata"]
    print(md["pmid"], round(hit["distance"], 3), md["title"])
```

Sample output (real data):

```
41503866 0.220 [Clinical considerations regarding the effect of intermittent fasting ...]
41693941 0.355 Effects of intermittent fasting on HbA1c and weight ...
41986966 0.360 Safety and efficacy of intermittent fasting with or without exercise ...
```

## RAG Q&A (`rag.py`)

Answers a clinician's question by **retrieving** the most relevant articles and
**generating** a grounded answer with Groq's Llama 3.

`answer_query()` runs the three steps from the brief:

1. **Query the vector store** — `retrieve_context()` calls `ChromaStore.query()`
   for the top-k articles.
2. **Build a prompt** with two variables — `build_prompt(context, query)` fills a
   template where `context` = the retrieved abstracts and `query` = the question.
3. **Generate** — `generate_answer()` sends it to the **Groq** client using the
   **Llama 3** model (`llama-3.3-70b-versatile`, override with `--model`).

The system prompt forces the model to answer **only from the retrieved context**,
**cite the PMIDs** it used, and say so when the context is insufficient (no
hallucinated medical claims).

### Setup — Groq API key

```bash
cp .env.example .env          # then edit .env and paste your Groq key
# get a key at https://console.groq.com/keys
```

`.env` holds `GROQ_API_KEY=...` and is **gitignored** (never committed).

### Run

```bash
python rag.py -q "Is 16:8 intermittent fasting effective for type 2 diabetes?"
```

Example (real run):

```
ANSWER:
According to the provided context, 16:8 time-restricted eating (TRE) has been
shown to improve insulin sensitivity in adults. The study [PMID 41351878] found
... HOMA-IR ... [PMID 41478229] reported reduced fasting blood sugar and HbA1c
... However, the context does not provide a direct clamp measure, so this should
be interpreted cautiously.

SOURCES:
  - PMID 41351878 (dist 0.361) Effect of 8-Hour Time-Restricted Eating (16/8 TRE) ...
  - PMID 41478229 (dist 0.396) Beneficial effects of time-restricted eating ...
```

## Notes

- **Rate limit:** retrieval is capped at **300 articles** (`MAX_ARTICLES_CAP`),
  with `sleep` between PubMed requests, per the brief.
- The `chroma_db/` store and `.env` are **not** committed (regenerate the store
  by running `ingest.py`; create `.env` from `.env.example`).
- **Pipeline order:** run `ingest.py` once to build the store, then `rag.py` to
  query it.
