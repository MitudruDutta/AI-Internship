# Week 4 — MediAssist AI: Streamlit UI

A client-facing **Streamlit** interface over the Week-3 RAG pipeline. Users
search PubMed, ingest articles into the vector store, and ask questions that are
answered from the ingested literature.

## Features

**Sidebar — find & ingest articles**
- Search PubMed by term (with a max-articles slider, capped at 300).
- **Preview** the found articles (PMID, title, journal, year) before storing.
- One **Ingest** button embeds the results into the Chroma vector store.
- Live count of documents currently in the knowledge base.

**Main — ask questions**
- A query bar to type a question about the ingested articles.
- The RAG pipeline retrieves the most relevant articles and generates an
  evidence-based answer with **Groq Llama 3**.
- Cited **sources** (PMID, title, link, similarity distance) are listed below
  the answer.

## How it reuses earlier work

This app does **not** duplicate the pipeline code — it imports the Week-3
modules by adding the `Week 3/` folder to the import path:

| From Week 3 | Used for |
|---|---|
| `pubmed.PubMedRetriever` | search + fetch articles |
| `ingest.article_to_document` | turn an article into (id, text, metadata) |
| `chroma_store.ChromaStore` | store + retrieve vectors |
| `rag.answer_query` | retrieve → prompt → Groq Llama 3 answer |

It shares the **same** vector store (`Week 3/chroma_db/`) and the **same** Groq
key (`Week 3/.env`), so anything ingested here is queryable by the Week-3 CLI and
vice versa.

## Run

```bash
pip install -r requirements.txt

# the Groq key must exist at ../Week 3/.env  (see Week 3 README)
streamlit run app.py
# opens http://localhost:8501
```

### Typical flow

1. In the sidebar, enter a term (e.g. *intermittent fasting*) and click
   **Search PubMed**.
2. Review the previewed articles, then click **Ingest … into vector store**.
3. In the main panel, type a question and click **Ask** to get a cited,
   evidence-based answer.

## Notes

- The knowledge base persists in `Week 3/chroma_db/` (not committed; rebuilt by
  ingesting).
- Requires a working `Week 3/.env` with `GROQ_API_KEY`.
