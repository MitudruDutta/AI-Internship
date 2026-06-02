"""
ingest.py — PubMed -> ChromaDB ingestion pipeline.

Steps:
  1. Search PubMed for a term and collect PMIDs (capped at MAX_ARTICLES <= 300).
  2. Iterate the PMIDs and fetch each article (title, abstract, journal, ...).
  3. Turn each article into an embeddable text document + metadata.
  4. Ingest everything into a Chroma vector collection (chroma_store.ChromaStore).

Run:
    python ingest.py
    python ingest.py --term "intermittent fasting" --max 300
"""

import argparse

from pubmed import PubMedRetriever
from chroma_store import ChromaStore, DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR

# Rate-limit safety: PubMed E-utilities allow ~3 req/s without an API key.
# The brief caps retrieval at 300 articles.
MAX_ARTICLES_CAP = 300
DEFAULT_TERM = "intermittent fasting"


def abstract_to_text(abstract) -> str:
    """
    Flatten the abstract (a dict of {section_label: text}) into one string.
    e.g. {"BACKGROUND": "...", "RESULTS": "..."} -> "BACKGROUND: ...\n\nRESULTS: ..."
    """
    if isinstance(abstract, dict):
        parts = []
        for label, text in abstract.items():
            if not text:
                continue
            if label and label != "SUMMARY":
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)
        return "\n\n".join(parts).strip()
    return str(abstract or "").strip()


def article_to_document(article: dict):
    """
    Convert one fetched article into (id, document_text, metadata) for Chroma.

    The embedded text is the title + abstract (what we search over). Structured
    fields go into metadata so they can be returned/filtered without re-parsing.
    """
    pmid = article["pmid"]
    title = article.get("title") or "No Title"
    abstract_text = abstract_to_text(article.get("abstract"))

    document = f"{title}\n\n{abstract_text}".strip()

    metadata = {
        "pmid": pmid,
        "title": title,
        "journal": article.get("journal", "Unknown Journal"),
        "authors": article.get("authors", "No Authors"),
        "publication_date": article.get("publication_date", "Unknown Year"),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "has_abstract": bool(abstract_text and abstract_text != "No Abstract"),
    }
    return pmid, document, metadata


def run_ingestion(term: str = DEFAULT_TERM, max_articles: int = MAX_ARTICLES_CAP,
                  persist_dir: str = DEFAULT_PERSIST_DIR,
                  collection_name: str = DEFAULT_COLLECTION):
    """End-to-end: search -> fetch -> transform -> ingest. Returns the store."""
    max_articles = min(max_articles, MAX_ARTICLES_CAP)

    # 1) search -> PMIDs
    print(f"[1/4] Searching PubMed for '{term}' (max {max_articles})...")
    pmids = PubMedRetriever.search_pubmed_articles(term, max_results=max_articles)
    print(f"      retrieved {len(pmids)} PMIDs")
    if not pmids:
        print("      no PMIDs found, aborting.")
        return None

    # 2) iterate PMIDs -> fetch articles
    print(f"[2/4] Fetching {len(pmids)} articles from PubMed...")
    articles = PubMedRetriever.fetch_pubmed_abstracts(pmids)
    print(f"      fetched {len(articles)} articles")

    # 3) transform to documents
    print("[3/4] Building documents + metadata...")
    ids, documents, metadatas = [], [], []
    for art in articles:
        pmid, doc, meta = article_to_document(art)
        ids.append(pmid)
        documents.append(doc)
        metadatas.append(meta)

    # 4) ingest into Chroma
    print(f"[4/4] Ingesting into Chroma collection '{collection_name}'...")
    store = ChromaStore(persist_dir=persist_dir, collection_name=collection_name)
    added = store.add_documents(ids=ids, documents=documents, metadatas=metadatas)
    print(f"      ingested {added} documents | collection now holds {store.count()}")
    return store


def main():
    parser = argparse.ArgumentParser(description="PubMed -> ChromaDB ingestion")
    parser.add_argument("--term", default=DEFAULT_TERM, help="PubMed search term")
    parser.add_argument("--max", type=int, default=MAX_ARTICLES_CAP,
                        help="max articles to retrieve (<= 300)")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    store = run_ingestion(
        term=args.term, max_articles=args.max,
        persist_dir=args.persist_dir, collection_name=args.collection,
    )

    # quick sanity query
    if store is not None and store.count() > 0:
        print("\nSample query: 'Does 16:8 fasting improve insulin sensitivity?'")
        for hit in store.query("Does 16:8 fasting improve insulin sensitivity?", n_results=3):
            md = hit["metadata"] or {}
            print(f"  - [{md.get('pmid')}] {md.get('title', '')[:80]}  "
                  f"(dist {hit['distance']:.3f})")


if __name__ == "__main__":
    main()
