"""
rag.py — Retrieval-Augmented Generation over the PubMed/Chroma store.

Pipeline:
  1. Retrieve the most relevant PubMed articles from the Chroma collection for
     the user's question (uses ChromaStore from the ingestion task).
  2. Build a prompt with two variables: the retrieved CONTEXT and the user QUERY.
  3. Generate an evidence-based answer with the Groq client (Llama 3).

The answer is grounded in the retrieved abstracts and cites the PMIDs it used,
which is exactly what MediAssist AI needs (no hallucinated medical claims).

Run:
    python rag.py
    python rag.py --question "Is 16:8 fasting effective for type 2 diabetes?"
"""

import argparse
import os

from dotenv import load_dotenv
from groq import Groq

from chroma_store import ChromaStore

# load GROQ_API_KEY from the .env next to this file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Llama 3 family on Groq. 70B versatile gives the best medical answers;
# swap to "llama-3.1-8b-instant" for faster/cheaper responses.
DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are MediAssist AI, a careful medical research assistant for clinicians. "
    "Answer questions about intermittent fasting strictly from the provided "
    "research-paper context. Be concise and evidence-based. If the context does "
    "not contain the answer, say so plainly instead of guessing. Cite the PMIDs "
    "of the papers you used."
)

PROMPT_TEMPLATE = """Use the following research-paper excerpts to answer the question.

CONTEXT:
{context}

QUESTION:
{query}

Answer using only the context above. Cite the relevant PMIDs in your answer.
If the context is insufficient, say what is missing."""


def retrieve_context(store: ChromaStore, query: str, n_results: int = 5):
    """Query the vector store and return (context_string, hits)."""
    hits = store.query(query, n_results=n_results)
    blocks = []
    for h in hits:
        md = h.get("metadata") or {}
        pmid = md.get("pmid", "?")
        title = md.get("title", "")
        journal = md.get("journal", "")
        year = md.get("publication_date", "")
        doc = h.get("document", "")
        blocks.append(
            f"[PMID {pmid}] {title} ({journal}, {year})\n{doc}"
        )
    context = "\n\n---\n\n".join(blocks) if blocks else "No relevant documents found."
    return context, hits


def build_prompt(context: str, query: str) -> str:
    """Fill the prompt template with the two variables: context and query."""
    return PROMPT_TEMPLATE.format(context=context, query=query)


def generate_answer(prompt: str, model: str = DEFAULT_MODEL,
                    temperature: float = 0.2) -> str:
    """Send the prompt to Groq (Llama 3) and return the generated answer."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
        )
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def answer_query(query: str, n_results: int = 5, model: str = DEFAULT_MODEL,
                 store: ChromaStore = None) -> dict:
    """
    Full RAG flow: retrieve -> build prompt -> generate.
    Returns {answer, sources, prompt}.
    """
    store = store or ChromaStore()
    context, hits = retrieve_context(store, query, n_results=n_results)
    prompt = build_prompt(context, query)
    answer = generate_answer(prompt, model=model)

    sources = [
        {
            "pmid": (h.get("metadata") or {}).get("pmid"),
            "title": (h.get("metadata") or {}).get("title"),
            "url": (h.get("metadata") or {}).get("url"),
            "distance": h.get("distance"),
        }
        for h in hits
    ]
    return {"answer": answer, "sources": sources, "prompt": prompt}


def main():
    parser = argparse.ArgumentParser(description="RAG Q&A over PubMed/Chroma")
    parser.add_argument(
        "--question", "-q",
        default="Is 16:8 intermittent fasting effective for type 2 diabetes?",
        help="the user's question",
    )
    parser.add_argument("--n", type=int, default=5, help="documents to retrieve")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    print(f"Q: {args.question}\n")
    result = answer_query(args.question, n_results=args.n, model=args.model)

    print("ANSWER:\n" + result["answer"] + "\n")
    print("SOURCES:")
    for s in result["sources"]:
        d = s["distance"]
        print(f"  - PMID {s['pmid']} (dist {d:.3f}) {s['title'][:70]}")
        print(f"    {s['url']}")


if __name__ == "__main__":
    main()
