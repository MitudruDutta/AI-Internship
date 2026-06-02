<div align="center">

# AI Internship — Applied Machine Learning Projects

**End-to-end ML projects built from real client briefs — from raw data to deployed, interactive demos.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B6B)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama%203-000000)

</div>

---

## Overview

This repository contains two client-driven projects, delivered week by week. Each
week builds on the last — from data pipelines and model training to RAG systems
and production-style demo apps.

| Project | Client | Weeks | What it does |
|---------|--------|:-----:|--------------|
| 🍎 **FreshHarvest** | FreshHarvest Logistics | 1–2 | Image model that classifies fruit as **fresh or spoiled** for conveyor-belt inspection |
| 🩺 **MediAssist AI** | MediInsight Health Solutions | 3–4 | **RAG Q&A** system answering clinician questions on intermittent fasting from PubMed literature |

---

## 🍎 FreshHarvest — Freshness Inspection (Weeks 1–2)

**Problem.** A cold-storage logistics company ships fruit whose freshness is
checked manually — inconsistent (lighting, fatigue) and costly (refunds, returns).
**Goal.** A deep-learning model that classifies fruit images as **fresh** or
**spoiled** in real time, plus a demo app for the client.

- **Week 1** — Full data pipeline (MD5 **deduplication**, augmentation, stratified
  split, visualization) and a **CNN trained from scratch**: a plain baseline, then
  a **regularized + hyperparameter-tuned** model.
- **Week 2** — **Transfer learning** with a pretrained **ResNet50**, served through
  a **FastAPI** backend with a **Streamlit** drag-and-drop demo frontend.

```
Streamlit UI ──HTTP /predict──▶ FastAPI backend ──▶ ResNet50 (fresh / spoiled)
```

**Result:** ~99.9 % test accuracy (on a leakage-free, deduplicated split).

---

## 🩺 MediAssist AI — Healthcare Q&A (Weeks 3–4)

**Problem.** Clinicians face a flood of conflicting intermittent-fasting studies
and have no time to sift them. **Goal.** A **Retrieval-Augmented Generation**
assistant that answers questions with evidence-based, **cited** summaries drawn
from peer-reviewed literature.

- **Week 3** — **Ingestion + RAG.** Pull articles from **PubMed**, embed and store
  them in **ChromaDB**, then answer questions by retrieving relevant abstracts and
  generating a grounded response with **Groq + Llama 3**. Includes an adversarial
  test suite (prompt-injection, hallucination, citation-integrity checks).
- **Week 4** — **Streamlit UI.** A client-facing app: search PubMed and ingest
  articles from a sidebar, then ask questions in a query bar and get cited answers.

```
Streamlit UI ──▶ PubMed search + ingest ──▶ ChromaDB
             ──▶ question ──▶ retrieve top-k ──▶ Groq Llama 3 ──▶ cited answer
```

**Safeguards:** answers are grounded in retrieved abstracts only, cite their
PMIDs, refuse off-topic / unanswerable questions, and resist prompt injection.

---

## Repository structure

```
.
├── dataset/                      # FRUIT-16K — images NOT committed (see dataset/README.md)
│   └── README.md
├── Week 1/                       # FreshHarvest: data pipeline + CNN from scratch
│   ├── FreshHarvest_Week1.ipynb
│   └── README.md
├── Week 2/                       # FreshHarvest: transfer learning + demo app
│   ├── FreshHarvest_Week2.ipynb
│   ├── fastapi_backend/          #   REST API serving the model
│   ├── streamlit_app/            #   drag-and-drop demo UI
│   └── README.md
├── Week 3/                       # MediAssist: PubMed → ChromaDB ingestion + RAG
│   ├── pubmed.py                 #   PubMed retriever (provided)
│   ├── chroma_store.py           #   Chroma collection management
│   ├── ingest.py                 #   ingestion pipeline
│   ├── rag.py                    #   RAG: retrieve → prompt → Groq Llama 3
│   ├── test_rag.py               #   adversarial RAG test suite
│   └── README.md
├── Week 4/                       # MediAssist: Streamlit UI over the RAG pipeline
│   ├── app.py
│   └── README.md
├── .gitignore
└── README.md                     # (this file)
```

---

## Tech stack

| Area | Tools |
|------|-------|
| Deep learning | PyTorch · torchvision (ResNet50) · scikit-learn |
| Vision serving | FastAPI · Streamlit |
| RAG / NLP | ChromaDB · sentence-transformers (MiniLM) · Groq (Llama 3) · PubMed E-utilities |
| Tooling | NumPy · Matplotlib · Pillow · python-dotenv |

---

## Getting started

```bash
# clone and enter the repo
git clone https://github.com/MitudruDutta/AI-Internship.git
cd AI-Internship

# create a virtual environment
python -m venv .venv && source .venv/bin/activate
```

### Run the FreshHarvest demo (Weeks 1–2)

```bash
pip install -r "Week 2/fastapi_backend/requirements.txt"
pip install -r "Week 2/streamlit_app/requirements.txt"

# place the dataset at dataset/FRUIT-16K/ (see dataset/README.md), then train:
jupyter nbconvert --to notebook --execute --inplace "Week 2/FreshHarvest_Week2.ipynb"
cp "Week 2/resnet50_freshharvest.pth" "Week 2/fastapi_backend/"

# run backend + frontend (two terminals)
cd "Week 2/fastapi_backend" && uvicorn main:app --port 8000
cd "Week 2/streamlit_app"   && streamlit run app.py
```

### Run MediAssist AI (Weeks 3–4)

```bash
pip install -r "Week 3/requirements.txt"

# add your Groq API key
cp "Week 3/.env.example" "Week 3/.env"   # then edit .env and paste your key

# build the knowledge base, then launch the UI
cd "Week 3" && python ingest.py          # PubMed → ChromaDB (up to 300 articles)
cd "../Week 4" && streamlit run app.py    # search, ingest, and ask questions
```

---

## A note on reproducibility

Large and sensitive files are **excluded from version control** (see
[`.gitignore`](.gitignore)) and regenerated by running the code:

- **Image dataset** (`FRUIT-16K`) — described in [`dataset/README.md`](dataset/README.md)
- **Trained model weights** (`*.pth`) — produced by the Week 1 / 2 notebooks
- **Vector store** (`chroma_db/`) — produced by `Week 3/ingest.py`
- **Secrets** (`.env`, API keys) — created from `.env.example`

---

<div align="center">
<sub>Each week has its own README with full details — start there for any module.</sub>
</div>
