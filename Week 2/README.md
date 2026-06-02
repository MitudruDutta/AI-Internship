# Week 2 — Transfer Learning + Demo App

This week replaces the from-scratch CNN with **transfer learning from a
pretrained ResNet50**, then wraps the trained model in a **FastAPI backend** and
a **Streamlit frontend** so the client can try it interactively.

## Contents

| Path | What it is |
|---|---|
| [`FreshHarvest_Week2.ipynb`](FreshHarvest_Week2.ipynb) | Transfer-learning training notebook |
| [`fastapi_backend/`](fastapi_backend/) | REST API serving the model |
| [`streamlit_app/`](streamlit_app/) | Drag-and-drop demo UI (calls the backend) |

## 1. Transfer-learning notebook

Loads **ResNet50** with ImageNet weights, swaps the head for 16 classes, and
trains in two phases:

1. **Feature extraction** — freeze the backbone, train only the new head (fast,
   cheap).
2. **Fine-tuning** — unfreeze the top residual block (`layer4`) and continue at
   a low learning rate.

The dataset is **deduplicated by MD5 hash before splitting** (see the
[dataset README](../dataset/README.md)) to avoid leakage. Transfer learning
reaches the target in only a few epochs per phase — far cheaper than training
from scratch.

**Result (held-out test set, after deduplication):**

| | 16-class | fresh / spoiled |
|---|---|---|
| Test accuracy | **99.9 %** | **99.9 %** |

The model is saved as `resnet50_freshharvest.pth` (state_dict + class names +
preprocessing metadata). Excluded from git — produce it by running the notebook.

> **Note on the near-perfect score:** FRUIT-16K is clean, well-lit studio
> imagery with visually distinct classes, which ResNet50's ImageNet features
> separate almost perfectly. The split is leakage-free (verified by content-hash
> assertions), so the score is honest for *this* dataset; real conveyor-belt
> imagery would be more varied.

## 2. FastAPI backend — [`fastapi_backend/`](fastapi_backend/)

REST API that loads the model once and serves predictions.

- `main.py` — routes (`GET /`, `GET /info`, `POST /predict`)
- `helper.py` — model loading, preprocessing, and inference logic

```bash
cd fastapi_backend
uvicorn main:app --port 8000
# docs at http://localhost:8000/docs
```

See [`fastapi_backend/README.md`](fastapi_backend/README.md).

## 3. Streamlit demo — [`streamlit_app/`](streamlit_app/)

Drag-and-drop UI. It does **no ML itself** — it posts the image to the FastAPI
backend and shows the predicted fruit, fresh/spoiled verdict, confidence, and a
top-k breakdown.

```bash
# start the backend first (above), then:
cd streamlit_app
streamlit run app.py
# opens http://localhost:8501
```

See [`streamlit_app/README.md`](streamlit_app/README.md).

## Architecture

```
 Streamlit frontend  ──HTTP /predict──▶  FastAPI backend  ──▶  ResNet50
 (streamlit_app)          requests         (fastapi_backend)      (.pth)
```

## Reproduce from scratch

```bash
# 1. train + save the model
jupyter nbconvert --to notebook --execute --inplace FreshHarvest_Week2.ipynb
# 2. copy the model next to the backend
cp resnet50_freshharvest.pth fastapi_backend/
# 3. run backend + frontend (see above)
```
