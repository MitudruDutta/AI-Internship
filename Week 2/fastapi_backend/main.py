"""
main.py — FastAPI backend for the FreshHarvest freshness-inspection model.

Endpoints
---------
GET  /            health check
GET  /info        model metadata (classes, accuracy, device)
POST /predict     upload an image -> fruit + fresh/spoiled prediction

Run:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import helper

app = FastAPI(
    title="FreshHarvest Freshness Inspection API",
    description="Classifies fruit images as fresh or spoiled using a ResNet50 model.",
    version="1.0.0",
)

# allow the Streamlit demo / browser clients to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# accepted image content types
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}


@app.on_event("startup")
def warmup():
    """Load the model once at startup so the first request is fast."""
    helper.model_info()


@app.get("/")
def root():
    return {"status": "ok", "service": "FreshHarvest Freshness Inspection API"}


@app.get("/info")
def info():
    """Model metadata."""
    return helper.model_info()


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    """
    Predict the fruit type and freshness for an uploaded image.

    - **file**: an image (jpeg / png / bmp / webp)
    - **top_k**: how many ranked classes to return (default 3)
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {sorted(ALLOWED_TYPES)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        result = helper.predict(data, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result["filename"] = file.filename
    return result
