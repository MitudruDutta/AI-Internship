# FreshHarvest — FastAPI Backend

REST API serving the ResNet50 freshness-inspection model. Upload a fruit image,
get back the predicted fruit and whether it is fresh or spoiled.

## Files

- `main.py` — FastAPI app and routes
- `helper.py` — model loading, preprocessing, and prediction logic
- `resnet50_freshharvest.pth` — trained model checkpoint
- `requirements.txt` — dependencies

## Run

```bash
# from this folder, with the project virtualenv active
uvicorn main:app --reload --port 8000
```

Interactive docs (Swagger UI): http://localhost:8000/docs

## Endpoints

| Method | Path       | Description                                  |
|--------|------------|----------------------------------------------|
| GET    | `/`        | Health check                                 |
| GET    | `/info`    | Model metadata (classes, accuracy, device)   |
| POST   | `/predict` | Upload an image → fruit + fresh/spoiled       |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/banana.jpg"
```

Response:

```json
{
  "label": "F_Banana",
  "fruit": "Banana",
  "state": "fresh",
  "confidence": 0.9998,
  "top_k": [
    {"label": "F_Banana", "fruit": "Banana", "state": "fresh", "probability": 0.9998},
    {"label": "S_Banana", "fruit": "Banana", "state": "spoiled", "probability": 0.0002},
    {"label": "F_Mango",  "fruit": "Mango",  "state": "fresh", "probability": 0.0000}
  ],
  "filename": "banana.jpg"
}
```

## Notes

- Model loads once at startup (`@lru_cache`), so requests are fast.
- Runs on GPU if available, otherwise CPU.
- CORS is open (`*`) so the Streamlit demo or a browser can call it directly.
