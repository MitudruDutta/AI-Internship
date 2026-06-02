# FreshHarvest — Streamlit Demo (frontend)

Interactive demo for the FreshHarvest conveyor-belt inspection model. Drag and
drop a fruit image; the app sends it to the **FastAPI backend**, which runs the
ResNet50 model and returns whether the fruit is fresh or spoiled.

The frontend does no ML itself — it only calls the backend API.

## Files

- `app.py` — the Streamlit application (calls the backend)
- `requirements.txt` — frontend dependencies (streamlit + requests)

## Run

1. **Start the backend** (in another terminal):

   ```bash
   cd ../fastapi_backend
   uvicorn main:app --port 8000
   ```

2. **Start this app:**

   ```bash
   streamlit run app.py
   ```

   Opens at http://localhost:8501. The sidebar shows whether the backend is
   connected.

## Configuration

The backend URL defaults to `http://localhost:8000`. Override with an env var:

```bash
FRESHHARVEST_API=http://my-host:8000 streamlit run app.py
```

## Supported classes

8 fruits × {fresh, spoiled}: Banana, Lemon, Lulo, Mango, Orange, Strawberry,
Tamarillo, Tomato.
