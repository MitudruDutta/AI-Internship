"""
FreshHarvest — Fruit Freshness Inspection Demo (Streamlit frontend)
===================================================================
Drag-and-drop a fruit image; the app sends it to the FastAPI backend, which runs
the trained ResNet50 model and returns the fruit type and fresh/spoiled verdict.

The frontend does NO machine learning itself — all inference happens in the
backend (see ../fastapi_backend). Start the backend first:

    cd ../fastapi_backend
    uvicorn main:app --port 8000

Then run this app:

    streamlit run app.py
"""

import os

import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Config — where the FastAPI backend lives
# --------------------------------------------------------------------------- #
API_URL = os.environ.get("FRESHHARVEST_API", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"
INFO_ENDPOINT = f"{API_URL}/info"

st.set_page_config(
    page_title="FreshHarvest — Freshness Inspection",
    page_icon="🍎",
    layout="centered",
)


# --------------------------------------------------------------------------- #
# Backend helpers
# --------------------------------------------------------------------------- #
def get_backend_info():
    """Fetch model metadata from the backend. Returns dict or None if unreachable."""
    try:
        r = requests.get(INFO_ENDPOINT, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def call_predict(file_bytes: bytes, filename: str, content_type: str):
    """POST the image to the backend /predict. Returns (json, error_message)."""
    try:
        files = {"file": (filename, file_bytes, content_type)}
        r = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
    except requests.RequestException as e:
        return None, f"Could not reach the backend at {API_URL}. Is it running?\n\n{e}"

    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        return None, f"Backend returned {r.status_code}: {detail}"
    return r.json(), None


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("🍎 FreshHarvest — Freshness Inspection")
st.caption(
    "Upload a fruit photo and the model predicts the fruit and whether it is "
    "**fresh** or **spoiled** — a demo of the conveyor-belt inspection system."
)

info = get_backend_info()

with st.sidebar:
    st.header("Backend")
    if info is None:
        st.error(
            f"Backend not reachable at\n`{API_URL}`.\n\n"
            "Start it first:\n\n"
            "```\ncd ../fastapi_backend\nuvicorn main:app --port 8000\n```"
        )
    else:
        st.success(f"Connected: {API_URL}")
        st.write(
            f"**Model:** {info.get('arch', 'resnet50')} (transfer learning)\n\n"
            f"**Classes:** {info.get('num_classes')} (8 fruits × fresh / spoiled)\n\n"
            f"**Test accuracy:** {(info.get('test_acc') or 0)*100:.1f}%\n\n"
            f"**Device:** {str(info.get('device', '')).upper()}"
        )
        st.divider()
        st.write("**Supported fruits:**")
        st.write(", ".join(info.get("fruits", [])))

if info is None:
    st.warning("Start the FastAPI backend, then reload this page.")
    st.stop()

# Drag-and-drop / file upload
uploaded = st.file_uploader(
    "Drag and drop a fruit image here, or click to browse",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("👆 Drop an image to get a prediction.")
    st.stop()

file_bytes = uploaded.getvalue()

col_img, col_pred = st.columns([1, 1])

with col_img:
    st.image(file_bytes, caption=uploaded.name, width="stretch")

with col_pred:
    with st.spinner("Asking the model…"):
        result, err = call_predict(file_bytes, uploaded.name, uploaded.type or "image/jpeg")

    if err:
        st.error(err)
        st.stop()

    fruit = result["fruit"]
    state = result["state"].capitalize()
    conf = result["confidence"]

    st.subheader("Prediction")
    badge = "🟢" if state.lower() == "fresh" else "🔴"
    st.markdown(f"### {badge} {fruit} — **{state}**")
    st.metric("Confidence", f"{conf*100:.1f}%")

    if state.lower() == "spoiled":
        st.warning("Flagged as **spoiled** — would be diverted off the belt.")
    else:
        st.success("Looks **fresh** — passes inspection.")

# Top-k breakdown from the backend response
st.divider()
st.subheader("Top predictions")
for item in result.get("top_k", []):
    label = f"{item['fruit']} — {item['state'].capitalize()}"
    prob = item["probability"]
    st.write(f"**{label}**")
    st.progress(min(max(prob, 0.0), 1.0), text=f"{prob*100:.1f}%")
