"""
helper.py — model loading and inference logic for the FreshHarvest API.

Keeps all the ML work (load checkpoint, preprocess, predict) out of the API
layer so `main.py` stays thin. The model is loaded once and reused.
"""

import io
import os
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

MODEL_PATH = os.path.join(os.path.dirname(__file__), "resnet50_freshharvest.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _load():
    """Load checkpoint, rebuild ResNet50, build preprocessing. Cached (loads once)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["norm_mean"], ckpt["norm_std"]),
    ])
    return model, transform, ckpt


def model_info() -> dict:
    """Metadata about the loaded model (for the /info endpoint)."""
    _, _, ckpt = _load()
    fruits = sorted({c.split("_", 1)[1] for c in ckpt["classes"]})
    return {
        "arch": ckpt.get("arch", "resnet50"),
        "num_classes": ckpt["num_classes"],
        "classes": ckpt["classes"],
        "fruits": fruits,
        "img_size": ckpt["img_size"],
        "test_acc": ckpt.get("test_acc"),
        "device": DEVICE.type,
    }


def split_label(raw_label: str):
    """'F_Banana' -> ('Banana', 'fresh');  'S_Mango' -> ('Mango', 'spoiled')."""
    state = "fresh" if raw_label.startswith("F_") else "spoiled"
    fruit = raw_label.split("_", 1)[1]
    return fruit, state


def load_image(data: bytes) -> Image.Image:
    """Decode raw image bytes to a PIL RGB image. Raises ValueError if invalid."""
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")


@torch.no_grad()
def predict(data: bytes, top_k: int = 3) -> dict:
    """
    Run inference on raw image bytes.

    Returns a dict with the top prediction (fruit, state, confidence, raw label)
    and the top-k class probabilities.
    """
    model, transform, ckpt = _load()
    classes = ckpt["classes"]

    image = load_image(data)
    x = transform(image).unsqueeze(0).to(DEVICE)
    probs = F.softmax(model(x), dim=1).squeeze(0).cpu().tolist()

    ranked = sorted(zip(classes, probs), key=lambda p: p[1], reverse=True)
    top_label, top_prob = ranked[0]
    fruit, state = split_label(top_label)

    return {
        "label": top_label,
        "fruit": fruit,
        "state": state,
        "confidence": round(top_prob, 4),
        "top_k": [
            {
                "label": lbl,
                "fruit": split_label(lbl)[0],
                "state": split_label(lbl)[1],
                "probability": round(p, 4),
            }
            for lbl, p in ranked[: max(1, top_k)]
        ],
    }
