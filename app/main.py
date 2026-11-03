from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.utils import normalize_text

app = FastAPI(title="Banking77 PEFT Service")
MODEL_PATH = os.getenv("MODEL_PATH", "./artifacts/final-model")
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()


class PredictionRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, str | float | int]:
    text = normalize_text(payload.text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_idx = int(logits.argmax(dim=-1).item())
    predicted_label = model.config.id2label.get(predicted_idx, str(predicted_idx))
    return {
        "text": text,
        "predicted_label": predicted_label,
        "predicted_index": predicted_idx,
        "confidence": float(logits.softmax(dim=-1).max(dim=-1).values.item()),
    }
