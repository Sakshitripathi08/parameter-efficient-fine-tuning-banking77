from __future__ import annotations

from typing import Any


def build_model_name() -> str:
    return "distilbert-base-uncased"


def get_label_mapping(labels: list[str]) -> dict[str, int]:
    return {label: idx for idx, label in enumerate(labels)}


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def format_example(example: dict[str, Any]) -> dict[str, Any]:
    text = normalize_text(example.get("text", ""))
    return {"text": text, "label": example.get("label", 0)}
