from __future__ import annotations

from datasets import load_dataset

from app.utils import format_example


DATASET_NAME = "banking77"


def load_banking77(split: str = "train"):
    dataset = load_dataset(DATASET_NAME, split=split)
    return dataset.map(format_example)


def get_dataset_splits():
    train = load_banking77("train")
    validation = load_banking77("validation")
    return train, validation
