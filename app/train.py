from __future__ import annotations

import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

from app.data import get_dataset_splits
from app.utils import build_model_name


MODEL_NAME = os.getenv("MODEL_NAME", build_model_name())
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./artifacts")


def label_to_id(dataset):
    labels = dataset.features["label"].names
    return {label: idx for idx, label in enumerate(labels)}


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predicted = predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predicted),
        "f1_macro": f1_score(labels, predicted, average="macro"),
    }


def train_model():
    train_dataset, validation_dataset = get_dataset_splits()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    id2label = {idx: label for idx, label in enumerate(train_dataset.features["label"].names)}
    label2id = {label: idx for idx, label in enumerate(train_dataset.features["label"].names)}

    tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_eval = validation_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_eval = tokenized_eval.remove_columns(["text"])
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_eval = tokenized_eval.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final-model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final-model"))


if __name__ == "__main__":
    train_model()
