#!/usr/bin/env python3
"""Evaluate distilbert-base-uncased-finetuned-sst-2-english on GLUE SST-2 validation set."""

import json

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
BATCH_SIZE = 64
DEVICE = torch.device("cpu")


def main() -> None:
    # 1. Load model and tokenizer (offline, from cache)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(DEVICE)

    # 2. Load GLUE SST-2 validation split (872 examples)
    dataset = load_dataset("glue", "sst2", split="validation")

    # 3. Tokenize and run batched inference
    sentences = dataset["sentence"]
    labels = dataset["label"]  # gold labels, already 0/1

    all_preds: list[int] = []

    for i in range(0, len(sentences), BATCH_SIZE):
        batch_sentences = sentences[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        preds = outputs.logits.argmax(dim=-1).tolist()
        all_preds.extend(preds)

    # 4. Compute accuracy
    correct = sum(1 for p, g in zip(all_preds, labels) if p == g)
    total = len(labels)
    accuracy = (correct / total) * 100.0

    # 5. Print exactly one strict-JSON REPRO_RESULT line
    result = {
        "metric": "accuracy",
        "actual": round(accuracy, 2),
        "num_examples": total,
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
