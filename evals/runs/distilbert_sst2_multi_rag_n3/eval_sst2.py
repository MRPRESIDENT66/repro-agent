#!/usr/bin/env python3
"""Evaluate distilbert-base-uncased-finetuned-sst-2-english on GLUE SST-2 validation set."""

import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset


def main():
    # Load model and tokenizer (offline, cached)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    model.eval()

    # Load GLUE SST-2 validation split (872 examples)
    dataset = load_dataset("glue", "sst2", split="validation")

    sentences = dataset["sentence"]
    labels = dataset["label"]

    # Tokenize in batch
    encodings = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # CPU inference
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Compute accuracy
    correct = (predictions == torch.tensor(labels)).sum().item()
    total = len(labels)
    accuracy = (correct / total) * 100.0

    # Print exactly one strict-JSON REPRO_RESULT line
    result = {
        "metric": "accuracy",
        "actual": accuracy,
        "num_examples": total,
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
