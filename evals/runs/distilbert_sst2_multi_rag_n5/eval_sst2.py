#!/usr/bin/env python3
"""Reproduce SST-2 accuracy for distilbert-base-uncased-finetuned-sst-2-english."""

import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

def main():
    # Load model and tokenizer from local cache (offline, CPU)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model.eval()

    # Load GLUE SST-2 validation split (872 examples)
    dataset = load_dataset("glue", "sst2", split="validation")

    # Tokenize all sentences
    texts = dataset["sentence"]
    labels = dataset["label"]

    # Tokenize with padding and truncation
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Run inference on CPU
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = logits.argmax(dim=-1).numpy()

    # Compute accuracy as percentage
    correct = (predictions == labels).sum()
    total = len(labels)
    accuracy = (correct / total) * 100.0

    # Print exactly one strict-JSON REPRO_RESULT line
    result = {
        "metric": "accuracy",
        "actual": round(accuracy, 2),
        "num_examples": total
    }
    print(f"REPRO_RESULT {json.dumps(result)}")

if __name__ == "__main__":
    main()
