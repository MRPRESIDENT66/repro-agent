#!/usr/bin/env python3
"""Evaluate distilbert-base-uncased-finetuned-sst-2-english on GLUE SST-2 validation set."""

import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

def main():
    # Load model and tokenizer from local cache (offline)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Load GLUE SST-2 validation split (872 examples)
    dataset = load_dataset("glue", "sst2", split="validation")

    correct = 0
    total = len(dataset)

    # Process in batches for efficiency
    batch_size = 32
    for i in range(0, total, batch_size):
        batch = dataset[i:i + batch_size]
        texts = batch["sentence"]
        labels = batch["label"]

        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Forward pass (CPU)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = logits.argmax(dim=-1).tolist()

        # Compare predictions to gold labels
        for pred, gold in zip(predictions, labels):
            if pred == gold:
                correct += 1

    accuracy = (correct / total) * 100.0

    # Output exactly one JSON line
    result = {
        "metric": "accuracy",
        "actual": accuracy,
        "num_examples": total
    }
    print(f"REPRO_RESULT {json.dumps(result)}")

if __name__ == "__main__":
    main()
