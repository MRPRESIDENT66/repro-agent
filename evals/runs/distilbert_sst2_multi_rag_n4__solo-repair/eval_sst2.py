#!/usr/bin/env python3
"""Evaluate distilbert-base-uncased-finetuned-sst-2-english on GLUE SST-2 validation set."""

import json

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    # Load model and tokenizer by name (cached offline)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Load GLUE SST-2 validation split
    dataset = load_dataset("glue", "sst2", split="validation")

    # Tokenize and run inference in batches
    batch_size = 64
    correct = 0
    total = len(dataset)

    for i in range(0, total, batch_size):
        batch = dataset[i : i + batch_size]
        texts = batch["sentence"]
        labels = batch["label"]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = logits.argmax(dim=-1).tolist()

        for pred, gold in zip(predictions, labels):
            if pred == gold:
                correct += 1

    accuracy = (correct / total) * 100.0

    result = {
        "metric": "accuracy",
        "actual": accuracy,
        "num_examples": total,
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
