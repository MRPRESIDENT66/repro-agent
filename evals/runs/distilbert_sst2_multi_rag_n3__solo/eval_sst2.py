#!/usr/bin/env python3
"""Evaluate distilbert-base-uncased-finetuned-sst-2-english on SST-2 validation set."""

import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

def main():
    # Load model and tokenizer (cached, offline)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Load SST-2 validation split (GLUE config)
    dataset = load_dataset("glue", "sst2", split="validation")
    # dataset has 872 examples, fields: 'sentence', 'label'

    # Tokenize in batches
    def tokenize_batch(batch):
        return tokenizer(batch["sentence"], padding=True, truncation=True, return_tensors="pt")

    # Run inference in batches (CPU)
    batch_size = 32
    correct = 0
    total = 0

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        inputs = tokenize_batch(batch)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        preds = logits.argmax(dim=-1).numpy()
        labels = batch["label"]
        correct += (preds == labels).sum().item()
        total += len(labels)

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
