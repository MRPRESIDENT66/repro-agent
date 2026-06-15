#!/usr/bin/env python3
"""Evaluate distilbert-base-uncased-finetuned-sst-2-english on SST-2 validation set."""

import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

def main():
    # Load model and tokenizer from cache (offline, by name)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Load SST-2 validation split (GLUE config)
    dataset = load_dataset("glue", "sst2", split="validation")
    # dataset has 872 examples; fields: 'sentence', 'label'

    # Tokenize all sentences with padding and truncation
    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], padding=True, truncation=True, return_tensors="pt")

    # Process in batches to avoid OOM on CPU
    batch_size = 32
    correct = 0
    total = 0

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        inputs = tokenize_fn(batch)
        # Move tensors to CPU (already on CPU)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        preds = logits.argmax(dim=-1).tolist()
        golds = batch["label"]
        correct += sum(p == g for p, g in zip(preds, golds))
        total += len(golds)

    accuracy = (correct / total) * 100.0
    result = {"metric": "accuracy", "actual": accuracy, "num_examples": total}
    print("REPRO_RESULT", json.dumps(result))

if __name__ == "__main__":
    main()
