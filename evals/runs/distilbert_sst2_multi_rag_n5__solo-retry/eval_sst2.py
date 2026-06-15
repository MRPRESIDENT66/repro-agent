#!/usr/bin/env python3
"""Evaluate distilbert-base-uncased-finetuned-sst-2-english on SST-2 validation set."""

import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

def main():
    # Load model and tokenizer by name (cached offline)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Load SST-2 validation split (GLUE config)
    dataset = load_dataset("glue", "sst2", split="validation")
    # dataset has 872 examples; fields: 'sentence', 'label'

    # Tokenize all sentences
    sentences = dataset["sentence"]
    labels = dataset["label"]

    # Tokenize with padding and truncation
    encodings = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Run inference in batches (CPU)
    batch_size = 32
    all_preds = []
    num_examples = len(sentences)

    with torch.no_grad():
        for i in range(0, num_examples, batch_size):
            batch_encodings = {k: v[i:i+batch_size] for k, v in encodings.items()}
            outputs = model(**batch_encodings)
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.tolist())

    # Compute accuracy
    correct = sum(1 for pred, gold in zip(all_preds, labels) if pred == gold)
    accuracy = (correct / num_examples) * 100.0

    # Print exactly one JSON line
    result = {
        "metric": "accuracy",
        "actual": accuracy,
        "num_examples": num_examples,
    }
    print(f"REPRO_RESULT {json.dumps(result)}")

if __name__ == "__main__":
    main()
