#!/usr/bin/env python3
"""Evaluate distilbert-base-uncased-finetuned-sst-2-english on GLUE SST-2 validation set."""

import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset


def main():
    # Load model and tokenizer (offline, from cache)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Load GLUE SST-2 validation split (872 examples)
    dataset = load_dataset("glue", "sst2", split="validation")

    # Tokenize all sentences
    sentences = dataset["sentence"]
    labels = dataset["label"]

    encodings = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on CPU
    correct = 0
    total = len(labels)

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = logits.argmax(dim=-1).tolist()

    for pred, gold in zip(predictions, labels):
        if pred == gold:
            correct += 1

    accuracy = (correct / total) * 100.0

    # Print exactly one strict-JSON REPRO_RESULT line
    result = {"metric": "accuracy", "actual": accuracy, "num_examples": total}
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
