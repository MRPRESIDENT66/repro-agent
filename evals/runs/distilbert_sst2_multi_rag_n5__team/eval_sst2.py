#!/usr/bin/env python3
"""Evaluate distilbert-base-uncased-finetuned-sst-2-english on GLUE SST-2 validation set."""

import json

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    # Load model and tokenizer (offline, cached weights)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Load GLUE SST-2 validation split
    dataset = load_dataset("glue", "sst2", split="validation")

    # Tokenize and run inference in batches
    sentences = dataset["sentence"]
    labels = dataset["label"]

    correct = 0
    total = len(sentences)
    batch_size = 64

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_sentences = sentences[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            inputs = tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)

            correct += (predictions == torch.tensor(batch_labels)).sum().item()

    accuracy = (correct / total) * 100.0

    # Print exactly one strict-JSON REPRO_RESULT line
    result = {"metric": "accuracy", "actual": accuracy, "num_examples": total}
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
