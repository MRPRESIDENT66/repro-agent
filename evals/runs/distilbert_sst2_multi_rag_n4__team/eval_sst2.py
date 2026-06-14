#!/usr/bin/env python3
"""Evaluate distilbert-base-uncased-finetuned-sst-2-english on GLUE SST-2 validation set (CPU)."""

import json

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    # 1. Load model and tokenizer (offline, from cache)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # 2. Load SST-2 validation split
    dataset = load_dataset("glue", "sst2", split="validation")

    # 3. Tokenize and run inference in batches
    sentences = dataset["sentence"]
    labels = dataset["label"]  # gold labels, already 0/1

    correct = 0
    total = len(sentences)
    batch_size = 64

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_sentences = sentences[i : i + batch_size]
            inputs = tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).tolist()

            for pred, gold in zip(preds, labels[i : i + batch_size]):
                if pred == gold:
                    correct += 1

    accuracy = correct / total * 100.0

    # 4. Print exactly one strict-JSON REPRO_RESULT line
    result = {
        "metric": "accuracy",
        "actual": accuracy,
        "num_examples": total,
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
