#!/usr/bin/env python3
"""CPU-safe evaluation of distilbert-base-uncased-finetuned-sst-2-english on GLUE SST-2 validation."""

import json

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    # 1. Load model and tokenizer (offline, cached)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # 2. Load dataset
    dataset = load_dataset("glue", "sst2", split="validation")
    sentences = dataset["sentence"]
    gold_labels = dataset["label"]

    # 3. Tokenize and run inference in batches
    batch_size = 64
    correct = 0
    total = len(sentences)

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
            gold_batch = gold_labels[i : i + batch_size]
            correct += sum(1 for p, g in zip(preds, gold_batch) if p == g)

    accuracy = (correct / total) * 100.0

    # 4. Print exactly one strict-JSON REPRO_RESULT line
    result = {
        "metric": "accuracy",
        "actual": accuracy,
        "num_examples": total,
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
