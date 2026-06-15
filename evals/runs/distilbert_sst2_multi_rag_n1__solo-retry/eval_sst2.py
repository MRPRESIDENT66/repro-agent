#!/usr/bin/env python3
"""Evaluate distilbert-base-uncased-finetuned-sst-2-english on GLUE SST-2 validation set."""

import json

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    # Load model and tokenizer by name (cached on disk)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Load GLUE SST-2 validation split
    dataset = load_dataset("glue", "sst2", split="validation")
    # 872 examples; fields: 'sentence' (str), 'label' (int, 0 or 1)

    sentences = dataset["sentence"]
    gold_labels = dataset["label"]

    # Tokenize in batches
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
            logits = outputs.logits
            predictions = logits.argmax(dim=-1).tolist()

            batch_gold = gold_labels[i : i + batch_size]
            correct += sum(p == g for p, g in zip(predictions, batch_gold))

    accuracy = (correct / total) * 100.0

    result = {
        "metric": "accuracy",
        "actual": accuracy,
        "num_examples": total,
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
