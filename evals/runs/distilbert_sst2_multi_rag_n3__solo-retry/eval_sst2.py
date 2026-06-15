#!/usr/bin/env python3
"""Evaluate distilbert-base-uncased-finetuned-sst-2-english on GLUE SST-2 validation set."""

import json

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
BATCH_SIZE = 64
DEVICE = torch.device("cpu")


def main() -> None:
    # 1. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(DEVICE)

    # 2. Load SST-2 validation split (GLUE)
    dataset = load_dataset("glue", "sst2", split="validation")
    # 872 examples; fields: 'sentence' (str), 'label' (int, 0/1)

    sentences = dataset["sentence"]
    gold_labels = dataset["label"]

    # 3. Tokenize and run inference in batches
    correct = 0
    total = len(sentences)

    for i in range(0, total, BATCH_SIZE):
        batch_sentences = sentences[i : i + BATCH_SIZE]
        batch_gold = gold_labels[i : i + BATCH_SIZE]

        inputs = tokenizer(
            batch_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = logits.argmax(dim=-1).cpu().tolist()

        correct += sum(p == g for p, g in zip(preds, batch_gold))

    accuracy = 100.0 * correct / total

    # 4. Print exactly one strict-JSON REPRO_RESULT line
    result = {"metric": "accuracy", "actual": accuracy, "num_examples": total}
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
