#!/usr/bin/env python3
"""Evaluate distilbert-base-uncased-finetuned-sst-2-english on GLUE SST-2 validation."""

import json

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    # 1. Load tokenizer and model (from local cache, CPU only)
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model.eval()
    model.to(device)

    # 2. Load SST-2 validation split (cached config name is 'sst2', not 'sst-2')
    dataset = load_dataset("glue", "sst2", split="validation")

    # 3. Tokenize all sentences
    sentences = dataset["sentence"]
    labels = dataset["label"]

    encodings = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # 4. Run inference
    correct = 0
    total = len(labels)

    with torch.no_grad():
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = logits.argmax(dim=-1)

        # 5. Compare against gold labels
        correct = (predictions.cpu() == torch.tensor(labels)).sum().item()

    # 6. Compute accuracy as a percentage
    accuracy = (correct / total) * 100.0

    # Print exactly one strict-JSON REPRO_RESULT line
    result = {"metric": "accuracy", "actual": accuracy, "num_examples": total}
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
