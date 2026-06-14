#!/usr/bin/env python3
"""CPU-safe SST-2 evaluation script. Prints REPRO_RESULT JSON line."""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

def main():
    # 1. Load model and tokenizer by name (cached locally)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # 2. Load SST-2 validation split (872 examples)
    dataset = load_dataset("glue", "sst2", split="validation")

    # 3. Tokenize and evaluate in batches for efficiency
    sentences = dataset["sentence"]
    labels = dataset["label"]

    # Tokenize all at once
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

    # 4. Compute accuracy
    correct = (predictions == torch.tensor(labels)).sum().item()
    total = len(labels)
    accuracy = correct / total * 100.0

    # 5. Print strict JSON result line
    result = {
        "metric": "accuracy",
        "actual": accuracy,
        "num_examples": total
    }
    print(f"REPRO_RESULT {json.dumps(result)}")

if __name__ == "__main__":
    main()
