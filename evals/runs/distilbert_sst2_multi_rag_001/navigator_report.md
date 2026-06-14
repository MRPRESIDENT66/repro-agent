```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# 1. Load model and tokenizer from local snapshot
model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

# 2. Verify label mapping from config.json
# Expected: id2label = {0: "NEGATIVE", 1: "POSITIVE"}, label2id = {"NEGATIVE": 0, "POSITIVE": 1}
# SST-2 gold: 0 = negative, 1 = positive → aligns perfectly
print("id2label:", model.config.id2label)
print("label2id:", model.config.label2id)

# 3. Load SST-2 validation split (872 examples)
dataset = load_dataset("glue", "sst2", split="validation")

# 4. Tokenize with padding + truncation
def tokenize_fn(examples):
    return tokenizer(examples["sentence"], padding=True, truncation=True, return_tensors="pt")

# 5. Evaluate on CPU
model.eval()
correct = 0
total = 0

for example in dataset:
    inputs = tokenizer(example["sentence"], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=-1).item()
    correct += (pred == example["label"])
    total += 1

accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")
```
