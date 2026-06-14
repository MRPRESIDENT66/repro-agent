## Navigator Handoff – Reproduce SST-2 Accuracy for distilbert-base-uncased-finetuned-sst-2-english

### Model
- **Class**: `AutoModelForSequenceClassification` (from `transformers`)
- **Name**: `"distilbert-base-uncased-finetuned-sst-2-english"` (load from local cache; offline, CPU-only)
- **Tokenizer**: `AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")` — use the model's own tokenizer
- **Label mapping** (from `model_card.md`):
  - `0` → `"NEGATIVE"`
  - `1` → `"POSITIVE"`
  - This aligns with SST-2 gold labels (0 = negative, 1 = positive).

### Dataset
- **Source**: GLUE / SST-2
- **Split**: validation
- **Number of examples**: 872
- **Fields**: `sentence` (text), `label` (integer, 0 or 1)

### Tokenization
- Use the model's own tokenizer.
- Apply `padding=True`, `truncation=True`, `return_tensors="pt"`.

### Evaluation
- **Device**: CPU (no GPU available; offline environment).
- **Metric**: Top-1 classification accuracy (percent).
- **Procedure**: For each example in the validation split, tokenize the `sentence`, forward through the model, take `argmax` over logits, compare to gold `label`, compute accuracy = (correct / 872) * 100.

### Key constraints
- Offline: load model and dataset from local cache (do not attempt downloads).
- CPU-only.
- Report accuracy as a percentage (e.g., `91.8%`).
