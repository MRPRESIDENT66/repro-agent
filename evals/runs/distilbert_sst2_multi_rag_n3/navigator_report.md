## Navigator Handoff – Reproduce SST-2 Accuracy for `distilbert-base-uncased-finetuned-sst-2-english`

### Model
- **Class**: `AutoModelForSequenceClassification` (from `transformers`)
- **Name**: `"distilbert-base-uncased-finetuned-sst-2-english"` – load from local cache (offline, CPU-only)
- **Tokenizer**: `AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")` – the model's own tokenizer

### Label Mapping (from `model_card.md`)
| id | label    |
|----|----------|
| 0  | NEGATIVE |
| 1  | POSITIVE |

This aligns with SST-2 gold labels (0 = negative, 1 = positive). No remapping needed.

### Dataset
- **Source**: GLUE / SST-2 validation split
- **Size**: 872 examples
- **Fields**: `sentence` (text), `label` (integer, 0 or 1)
- **Loading**: `load_dataset("glue", "sst2", split="validation")` – pre-cached locally

### Tokenization
- Use the model's own tokenizer
- Apply `padding=True`, `truncation=True` (and `return_tensors="pt"`)
- Tokenize the `sentence` field

### Evaluation
- **Device**: CPU (no GPU available)
- **Mode**: `model.eval()`, `torch.no_grad()`
- **Metric**: Top-1 classification accuracy = (correct predictions / 872) * 100, reported as a percentage
- **Prediction**: `torch.argmax(logits, dim=-1)` compared against ground-truth `label`

### Key Constraints
- Offline environment – model and dataset are pre-cached; load by name from cache
- CPU-only – no `.to("cuda")`
- Report final accuracy as a percentage string (e.g., `"91.3%"`)
