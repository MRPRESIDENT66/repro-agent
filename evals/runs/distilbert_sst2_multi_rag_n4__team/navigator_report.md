## Navigator Handoff – Reproduce SST-2 Accuracy for distilbert-base-uncased-finetuned-sst-2-english

### Model
- **Class**: `AutoModelForSequenceClassification` (from `transformers`)
- **Name**: `"distilbert-base-uncased-finetuned-sst-2-english"` (load from local cache; offline, CPU-only)
- **Tokenizer**: `AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")` — the model's own tokenizer

### Label Mapping (from `model_card.md`)
| id | label    |
|----|----------|
| 0  | NEGATIVE |
| 1  | POSITIVE |

This aligns with SST-2 gold labels (0 = negative, 1 = positive). No remapping needed.

### Dataset
- **Source**: GLUE / SST-2 (pre-cached locally)
- **Split**: `validation`
- **Size**: 872 examples
- **Fields**: `sentence` (text), `label` (integer, 0 or 1)

### Tokenization
- Use the model's own tokenizer
- Apply `padding=True`, `truncation=True` (and `return_tensors="pt"`)
- Run on CPU (no device mapping to GPU)

### Evaluation
- Compute top-1 classification accuracy: `(correct predictions) / 872 * 100`
- Report as a percentage (e.g., `91.8%`)

### Execution Notes
- Offline environment: model and dataset are pre-cached; load by name from cache.
- CPU-only: do not move model to CUDA.
- Use `datasets.load_dataset("glue", "sst2", split="validation")` for the SST-2 validation set.
