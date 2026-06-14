## Navigator Handoff – Reproduce SST-2 Accuracy for distilbert-base-uncased-finetuned-sst-2-english

### Model
- **Class**: `AutoModelForSequenceClassification` (from `transformers`)
- **Name**: `"distilbert-base-uncased-finetuned-sst-2-english"` – load from local cache (offline, CPU-only)
- **Tokenizer**: `AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")` – the model's own tokenizer

### Label Mapping (from `model_card.md`)
| id | label    |
|----|----------|
| 0  | NEGATIVE |
| 1  | POSITIVE |

This aligns with SST-2 gold labels (0 = negative, 1 = positive).

### Dataset
- **Source**: GLUE / SST-2 validation split
- **Size**: 872 examples
- **Fields**: `sentence` (text), `label` (integer, 0 or 1)
- **Loading**: `load_dataset("glue", "sst2", split="validation")` – pre-cached locally

### Tokenization
- Use the model's own tokenizer
- Apply `padding=True`, `truncation=True` (return tensors="pt")
- Run on CPU only (no device mapping to GPU)

### Evaluation
- Compute top-1 classification accuracy: `(predictions == labels).mean() * 100`
- Report as a percentage (e.g., `91.2%`)

### Constraints
- **CPU-only**, **offline** – no internet, no GPU
- Load model and dataset from local cache only
