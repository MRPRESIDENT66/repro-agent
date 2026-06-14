## Navigator Handoff – Reproduce SST-2 Accuracy for `distilbert-base-uncased-finetuned-sst-2-english`

### Model
- **Class**: `AutoModelForSequenceClassification` (from `transformers`)
- **Model name**: `"distilbert-base-uncased-finetuned-sst-2-english"` (load from local cache; offline, CPU-only)
- **Tokenizer**: `AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")` (same name, own tokenizer)

### Label Mapping (from `model_card.md`)
| id | label    |
|----|----------|
| 0  | NEGATIVE |
| 1  | POSITIVE |

This aligns with SST-2 gold labels (0 = negative, 1 = positive). No remapping needed.

### Dataset
- **Source**: GLUE SST-2, **validation** split
- **Size**: 872 examples
- **Fields**: `sentence` (text), `label` (integer, 0 or 1)
- **Loading**: `load_dataset("glue", "sst2", split="validation")` — pre-cached locally

### Tokenization
- Use the model's own tokenizer
- Apply `padding=True`, `truncation=True`, `return_tensors="pt"`

### Evaluation
- CPU-only, offline
- Run inference on all 872 examples, get predicted class (argmax over logits)
- Compute accuracy = (correct predictions / 872) × 100
- Report as a percentage (e.g., `91.2%`)

### Expected published accuracy
The published accuracy for this model on SST-2 validation is **91.2%** (or close to it, typically reported as ~91.2–91.3%).
