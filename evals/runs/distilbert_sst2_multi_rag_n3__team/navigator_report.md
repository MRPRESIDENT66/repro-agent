## Navigator Handoff – Reproduce SST-2 Accuracy for distilbert-base-uncased-finetuned-sst-2-english

### Model
- **Class**: `AutoModelForSequenceClassification` (from `transformers`)
- **Name**: `"distilbert-base-uncased-finetuned-sst-2-english"` (load from local cache; offline)
- **Tokenizer**: `AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")`
- **Device**: CPU only

### Label Mapping (from `model_card.md`)
| id | label     |
|----|-----------|
| 0  | NEGATIVE  |
| 1  | POSITIVE  |

This aligns with SST-2 gold labels (0 = negative, 1 = positive). No remapping needed.

### Dataset
- **Source**: GLUE / SST-2 validation split
- **Size**: 872 examples
- **Fields**: `sentence` (text), `label` (integer, 0 or 1)
- **Loading**: `load_dataset("glue", "sst2", split="validation")` — pre-cached locally

### Tokenization
- Use the model's own tokenizer
- Apply `padding=True`, `truncation=True`, `return_tensors="pt"`
- Tokenize the `sentence` field

### Evaluation
- Run inference in `torch.no_grad()` context
- Get logits, take `argmax(dim=-1)` for predictions
- Compare predictions against gold `label` field
- Compute accuracy = (correct / 872) * 100, report as percentage

### Expected Output
A single float: the top-1 classification accuracy in percent (e.g., ~91–93% based on published results).
