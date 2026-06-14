## Navigator Handoff – distilbert-base-uncased-finetuned-sst-2-english on SST-2

### Model
- **Class**: `AutoModelForSequenceClassification` (HuggingFace transformers)
- **Model name**: `"distilbert-base-uncased-finetuned-sst-2-english"` – load from local cache (offline, CPU-only)
- **Tokenizer**: `AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")` – the model's own tokenizer

### Label Mapping (from `model_card.md`)
| id | label    |
|----|----------|
| 0  | NEGATIVE |
| 1  | POSITIVE |

This aligns with SST-2 gold labels (0 = negative, 1 = positive).

### Dataset
- **Source**: GLUE / SST-2
- **Split**: validation
- **Size**: 872 examples
- **Fields**: `sentence` (text), `label` (integer, 0 or 1)

### Tokenization
- Use the model's own tokenizer
- Apply `padding=True`, `truncation=True`
- Return PyTorch tensors

### Evaluation
- **Device**: CPU (no GPU available)
- **Metric**: Top-1 classification accuracy (percent)
- **Procedure**: For each example in the validation set, tokenize the `sentence`, run through the model, take argmax over logits, compare to gold `label`, compute accuracy = (correct / 872) * 100.

### Grounding
All facts above are sourced from the provided `model_card.md` (label mapping, dataset description, field names, split size). The model class and tokenizer are standard for this HuggingFace model card.
