## Navigator Handoff – distilbert-base-uncased-finetuned-sst-2-english on SST-2

### Model
- **Class**: `AutoModelForSequenceClassification` (from `transformers`)
- **Name**: `"distilbert-base-uncased-finetuned-sst-2-english"`
- **Tokenizer**: `AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")`
- **Head**: Sequence classification (binary)
- **Load**: Use the model name string; weights are pre-cached locally. CPU-only, offline.

### Label Mapping (from `model_card.md`)
| id | label    |
|----|----------|
| 0  | NEGATIVE |
| 1  | POSITIVE |

This aligns with SST-2 gold labels (0 = negative, 1 = positive). No remapping needed.

### Dataset
- **Source**: GLUE SST-2
- **Split**: `validation`
- **Size**: 872 examples
- **Fields**: `sentence` (text), `label` (integer 0/1)

### Tokenization
- Use the model's own tokenizer.
- Apply `padding=True`, `truncation=True`, `return_tensors="pt"`.
- Run on CPU (no device mapping to GPU).

### Evaluation
- Compute top-1 classification accuracy: `(correct / total) * 100`.
- Report as a percentage (e.g., `91.8`).

### Grounding
All facts above are sourced from the provided `model_card.md` (label mapping, dataset description, model identity). No private or external information was used.
