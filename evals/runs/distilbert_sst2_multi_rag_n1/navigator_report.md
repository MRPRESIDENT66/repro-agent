## Navigator Handoff: Reproduce SST-2 Accuracy for distilbert-base-uncased-finetuned-sst-2-english

### Model
- **Class**: `AutoModelForSequenceClassification` (from `transformers`)
- **Model name**: `"distilbert-base-uncased-finetuned-sst-2-english"` — load from local cache (offline, CPU-only)
- **Tokenizer**: `AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")` — the model's own tokenizer
- **Label mapping** (from `model_card.md`):
  - `0` → NEGATIVE
  - `1` → POSITIVE
  - Aligns with SST-2 gold labels (0 = negative, 1 = positive)

### Dataset
- **Dataset**: GLUE SST-2, **validation** split
- **Number of examples**: 872
- **Text field**: `"sentence"`
- **Label field**: `"label"` (integer, 0 or 1)

### Tokenization
- Use the model's own tokenizer
- Apply `padding=True`, `truncation=True` (and `return_tensors="pt"`)
- Run on CPU only

### Evaluation
- Compute top-1 classification accuracy: `(correct_predictions / 872) * 100`
- Report as a percentage (e.g., `91.33%`)

### Implementation Steps
1. Load tokenizer and model via `AutoTokenizer` / `AutoModelForSequenceClassification` from the local cache.
2. Load SST-2 validation split from `datasets.load_dataset("glue", "sst-2", split="validation")`.
3. Tokenize all sentences with padding/truncation.
4. Run inference (no gradient, model.eval()).
5. Compare `argmax(logits, dim=1)` against gold labels.
6. Compute and print accuracy as a percentage.
