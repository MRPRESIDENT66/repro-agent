## Audit of `eval_sst2.py`

### Contract checks

| Requirement | Status | Evidence |
|---|---|---|
| Accuracy is a percentage (0–100) | ✅ PASS | `accuracy = (correct / total) * 100.0` → 91.055% |
| num_examples = 872 (full validation split) | ✅ PASS | `total = len(labels)` → 872 |
| Label mapping correct (not inverted) | ✅ PASS | Model card: 0=NEGATIVE, 1=POSITIVE. Accuracy ~91% (>50%) confirms no argmax inversion. |
| Real model inference (not hardcoded) | ✅ PASS | Code loads model, tokenizes, runs `model(**encodings)`, computes argmax. Log shows actual execution. |

### Conclusion

All deterministic public-contract requirements are satisfied. No repair needed.

REVIEW_STATUS: PASS
