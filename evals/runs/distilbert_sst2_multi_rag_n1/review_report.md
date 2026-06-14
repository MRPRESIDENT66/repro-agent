## Audit Summary

**Execution Error (Command 2):**  
`ValueError: Couldn't find cache for glue for config 'sst-2'`  
The cached dataset config is named `sst2`, not `sst-2`. The script was corrected to use `"sst2"` and subsequently succeeded.

**Successful Execution (Command 4):**  
`REPRO_RESULT {"metric": "accuracy", "actual": 91.05504587155964, "num_examples": 872}`

**Contract Checks:**
- ✅ Accuracy is a percentage (91.06%), not a fraction.
- ✅ num_examples = 872 (full validation split).
- ✅ Label mapping: model_card.md confirms 0=NEGATIVE, 1=POSITIVE, matching SST-2 gold labels. Accuracy ~91% is well above 50%, so no inversion.
- ✅ Result came from real model inference (not a hardcoded constant) — the script loads the model and runs `model()` with tokenized inputs.

**No repair is needed.** The script now correctly loads the cached dataset config `sst2`, runs inference, and reports accuracy as a percentage over all 872 examples.

REVIEW_STATUS: PASS
