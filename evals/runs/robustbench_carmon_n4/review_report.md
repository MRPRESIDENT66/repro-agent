REVIEW_STATUS: REPAIR_REQUIRED

**Audit findings:**

1. **No valid REPRO_RESULT line** – The execution never reached the `print(f"REPRO_RESULT ...")` line because the script crashed with a `RuntimeError` in all successful attack runs (Commands 4, 8, 10). The error is:
   ```
   RuntimeError: The size of tensor a (32) must match the size of tensor b (50) at non-singleton dimension 2
   ```
   This indicates a shape mismatch: `x_adv.argmax(1)` has shape `[32]` (batch size from AutoAttack's internal batching) while `y_test` has shape `[50]` (the requested 50 examples). The `run_standard_evaluation` method processes data in batches internally, and the returned `x_adv` may not align with the original `y_test` tensor when `n_examples` is not a multiple of the batch size.

2. **robust accuracy is percentage (0–100)** – The code correctly converts to percentage with `robust_acc_pct = robust_acc * 100.0`, so this requirement is met in the implementation.

3. **num_examples=50 in REPRO_RESULT** – The implementation includes `"num_examples": args.n_examples` in the result dict, which would be 50. However, no REPRO_RESULT was ever printed.

4. **AutoAttack was actually run** – The logs show AutoAttack ran successfully (APGD-CE and APGD-DLR completed), so this requirement is satisfied.

5. **Result came from actual model evaluation** – The attack ran and produced robust accuracy values (e.g., 52.00%), but the final computation failed due to the shape mismatch, so the result was never captured.

**Root cause:** The `run_standard_evaluation` method returns adversarial examples in batches, and the final `x_adv` tensor may have a different first dimension than `y_test` when `n_examples` is not a multiple of the internal batch size. The comparison `x_adv.argmax(1) == y_test` fails because `x_adv` has shape `[32, 3, 32, 32]` (last batch of 32) while `y_test` has shape `[50]`.

**Required repair:** The evaluation must handle the batching correctly. Either:
- Use `adversary.run_standard_evaluation` with a batch size that divides `n_examples` evenly, or
- Recompute predictions on the full `x_adv` tensor after the attack, or
- Use the robust accuracy reported by AutoAttack's internal logging (e.g., "robust accuracy: 52.00%") instead of recomputing.

The script must be fixed to produce a valid `REPRO_RESULT` with `metric="robust_accuracy"`, `actual` as a percentage (0–100), and `num_examples=50`.

REVIEW_STATUS: REPAIR_REQUIRED
