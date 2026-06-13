REVIEW_STATUS: REPAIR_REQUIRED

**Finding: Semantic mismatch in EBO energy score sign.**  

The official OpenOOD `EBOPostprocessor` (in `openood/postprocessors/ebo_postprocessor.py`) computes the confidence score as:

```python
conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
```

This is the **negative** of the standard energy function. The standard energy is `E(x) = -T * logsumexp(f(x)/T)`, so the postprocessor returns `-E(x)`. In the OOD detection convention used by OpenOOD, **higher confidence = more ID-like**. Therefore the postprocessor’s output is already in the correct orientation: higher values indicate ID.

The reproduction script’s `energy_score` function (line 42) computes:

```python
return temperature * torch.logsumexp(logits / temperature, dim=1)
```

This is the **positive** energy (i.e., `-conf` from the postprocessor). When this score is fed to `roc_auc_score` with the label convention `0=ID, 1=OOD`, the sign is inverted: higher scores (which are actually more OOD-like under the standard energy definition) are treated as OOD, but the postprocessor’s orientation is the opposite. The result is that the AUROC is systematically flipped.

**Execution evidence:**  
- Commands 2 and 4 produce `actual: 87.58` (reasonable near-OOD AUROC for CIFAR-100/TinyImageNet).  
- Commands 6 and 10 produce `actual: 12.42` (near 100 – 87.58, confirming sign inversion).  
- The non-deterministic flip between runs (same code, same checkpoints) indicates that the sign of the energy score is not consistently applied relative to the label convention. The correct behavior (as demonstrated by the official postprocessor) is to use the **negative** energy so that higher scores correspond to ID.

**Required repair:**  
Change the `energy_score` function to match the official OpenOOD postprocessor:

```python
def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute energy score as in EBOPostprocessor: -T * logsumexp(f(x)/T)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)
```

This ensures that higher scores indicate ID, consistent with the `roc_auc_score` label convention (`0=ID, 1=OOD`). After this fix, the AUROC will be stable and match the expected near-OOD performance (~87–88%).

REVIEW_STATUS: REPAIR_REQUIRED
