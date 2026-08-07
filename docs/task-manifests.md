# Declarative Oracle Manifests

Standard evaluation tasks can be added without implementing a complete Python
Oracle adapter. An Oracle-side YAML manifest is loaded by `evals/manifest.py`, which
creates the same `OracleConfig` consumed by the existing agent pipeline.

```text
Oracle-side YAML manifest
        |
        v
make_oracle_config()
        |
        +-- public task, command, and output contract --> Agent
        +-- hidden gold, expected value, tolerance ----> verifier only
        |
        v
existing ReproductionPipeline
```

The migrated STS-B example is
[`evals/tasks/sentence_transformers_stsb.yaml`](../evals/tasks/sentence_transformers_stsb.yaml).
Its Python wrapper only preserves the existing `make_config(attempt)` import.

## Standard Profile

The generic runtime currently supports:

- a pinned Git repository copied into a clean workspace;
- one public JSONL input copied into the workspace;
- one model directory mounted with a symbolic link;
- a local, secret-scrubbed `Session` with declared environment variables;
- JSON output containing a fixed-size list of objects;
- exact fields, primitive types, sequential IDs, numeric ranges, and sample count;
- `accuracy`, `auroc`, `pearson`, and `spearman` metric recomputation;
- repository commit and required-asset checks;
- fail-closed target/gold isolation and normal audit artifacts.

The manifest itself remains Oracle-side and is never copied into the agent's
workspace. Only `task.description`, the execution command, and the generated
output contract enter the LLM context.

`accuracy` and `auroc` return fractions in `[0, 1]`; correlations use their
standard `[-1, 1]` scale. The private `verification.expected` value must use the
same scale.

## Optional Hooks

Tasks with unusual setup or grading can reuse the standard path and add a small
explicit hook:

```python
from evals.manifest import OracleHooks, make_oracle_config


def provision(manifest, workdir):
    # Add task-specific links, generated configs, or converted public assets.
    ...


def verifier(manifest, workdir):
    # Recompute a task-specific aggregate from verifier-visible artifacts.
    return score, num_examples


def make_config(attempt):
    return make_oracle_config(
        MANIFEST,
        attempt,
        hooks=OracleHooks(provision=provision, verifier=verifier),
    )
```

Hooks run on the Oracle side. They are not LLM tools and are not visible to the
agent. A provisioning hook extends common workspace setup; a verifier hook
replaces the registered metric recomputation.

## Limits

Docker sessions, multiple model/data mounts, nested task-specific output shapes,
and complex grouped metrics are not forced into the standard profile. They can
use hooks or retain a custom adapter. The goal is to remove repeated glue from
ordinary tasks, not to encode arbitrary Python programs in YAML.
