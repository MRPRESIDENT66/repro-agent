# Task Manifests

Every benchmark task enters through an Oracle-side YAML file in `evals/tasks/`.
`evals.catalog.make_config(task, attempt)` loads the manifest, resolves its
optional named hook, and creates the `OracleConfig` consumed by the adaptive
pipeline.

```text
YAML manifest -> catalog -> ManifestRuntime -> OracleConfig -> adaptive pipeline
                       \-> optional Oracle-side hook
```

## What Belongs in YAML

- task identity and public description;
- pinned repository, model, dataset, and selectively copied or linked assets;
- generated script name, command, environment, backend profiles, and timeout;
- output file, structure, fields, ranges, and sample count;
- private metric, gold path, expected value, tolerance, and chance floor.

The manifest remains outside the blind workspace. Agent context receives only
the public description, command, and output protocol. Private verification
fields stay in `OracleConfig` and are used after the agent workflow.

## Reusable Capabilities

`assets` provisions files or directories with `copy` or `symlink`. An optional
`include` list selects only required checkpoint/data subtrees, while `exclude`
keeps generated or private files out. `mount_as: .` merges a repository into the
workspace root.

`execution` defines the generated script, command, timeout, Python environment,
and a local or Docker runtime. Optional named `profiles` inherit those defaults
and can be selected by `profile_env`; OpenOOD uses this to share one task
definition between the isolated Docker backend and trusted host MPS backend.

`privacy.scrub_globs` removes the private target from matching public text after
provisioning. The blind-workspace check still scans the result and fails closed
if a target or hidden file remains.

## Output Structures

- `records`: a JSON list of objects, such as STS-B `{id, similarity}` rows;
- `values`: a JSON list of scalar predictions for classification tasks;
- `grouped_scores`: named groups of fixed-length score series;
- `custom`: a task-defined shape validated by a `public_check` hook.

Generic metrics include `accuracy`, `top1_accuracy`, `auroc`, `pearson`, and
`spearman`. Grouped scores support mean AUROC across positive series and groups,
plus public score-direction diagnostics. `verification.scale` converts a
fraction to percentage, and `gold_limit` selects a declared private-label prefix.

## Hooks

Hooks live in `evals/hooks/` and are selected by the manifest's top-level `hook`
name. Available extension points are deliberately small:

- `provision` or `provision_override`;
- `session` and `execute`;
- `public_check` and `public_diagnostics`;
- `blind_check`;
- `verifier`.

Use a hook only for behavior that is genuinely task-specific. OpenOOD,
mmpretrain, RobustBench, and STS-B now use no hook; DistilBERT and detectors keep
small hooks for dynamic or filtered model-card creation. Hooks are deterministic
Oracle code, not LLM tools.

## Current Coverage

All shipped runners are manifest-backed: STS-B, DistilBERT, two detectors tasks,
mmpretrain, RobustBench, and OpenOOD. The tiny modules in `evals/oracles/` are
compatibility imports only; task logic lives in YAML plus optional hooks.

## Design Limit

Manifest-first does not mean arbitrary-repository zero-config. New standard
tasks should require only YAML. A genuinely new shared backend, output family,
or metric belongs in a focused reusable module; one-off source discovery or
preprocessing belongs in a short hook rather than turning YAML into a general-
purpose programming language.
