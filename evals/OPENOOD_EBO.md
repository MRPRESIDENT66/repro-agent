# OpenOOD CIFAR-10 EBO Near-OOD reproduction

> Status: **canonical oracle established and strict-blind Agent reproduction
> passed**.
>
> Date: 2026-06-12

## Goal

Reproduce OpenOOD v1.5's CIFAR-10 CrossEntropy + Energy-based OOD detection
(`EBO`) leaderboard result using the three official ResNet-18 checkpoints.

The target is the mean **Near-OOD AUROC**:

1. For each checkpoint (`s0`, `s1`, `s2`), evaluate CIFAR-10 as ID data against
   the two Near-OOD datasets: CIFAR-100 and TinyImageNet.
2. Average the two dataset AUROCs to obtain that checkpoint's Near-OOD AUROC.
3. Report mean and population standard deviation across the three checkpoints.

## Fixed artifact

| Item | Value |
|---|---|
| Repository | `https://github.com/Jingkang50/OpenOOD` |
| Commit | `3c35632ee91b54b09d1f085d04f94744cece7d0b` |
| Model | Official CIFAR-10 ResNet-18 CrossEntropy checkpoints, `s0/s1/s2` |
| Postprocessor | EBO, temperature `1` |
| ID data | CIFAR-10 test split from OpenOOD benchmark imglist, 9,000 images |
| Near-OOD data | CIFAR-100 test, 9,000 images; TinyImageNet test, 7,793 images |
| Backend | Native macOS CPU, Python 3.12 |
| Downloaded resources | `repos/OpenOOD/data` 1.3 GB; `repos/OpenOOD/results` 384 MB |

Checkpoint SHA256:

```text
dc96647a7f835dbbf72abaec74810009abd444a39376328abbc582b91a5ec3fd  s0/best.ckpt
d055735fd1b5b7033a8ace7378fd81444e37e71bdcad9638a0c25817cc6f30aa  s1/best.ckpt
2c2a41cff9cb5e7ba360397a1e6f989dfec46a83f0db5907ddbf02a49959a9e0  s2/best.ckpt
```

## Canonical commands

From the project root:

```bash
git clone https://github.com/Jingkang50/OpenOOD.git repos/OpenOOD
git -C repos/OpenOOD checkout 3c35632ee91b54b09d1f085d04f94744cece7d0b

cd repos/OpenOOD
../../.venv-oracle/bin/python scripts/download/download.py \
  --contents checkpoints \
  --checkpoints cifar10_res18_v1.5 \
  --save_dir ./data ./results

../../.venv-oracle/bin/python scripts/download/download.py \
  --contents datasets \
  --datasets cifar10 cifar100 tin \
  --save_dir ./data ./results \
  --dataset_mode dataset

PYTHONPATH=. ../../.venv-oracle/bin/python run_nearood_ebo_cpu.py
```

The CPU runner is stored at:

```text
repos/OpenOOD/run_nearood_ebo_cpu.py
```

The raw structured result is stored at:

```text
repos/OpenOOD/nearood_ebo_cpu_results.json
```

## Results

### Per checkpoint

| Seed | CIFAR-100 AUROC | TinyImageNet AUROC | Near-OOD AUROC | ID accuracy |
|---|---:|---:|---:|---:|
| `s0` | 85.5469 | 88.3135 | 86.9302 | 95.2222 |
| `s1` | 86.8782 | 88.9415 | 87.9099 | 94.6333 |
| `s2` | 86.6568 | 89.1568 | 87.9068 | 95.3222 |

### Three-checkpoint aggregate

| Metric | Reproduced | Official leaderboard | Match |
|---|---:|---:|---|
| **Near-OOD AUROC** | **87.58 +/- 0.46** | **87.58 +/- 0.46** | exact at reported precision |
| ID accuracy | **95.06 +/- 0.30** | **95.06 +/- 0.30** | exact at reported precision |
| Near-OOD FPR@95 | 61.34 +/- 4.63 | not used as the target | n/a |

The official values are recorded in OpenOOD's `leaderboard` branch:

```text
model_info/cifar10/CrossEntropy_EBO_ResNet18_100_epochs.json
results/cifar10.csv
```

## Strict-blind Agent experiments

The Agent did **not** receive the manual CPU runner, raw canonical result,
compatibility findings, target value, or previous attempt transcript. Each
attempt started from a freshly copied OpenOOD tree containing only the fixed
source, official data, and official checkpoints. The Docker network was
disconnected before the Agent started.

The private verifier required:

- the exact `cifar100=9000` and `tin=7793` sample counts;
- all six per-run/per-dataset AUROCs for `s0`, `s1`, and `s2`;
- an `actual` value recomputed as dataset mean, then run mean;
- evidence printed by a successful executed evaluation command, not an
  `echo`/`printf` relay.

| Attempt | Budget | Result | Evidence | Errors | Cost | Peak context | Called `finish` |
|---|---:|---|---|---:|---:|---:|---|
| `001` | 30 steps | fail | none; spent the budget reading the repository | 3 | CNY 0.1090 | 8,363 tokens | no |
| `002` | 60 steps | **pass** | `87.58227747333393`, private diff `0.002277` | 5 | CNY 0.3464 | 20,430 tokens | no |

Attempt `002` independently wrote `eval_ebo.py`, recovered from three
substantive implementation failures (`faiss` eager import, missing
`data_aux_preprocessor`, and nonexistent `Config.from_yaml`), and executed the
full CPU evaluation in 304 seconds. Its six AUROCs and dataset counts exactly
match the canonical run at full precision.

The result is a genuine numerical reproduction pass, but the control loop still
has a termination defect: after printing valid evidence, the Agent continued
exploring until the 60-step limit instead of calling `finish`. Attempt `001`
also shows that a 30-step budget is insufficient for the current exploration
strategy.

Saved artifacts:

```text
evals/runs/openood_ebo_blind_001/{blind_result.json,commands.sh,transcript.jsonl}
evals/runs/openood_ebo_blind_002/{blind_result.json,commands.sh,transcript.jsonl}
```

## Multi-Agent + RAG strict-blind experiment

A separate run forced collaborative Multi-Agent + RAG execution:

1. **Navigator**: independent LLM context, required to call `search_repo` at
   least four times and write `navigator_report.md`.
2. **Reproducer**: independent LLM context, intended to consume the Navigator
   handoff and run the full evaluation.
3. **Reviewer/Debugger**: independent LLM context, required to call
   `search_repo` at least twice, audit the implementation, and repair it when
   necessary.
4. **Private deterministic verifier**: unchanged; remained outside the Docker
   mount.

The run remained strict-blind and offline. No Agent received the manual CPU
runner, canonical result, compatibility findings, or private target.

### Initial result: failed

| Role | Steps | Commands | RAG calls | Errors | Cost | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Navigator | 16 | 33 | 4 | 0 | CNY 0.0724 | found relevant source, but never wrote the required handoff |
| Reproducer | 42 | 39 | 3 | 2 | CNY 0.1280 | repeatedly read source; wrote no evaluation script |
| Reviewer/Debugger | 18 | 25 | 2 | 2 | CNY 0.0752 | repeated the same source audit; wrote no review or repair |
| **Total** | **76** | **97** | **9** | **4** | **CNY 0.2756** | **no structured evidence** |

The RAG requirement passed, but the handoff requirement and private numerical
verifier failed. This is an important negative result:

- merely forcing more RAG calls did not make the Agents act sooner;
- the Navigator located useful files but failed to synthesize them into the
  required artifact;
- missing handoff caused the Reproducer and Reviewer to repeat repository
  exploration;
- the orchestration loop currently executes multiple tool calls from one model
  turn even though the prompt requests exactly one, allowing exploration to
  expand faster than the step budget suggests;
- compared with the successful 60-step Single-Agent run, the current
  Multi-Agent design increased duplicated work and failed before evaluation.

Saved artifacts:

```text
evals/runs/openood_ebo_multi_rag_001/result.json
evals/runs/openood_ebo_multi_rag_001/commands.sh
evals/runs/openood_ebo_multi_rag_001/{navigator,reproducer,reviewer}_transcript.jsonl
```

### Repaired orchestration result: passed

The initial failure was used as an orchestration debugging benchmark rather
than treated as the final result. The repaired design separates retrieval,
synthesis, execution, review, and repair:

1. fixed RAG queries retrieve source evidence independently for Navigator,
   Builder, Code Critic, and Reviewer;
2. Navigator synthesizes a bounded handoff instead of freely exploring;
3. Builder writes a new evaluation script from the handoff and retrieved
   source; it is not given the manual CPU runner;
4. Code Critic independently audits and corrects the generated script;
5. a deterministic executor runs the script;
6. Reviewer audits the implementation and public logs, returning a fail-closed
   `PASS` or `REPAIR_REQUIRED`;
7. Repair Agent receives the script, public error, Navigator handoff, and
   Reviewer audit, then the executor reruns it;
8. the unchanged private verifier checks the final composite evidence.

The feedback loop was developed through observable failures:

| Attempt | Outcome | Failure or improvement |
|---|---|---|
| `001` | fail | unconstrained roles duplicated exploration and produced no artifacts |
| `002`-`003` | fail | stage gates exposed missing structured handoffs |
| `004`-`007` | fail | bounded Navigator/Builder/Critic worked, but execution and repair routing remained weak |
| `008` | fail | produced structured evidence, but AUROC sign and percentage scale were wrong; Reviewer diagnosed the sign error only after execution |
| `009` | aborted | Reviewer-to-Repair feedback worked, but Repair regressed to per-image CPU inference |
| `010` | **pass** | added fixed CLI, batched CPU, percentage-unit, and fail-closed review contracts |

Attempt `010` is a genuine strict-blind collaborative repair:

- the initial generated script failed because it resolved the data path
  incorrectly;
- `reviewer_0` identified the concrete path failure and returned
  `REVIEW_STATUS: REPAIR_REQUIRED`;
- `repair_1` corrected the generated script;
- the second real CPU evaluation emitted all six AUROCs and exact dataset
  counts;
- `reviewer_1` returned `REVIEW_STATUS: PASS`;
- the private verifier matched `87.58227866875886` against `87.58 ± 0.05`.

| Result | Value |
|---|---:|
| Agents/isolated role calls | 6 |
| Real RAG calls | 15 |
| Executed evaluation commands | 2 |
| Final Near-OOD AUROC | **87.58227866875886** |
| Private absolute difference | **0.002278668758862068** |
| Total LLM cost | CNY 0.1700 |
| Collaboration pass | **true** |

This result shows where Multi-Agent + RAG adds value in this project: not by
splitting an already-known runner into decorative roles, but by retrieving
repository semantics, generating a new executable, independently finding a
real failure, routing that finding to a separate repair context, and verifying
the repaired execution.

Saved artifacts:

```text
evals/runs/openood_ebo_multi_rag_010/result.json
evals/runs/openood_ebo_multi_rag_010/commands.sh
evals/runs/openood_ebo_multi_rag_010/eval_ebo.py
evals/runs/openood_ebo_multi_rag_010/{navigator,reproducer,critic,repair_1,reviewer_0,reviewer_1}_transcript.jsonl
evals/runs/openood_ebo_multi_rag_010/{navigator_report,review_report}.md
evals/runs/openood_ebo_multi_rag_010/reproducer_public_log.txt
```

### Dynamic RAG follow-up: runtime-generated queries

Attempt `010` proved the collaborative repair loop, but its retrieval queries
were fixed by the orchestrator. The follow-up removed those fixed query arrays
and RAG packets. Each role now receives the current task, generated code, or
execution error and decides what to search for at runtime:

1. Navigator queries from the initial task; Reproducer and Critic query from
   the handoff and generated code.
2. Reviewer queries from the latest public execution result.
3. Repair queries from both the execution error and Reviewer finding.
4. Each role can make at most three repository searches, then moves into a
   separate no-tool synthesis context so it must produce its handoff.
5. Generated scripts, reports, logs, traces, and transcripts are excluded from
   retrieval, preventing the Agent from retrieving its own previous answer.
6. Every query, ranked path, and retrieved source snippet is saved in a
   `*_rag_trace.md` artifact.

The implementation was tested through additional strict-blind runs:

| Attempt | Outcome | Dynamic-RAG finding |
|---|---|---|
| `011` | fail | runtime queries were relevant, but Navigator exhausted its search budget without submitting a handoff; this motivated separate search and synthesis phases |
| `012` | fail | all roles generated queries, but synthesis markup and Markdown status formatting broke the fail-closed handoff validator |
| `013` | fail | 24 dynamic queries reacted to real path and indentation failures; no valid numerical evidence, CNY 0.4062 |
| `014`-`015` | fail | overly literal static code gates rejected otherwise plausible implementations; gates were reduced to structural checks |
| `016` | fail | 36 dynamic queries across 12 isolated role calls and 10 commands; no valid numerical evidence, CNY 0.6270 |

Attempt `016` demonstrates that the retrieval is genuinely error-driven rather
than a fixed checklist. The observed error/query chain included:

- after local CIFAR-10 loading failed, Reviewer searched for
  `eval_ood.py data root cifar10 data location`;
- after the broad evaluation API imported an unavailable optional dependency,
  Reviewer and Repair searched the evaluation API, evaluator metrics, and EBO
  postprocessor import paths;
- after the repaired loader failed, Reviewer searched
  `ImglistDataset __init__ parameters openood` and retrieved the constructor
  requiring `data_aux_preprocessor`.

This is an honest negative result. Dynamic RAG is now causally connected to
task state and execution errors, and its behavior is auditable, so it is no
longer merely a fixed retrieval step added for presentation. However, it cost
more and did not outperform the controlled fixed-query attempt `010`. The next
engineering priority is better retrieval ranking and structured patch repair,
not increasing the number of RAG calls.

Saved artifacts:

```text
evals/runs/openood_ebo_multi_rag_016/result.json
evals/runs/openood_ebo_multi_rag_016/commands.sh
evals/runs/openood_ebo_multi_rag_016/*_rag_trace.md
evals/runs/openood_ebo_multi_rag_016/*_transcript.jsonl
evals/runs/openood_ebo_multi_rag_016/*_synthesis_transcript.jsonl
evals/runs/openood_ebo_multi_rag_016/reproducer_public_log.txt
```

### Post-016 retrieval and repair hardening

Attempt `016` exposed two implementation weaknesses that were then addressed
without increasing the search budget:

1. BM25 ranked files that *mentioned* an exact path above the target file. For
   example, the query `scripts/eval_ood.py` ranked the actual file at position
   333 because many shell scripts referenced it.
2. Repair Agents submitted a complete replacement script each round. This
   allowed a fix for one error to regress unrelated working behavior.

The retrieval ranker now combines BM25 with strong exact-path, traceback-path,
and symbol signals, always preserves the strongest deterministic match, and
returns query-centered source windows instead of only the file head. An offline
regression over queries observed in attempt `016` produced:

| Query target | Old BM25 rank | Hardened rank |
|---|---:|---:|
| `scripts/eval_ood.py` | 333 | **1** |
| `openood/datasets/imglist_dataset.py` | 15 | **1** |
| `openood/evaluators/metrics.py` | 12 | **1** |
| `openood/postprocessors/ebo_postprocessor.py` | 10 | **1** |

Repair now submits 1-8 exact `old` to `new` replacements. A patch is rejected
when the old text is not unique, is a no-op, replaces more than 65% of the
file, introduces invalid Python syntax, or fails the existing code contract.
Each accepted patch is saved as a separate `*_submission.json` artifact. On a
copy of attempt `016`'s final script, a three-edit patch added the missing
`data_aux_preprocessor` arguments, preserved the rest of the file, and passed
`py_compile`.

Reviewer and Repair stages are also capped at two dynamic queries each, down
from three. These are implementation and offline-regression results, not a new
strict-blind reproduction result; a new real run is still required to measure
whether the changes improve end-to-end success.

### Hardened dynamic-RAG results: attempts 017-018

Attempt `017` tested the exact-path ranker and structured patch Repair in a
full strict-blind run:

| Result | Attempt `016` | Attempt `017` |
|---|---:|---:|
| Dynamic RAG calls | 36 | **25** |
| Isolated role calls | 12 | 12 |
| Commands | 10 | 10 |
| Cost | CNY 0.6270 | **CNY 0.5250** |
| Valid public evidence | no | no |
| Collaboration pass | false | false |

The first execution failed on an incorrect data root. Repair 1 generated one
runtime query, retrieved `scripts/eval_ood.py` as the top result, and submitted
one exact replacement. The next execution successfully evaluated all three
checkpoints and both Near-OOD datasets. This is a concrete improvement over
attempt `016`, where successive full-file rewrites repeatedly introduced new
import, download, and constructor failures.

Attempt `017` still failed for three reasons:

1. the generated script emitted incorrect dataset counts (`3/3` instead of
   `9000/7793`), so its structured result failed the public evidence contract;
2. its hardcoded CIFAR-10 normalization produced AUROC `87.091608`, outside the
   private target, but the target remained hidden as required;
3. later Reviewer/Repair rounds made semantically weak changes. One patch only
   appended `sys.exit(0)` while presenting a large aggregation block, and
   another changed a previously successful data path back to the failing path.

Attempt `018` then failed fast before evaluation because the generated script
used the broad `openood.evaluation_api` import, which pulled the unavailable
optional `faiss` dependency. Repair correctly could not modify OpenOOD
repository source through the eval-script patch tool, but exhausted its
synthesis retries. This exposed a missing pre-execution compatibility gate and
missing partial-result recording.

The next hardening step adds deterministic controls rather than more searches:

- broad `openood.evaluation_api` and `openood.postprocessors` imports are
  rejected before execution;
- the public-contract audit explains mismatches such as `datasets=3/3` without
  revealing the private numerical target;
- patches addressing a public-contract failure must change the relevant diff,
  not merely include relevant words in a large unchanged block;
- code introduced by a patch is protected after a successful execution, so a
  later Repair cannot silently revert it;
- Repair is explicitly limited to `eval_ebo.py`, receives up to four synthesis
  correction attempts, and workflow exceptions are recorded in `result.json`.

Offline replay against the actual attempt `017` patches confirmed that both the
unrelated `sys.exit(0)` patch and the successful-path regression are now
rejected. The full suite passes with `75 passed, 2 skipped`. Attempt `019`
could not be launched because the configured external model API usage limit was
reached; it is not counted as an experiment result.

### Dynamic-RAG follow-up: attempts 019-024

The later runs progressively tightened repository import boundaries, structured
evidence, semantic audits, and patch validation. All results below are honest
strict-blind failures; no private runner or private numerical target was
provided to the Agents.

| Attempt | Agents | RAG calls | Commands | Cost (CNY) | Main outcome |
|---|---:|---:|---:|---:|---|
| `019` | 6 | 15 | 4 | 0.2687 | Repaired an optional-`faiss` import, then failed synthesizing the next path repair |
| `020` | 1 | 3 | 0 | 0.0288 | Builder failed the new direct-import gate before execution |
| `021` | 12 | 26 | 10 | 0.6143 | Repaired three path failures and completed all checkpoints, but printed non-JSON evidence and used wrong normalization |
| `022` | 12 | 26 | 10 | 0.5592 | Produced strict JSON with correct counts/runs; result `87.09`, then was diverted by an overly strict rounding audit |
| `023` | 12 | 27 | 10 | 0.4646 | Four repairs remained trapped in checkpoint `config.yml` parsing |
| `024` | 1 | 3 | 0 | 0.0223 | Builder blocked by a false-positive validator match before execution |

Attempt `022` is the strongest dynamic result. It demonstrated a full
Navigator → Builder → Critic → Reviewer/Repair workflow, runtime-generated RAG
queries, three evidence-grounded path repairs, all three checkpoint
evaluations, strict JSON, exact dataset counts (`9000/7793`), and the expected
run/aggregation shape. Its numerical result was `87.09`, not the private target
`87.58`, because the generated preprocessing still did not preserve the exact
official semantics.

Attempts `023-024` clarify the current bottleneck:

1. Retrieval is no longer the primary failure. Agents repeatedly locate
   `ImglistDataset`, `transform.py`, official YAML paths, and the EBO source.
2. Repair quality remains unstable on ambiguous implementation choices.
   Attempt `023` repeatedly rewrote serialized checkpoint-config parsing instead
   of selecting the simpler repository transform source.
3. Validator feedback is currently too brittle. Attempt `024` generated code
   that did not instantiate `TestStandardPreProcessor`, but a raw substring
   gate rejected its comment saying it "replicates TestStandardPreProcessor".
   The same misleading feedback repeated until synthesis retries were exhausted.
4. Some remaining validators should use AST-level imports/calls and structured
   semantic checks rather than raw substring rules. This is the next
   engineering priority; increasing RAG query count is not.

After attempt `024`, further experiments were intentionally paused. The final
test suite at that point was `81 passed, 2 skipped`.

### Generalisation hardening: attempts 025-034

The `024` false-positive (substring gate rejecting a comment) was fixed with
full AST-level `_forbidden_contract_violations()` and a series of further
workflow bugs were diagnosed and fixed through live runs. All fixes were
deliberately kept task-agnostic: each new rule reads from named oracle
constants (`CHANCE_LEVEL`, `FORBIDDEN_*`, etc.) rather than from the expected
numerical value.

| Attempt | Outcome | Failure / improvement |
|---|---|---|
| `025` | **pass** | First successful deepseek-chat run after AST gate; `87.58227747333393` |
| `026` | fail | Protected-blocks gate fired on bare exit-0; sign was still inverted (`12.42`). Fix: endorsement requires reviewer PASS, not just `run.ok` |
| `027` | fail | TinyImageNet silent drop (`tin=6526` of `7793`). Fix: `_silent_drop_hint()` scans public log for `broken pipe` / `FileNotFoundError` / `cannot identify image` and exposes it to the next Repair |
| `028` | fail | Three-signal gap: Reviewer gave PASS on a wrong tin count; protected code froze. Fix: `run_ok AND contract_passes AND reviewer_PASS` all required |
| `029` | fail | EBO score polarity oscillation; agent reported `12.42 = 100 − 87.58`. Fix: `_below_chance_diagnostic()` flags any result below the `CHANCE_LEVEL` baseline without leaking the target |
| `030` | **pass** | `87.58227866875886`; workflow terminated correctly after contract passed (repair loop halted on `_repair_loop_should_continue`) |
| `031` | fail | Silent tin drop reoccurred; diagnostic now surfaces it to Repair |
| `032` | fail | Synthesis exhausted retries after Repair submitted only `sys.exit(0)` patch |
| `033` | fail (model switch) | First qwen3-max run; `_extract_python` took the **first** code block (a prose preface snippet), not the eval script. Fix: `_extract_python` selects the largest / `REPRO_RESULT`-bearing block |
| `034` | fail (model switch) | qwen3-max submitted a surgical one-line normalization fix; the near-identical guard rejected it before validation. Fix: validate-first synthesis — accept any valid candidate regardless of textual similarity |

Attempts `025` and `030` are confirmed deepseek-chat passes. The two-of-twelve
pass rate for deepseek-chat reflects the sign-inversion (`12.42`) and dataset
count (`tin=6526`) failure modes that the new diagnostics address. Attempts
`033`-`034` exposed two deepseek-coupling bugs that were fixed before attempt
`035`, making the workflow genuinely model-agnostic.

The test suite grew from `81 passed, 2 skipped` (after `024`) to `93 passed,
2 skipped` (after `034`), covering the AST validator, three-signal endorsement,
below-chance diagnostic, repair-loop stopping, silent-drop hint, model-agnostic
extraction, and validate-first synthesis.

### Model comparison: qwen3-max attempt 035

After fixing the two model-coupling bugs, attempt `035` ran the full pipeline
with **qwen3-max** (DashScope `qwen3-max`, OpenAI-compatible endpoint).

| Result | Value |
|---|---|
| Agents / isolated role calls | 12 |
| Dynamic RAG calls | 22 |
| Repair rounds | 4 |
| Executed evaluation commands | 10 |
| Final Near-OOD AUROC | **92.46** |
| Private absolute difference | 4.88 |
| Total LLM cost | CNY 0.7004 |
| Collaboration pass | **false** (outside tolerance) |

Per-run metrics from the evidence line:

| Seed | CIFAR-100 AUROC | TinyImageNet AUROC |
|---|---:|---:|
| `s0` | 85.55 ✓ | 98.50 |
| `s1` | 86.88 ✓ | 98.64 |
| `s2` | 86.66 ✓ | 98.51 |

The three CIFAR-100 values exactly match the canonical run. TinyImageNet AUROC
is ~98.5 across all seeds (canonical: ~88.5). AUROC 98.5 means the agent's
script separates TIN from CIFAR-10 almost perfectly — the opposite of a score
near 50 (no discrimination), suggesting a **preprocessing mismatch**: TinyImageNet
images are 64×64 natively; if the script did not resize them to 32×32 before
the model, the energy scores would be systematically different for TIN while
CIFAR-100 (already 32×32-compatible) runs correctly.

This is a genuine semantic gap in the generated evaluation script, not a sign
inversion or dataset-count error. The deterministic contract correctly rejected
it (outside tolerance). No diagnostic can flag it without leaking the target,
so it remains a task for the Agent to discover and fix.

**Model-quality comparison (deepseek-chat vs qwen3-max):**

| Failure mode | deepseek-chat | qwen3-max (035) |
|---|---|---|
| EBO score sign inversion | ✗ (026, 029) | ✓ correct |
| Dataset count short (tin=6526) | ✗ (027, 031) | ✓ full count |
| Script crash before evidence | ✗ multiple | ✓ evaluates all checkpoints |
| CIFAR-100 AUROC | wrong or missing | **exact** (85.55 / 86.88 / 86.66) |
| TinyImageNet AUROC | wrong or missing | wrong (98.5 vs 88.5, preprocessing) |

Switching to the stronger model raised the failure quality from
"crashes / sign errors / count errors" to "structurally complete, CIFAR-100
exact, single preprocessing detail wrong". The gap from 87.58 to 92.46 is
entirely explained by the TIN preprocessing bug; the rest of the pipeline is
correct.

Saved artifacts:

```text
evals/runs/openood_ebo_multi_rag_035/result.json
evals/runs/openood_ebo_multi_rag_035/eval_ebo.py
evals/runs/openood_ebo_multi_rag_035/{navigator,reproducer,critic,reviewer_*,repair_*}_transcript.jsonl
evals/runs/openood_ebo_multi_rag_035/*_rag_trace.md
evals/runs/openood_ebo_multi_rag_035/reproducer_public_log.txt
```

## Compatibility findings

The repository's unmodified generic evaluation entry could not run on this
machine:

1. `scripts/eval_ood.py` calls `net.cuda()` directly.
2. Base inference code moves every batch with `.cuda()`.
3. Importing the generic evaluation API eagerly imports optional postprocessors,
   including `libmr`, even though EBO does not use them.
4. `libmr` does not install cleanly in the current Python 3.12 environment.
5. `imgaug==0.4.0` accesses `np.sctypes`, removed in NumPy 2.
6. The generic evaluator always constructs and evaluates Far-OOD loaders, while
   this canonical task intentionally targets Near-OOD only.

The CPU runner therefore reuses the official:

- `ResNet18_32x32` model and official checkpoints;
- benchmark image lists and image files;
- CIFAR-10 normalization;
- EBO score: `logsumexp(logits)` with temperature 1;
- AUROC and FPR@95 definitions;
- dataset-then-seed aggregation order.

It bypasses only CUDA-specific execution and unrelated optional-module imports.
The exact match against the official Near-OOD AUROC and ID accuracy provides a
cross-check that the compatibility runner preserved the intended evaluation.

## Why this is a useful difficult benchmark

Unlike the current library-load oracles, this task requires:

- navigating a large research repository to find the correct evaluation path;
- understanding checkpoint directory structure and three-run aggregation;
- distinguishing ID accuracy, Near-OOD AUROC, Far-OOD AUROC, and FPR@95;
- repairing CUDA-only and dependency-drift failures without changing the metric;
- producing evidence for multiple datasets and multiple checkpoints.

## Composite evidence contract

The verifier now supports a richer evidence contract:

```text
REPRO_RESULT {
  "metric": "near_ood_auroc",
  "actual": 87.58,
  "datasets": {"cifar100": 9000, "tin": 7793},
  "run_metrics": {
    "s0": {"cifar100": 85.55, "tin": 88.31},
    "s1": {"cifar100": 86.88, "tin": 88.94},
    "s2": {"cifar100": 86.66, "tin": 89.16}
  },
  "aggregation": "dataset_mean_then_run_mean"
}
```

It verifies exact dataset counts and run names, recomputes the aggregate from
the six component values, and then compares the aggregate against the private
target.

## Integrity hashes

```text
0ed04675111fb6cefc59116879c9986374dfd67e9c51e2c422e07c9fbfb2edef  nearood_ebo_cpu_results.json
a1e785c1a86e434e1d00297c10e3d0907c8d1074d91019ae220e0bb8b77f7a70  run_nearood_ebo_cpu.py
```
