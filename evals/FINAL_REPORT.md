# Repro-Agent — Final Report

An agent that reproduces published ML results **blind**: it reads a repository,
writes and runs an evaluation, and a deterministic verifier compares the number
against a target the agent never sees. This report is organized around three
research questions and three experiments. (The chronological lab notebook lives
in [`RESULTS.md`](RESULTS.md); the OpenOOD development trace in
[`OPENOOD_EBO.md`](OPENOOD_EBO.md). This document is the clean, self-contained
summary.)

Agent LLM: deepseek-chat. Execution: a persistent local subprocess session, or a
sandboxed `linux/amd64` Docker container when the environment can't be built on
the host. Costs in CNY (¥). All pass rates are N=5, blind.

---

## Research questions

- **RQ1 — Feasibility.** Can a blind, cheat-resistant agent reproduce published
  numbers at all?
- **RQ2 — Generality.** Does one agent design handle many domains / frameworks /
  execution backends without per-task orchestration code?
- **RQ3 — Where does the complexity pay off?** The agent is a pipeline
  (Navigator → Reproducer → Critic → execute → Reviewer → Repair). Which parts
  actually buy success, and on what kind of task?

---

## Evaluation protocol (what makes a "pass" trustworthy)

1. **Blind.** The target value and tolerance live outside the agent's workspace.
   The task states only *what* to reproduce, never *how* or the number.
2. **Structured evidence.** A run passes only if a *successful* command prints a
   strict-JSON line `REPRO_RESULT {metric, actual, num_examples, …}`.
3. **Provenance gate.** That evidence must come from a real evaluation — a script
   that loads a model + data and computes the metric (inline), or one that
   delegates to the repository's own eval entry against the checkpoint. A bare
   `echo`/`printf` of a number is rejected.

This gate is the spine of trust, and tightening it repeatedly caught real
problems (a `0.91` vs `91.0` unit ambiguity; an echo relay; and — during this
study — two false-negatives that were fixed, see Appendix B).

---

## The reproduction-task suite

Five tasks spanning four task-types and two backends, with a difficulty label set
by the blind first-try behaviour:

| Task | Type / domain | Backend | Target | Difficulty |
|---|---|---|---|---|
| DistilBERT SST-2 | NLP sentiment | subprocess | 91.06 acc | easy |
| mmpretrain ResNet-18 | image cls — clone & navigate (mmcv) | Docker | 94.82 top-1 | medium |
| detectors ResNet-18 CIFAR-100 | image cls — timm registration | subprocess | 79.26 top-1 | medium |
| OpenOOD EBO | OOD detection (composite AUROC) | Docker | 87.58 AUROC | hard |
| RobustBench Carmon2019 | adversarial robustness (AutoAttack) | subprocess | 52.0 robust acc | hard |

Every target's offline truth was independently re-verified before any agent run.
Adding a task is a ~200-line config plus a 9-line run script; the orchestrator is
untouched.

---

## E1 — Feasibility & generality (RQ1, RQ2)

Full pipeline, N=5, provenance-gated. Tables are generated from the run
artifacts by `python evals/report_tables.py` (single source of truth). `blind`:
*strict* = the target is absent from the agent's workspace; *soft* = present in
the public repo (never surfaced by the task/verifier) — the agent must still run
the real eval to produce it.

| Task | blind | passed | repair fired | ~cost/run |
|---|---|---|---|---|
| DistilBERT SST-2 | strict | **5/5** | 1/5 | ¥0.029 |
| mmpretrain ResNet-18 | **soft** | **5/5** | 0/5 | ¥0.093 |
| detectors ResNet-18 CIFAR-100 | strict | **4/5** | 2/5 | ¥0.086 |
| OpenOOD EBO | strict | **3/5** | 3/5 | ¥0.262 |
| RobustBench Carmon2019 | strict | **4/5** | 4/5 | ¥0.268 |
| **total** | | **21/25** | | |

**Read:** blind reproduction works across four task-types and both backends —
the same agent, no orchestration changes. Difficulty tracks the repair rate:
easy tasks pass first-try (repair idle); the hard tasks fail first and lean on the
repair loop. *Honest correction:* an earlier draft reported RobustBench at 5/5,
but that run was **soft-blind** — the repo README leaked a `52.00%` worked
example. After scrubbing it (strict-blind) and re-running, RobustBench is **4/5**;
the total is **21/25**, not 22/25.

---

## E2 — Where the complexity pays off (RQ3) — the controlled ablation

The same task, run under five conditions that **share one execution budget**
(≤5 evaluations) so "more attempts" is held constant, N=5 each:

- **solo** — Reproducer only → 1 execution. Single-agent baseline.
- **team** — Navigator + Reproducer + Critic → 1 execution. Pre-execution
  collaboration, no follow-ups.
- **solo-retry** — Reproducer; on failure RE-GENERATE with **no execution
  feedback**, ≤5 executions. The budget-matched control for "just more tries".
- **solo-repair** — Reproducer; on failure a Repair role fixes it **with the
  real error**, ≤5 executions. Single agent + feedback repair, no Navigator/
  Critic/Reviewer.
- **full** — Navigator + Reproducer + Critic + Reviewer + Repair loop, ≤5 executions.

| condition | easy — DistilBERT | hard — OpenOOD | execs (hard) |
|---|---|---|---|
| solo | 5/5 | 0/5 | 1 |
| team | 5/5 | 0/5 | 1 |
| solo-retry | 5/5 | **0/5** | 5 |
| solo-repair | 5/5 | **0/5** | 5 |
| **full** | 5/5 | **3/5** | ≤5 |

**The decisive result — and a correction of an earlier, premature claim.** On the
easy task every condition passes 5/5: the machinery is unnecessary, and the extra
roles only add cost. On the hard task, **no reduced condition produces any
success**: not more attempts (solo-retry **0/5** at the full 5-execution budget),
not single-agent feedback repair (solo-repair **0/5**, also 5 executions), not
pre-execution collaboration (team **0/5**). **Only the full pipeline reaches
3/5.**

So the components are **complementary on hard tasks, not redundant**: success
requires the pre-execution grounding (Navigator + Critic) **and** the
reviewer-guided repair loop *together*; removing any major piece drops it to 0.
An earlier draft of this report — comparing only solo / team / full, where solo
and team ran *once* and full ran up to *five* times — concluded that "the value is
all in the repair loop, the Critic is overhead." The budget-matched controls
(solo-retry, solo-repair) **overturn that**: feedback-repair *alone* (solo-repair)
is 0/5, so the repair loop is necessary but **not sufficient** — it only works
on top of the full multi-agent pipeline.

What this does **not** isolate: which of Navigator / Critic / Reviewer is the
critical addition over solo-repair (the difference is the whole pre-execution team
+ the reviewer's anomaly audit at once), and the result rests on a single hard
task (OpenOOD). See Limitations.

This refines the earlier multi-agent ablation (Appendix A, "M5"), which found
that on *easy* independent tasks multi-agent buys parallelism + isolation but not
success — consistent with the easy row here.

---

## E3 — Supporting component ablations

Compact, reused from the development study (illustrative, small-N).

- **Retrieval ladder** (navigating `mmpretrain`, 1858 files, recall@5):
  keyword ≈ BM25 ≈ dense ≈ hybrid ≈ **60%**; **+ LLM reranker → 80%**. The
  retrieval *algorithm* barely matters for code; the LLM reranker (+20pp), which
  disambiguates the true entry script from look-alikes, is the win.
- **Transport** (function-calling vs text protocol, N=5): both reproduce 5/5
  (null on success), but FC removes a failure mode — the text protocol drops into
  an unparseable reply ~once/run, costing **2.2× steps and 2.9× cost**
  (¥0.0143 vs ¥0.0050). FC is the right default.
- **Context compression**: a long debug trajectory shrinks **27.3k → 11.1k
  chars (−59%)** with recent turns intact; a no-op on short reproductions.

---

## Key findings

1. **Blind reproduction is feasible and provenance-gated** — 21/25 across four
   task-types and two backends, behind a gate (bound to the evidence-emitting
   command, AST-checked) that rejects echoed/decoy evidence. Not proven secure
   against a determined adversary, but the listed forgeries fail closed.
2. **One design runs many tasks with no orchestration changes** — adding a task is
   config. (Generalization is scoped to *development* tasks; see Limitations.)
3. **On hard tasks the multi-agent components are complementary, not redundant**
   (E2): at a shared execution budget, no reduced condition produces any success
   (more attempts 0/5, single-agent feedback-repair 0/5, pre-execution
   collaboration 0/5); only the full pipeline reaches 3/5. On easy tasks the whole
   apparatus is unnecessary.
4. **Difficulty, not luck, is the variable** — easy tasks are first-try; the hard
   tasks fail first and need the full pipeline.
5. **For code navigation, the LLM reranker matters; the retrieval algorithm does
   not.**

This **corrects** an earlier draft's claim that "the Critic is overhead, keep only
the Reproducer + repair loop." The budget-matched controls show feedback-repair
alone (solo-repair) is 0/5 on the hard task — the pipeline cannot be reduced to it.

---

## Limitations (stated plainly)

- **Small N** (5). Pass rates are indicative, not significance-tested. All five
  tasks are **development** tasks (prompts were iterated against them) — there is
  **no held-out split**, so the generality claim is scoped to this suite.
- **The hard-task E2 result rests on ONE hard task (OpenOOD).** The
  complementarity finding (only `full` succeeds) is a single data point at N=5;
  RobustBench would be a useful second hard task for E2 but was not run under the
  five conditions.
- **E2 does not pin down *which* component is critical.** It shows the full
  pipeline beats every reduced condition, but the gap between solo-repair (0/5)
  and full (3/5) bundles the Navigator, the Critic, and the Reviewer together; a
  finer ablation (e.g. Reproducer + Reviewer + repair, no Navigator/Critic) is not
  run.
- **Oracle specialization is real.** The per-task prompts hand the agent
  substantial task knowledge (APIs, field names, known gotchas); "generalizes" must
  be read with that caveat. (A `prompt_mode=generic` path that strips this is under
  development.)
- **The two `detectors` tasks are not new papers** — same library as the existing
  `resnet18_cifar100` task; included to exercise repair, not for paper breadth.
- **mmpretrain blindness is soft** — its 94.82 is in the public repo's own
  model-zoo metafile; the task/verifier never reveal it and the agent must still
  run the real `tools/test.py`, but it is not strict-blind.
- **RobustBench truth is `52.0` on n=50 examples** (a CPU-feasible slice), not the
  full-test-set headline number.
- **The provenance gate is a heuristic, not a security boundary.** It fail-closes
  the known forgeries (decoy files, `python -c` prints, comment markers, fake
  wrappers — see `tests/test_verify.py`) but is not proven robust to an adaptive
  attacker; the subprocess backend is not a sandbox.

---

## Appendix A — earlier ablations (development study)

- **Reproduction reliability (N=5):** `cifar10_resnet20` 5/5 (easy),
  `distilbert_sst2` 5/5 (medium), `resnet18_cifar100` 4/5 (hard; the timm
  registration gotcha, moved 0→80% by a transferable "read the model card /
  install+import the helper" strategy).
- **Multi-agent isolation/concurrency (M5):** parallel **34.9s vs serial 112.3s
  ≈ 3.2×** (near-linear); per-agent context **~18–20k vs 25.9k**; success single
  **3/3** vs multi **2/3** — multi-agent buys parallelism + isolation, not success.
- **OpenOOD development:** ~40 attempts of prompt/repair hardening are recorded in
  `OPENOOD_EBO.md`; only the final clean N=5 runs feed E1/E2 above.

## Appendix B — framework bugs surfaced by the clean re-run

Building the ablation and running the full N=5 matrix surfaced four latent,
general bugs (all masked because the post-refactor repair/teardown paths were
only unit-tested):

1. `Session` had no `close()` → teardown crashed subprocess-backed tasks.
2. Repair instructions embed literal `EVIDENCE` JSON; injecting the round number
   with `str.format` mis-parsed the braces (`KeyError '"metric"'`) and aborted the
   repair loop for every task. → use `.replace`.
3. The provenance gate recognized a delegated repo eval only on the command line,
   not inside a wrapper `.py` (mmpretrain). → scan the agent's eval scripts too.
4. The provenance gate missed library-API evals (RobustBench: `load_model` +
   `AutoAttack` + `clean_accuracy`, no `argmax`/`logits`), false-rejecting a
   correct 52.0. → added the library eval-call markers. This alone moved
   RobustBench from a misleading 1/5 to 5/5.

Also corrected: a DistilBERT provisioning trap (a weightless `model/` dir that
baited the Critic into a local-load crash) that had confounded the E2 easy-task
team condition; replaced with a prose `model_card.md`.

## Appendix C — hardening after an external review

An external critique of an earlier draft was largely correct and prompted these
fixes (all with regression/attack tests):

- **Provenance was forgeable.** The gate scanned *any* workspace `.py` for marker
  substrings, so a decoy file + a hardcoded print could pass. It now binds to the
  source the **evidence-emitting command actually ran**, requires AST-proven
  load+predict calls (or a real subprocess delegation to the repo entry), and
  fail-closes decoys / `python -c` prints / comment markers (`tests/test_verify.py`).
- **The ablation budget was unfair** (solo/team ran once, full up to five). Added
  the budget-matched `solo-retry` and `solo-repair` conditions; this **overturned**
  the earlier "drop the Critic" conclusion (see E2).
- **RobustBench was only soft-blind** (README `52.00%` leak). Added a provisioning
  scrub + a uniform blind-workspace check; re-running strict-blind gives 4/5, and
  the suite total is restated as 21/25.
- **Reproducibility:** added `evals/FREEZE.md` (pinned repo SHAs, checkpoint
  SHA-256s, per-task blind level) and `evals/report_tables.py` (tables generated
  from `result.json`, so prose can't drift from artifacts).

All **107** unit tests pass (orchestration + provenance-attack suites added).
Still open: a true held-out split, a finer component ablation, a second hard task
for E2, and the in-progress `prompt_mode=generic` path to reduce specialization.
