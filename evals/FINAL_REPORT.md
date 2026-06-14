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

Full pipeline, N=5 blind, provenance-gated.

| Task | difficulty | passed | repair fired | ~cost/run |
|---|---|---|---|---|
| DistilBERT SST-2 | easy | **5/5** | 1/5 | ¥0.029 |
| mmpretrain ResNet-18 | medium | **5/5** | 0/5 | ¥0.093 |
| detectors ResNet-18 CIFAR-100 | medium | **4/5** | 2/5 | ¥0.086 |
| OpenOOD EBO | hard | **3/5** | 3/5 | ¥0.262 |
| RobustBench Carmon2019 | hard | **5/5** | 5/5 | ¥0.242 |
| **total** | | **22/25 (88%)** | | |

**Read:** blind reproduction works across all four task-types and both backends —
the same agent, no orchestration changes. Difficulty tracks the repair rate:
easy tasks pass first-try (repair idle); the hard tasks fail first and lean on the
repair loop (OpenOOD 3/3 of its passes used repair; RobustBench fired repair on
all 5 and recovered all 5 — its first attempt reliably gets the unit/output wrong,
then repair fixes it).

---

## E2 — Where the complexity pays off (RQ3) — the controlled ablation

The same task, run under three pipeline depths, N=5 each:

- **solo** — Reproducer only (its own RAG) → execute. Single-agent baseline.
- **team** — Navigator + Reproducer + Critic → execute. Pre-execution
  collaboration (handoff + audit), no repair.
- **full** — adds the Reviewer + Repair loop.

| | solo | team | full |
|---|---|---|---|
| **easy — DistilBERT** | 5/5 | 5/5 | 5/5 |
| ↳ cost/run | ¥0.005 | ¥0.015 | ¥0.029 |
| **hard — OpenOOD** | 0/5 | 0/5 | 3/5 |
| ↳ cost/run | ¥0.036 | ¥0.122 | ¥0.262 |

**The decisive result, in one paragraph:** on the easy task the extra machinery
buys **nothing** — solo already passes 5/5, and team and full add 0 success at
**6× the cost**. On the hard task, pre-execution collaboration also buys
**nothing** — team is still **0/5**, same as solo. Success appears **only** when
the post-execution **repair loop** is switched on: **0/5 → 3/5**. So the value of
the multi-agent design is concentrated entirely in the **repair loop**, and only
on tasks that fail first; the Navigator handoff + Critic audit do not move the
success rate on either end of the difficulty range.

This sharpens the earlier multi-agent ablation (Appendix A, "M5"), which already
found that on easy independent tasks multi-agent buys parallelism and context
isolation but **not** success (single 3/3 vs multi 2/3).

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

1. **Blind reproduction is feasible and trustworthy** — 22/25 across four
   task-types and two backends, behind a provenance gate that rejects fabricated
   evidence.
2. **One design generalizes** — adding a task is config, not orchestration code.
3. **The repair loop is where the multi-agent complexity earns its keep, and only
   on hard tasks** (E2): easy tasks need none of it; on the hard task, success
   appears only with repair (0→3/5), while pre-execution collaboration adds none.
4. **Difficulty, not luck, is the variable** — repair rate rises monotonically
   with difficulty; easy tasks are first-try, hard tasks are repair-driven.
5. **For code navigation, the LLM reranker matters; the retrieval algorithm does
   not.**

A practical implication: the pipeline could be **simplified** — Reproducer +
repair loop carry the weight; the pre-execution Critic is cost without measured
success benefit on this suite.

---

## Limitations (stated plainly)

- **Small N** (5). Pass rates are indicative, not significance-tested. Tasks are
  development tasks, **not held-out**.
- **E2 separates *pre-execution collaboration* from *repair*, but not *Critic*
  from *Navigator***, and does not isolate "Reproducer + repair (no Critic)". The
  clean claim is: repair is necessary for hard-task success; the Critic's
  marginal value *given* repair is untested.
- **The two `detectors` tasks are not new papers** — same library as the existing
  `resnet18_cifar100` task; included to exercise repair, not for paper breadth.
  (Only `resnet18_cifar100` is in the suite above; `vgg16_bn_cifar10` is an extra.)
- **mmpretrain blindness is soft** — its 94.82 is in the public repo's own
  model-zoo metafile; "blind" means the task/verifier never reveal it, and the
  agent must still run the real `tools/test.py`.
- **RobustBench truth is `52.0` on n=50 examples** (a small slice chosen for CPU
  feasibility), not the full-test-set headline number.

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

All 98 unit tests pass after the fixes.
