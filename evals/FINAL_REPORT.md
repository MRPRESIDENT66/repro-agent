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

## E4 — Stripping the specialization: the `generic` prompt path

E1/E2 use *specialized* per-task prompts that hand the agent substantial task
knowledge (APIs, field names, known gotchas). The fair question is what survives
when that is removed. `prompt_mode=generic` swaps every role prompt for a single
**task-agnostic** set (no task identity, API, or gotcha) — only the public result
contract and the repo itself remain. On two hard tasks the agent reaches the
target under generic prompts (with one caveat, noted in the table):

| Task | backend | target | generic, full pipeline |
|---|---|---|---|
| RobustBench Carmon2019 | subprocess | 52.0 robust acc | verifier matched (recomputed 52.0); run then hit a 402 balance error in a later step |
| OpenOOD EBO | Docker | 87.58 Near-OOD AUROC | **pass** (verifier-recomputed 87.58, no workflow error) |

The RobustBench row is reported honestly: the deterministic verifier recomputed
52.0 from the agent's adversarial predictions (the reproduction target was met),
but the run later raised a `402 Insufficient Balance` API error in a downstream
step — so it is "verifier matched, workflow interrupted", not a clean end-to-end
pass. OpenOOD completed with no workflow error.

Getting there was itself a debugging study, and the failures were *orchestrator*
defects, not agent-capability limits:

- **RobustBench was a pure wall-clock timeout.** Full AutoAttack (apgd-ce +
  apgd-dlr, n=50) on a WRN-28-10 measured **~1484 s on CPU**; the eval timeout was
  900 s, so it was killed mid-attack every time. Raising the budget to 2700 s —
  **the attack configuration is unchanged**, so the `52.0` semantics are intact —
  let it finish and pass.
- **OpenOOD exposed a three-stage cascade.** (1) In specialized mode the Navigator
  *hallucinated* a CIFAR-10 normalization `std` (a value absent from the entire
  repo) and attributed it to a file that contains no such number; the repair loop
  re-fed that bad handoff every round, so a sign + normalization error produced
  AUROC **8.49**. (2) The generic diagnostics path checked only artifact *shape*,
  so a structurally-valid-but-semantically-wrong `predictions.json` drew no
  corrective feedback. (3) After fixing (1) with an anti-hallucination rule ("a
  reported constant must be read from a named file, never filled from memory"), the
  agent stopped inventing values but instead tried to *import* the repo's
  CUDA/optional-dependency API chain (`libmr`), failing five rounds without
  producing any artifact.

The fixes are all **task-agnostic** and stay inside the generic contract (no
hidden target, no task identity):

1. **Anti-hallucination grounding** (Navigator): concrete constants must be quoted
   from a specific file actually read, or flagged as unresolved — never supplied
   from convention.
2. **Inline-fallback strategy** (Reproducer + Repair): when a high-level API can't
   be imported because of an absent/incompatible dependency, stop retrying the
   import — read the constants and computation *logic* from source and reimplement
   that minimal slice inline with stable base libraries. Reuse the repo's **values
   and semantics, not its import surface.**
3. **Sanity diagnostics in the generic path**, of two kinds with an honest
   boundary between them:
   - *Framework-level (truly task-agnostic):* a verifier-recomputed
     higher-is-better metric **below its random-chance floor** signals an inverted
     decision direction. The floor is declared once per task as
     `OracleConfig.chance_level` (50.0 for binary AUROC, 100/num_classes for
     balanced top-1; left unset for robust-accuracy, where sub-chance is a
     legitimate attack outcome, not an error). The value comes from the verifier's
     own recomputation — never the hidden target — so the check generalizes to any
     task that declares a floor.
   - *Was oracle-specific, now removed:* an earlier version also carried a
     "hardcoded normalization disagrees with the repo's own source" check that
     named OpenOOD's source file and dict key. An ablation (below) showed it was
     **not load-bearing**, so it was **removed** — the generic path now carries
     **zero** oracle-specific feedback. (The *specialized* path still uses that
     check; it lives in `_normalization_diagnostics_for_code`, wired only into the
     specialized contract diagnostics.)

With these, on the run that reached the target the agent hit 87.58 **on its own**
(4 executions, ¥0.58, no human edit to its code) — at that point the generic path
still included the normalization feedback. Independently re-verified: the official
CPU reference script's held-out result is 87.5823 (s0/s1/s2 = 86.93/87.91/87.91),
and that script is scrubbed from the agent's workspace, so blindness holds.

### Ablation: is the hand-authored normalization check load-bearing?

To test whether that one oracle-specific diagnostic was actually *needed*, I
removed its wiring from the generic path (the config no longer passes
`generic_safe_diagnostics`) and re-ran OpenOOD generic — leaving only
task-agnostic machinery (anti-hallucination prompt + framework-level below-chance
+ inline-fallback). The result is more informative than a pass/fail:

- The agent **corrected the `std` on its own** to the repo's exact
  `[0.2470, 0.2435, 0.2616]` (the anti-hallucination rule drove it to read
  `transform.py` instead of inventing a value) **and** negated the energy so OOD
  scores rank higher (the framework below-chance signal). So the hand-authored
  normalization feedback is **not load-bearing** — the generic machinery recovers
  both the value and the direction without it.
- It nonetheless **missed at 92.46** (target 87.58): it omitted the official
  `Resize(32)+CenterCrop(32)` step, a third, subtler preprocessing detail that has
  **no diagnostic at all** — not the (removed) normalization check, and no
  framework signal, since 92.46 is above chance. This is the honest frontier of
  the purely-generic path: it self-corrects failures that leave a *signal* (a
  below-chance metric) or a *checkable constant* (std vs the repo), but a missing
  transform that merely shifts an already-plausible score still goes uncaught.

On the strength of this ablation the normalization wiring was **removed for good**,
so the generic path is now fully task-agnostic. The honest trade-off: the one
recorded 87.58 generic pass was obtained *with* that feedback present; without it
the agent recovers std + sign unaided but, on this iterated task, lands at 92.46
short of full convergence (the uncovered `Resize/CenterCrop`). "Given feedback
with the right *form* the agent self-corrects" holds; "the unaided generic system
nails every preprocessing detail" does not yet.

**Caveats:** these are single N=1 generic passes per task, not an N=5 rate, and
the development above was iterated against these same tasks — read as feasibility,
not a generalization rate.

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
  be read with that caveat. A `prompt_mode=generic` path that strips this exists
  and now passes two hard tasks (RobustBench, OpenOOD) as single N=1 runs — see
  E4 — but the broad N=5 generality claim still rests on the specialized prompts.
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
for E2, and scaling the `prompt_mode=generic` path (E4) from single N=1 passes to
a full N=5 rate.
