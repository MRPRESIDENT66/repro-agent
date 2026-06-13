# Benchmark results

Agent: deepseek-chat (OpenAI-compatible) · execution: persistent subprocess
session (MPS) · every match is **blind + provenance-gated** (see protocol below):
the agent never sees the target, and a match requires structured evidence from a
real eval that loads data and predicts.

> New difficult oracle: the strict-blind Agent independently reproduced OpenOOD
> CIFAR-10 EBO **Near-OOD AUROC 87.582277**, matching the private target `87.58`
> across two datasets and three checkpoints. A 30-step attempt failed before
> evaluation; a 60-step attempt passed numerically but did not call `finish`.
> See [`OPENOOD_EBO.md`](OPENOOD_EBO.md) for the isolation protocol, full record,
> compatibility findings, hashes, and saved transcripts.

> **OpenOOD Multi-Agent + RAG repair result:** the initial unconstrained
> three-role run failed with 97 duplicated commands and no evidence. After
> replacing free exploration with bounded RAG handoffs and a fail-closed
> Reviewer-to-Repair feedback loop, strict-blind attempt `010` passed:
> the first generated script failed on data-path resolution, Reviewer diagnosed
> it, Repair Agent corrected it, and the second real CPU evaluation reproduced
> **87.582279** with `collaboration_pass=true`. The passing run used 15 real RAG
> calls, 6 isolated role calls, 2 executed evaluations, and CNY 0.1700. Both the
> initial negative result and repaired result are recorded in
> [`OPENOOD_EBO.md`](OPENOOD_EBO.md).

> **Dynamic-RAG follow-up:** fixed retrieval queries were then removed. In
> strict-blind attempt `016`, 12 isolated role calls generated 36 repository
> queries at runtime in response to the task, generated code, and successive
> execution errors. The query chain adapted from missing local CIFAR data, to
> an unavailable `faiss` dependency, to the required `ImglistDataset`
> constructor arguments. The run still failed to produce valid numerical
> evidence and cost CNY 0.6270. This establishes that retrieval is genuinely
> adaptive and auditable, but does **not** establish a success-rate benefit over
> the fixed-query attempt `010`.

> **Retrieval/Repair hardening follow-up:** attempt `017` reduced dynamic RAG
> calls from 36 to 25 and cost from CNY 0.6270 to CNY 0.5250. Its first
> structured patch fixed the data path and enabled a complete three-checkpoint
> evaluation, but invalid dataset counts and later semantic regressions still
> prevented valid evidence. Attempt `018` failed fast on a broad import that
> pulled optional `faiss`. Deterministic public-contract diagnostics,
> diff-scoped patch validation, confirmed-code regression protection, and
> broad-import gates were added afterward; a new real run remains pending.

> **Blind verification V1:** expected/tolerance stay private; the agent never
> sees the target value. Only successful-command stdout containing a structured
> `REPRO_RESULT` line (metric, actual, num_examples) — from a real eval (not an
> `echo`/`printf` relay), with eval-script/command provenance — can match.
> (Tightening it surfaced + fixed a unit ambiguity `0.91055` vs `91.055` and an
> echo relay.) The numbers below are measured under this blind protocol; they are
> development tasks, **not held-out**.

## Reproduction reliability — blind, N=5 each (deepseek-chat)

The agent is told only *what* to reproduce, never *how*, and **never the target
value**.

| Oracle | Domain | Difficulty | matched | avg steps | avg errors |
|---|---|---|---|---|---|
| `cifar10_resnet20` (92.60) | vision | easy — torch.hub | **5/5 = 100%** | 4.8 | 0.0 |
| `distilbert_sst2` (91.06) | NLP | medium — transformers | **5/5 = 100%** | 5.8 | 0.2 |
| `resnet18_cifar100` (79.26) | vision | hard — `import detectors` + timm | **4/5 = 80%** | 9.6 | 1.6 |

All matched runs reproduced the published value **exactly**. The blind rates
**match the earlier non-blind baseline** — the agent reproduces just as reliably
without seeing the target, so it never relied on knowing the answer.

## Transport ablation — native function calling vs text protocol

Same loop, same evidence rules, two transports: native tool calls (`bash` /
`search_repo` / `finish` schemas) vs the original regex-parsed ```bash text
protocol (`--no-fc`). `cifar10_resnet20`, blind, N=5 each. `fmt_errors` =
turns the model violated the action protocol (text: unparseable reply; FC:
no/empty/hallucinated tool call).

| transport | matched | avg steps | avg errors | **avg fmt_errors** | avg cost | wall |
|---|---|---|---|---|---|---|
| **function calling** | 5/5 | **4.0** | 0.0 | **0.00** | **¥0.0050** | 158s |
| text protocol | 5/5 | 8.8 | 0.4 | **1.00** | ¥0.0143 | 224s |

**Finding:** function calling does **not** change *whether* the agent
reproduces the number (both 5/5) — the honest null result. What it changes is
*how cleanly*: the text protocol drops into an unparseable reply **once every
single run** (1.00 fmt_errors), and pays for it in **2.2× the steps and ~2.9×
the cost** (¥0.0143 vs ¥0.0050). FC removes a whole failure mode (format
parsing) and is materially cheaper — the right default; the text path is kept
as the measured ablation twin, not deleted. (n=5, illustrative.)

## Difficulty is the variable, not luck

- Easy/medium artifacts (standard loaders) are reproduced **reliably in 2–4
  steps with no errors**.
- The hard artifact (a non-obvious registration-via-helper-library load) costs
  **~10 steps, ~2 errors, and fails 1/5** — the agent must read the model card,
  discover it needs `import detectors`, and avoid rabbit-holing into manual
  architecture reconstruction. A targeted, *transferable* strategy fix (read the
  README prose; "not registered" → install+import the helper) moved this oracle
  from **0 → 80%**.

## Clone-and-navigate reproduction — mmpretrain ResNet-18 CIFAR-10 (the main act)

The first oracle that is **not** a one-line library load. The agent is dropped
into the **cloned 1858-file mmpretrain repo** (env pre-provisioned in the
`repro-mmpretrain` Docker image — the irreducible mmcv hell, see above) and must
**navigate** to the eval entry + config, run it, and report the number — blind,
behind a **two-phase network** (provision online → `go_offline()` → eval offline).

| | result |
|---|---|
| reproduced | **2/3 blind runs → 94.82 / 94.82 exact** over the full 10000-example test set |
| winning trajectory | 10–11 steps, 0 errors; `search_repo` + `cat`s to find `tools/test.py` + the cifar10 config, then ran the repo's **own** eval harness via a wrapper |
| failure mode | 1/3 rabbit-holed **offline**: tried `datasets.load_dataset` / `pip install` (no network) instead of the on-disk data → 4 errors, no evidence |
| cost | ¥0.025–0.034 / run (peak ctx ~8.5k tok) |
| isolation | amd64 Docker sandbox, network cut before the eval |

**The provenance gate, doing its job (two honest findings):**

1. **Delegation is the right behaviour — and the V1 gate under-credited it.** The
   agent did **not** reinvent the eval; it wrote a wrapper that
   **subprocess-invokes the repo's own `tools/test.py <config> <checkpoint>`**
   (data-load + argmax live in mmpretrain's library code) and parsed
   `accuracy/top1`. The V1 gate, calibrated for *inline-eval* oracles (agent
   writes a self-contained `eval.py` with `argmax`), **false-negatived** this —
   the emitting command has no `argmax`/`dataset` literal. Fix: provenance now
   also accepts **delegation** (evidence from a command that ran the repo's eval
   entry against the checkpoint). Re-verified the **same saved transcript**
   offline → match; no eval re-run.
2. **The gate is not foolable by an echo, even when the agent knows the value.**
   In one winning run the agent — having already run `test.py` and seen 94.82 —
   first tried `echo 'REPRO_RESULT …94.82…'`. The gate **rejected the echo**
   (`_is_direct_result_echo`), forcing the agent to emit the number from a
   **real eval** (`run_eval.py` → `tools/test.py`) instead. The blind protocol
   converts "knowing the answer" into "having to actually run it."

## Failure taxonomy (observed)

`nonobvious_loading` (registration helper not discovered) · `dataset_id` (HF
ids need `namespace/config`) · `dataset_link_dead` (canonical CIFAR URL 403s) ·
`preprocessing` (wrong normalization → close-but-wrong number) · `label_field`
(CIFAR-100 fine vs coarse).

## Artifact-blocked (excluded from the denominator)

`RepDistiller` (CRD, a stronger *navigation* candidate) — checkpoint host
`shape2prog.csail.mit.edu` is **dead** → `outcome=artifact_blocked`, reported
separately, not counted as an agent failure.

## Repo navigation — retrieval ladder (M3)

Can retrieval locate the eval entry + config in a **large** repo
(`mmpretrain`, 1858 files)? 5 hint-light queries, gold = the file(s) you must
land on; metric = recall@k. (mmpretrain's full reproduction is `env_blocked` and
**navigation is measured on the cloned source regardless** — see below.)

> **mmpretrain env-block, fully chased — then SOLVED (2026-06-12):** its core
> dep **mmcv** won't build on **Python 3.12** (`pkgutil.ImpImporter`, removed in
> 3.12) and has **no arm64 prebuilt wheel** → native source build fails (numpy-2
> ABI). The fix that the `DockerSession` backend makes possible: run a
> **`linux/amd64`** image (qemu-emulated on the arm64 Mac) and install mmcv's
> **prebuilt x86_64 wheel** — `torch==2.1.0-cpu` + `mmcv==2.1.0` via `mim`, the
> opencv system libs, and `numpy==1.26` pinned **last** (mmpretrain's deps
> re-upgrade it to 2.x and break the torch ABI). Baked into image
> `repro-mmpretrain:latest`. The env-block was never fundamental — it was a
> wheel/platform/ABI puzzle, and the right *isolation backend* is what let us
> pin the exact combo. **This converts the oracle from `env_blocked` to a real
> clone-and-navigate reproduction** (next section).

| Rung | recall@5 | recall@10 |
|---|---|---|
| keyword (fair grep: path-weighted presence) | 60% | 60% |
| BM25 | 60% | 60% |
| dense (text-embedding-v4) | 50% | 60% |
| hybrid (BM25 + dense, RRF) | 60% | 60% |
| **+ LLM rerank** | **80%** | **80%** |

**Finding:** for code navigation the *retrieval algorithm barely matters* —
keyword ≈ BM25 ≈ dense ≈ hybrid (~60%), because query terms (model/dataset
names) are literal in paths, and they all miss `tools/test.py` (the entry,
drowned by `slurm_test.sh` / `mim_dist_test.sh` / test docs). **Dense embeddings
do not beat BM25 here.** The win is the **LLM reranker (+20pp)**, which
disambiguates the true entry script from look-alikes — retrieval recalls
candidates, the LLM judges which is the entry. (n=5, illustrative not
significant.)

## Multi-agent — isolation AND concurrency ablation (M5)

Reproduce 3 CIFAR-10 ResNets (resnet20/32/56) from one repo. Multi-agent: Lead
splits into 3, each Reproducer runs in an **isolated** context (own session, own
LLM client), a deterministic Verifier checks each. Single: one agent does all 3
in one shared context. Multi is run both **parallel** (thread pool) and
**serial** to isolate the concurrency effect from the isolation effect.

| mode | agents | matched | max ctx (chars) | steps | cost | **wall** |
|---|---|---|---|---|---|---|
| multi-parallel | 3 | 2/3 | **17.8k** | 20 | ¥0.040 | **34.9s** |
| multi-serial | 3 | 2/3 | 20.1k | 28 | ¥0.057 | 112.3s |
| single | 1 | 3/3 | 25.9k | 18 | ¥0.042 | 108.6s |

**Three honest findings, each measured:**

1. **Concurrency is a real ~3.2× win — and it overturns my own prediction.** The
   design *predicted* single-box MPS/RAM contention would eat the parallel
   advantage. It didn't: parallel **34.9s vs serial 112.3s ≈ 3.2× ≈ N**, near
   linear. For light CIFAR evals the bottleneck is LLM I/O + load, not GPU
   contention — so the thread pool scales. (Heavier evals could still contend;
   reported for *this* workload.)
2. **Isolation holds:** each Reproducer's peak context stays **~18–20k vs 25.9k**
   for the single agent juggling all three (~25–30% smaller) — the designed
   property, as a number.
3. **Multi-agent does NOT improve success.** Single got **3/3**; multi got
   **2/3** (one isolated Reproducer flaked — the same per-agent stochasticity the
   solo oracles show). On independent *easy* sub-tasks, multi-agent buys
   parallelism + isolation, **not** reliability. Reported as-is — the design
   said value is isolation/concurrency, not success, and that's what the data
   shows.

## Context compression (M4)

The debug trajectory grows over many repair cycles. Compression keeps
system+task+the last few turns full and shrinks older observations/tracebacks.

- On a representative 15-step env-hell trajectory: **27.3k → 11.1k chars (−59%)**,
  recent context untouched (unit-tested).
- Live ablation on `resnet18_cifar100` was **inconclusive**: both compress on/off
  reproduced (success unaffected), but the agent happened to solve it in 4–5
  steps — a *short* trajectory, so compression never engaged. Reported as-is:
  compression bounds context on **long** debug sequences (env hell); on quick
  reproductions it's a no-op.

## Repo-search tool wired into the loop

The agent can emit a ` ```search ` block (a natural-language query) once it has
cloned a large repo; it returns the most relevant files via **BM25 + LLM rerank
(no embeddings — the ladder showed dense doesn't help)**. Verified standalone on
mmpretrain ("evaluate resnet18 on cifar10" → the right config ranks #1). Only
exercised when an oracle requires cloning a large repo (currently env-blocked).

## Honest caveats

- Small n (5–8 runs/oracle); single model (deepseek-chat). Numbers are
  effect-size + pass-rate, not significance claims.
- These oracles are **library-load** artifacts; a true *clone-and-navigate*
  large-repo oracle (where RAG would earn its place) is not yet in the set —
  the best candidate (RepDistiller) was artifact-blocked.
- Execution is a subprocess session (MPS), not the Docker/VM isolation the
  design specifies for untrusted repos (security debt, deferred).

## Dynamic OpenOOD Multi-Agent + RAG status

Attempts `019-024` are recorded in detail in `evals/OPENOOD_EBO.md`. The best
dynamic run, `022`, completed all three checkpoints after evidence-grounded
repairs and emitted strict structured evidence with correct counts and run
shape, but produced `87.09` rather than `87.58`. Attempt `024` then failed
before execution because a raw substring validator rejected a forbidden class
name appearing only in a comment.

Current conclusion: dynamic RAG and role isolation are meaningful and
observable, but end-to-end reliability is now limited mainly by repair
selection and brittle validator feedback, not by insufficient retrieval calls.
Experiments were paused after `024`.
