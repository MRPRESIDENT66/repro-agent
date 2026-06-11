# Repro Agent

> An agent that takes an unfamiliar ML research repo + a target metric + the
> paper's claimed value, and autonomously **navigates the repo (RAG), sets up an
> isolated environment, runs the public lightweight evaluation, verifies the
> result deterministically, and emits replayable, auditable evidence.**

It does **not** reproduce whole papers — only public, lightweight eval results
(mostly "load the released checkpoint and run eval"). The point is to **honestly
quantify how far an agent gets** on a real, messy artifact, stage by stage.

**Stars:** RAG (large-repo navigation) + context management (long debug
trajectories + isolation).
**Supporting:** multi-agent (added later; value = context/role isolation,
measured honestly).
**Foundation (lean, validated first):** persistent, isolated execution sessions
with shell/file actions.

Full design: [`docs/DESIGN.md`](docs/DESIGN.md).

## Status

Early M1 — de-risking the persistent execution closed loop on one offline,
pre-cached repo. Nothing claimed here until it's measured.
