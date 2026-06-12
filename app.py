"""Gradio demo: pick an ML artifact, watch the agent reproduce its published
metric — and see the deterministic verdict + the replayable evidence it leaves.

    python app.py     # (run with an env that has gradio + openai)
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from run_repro import reproduce

ROOT = Path(__file__).resolve().parent
MANIFESTS = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "evals/benchmark").glob("*.yaml"))


def _run(manifest: str):
    r = reproduce(manifest)
    v = r["verdict"]
    badge = "✅ REPRODUCED" if v["match"] else "❌ not matched"

    stages = "\n".join(f"- [{'x' if ok else ' '}] {k}" for k, ok in r["stages"].items())
    cmds = "\n".join(f"{i}. `{c}`" for i, c in enumerate(r["commands"], 1))

    summary = (
        f"### {badge} — {r['name']}\n\n"
        f"**Task (hint-light):** {r['task']}\n\n"
        f"**Verdict (deterministic):** reproduced **{v['actual']}** vs published "
        f"**{v['expected']}** (±{v['tolerance']}) — abs_diff `{v['abs_diff']}`\n\n"
        f"**Stages** ({r['steps']} steps, {r['errors']} errors):\n{stages}\n\n"
        f"**Replayable commands the agent ran:**\n{cmds}"
    )
    script = r["eval_script"] or "(no script)"
    return summary, script


with gr.Blocks(title="Repro Agent") as demo:
    gr.Markdown(
        "# Repro Agent\n"
        "Pick an ML artifact. The agent is told only **what** to reproduce (model + "
        "dataset + metric), never **how** or the private published value. It sets up "
        "the env, writes and runs the eval, self-repairs, and command evidence is "
        "verified by code (not the LLM). Runs live — takes ~1–5 min."
    )
    with gr.Row():
        manifest = gr.Dropdown(MANIFESTS, value=MANIFESTS[0] if MANIFESTS else None, label="Artifact")
        go = gr.Button("Reproduce", variant="primary")
    summary = gr.Markdown()
    script = gr.Code(label="The eval script the agent wrote (re-runnable)", language="python")
    go.click(_run, inputs=manifest, outputs=[summary, script])


if __name__ == "__main__":
    demo.launch()
