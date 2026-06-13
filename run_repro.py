"""Run the agent on a benchmark manifest. Hint-light: the task is built from the
manifest's model/dataset/claim only — never the loading mechanism or gotchas.

    python run_repro.py evals/benchmark/resnet18_cifar100.yaml          # native tool calls
    python run_repro.py evals/benchmark/resnet18_cifar100.yaml --no-fc  # text-protocol ablation
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

from agent.llm import ChatLLM
from agent.loop import run_agent
from exec.session import Session
from verify.check import verify_run

ROOT = Path(__file__).resolve().parent
REPRO_PY = ROOT / ".venv-oracle" / "bin" / "python"


def build_task(m: dict) -> str:
    """A fair, hint-light task: what to reproduce, not how."""
    metric_id = m["target"]["metric"]
    metric = metric_id.replace("top1_accuracy", "top-1 accuracy").replace("_", " ")
    if m.get("local_repo"):
        # Clone-and-navigate oracle: the repo is already cloned into the workdir;
        # the agent must NAVIGATE it to find the eval entry + config (the RAG
        # thesis), then run it. Hint-light: we name the model + dataset + the
        # on-disk locations, never the entry script, the CLI, or the value.
        repo_dir = Path(m["local_repo"]).name
        return (
            f"A clone of the '{m['repo']}' repository is in the './{repo_dir}' "
            f"directory, and its Python env (mmcv/mmengine/mmpretrain etc.) is "
            f"already installed. A pretrained checkpoint is at './ckpt.pth' and "
            f"the {m['dataset']['name']} is already downloaded under "
            f"'./{repo_dir}/data'. Navigate the repo to find the evaluation entry "
            f"script and the config for model '{m['model']}', run it against the "
            f"checkpoint, and reproduce its published {metric} (in percent). Use "
            f"the machine-readable metric id '{metric_id}' in REPRO_RESULT, and "
            f"use the search_repo tool to locate the entry script in this large "
            f"repo.\n"
            f"IMPORTANT — you are fully OFFLINE: there is NO network. Do NOT "
            f"`pip install` and do NOT fetch datasets (no `datasets.load_dataset`, "
            f"no downloads) — everything is already on disk. Use the repo's OWN "
            f"evaluation entry script (it loads the on-disk data itself); parse "
            f"its printed metric and emit REPRO_RESULT from that."
        )
    if m.get("hf_model"):
        model_desc = f"the HuggingFace model '{m['hf_model']}'"
    else:
        model_desc = f"the model '{m['model']}' from the torch.hub repository '{m['repo']}'"
    return (
        f"Reproduce the published {metric} (in percent) of {model_desc} "
        f"on the {m['dataset']['name']}. Use the machine-readable metric id "
        f"'{metric_id}' in REPRO_RESULT."
    )


def _require_docker_image(image: str) -> None:
    """Fail early with a buildable fix if the pre-provisioned env image is absent
    (it is NOT created by the agent — see docker/mmpretrain.Dockerfile)."""
    import subprocess

    have = subprocess.run(["docker", "image", "inspect", image],
                          capture_output=True).returncode == 0
    if not have:
        raise SystemExit(
            f"Docker image '{image}' not found. Build it (once):\n"
            f"  docker build --platform linux/amd64 "
            f"-f docker/mmpretrain.Dockerfile -t {image} .")


def _provision_docker(m: dict, workdir: Path):
    """Provision a DockerSession for a clone-and-navigate oracle, two-phase
    network: copy the repo into the mounted workdir, download checkpoint +
    dataset ONLINE, then cut the network so the agent's eval runs OFFLINE."""
    from exec.docker_session import DockerSession

    _require_docker_image(m["docker_image"])
    repo_dst = workdir / Path(m["local_repo"]).name
    shutil.copytree(ROOT / m["local_repo"], repo_dst, ignore=shutil.ignore_patterns(".git"))
    session = DockerSession(workdir, image=m["docker_image"], mem="6g", cpus=6.0,
                            default_timeout=5400)  # emulated CPU eval is slow

    ckpt_url = m["checkpoint"]["source"].split()[0]
    repo = Path(m["local_repo"]).name
    # Provision phase (online): checkpoint + CIFAR-10 onto disk where the offline
    # run will find them (mmpretrain's CIFAR data_root is <cwd>/data/cifar10).
    session.shell(f"python -c \"import urllib.request as u; "
                  f"u.urlretrieve('{ckpt_url}', '/workspace/ckpt.pth')\"", timeout=600)
    session.shell(
        f"mkdir -p /workspace/{repo}/data/cifar10 && cd /workspace/{repo}/data/cifar10 && "
        f"python -c \"import urllib.request as u; u.urlretrieve("
        f"'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 'cifar-10-python.tar.gz')\" && "
        f"tar xzf cifar-10-python.tar.gz", timeout=600)
    # The config's data_root is '<cwd>/data/cifar10'; expose it from the repo root
    # too, so the eval finds CIFAR whether the agent runs from /workspace or the
    # repo dir. (A provisioning convenience — not a navigation hint.)
    session.shell(f"ln -sfn /workspace/{repo}/data /workspace/data")
    session.go_offline()  # Execution phase: no network — no exfiltration
    return session


def reproduce(manifest_path: str, use_tools: bool = True) -> dict:
    """Run the full agent → verify pipeline on one manifest; return structured result."""
    m = yaml.safe_load((ROOT / manifest_path).read_text())
    expected, tol = float(m["target"]["expected"]), float(m["target"]["tolerance"])
    name = Path(manifest_path).stem
    task = build_task(m)

    workdir = ROOT / f"workspaces/{name}"
    # A prior run's audit files contain the private verdict. Start clean so the
    # next Agent cannot discover the expected value from stale workspace state.
    shutil.rmtree(workdir, ignore_errors=True)

    docker = m.get("backend") == "docker"
    if docker:
        workdir.mkdir(parents=True, exist_ok=True)
        session = _provision_docker(m, workdir)
        max_steps = 16
    else:
        session = Session(workdir, venv_python=REPRO_PY, default_timeout=400)
        max_steps = 20
    try:
        result = run_agent(task, session, ChatLLM(), max_steps=max_steps, use_tools=use_tools)
    finally:
        if docker:
            session.close()

    v = verify_run(
        session.transcript,
        session.workdir,
        expected=expected,
        tolerance=tol,
        metric=m["target"]["metric"],
        expected_num_examples=int(m["dataset"]["num_examples"]),
    )
    stages = {
        "repo_inspected": len(session.transcript) > 0,
        "evaluation_completed": v.evidence_line is not None,
        "metric_extracted": v.actual is not None,
        "claim_matched": v.match,
    }
    scripts = sorted(session.workdir.glob("*.py"), key=lambda p: p.stat().st_mtime)
    eval_script = scripts[-1].read_text(errors="replace") if scripts else ""

    output = {
        "name": name, "task": task, "protocol": "tool_calls" if use_tools else "text",
        "stages": stages, "verdict": v.as_dict(),
        "steps": result.steps, "errors": result.errors,
        "usage": result.usage, "peak_ctx_tokens": result.peak_ctx_tokens,
        "commands": [r.command.splitlines()[0][:100] for r in session.transcript],
        "eval_script": eval_script,
    }
    (session.workdir / "result.json").write_text(json.dumps(output, indent=2))
    (session.workdir / "commands.sh").write_text(session.replay_script() + "\n")
    with (session.workdir / "transcript.jsonl").open("w") as f:
        for message in result.transcript:
            f.write(json.dumps(message) + "\n")
    return output


def main() -> None:
    argv = sys.argv[1:]
    use_tools = "--no-fc" not in argv
    paths = [a for a in argv if not a.startswith("--")]
    manifest = paths[0] if paths else "evals/benchmark/cifar10_resnet20.yaml"
    r = reproduce(manifest, use_tools=use_tools)
    print(f"\n========== {r['name']} ({r['protocol']}) ==========")
    print(f"task: {r['task']}")
    print(f"steps={r['steps']} errors={r['errors']}")
    if r["usage"]:
        u = r["usage"]
        print(f"tokens: {u['prompt_tokens']} in ({u['cache_hit_tokens']} cached) + "
              f"{u['completion_tokens']} out over {u['llm_calls']} calls = "
              f"¥{u['cost_yuan']}  (peak ctx {r['peak_ctx_tokens']} tok)")
    for k, ok in r["stages"].items():
        print(f"  [{'x' if ok else ' '}] {k}")
    print(f"verdict: {r['verdict']}")
    for i, c in enumerate(r["commands"], 1):
        print(f"  {i}. {c}")


if __name__ == "__main__":
    main()
