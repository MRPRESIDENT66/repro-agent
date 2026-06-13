from agent.llm import ScriptedLLM
from retrieval.search import relevant_snippet, search_repo


def test_search_repo_excludes_generated_artifacts(tmp_path) -> None:
    (tmp_path / "source.py").write_text("official evaluator implementation")
    (tmp_path / "eval_ebo.py").write_text("generated evaluator implementation")
    llm = ScriptedLLM(["eval_ebo.py\nsource.py"])

    result = search_repo(
        "evaluator implementation",
        tmp_path,
        llm,
        exclude_paths={"eval_ebo.py"},
    )

    assert "source.py" in result
    assert "eval_ebo.py" not in result


def test_search_repo_keeps_exact_path_ahead_of_files_that_mention_it(tmp_path) -> None:
    exact = tmp_path / "scripts" / "eval_ood.py"
    exact.parent.mkdir()
    exact.write_text("official evaluator")
    wrapper = tmp_path / "run_eval.sh"
    wrapper.write_text("python scripts/eval_ood.py\n" * 20)
    llm = ScriptedLLM(["run_eval.sh"])

    result = search_repo("scripts/eval_ood.py", tmp_path, llm)

    paths = [line.split("  —", 1)[0].strip() for line in result.splitlines()[1:]]
    assert paths[0] == "scripts/eval_ood.py"


def test_search_repo_uses_traceback_context_for_ranking(tmp_path) -> None:
    target = tmp_path / "openood" / "datasets" / "imglist_dataset.py"
    target.parent.mkdir(parents=True)
    target.write_text("class ImglistDataset: pass")
    (tmp_path / "notes.md").write_text("fix constructor error " * 20)
    llm = ScriptedLLM(["notes.md"])

    result = search_repo(
        "fix constructor error",
        tmp_path,
        llm,
        context='File "/workspace/openood/datasets/imglist_dataset.py", line 20',
    )

    assert result.splitlines()[1].lstrip().startswith(
        "openood/datasets/imglist_dataset.py"
    )


def test_relevant_snippet_centers_late_symbol(tmp_path) -> None:
    source = tmp_path / "large.py"
    source.write_text("\n".join(["irrelevant = 1"] * 400 + [
        "class ImglistDataset:",
        "    def __init__(self, data_aux_preprocessor):",
        "        self.aux = data_aux_preprocessor",
    ]))

    snippet = relevant_snippet(source, "ImglistDataset __init__ data_aux_preprocessor")

    assert "data_aux_preprocessor" in snippet
    assert "# Lines " in snippet
