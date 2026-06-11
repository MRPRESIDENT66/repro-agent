"""Load a repo as a retrieval corpus — one document per navigable file.

For repo navigation we retrieve whole FILES (the agent wants to land on the
right config / entry script), so each file is a document: its path plus a head
of its content. Paths carry a lot of signal in real repos
(``configs/resnet/resnet18_8xb16_cifar10.py``), so they're part of the document.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EXTS = {".py", ".md", ".yml", ".yaml", ".txt", ".sh", ".cfg", ".rst"}
SKIP_DIRS = {".git", "__pycache__", ".github", "node_modules"}
HEAD_CHARS = 2000  # how much of each file goes into the document


@dataclass
class Doc:
    path: str          # repo-relative path
    text: str          # "path\n\n<head of content>"


def load_corpus(repo_root: str | Path) -> list[Doc]:
    root = Path(repo_root)
    docs: list[Doc] = []
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in EXTS:
            continue
        if any(part in SKIP_DIRS for part in f.parts):
            continue
        rel = str(f.relative_to(root))
        try:
            content = f.read_text(encoding="utf-8", errors="replace")[:HEAD_CHARS]
        except Exception:
            content = ""
        docs.append(Doc(path=rel, text=f"{rel}\n\n{content}"))
    return docs
