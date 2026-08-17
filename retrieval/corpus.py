"""Load a repository as a chunk-level retrieval corpus.

Python files are split at top-level functions and classes, then long sections
are split again every fixed number of lines. Other navigable files use fixed
line chunks. Each document keeps its file path and source range, so retrieval
can find a symbol deep in a file without losing the file navigation cue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

EXTS = {".py", ".md", ".yml", ".yaml", ".txt", ".sh", ".cfg", ".rst"}
SKIP_DIRS = {".git", "__pycache__", ".github", "node_modules"}
MAX_LINES_PER_CHUNK = 100
_PYTHON_BOUNDARY = re.compile(r"^(?:async\s+def|def|class)\s+")


@dataclass
class Doc:
    path: str
    start_line: int
    end_line: int
    text: str

    @property
    def key(self) -> tuple[str, int, int]:
        return self.path, self.start_line, self.end_line


def _split_ranges(lines: list[str], suffix: str) -> list[tuple[int, int]]:
    """Return one-based source ranges, preferring Python definition boundaries."""
    if not lines:
        return [(1, 1)]

    if suffix == ".py":
        starts = [
            0,
            *(index for index, line in enumerate(lines) if _PYTHON_BOUNDARY.match(line)),
        ]
        starts = list(dict.fromkeys(starts))
        sections = zip(starts, [*starts[1:], len(lines)])
    else:
        sections = [(0, len(lines))]

    ranges: list[tuple[int, int]] = []
    for start, end in sections:
        for chunk_start in range(start, end, MAX_LINES_PER_CHUNK):
            chunk_end = min(chunk_start + MAX_LINES_PER_CHUNK, end)
            if chunk_start < chunk_end:
                ranges.append((chunk_start + 1, chunk_end))
    return ranges


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
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for start_line, end_line in _split_ranges(lines, f.suffix.lower()):
            chunk = "\n".join(lines[start_line - 1 : end_line])
            docs.append(
                Doc(
                    path=rel,
                    start_line=start_line,
                    end_line=end_line,
                    text=f"{rel}\n# Lines {start_line}-{end_line}\n\n{chunk}",
                )
            )
    return docs
