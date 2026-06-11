"""Embedder for repo navigation — DashScope text-embedding-v4 (1024-dim).

The LLM is DeepSeek (no embedding API), so embeddings come from DashScope; the
two providers are independent. Batched at 10 (DashScope's per-request cap).
"""

from __future__ import annotations


class DashScopeEmbedder:
    def __init__(self) -> None:
        from langchain_openai import OpenAIEmbeddings

        from agent.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, EMBEDDING_MODEL

        self.dim = 1024
        self._batch = 10  # DashScope caps embedding batch at 10/request
        self._emb = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            check_embedding_ctx_length=False,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            out.extend(self._emb.embed_documents(texts[i : i + self._batch]))
        return out
