"""Configuration. Reads .env once at import.

LLM and embedding providers are kept SEPARATE on purpose: the chat LLM is
DeepSeek (no embedding API), while embeddings (used from M3's RAG) come from
DashScope. They are independent — mixing providers is fine.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# --- Chat LLM (DeepSeek, OpenAI-compatible). Falls back to the old DashScope
#     vars so nothing breaks if only those are set. ---
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("CHAT_MODEL", "deepseek-chat")

# --- Cost accounting: yuan per 1M tokens. Defaults = deepseek-chat list price;
#     override in .env when the provider or its pricing changes. ---
PRICE_INPUT_MISS = float(os.getenv("PRICE_INPUT_MISS", "2.0"))   # input, cache miss
PRICE_INPUT_HIT = float(os.getenv("PRICE_INPUT_HIT", "0.2"))     # input, cache hit
PRICE_OUTPUT = float(os.getenv("PRICE_OUTPUT", "3.0"))           # output

# --- Embeddings (DashScope text-embedding-v4) — used from M3 (RAG), not yet. ---
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
