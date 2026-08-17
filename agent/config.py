"""Configuration. Reads .env once at import."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# --- Chat LLM (OpenAI-compatible). Falls back to the old DashScope vars so
#     existing local .env files keep working. ---
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("CHAT_MODEL", "deepseek-v4-flash")
LLM_THINKING = os.getenv("LLM_THINKING", "disabled").strip().lower()
if LLM_THINKING not in {"enabled", "disabled"}:
    raise ValueError("LLM_THINKING must be 'enabled' or 'disabled'")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
if LLM_TIMEOUT_SECONDS <= 0:
    raise ValueError("LLM_TIMEOUT_SECONDS must be positive")
if LLM_MAX_RETRIES < 0:
    raise ValueError("LLM_MAX_RETRIES must be non-negative")

# --- Cost accounting: yuan per 1M tokens. Defaults = deepseek-chat list price;
#     override in .env when the provider or its pricing changes. ---
PRICE_INPUT_MISS = float(os.getenv("PRICE_INPUT_MISS", "2.0"))   # input, cache miss
PRICE_INPUT_HIT = float(os.getenv("PRICE_INPUT_HIT", "0.2"))     # input, cache hit
PRICE_OUTPUT = float(os.getenv("PRICE_OUTPUT", "3.0"))           # output
