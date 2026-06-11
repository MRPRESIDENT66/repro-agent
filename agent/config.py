"""Configuration. Reads .env once at import."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# DashScope (Qwen), OpenAI-compatible endpoint.
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")
