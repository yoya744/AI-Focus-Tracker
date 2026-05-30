"""Ollama API クライアント。接続先は環境変数のみ（ハードコードしない）。"""

from __future__ import annotations

import requests

from focus_tracker.config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL


def ollama_generate(prompt: str, *, model: str | None = None) -> str:
    """Ollama /api/generate を呼び出し、応答テキストを返す。"""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("response", ""))
