"""環境変数と .env から読み込むアプリケーション設定。

機密情報（APIキー、Ollama URL など）はサーバー側のみで使用し、
静的ファイルやフロントエンドには渡さないこと。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(_APP_ROOT / ".env", override=False)
load_dotenv(_REPO_ROOT / ".env", override=False)


def _env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value.strip()


_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_DEFAULT_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

# 公開 CDN（秘密情報ではないが、ミラー等のため .env で上書き可能）
FACE_LANDMARKER_MODEL_URL: str = (
    _env("FACE_LANDMARKER_MODEL_URL", _DEFAULT_MODEL_URL) or _DEFAULT_MODEL_URL
)
FACE_LANDMARKER_MODEL_SHA256: str = (
    _env("FACE_LANDMARKER_MODEL_SHA256", _DEFAULT_MODEL_SHA256) or _DEFAULT_MODEL_SHA256
)

_cache_dir = _env("FACE_LANDMARKER_MODEL_CACHE_DIR")
FACE_LANDMARKER_MODEL_CACHE_DIR: Path = (
    Path(_cache_dir).expanduser()
    if _cache_dir
    else Path.home() / ".cache" / "ai_focus_tracker"
)

# 以下はバックエンド専用（フロントエンドの JS/HTML に埋め込まない）
OLLAMA_BASE_URL: str = _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "http://127.0.0.1:11434"
OLLAMA_MODEL: str = _env("OLLAMA_MODEL", "qwen2.5:7b") or "qwen2.5:7b"

FLASK_HOST: str = _env("FLASK_HOST", "127.0.0.1") or "127.0.0.1"
FLASK_PORT: int = int(_env("FLASK_PORT", "5000") or "5000")
FLASK_DEBUG: bool = (_env("FLASK_DEBUG", "false") or "false").lower() in (
    "1",
    "true",
    "yes",
)
