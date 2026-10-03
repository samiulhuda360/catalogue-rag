"""Settings, read from environment variables (and a .env file in the project root)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv is optional
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


def _default_parsed() -> Path:
    """Your own parsed catalogues if there are any, otherwise the bundled fictional sample."""
    own = ROOT / "data" / "parsed_documents"
    return own if own.exists() and any(own.glob("*.md")) else ROOT / "examples" / "parsed"


@dataclass(frozen=True)
class Settings:
    # Where things live
    documents_dir: Path = field(default_factory=lambda: Path(_env("CATALOGUE_DOCUMENTS_DIR", str(ROOT / "data" / "documents"))))
    parsed_dir: Path = field(default_factory=lambda: Path(_env("CATALOGUE_PARSED_DIR", str(_default_parsed()))))
    index_dir: Path = field(default_factory=lambda: Path(_env("CATALOGUE_INDEX_DIR", str(ROOT / "index"))))

    # Language model: any OpenAI-compatible endpoint. OpenRouter by default; LLM_BASE_URL points it at a
    # self-hosted server instead (vLLM, TGI, Ollama, LM Studio), which usually needs no key.
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", _env("OPENROUTER_API_KEY", "")))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", _env("OPENROUTER_MODEL", "qwen/qwen3-235b-a22b-2507")))
    llm_temperature: float = 0.0
    llm_max_tokens: int = 700

    # PDF parsing (LlamaParse); only needed to add new PDFs
    llama_cloud_api_key: str = field(default_factory=lambda: _env("LLAMA_CLOUD_API_KEY", ""))

    # Retrieval
    chunk_chars: int = 1200
    top_k: int = 8
    candidates: int = 40  # per retriever, before fusion
    rrf_k: int = 60
    bm25_weight: float = float(os.getenv("CATALOGUE_BM25_WEIGHT") or 1.5)  # chosen from eval/sweep_weights.py

    # Adding catalogues through the API / UI: switched off unless an admin token is set
    admin_token: str = field(default_factory=lambda: _env("CATALOGUE_ADMIN_TOKEN", ""))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("CATALOGUE_MAX_UPLOAD_MB") or 50))


def settings() -> Settings:
    return Settings()
